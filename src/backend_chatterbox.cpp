// Family "chatterbox": Resemble AI's Chatterbox (English) in the layout of build_chatterbox.py.
// A model without per-frame codes for the engine (the "direct" mode): every piece of text goes T3 (speech tokens, CFG) →
// S3Gen (flow matching + HiFT, the CosyVoice2 networks) at once, as ChatterboxTTS.generate does for its whole input.
// The official code takes the whole text in one go; here a segment is cut into groups of sentences of up to ~300 characters
// (option "max_chars") so long texts stay within what the model was trained on. Voices: cloned from a recording without its
// text (the speaker embedding, the first 6 s of speech tokens for T3, the first 10 s for S3Gen), or the built-in voice
// ("default", from conds.pt). Options: exaggeration, cfg_weight, temperature, repetition_penalty, min_p, top_p, max_new_tokens,
// max_chars. cfg_weight 0 runs T3 once without guidance (the official code needs cfg_weight > 0).
#include <algorithm>
#include <cmath>
#include <cstring>

#include "backend.hpp"
#include "gpu.hpp"
#include "chatterbox.hpp"
#include "cosyvoice.hpp"
#include "family.hpp"
#include "nn.hpp"

namespace vox {
namespace {

std::vector<int32_t> ints(const Tensor &t) {
    if (t.dtype != DType::I32) fail(VOX_ERR_ARG, "Chatterbox: token features must be I32");
    std::vector<int32_t> v(t.bytes.size() / 4);
    std::memcpy(v.data(), t.bytes.data(), t.bytes.size());
    return v;
}
std::vector<float> floats(const Tensor &t) {
    if (t.dtype != DType::F32) fail(VOX_ERR_ARG, "Chatterbox: features must be F32");
    std::vector<float> v(t.bytes.size() / 4);
    std::memcpy(v.data(), t.bytes.data(), t.bytes.size());
    return v;
}

// sentences (. ! ? and the punc_norm enders, closing quotes kept) grouped up to max_chars; complete = no more text will come
std::vector<std::string> groups(const std::string &text, size_t max_chars, bool complete) {
    std::vector<std::string> sent;
    size_t st = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '\n' || c == '.' || c == '!' || c == '?') {
            size_t j = i + 1;
            while (j < text.size() && (text[j] == '.' || text[j] == '!' || text[j] == '?' || text[j] == '"' || text[j] == '\'')) ++j;
            if (j < text.size() && !(text[j] == ' ' || text[j] == '\n' || text[j] == '\t')) { i = j - 1; continue; }   // 3.5, e.g.
            if (j >= text.size() && !complete) break;   // the next character decides
            sent.push_back(text.substr(st, j - st));
            st = j;
            i = j - 1;
        }
    }
    std::string rest = text.substr(std::min(st, text.size()));
    if (complete) sent.push_back(rest);
    std::vector<std::string> out;
    std::string cur;
    for (const std::string &s : sent) {
        if (!cur.empty() && cur.size() + s.size() > max_chars) { out.push_back(cur); cur.clear(); }
        cur += s;
    }
    if (complete) out.push_back(cur);
    std::vector<std::string> res;
    for (auto &g : out) {
        bool blank = true;
        for (char c : g) if (!(c == ' ' || c == '\n' || c == '\t' || c == '\r')) { blank = false; break; }
        if (!blank) res.push_back(g);
    }
    return res;
}

struct CbState : GenState {
    std::vector<int32_t> s3_tokens, t3_tokens;
    std::vector<float> s3_mel, xvec, ve;
    T3Sampling sp;
    float exaggeration = 0.5f;
    std::vector<float> cond;
    size_t max_chars = 300;
    uint64_t seed = 1;
    std::string text;
    size_t emitted = 0;
    std::vector<std::string> queue;
    uint32_t piece_counter = 0;
    bool sealed = false;
    std::map<std::string, Tensor> injected;
    std::vector<std::string> done;
};

class CbBackend final : public Backend {
public:
    CbBackend(std::shared_ptr<TmodFile> f, const ResourcePlan &plan) : file_(f), s3gen_(f), t3_(f) {
        if (plan.uses_gpu() && plan.place_of("s3gen") == Place::Gpu) {
            gpu_ = s3gen_.enable_gpu(plan.release_host_weights);
            if (!gpu_ && plan.strict) fail(VOX_ERR_DEVICE, "GPU strict mode: the flow estimator did not go to the GPU (" + gpu_stats().last_error + ")");
        }
        if (plan.uses_gpu() && plan.place_of("t3") == Place::Gpu) {
            gpu_t3_ = t3_.enable_gpu(plan.release_host_weights);
            if (!gpu_t3_ && plan.strict) fail(VOX_ERR_DEVICE, "GPU strict mode: T3 did not go to the GPU (" + gpu_stats().last_error + ")");
        }
        info_.sample_rate = s3gen_.cfg().sample_rate;
        info_.samples_per_frame = 480;
        info_.num_codebooks = 1;
        info_.codebook_size = 1;
    }
    std::string name() const override { return gpu_ || gpu_t3_ ? "vulkan-chatterbox (" + gpu_device_name() + ")" : "cpu-chatterbox"; }
    const ModelInfo &info() const override { return info_; }
    std::string fingerprint() const override {
        const json &m = file_->manifest();
        return "cpu-chatterbox-" + hex64(fnv1a64(m.at("model").value("weights_sha256", "") + m.value("precision", json::object()).dump())).substr(0, 12);
    }
    static Capabilities caps(const TmodFile &f) {
        Capabilities c;
        const json &cv = f.manifest().at("cosyvoice");
        c.clone_xvector = cv.value("campplus", false) && cv.contains("s3tokenizer") && f.has("ve.lstm.weight_ih_l0");
        c.named_speakers = f.has("voice.default.s3_tokens");
        if (c.named_speakers) c.speakers = {"default"};
        c.streaming_text = true;
        c.text_in = "chunk";
        c.audio_out = "chunks";
        c.stream_exact = true;
        c.languages = {"en"};
        return c;
    }
    Capabilities capabilities() const override { return caps(*file_); }
    SessionCost session_cost(const VoiceData &, const SessionOptions &) const override { return {512u << 20, 0}; }
    json stats() const override {
        return {{"t3_ms", t3_.ms_gen}, {"flow_ms", s3gen_.ms_flow}, {"hift_ms", s3gen_.ms_hift}, {"weights_bytes", s3gen_.weight_bytes()}, {"device", gpu_ || gpu_t3_ ? "vulkan" : "cpu"},
                {"flow_device", gpu_ ? "vulkan" : "cpu"}, {"t3_device", gpu_t3_ ? "vulkan" : "cpu"}, {"gemm", gpu_ ? gpu_gemm_choice() : ""}};
    }

    std::vector<int32_t> tokenize(const std::string &utf8) override { return t3_.text_tokens(utf8); }
    std::string phonemize(const std::string &text, const std::string &) override {
        json pieces = json::array(), toks = json::array();
        for (const auto &p : groups(text, 300, true)) {
            pieces.push_back(ChatterboxT3::punc_norm(p));
            toks.push_back(t3_.text_tokens(p));
        }
        return json{{"pieces", pieces}, {"tokens", toks}}.dump();
    }
    std::vector<float> speaker_embedding(const float *, size_t, uint32_t) override { no_codes(); return {}; }
    std::vector<int16_t> encode_codes(const float *, size_t, uint32_t, int64_t &) override { no_codes(); return {}; }
    void append_text(GenState &, const std::vector<int32_t> &) override { no_codes(); }
    bool needs_text(const GenState &, bool) const override { return false; }
    void talker_logits(GenState &, bool, std::vector<float> &) override { no_codes(); }
    void accept_code0(GenState &, int32_t) override { no_codes(); }
    void predictor_logits(GenState &, uint32_t, std::vector<float> &) override { no_codes(); }
    void accept_code(GenState &, uint32_t, int32_t) override { no_codes(); }
    void end_frame(GenState &) override { no_codes(); }
    void decode(GenState &, const int16_t *, size_t, std::vector<float> &) override { no_codes(); }
    void flush(GenState &, std::vector<float> &) override {}
    bool direct() const override { return true; }

    // ChatterboxTTS.prepare_conditionals: 24 kHz (first 10 s) → S3Gen prompt (mel, speech tokens, CAM++ x-vector);
    // 16 kHz → T3 prompt tokens (first 6 s, at most 150) and the VoiceEncoder embedding
    void prepare_voice(VoiceData &v, const float *pcm, size_t n, uint32_t sr) override {
        if (!caps(*file_).clone_xvector) fail(VOX_ERR_UNSUPPORTED, "Chatterbox: this .tmod cannot clone voices (no speaker encoders / tokenizer)");
        if (sr < 16000) fail(VOX_ERR_ARG, "Chatterbox: the recording must be sampled at 16 kHz or more");
        std::vector<float> w24 = nn::resample_ta(pcm, n, sr, 24000);
        const std::vector<float> w16full = nn::resample_ta(w24.data(), w24.size(), 24000, 16000);
        if (w24.size() > size_t(10 * 24000)) w24.resize(size_t(10 * 24000));
        int fm = 0, ff = 0, fw = 0;
        const std::vector<float> mel = s3gen_.mel24(w24.data(), w24.size(), fm);
        const std::vector<float> w16 = nn::resample_ta(w24.data(), w24.size(), 24000, 16000);
        const std::vector<float> fb = s3gen_.fbank16(w16.data(), w16.size(), ff);
        const std::vector<float> xv = s3gen_.xvector(fb, ff);
        const std::vector<float> wm = s3gen_.whisper_mel(w16.data(), w16.size(), fw);   // (fw is set here: not in one expression)
        std::vector<int32_t> tk = s3gen_.speech_tokens(wm, fw);
        if (fm != 2 * int(tk.size()) && int(tk.size()) > fm / 2) tk.resize(size_t(fm / 2));
        const std::vector<int32_t> t3t = t3_prompt_tokens(w16full.data(), w16full.size());
        const std::vector<float> ve = t3_.voice_embedding(w16full.data(), w16full.size());
        v.features.clear();
        v.features["feat.cb_s3_tokens"] = Tensor::from(DType::I32, {int64_t(tk.size())}, tk);
        v.features["feat.cb_s3_mel"] = Tensor::from(DType::F32, {int64_t(fm), int64_t(s3gen_.cfg().mel_dim)}, mel);
        v.features["feat.cb_xvec"] = Tensor::from(DType::F32, {int64_t(xv.size())}, xv);
        v.features["feat.cb_t3_tokens"] = Tensor::from(DType::I32, {int64_t(t3t.size())}, t3t);
        v.features["feat.cb_ve"] = Tensor::from(DType::F32, {int64_t(ve.size())}, ve);
    }

    std::unique_ptr<GenState> begin(const VoiceData &voice, const std::string &) override {
        auto st = std::make_unique<CbState>();
        auto get = [&](const char *k) -> const Tensor & {
            auto it = voice.features.find(k);
            if (it == voice.features.end()) fail(VOX_ERR_ARG, "this voice has no Chatterbox features: make it from a recording with this model (voice clone)");
            return it->second;
        };
        if (voice.mode == VOX_VOICE_SPEAKER) {
            if (!voice.speaker.empty() && voice.speaker != "default") fail(VOX_ERR_ARG, "Chatterbox: unknown speaker '" + voice.speaker + "' (the built-in one is 'default')");
            const TmodFile &F = *file_;
            if (!F.has("voice.default.s3_tokens")) fail(VOX_ERR_ARG, "Chatterbox: this .tmod has no built-in voice (conds.pt)");
            auto ti = [&](const char *k) { const TensorView &t = F.get(k); std::vector<int32_t> v(size_t(t.numel())); std::memcpy(v.data(), t.data, v.size() * 4); return v; };
            st->s3_tokens = ti("voice.default.s3_tokens");
            st->t3_tokens = ti("voice.default.t3_tokens");
            st->s3_mel = nn::to_float(F.get("voice.default.s3_mel"));
            st->xvec = nn::to_float(F.get("voice.default.xvec"));
            st->ve = nn::to_float(F.get("voice.default.ve"));
            const json &m = F.manifest().at("chatterbox");
            st->exaggeration = m.value("default_exaggeration", 0.5f);
        } else {
            st->s3_tokens = ints(get("feat.cb_s3_tokens"));
            st->t3_tokens = ints(get("feat.cb_t3_tokens"));
            st->s3_mel = floats(get("feat.cb_s3_mel"));
            st->xvec = floats(get("feat.cb_xvec"));
            st->ve = floats(get("feat.cb_ve"));
        }
        return st;
    }

    void configure(GenState &g, const SessionOptions &o) override {
        CbState &s = static_cast<CbState &>(g);
        s.seed = o.seed;
        const json &op = o.options;
        auto num = [&](const char *k, float &v) { if (op.contains(k) && op[k].is_number()) v = op[k].get<float>(); };
        num("exaggeration", s.exaggeration);
        num("cfg_weight", s.sp.cfg_weight);
        num("temperature", s.sp.temperature);
        num("repetition_penalty", s.sp.repetition_penalty);
        num("min_p", s.sp.min_p);
        num("top_p", s.sp.top_p);
        if (s.sp.temperature <= 0.f) fail(VOX_ERR_ARG, "Chatterbox: temperature must be positive");
        if (op.contains("max_chars") && op["max_chars"].is_number_integer()) s.max_chars = size_t(std::max(20, op["max_chars"].get<int>()));
        if (op.contains("max_new_tokens") && op["max_new_tokens"].is_number_integer()) s.sp.max_new_tokens = std::max(1, op["max_new_tokens"].get<int>());
        if (op.contains("noise_file") && op["noise_file"].is_string()) s.injected = safetensors_load(op["noise_file"].get<std::string>()).tensors;
        s.cond = t3_.cond_embeds(s.ve, s.t3_tokens, s.exaggeration);
    }

    void direct_text(GenState &g, const std::string &utf8) override { static_cast<CbState &>(g).text += utf8; }
    void direct_seal(GenState &g) override { static_cast<CbState &>(g).sealed = true; }

    bool direct_step(GenState &g, bool text_ended, std::vector<float> &pcm) override {
        CbState &s = static_cast<CbState &>(g);
        for (;;) {
            if (!s.queue.empty()) {
                const std::string piece = s.queue.front();
                s.queue.erase(s.queue.begin());
                const size_t before = pcm.size();
                synth(s, piece, pcm);
                if (pcm.size() > before) return false;
                continue;   // a piece without speech (nothing sampled): on to the next
            }
            const bool complete = text_ended || s.sealed;
            std::vector<std::string> gs = groups(s.text, s.max_chars, complete);
            if (!complete && !gs.empty()) gs.pop_back();   // the last group can still grow
            if (s.emitted < gs.size()) {
                for (; s.emitted < gs.size(); ++s.emitted) s.queue.push_back(gs[s.emitted]);
                continue;
            }
            if (s.sealed && !text_ended) {
                s.text.clear();
                s.emitted = 0;
                s.sealed = false;
                return false;
            }
            return text_ended;
        }
    }

    void synth(CbState &s, const std::string &piece, std::vector<float> &pcm) {
        const uint32_t idx = s.piece_counter++;
        const std::string key = std::to_string(idx);
        const std::vector<int32_t> text = t3_.text_tokens(piece);
        std::vector<int32_t> gen;
        auto it = s.injected.find(key + ".tokens");
        if (it != s.injected.end()) gen = ints(it->second);
        else gen = t3_.generate(s.cond, text, s.sp, s.seed, idx);
        // drop_invalid_tokens: after the last start-of-speech, before the end; then only real codes
        auto sos = std::find(gen.begin(), gen.end(), t3_.cfg().start_speech);
        if (sos != gen.end()) gen.erase(gen.begin(), sos + 1);
        auto eos = std::find(gen.begin(), gen.end(), t3_.cfg().stop_speech);
        if (eos != gen.end()) gen.erase(eos, gen.end());
        gen.erase(std::remove_if(gen.begin(), gen.end(), [&](int32_t t) { return t >= s3gen_.cfg().speech_tokens; }), gen.end());
        std::vector<float> mel;
        if (!gen.empty()) {
            std::vector<float> z0;
            auto zi = s.injected.find(key + ".flow_noise");
            if (zi != s.injected.end()) z0 = floats(zi->second);
            s3gen_.flow(s.s3_tokens, gen, s.s3_mel, s.xvec, mel, nullptr, s.seed, idx, z0.empty() ? nullptr : &z0);
        }
        const int M = s3gen_.cfg().mel_dim;
        const int T = int(mel.size() / size_t(M));
        CosyNoise nz;
        const CosyNoise *inj = nullptr;
        auto a = s.injected.find(key + ".rand_ini"), b = s.injected.find(key + ".sine_noise");
        if (a != s.injected.end() && b != s.injected.end()) { nz.rand_ini = floats(a->second); nz.sine_noise = floats(b->second); inj = &nz; }
        std::vector<float> wave;
        if (T > 0) s3gen_.hift(mel, T, s.seed, idx, inj, wave);
        // trim_fade: 20 ms of silence, then a 20 ms raised-cosine fade in
        const int nt = int(s3gen_.cfg().sample_rate / 50);
        for (int i = 0; i < 2 * nt && i < int(wave.size()); ++i) {
            float f = 0.f;
            if (i >= nt) {
                const int k = i - nt;
                const double lin = nt > 1 ? kPiD() * (1.0 - double(k) / double(nt - 1)) : 0.0;   // linspace(π, 0, n)
                f = float((std::cos(lin) + 1.0) / 2.0);
            }
            wave[size_t(i)] *= f;
        }
        pcm.insert(pcm.end(), wave.begin(), wave.end());
        s.done.push_back(piece);
    }

    json state_info(const GenState &g) const override {
        const CbState &s = static_cast<const CbState &>(g);
        return {{"pieces", s.done}, {"exaggeration", s.exaggeration}};
    }

    bool debug_check(const std::string &dump, double tol, std::string &report) override;

    // T3's speech prompt: s3_tokzr.forward([wav16[:6 s]], max_len=150) — the log-mel cut at 4 × 150 frames, then the tokenizer
    std::vector<int32_t> t3_prompt_tokens(const float *w16, size_t n) const {
        const size_t n6 = std::min(n, size_t(6 * 16000));
        int f6 = 0;
        std::vector<float> wm6 = s3gen_.whisper_mel(w16, n6, f6);
        const int maxf = 4 * t3_.cfg().prompt_len;
        if (f6 > maxf) {
            const int M = s3gen_.cfg().s3_mels;
            std::vector<float> cut(size_t(M) * size_t(maxf));
            for (int m = 0; m < M; ++m) std::memcpy(&cut[size_t(m) * maxf], &wm6[size_t(m) * f6], size_t(maxf) * 4);
            wm6.swap(cut);
            f6 = maxf;
        }
        return s3gen_.speech_tokens(wm6, f6);
    }

private:
    static double kPiD() { return 3.14159265358979323846; }
    [[noreturn]] void no_codes() const { fail(VOX_ERR_UNSUPPORTED, "Chatterbox does not use audio codes"); }
    std::shared_ptr<TmodFile> file_;
    bool gpu_ = false;      // the flow estimator runs on the GPU
    bool gpu_t3_ = false;   // T3 runs on the GPU
    CosyVoiceModel s3gen_;
    ChatterboxT3 t3_;
    ModelInfo info_;
};

bool CbBackend::debug_check(const std::string &dump, double tol, std::string &report) {
    const SafetensorsFile f = safetensors_load(dump);
    auto get = [&](const char *k) -> std::vector<float> {
        auto it = f.tensors.find(k);
        if (it == f.tensors.end()) return {};
        const Tensor &t = it->second;
        std::vector<float> v;
        if (t.dtype == DType::I32) { for (int32_t x : ints(t)) v.push_back(float(x)); return v; }
        return floats(t);
    };
    auto geti = [&](const char *k) { const auto v = get(k); return std::vector<int32_t>(v.begin(), v.end()); };
    auto meta = [&](const char *k) { auto it = f.metadata.find(k); return it == f.metadata.end() ? std::string() : it->second; };
    bool ok = true;
    std::string out;
    auto cmp = [&](const char *name, const std::vector<float> &a, const std::vector<float> &b, double t) {
        if (a.size() != b.size()) {
            out += std::string("  ") + name + ": size mismatch " + std::to_string(a.size()) + " vs " + std::to_string(b.size()) + "  FAIL\n";
            ok = false;
            return;
        }
        double mx = 0, ref = 0;
        for (size_t i = 0; i < a.size(); ++i) { mx = std::max(mx, std::fabs(double(a[i]) - double(b[i]))); ref = std::max(ref, std::fabs(double(b[i]))); }
        const double rel = mx / (ref + 1e-6);
        char line[220];
        std::snprintf(line, sizeof line, "  %-16s max|diff| %.3e  max|ref| %.3e  rel %.3e  %s\n", name, mx, ref, rel, rel <= t ? "ok" : "FAIL");
        out += line;
        if (rel > t) ok = false;
    };
    auto cmpi = [&](const char *name, const std::vector<int32_t> &a, const std::vector<int32_t> &b, double min_match) {
        size_t same = 0;
        for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) same += a[i] == b[i];
        const double frac = b.empty() ? (a.empty() ? 1.0 : 0.0) : double(same) / double(std::max(a.size(), b.size()));
        char line[220];
        std::snprintf(line, sizeof line, "  %-16s %zu / %zu equal (%zu vs %zu)  %s\n", name, same, std::max(a.size(), b.size()), a.size(), b.size(),
                      frac >= min_match ? "ok" : "FAIL");
        out += line;
        if (frac < min_match) ok = false;
    };
    const std::string text = meta("text");
    cmpi("text tokens", t3_.text_tokens(text), geti("text_tokens"), 1.0);
    const auto w16 = get("ref16");
    if (!w16.empty()) {
        cmp("voice encoder", t3_.voice_embedding(w16.data(), w16.size()), get("ve"), std::max(tol, 1e-3));
        cmpi("t3 prompt tokens", t3_prompt_tokens(w16.data(), w16.size()), geti("t3_prompt_tokens"), 1.0);
    }
    const float ex = std::stof(meta("exaggeration").empty() ? "0.5" : meta("exaggeration"));
    const auto cond = t3_.cond_embeds(get("ve"), geti("t3_prompt_tokens"), ex);
    cmp("t3 cond", cond, get("t3_cond"), tol);
    {
        T3Sampling sp;
        sp.cfg_weight = std::stof(meta("cfg_weight").empty() ? "0.5" : meta("cfg_weight"));
        const auto toks = geti("t3_tokens");   // the official sampled tokens, ending with [STOP_SPEECH]
        ChatterboxT3::Trace tr;
        t3_.generate(get("t3_cond"), geti("text_tokens"), sp, 1, 0, &toks, &tr);
        std::vector<float> a, b;
        const auto ref = get("t3_logits");
        if (tr.logits.size() == ref.size()) {
            const float mx = *std::max_element(ref.begin(), ref.end());
            for (size_t i = 0; i < ref.size(); ++i)
                if (ref[i] > mx - 30.f) { a.push_back(tr.logits[i]); b.push_back(ref[i]); }
        } else {
            a = tr.logits;
            b = ref;
        }
        cmp("t3 logits", a, b, std::max(tol, 1e-3));
    }
    {
        const auto s3t = geti("s3_prompt_tokens"), gen = geti("speech_tokens");
        const auto pm = get("s3_prompt_mel"), xv = get("xvec"), z0 = get("flow_noise");
        std::vector<float> mel;
        CosyVoiceModel::FlowTrace tr;
        s3gen_.flow(s3t, gen, pm, xv, mel, &tr, 1, 0, z0.empty() ? nullptr : &z0);
        cmp("flow mu", tr.mu, get("flow_mu"), std::max(tol, 1e-3));
        cmp("flow mel", mel, get("flow_mel"), std::max(tol, 2e-3));
        const auto rmel = get("flow_mel");
        CosyNoise nz;
        nz.rand_ini = get("rand_ini");
        nz.sine_noise = get("sine_noise");
        std::vector<float> wav;
        CosyVoiceModel::HiftTrace ht;
        s3gen_.hift(rmel, int(rmel.size() / 80), 1, 0, &nz, wav, &ht);
        cmp("hift f0", ht.f0, get("hift_f0"), std::max(tol, 1e-4));
        const int nt = 480;
        for (int i = 0; i < 2 * nt && i < int(wav.size()); ++i) {
            const double lin = kPiD() * (1.0 - double(std::max(0, i - nt)) / double(nt - 1));
            wav[size_t(i)] *= i < nt ? 0.f : float((std::cos(lin) + 1.0) / 2.0);
        }
        cmp("audio", wav, get("audio"), std::max(tol, 2e-3));
    }
    report = out;
    return ok;
}

class CbFamily final : public Family {
public:
    std::string name() const override { return "chatterbox"; }
    std::string description() const override { return "Chatterbox (Resemble AI): Llama T3 over speech tokens + S3Gen flow matching + HiFT; English, voice cloning, exaggeration"; }
    ModelFootprint footprint(const TmodFile &f) const override {
        const json &m = f.manifest();
        ModelFootprint fp;
        fp.family = name();
        fp.variant = m.at("model").value("variant", "");
        StageDesc t3;
        t3.name = "t3";
        t3.prefixes = {"t3."};
        t3.gpu_capable = true;   // the GPU transformer (gpu_vulkan.cpp), both CFG passes' KV caches in video memory
        t3.session_vram_bytes = 800u << 20;   // 2 × 30 layers × 1024 × (K + V) × f32 × ~1500 positions
        t3.bytes_per_frame = double(blob_bytes(f, t3.prefixes));
        t3.session_ram_bytes = 256u << 20;
        fp.stages.push_back(t3);
        StageDesc fl;
        fl.name = "s3gen";
        fl.prefixes = {"flow."};
        fl.gpu_capable = true;   // the flow estimator (gpu_f5.cpp: gpu_cosy_*); the encoder runs on the CPU
        fl.bytes_per_frame = double(blob_bytes(f, fl.prefixes)) * 20.0 / 50.0;
        fl.session_ram_bytes = 384u << 20;
        fl.session_vram_bytes = 96u << 20;
        fp.stages.push_back(fl);
        StageDesc hf;
        hf.name = "hift";
        hf.prefixes = {"hift."};
        hf.gpu_capable = false;
        hf.bytes_per_frame = double(blob_bytes(f, hf.prefixes)) / 50.0;
        fp.stages.push_back(hf);
        fp.always_ram_bytes = blob_bytes(f, {"camp.", "s3.", "front.", "ve.", "voice.", "tokenizer."}) + (32u << 20);
        return fp;
    }
    Capabilities capabilities(const TmodFile &f) const override { return CbBackend::caps(f); }
    std::unique_ptr<Backend> create(std::shared_ptr<TmodFile> f, const ResourcePlan &plan) const override {
        return std::make_unique<CbBackend>(std::move(f), plan);
    }
};

}  // namespace

std::shared_ptr<Family> make_chatterbox_family() { return std::make_shared<CbFamily>(); }

}  // namespace vox
