// Family "kokoro": Kokoro-82M and other StyleTTS2 + iSTFTNet checkpoints in the hexgrad/kokoro layout.
// A model without audio codes, run through the "direct" mode of Backend like F5-TTS: the text is split as
// kokoro.KPipeline does (lines, then pieces of at most 510 phonemes cut at the last sentence / clause / comma), every
// piece goes through the network at once, the pieces follow each other without crossfade.
// Text → phonemes: the built-in English G2P (misaki's lexicons, kokoro_g2p.cpp), or phonemes given directly
// (option "phonemes": true — the Python frontend does this with misaki itself).
// Voices: the packs stored in the .tmod are named speakers ("af_heart"; "af_heart,am_adam" averages packs);
// a voice may also carry its own pack (feature "feat.kokoro_pack", [rows][256]).
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "backend.hpp"
#include "family.hpp"
#include "kokoro.hpp"
#include "kokoro_g2p.hpp"

namespace vox {
namespace {

std::vector<std::string> split_lines(const std::string &text) {   // re.split(r'\n+', text.strip())
    std::vector<std::string> out;
    const std::string ws = " \t\n\r\f\v";
    const size_t a = text.find_first_not_of(ws);
    if (a == std::string::npos) return out;
    const std::string t = text.substr(a, text.find_last_not_of(ws) - a + 1);
    size_t i = 0;
    while (i <= t.size()) {
        size_t j = t.find('\n', i);
        if (j == std::string::npos) j = t.size();
        out.push_back(t.substr(i, j - i));
        while (j < t.size() && t[j] == '\n') ++j;
        i = j + (j == t.size() ? 1 : 0);
        if (j == t.size()) break;
    }
    return out;
}

struct KokoroState : GenState {
    std::string voice;                  // speaker spec, or empty when the voice carries its own pack
    std::vector<float> own_pack;        // [rows][256]
    int own_rows = 0;
    bool british = false, phonemes = false;
    float speed = 1.0f;
    uint64_t seed = 1;
    std::string text;
    size_t next_line = 0;               // lines of the current segment already turned into chunks
    std::vector<std::string> queue;     // phoneme chunks waiting for synthesis
    uint32_t chunk_counter = 0;
    bool sealed = false;
    std::map<std::string, Tensor> injected_noise;
    std::vector<std::string> done_chunks;   // phonemes actually synthesized (session info / tests)
};

class KokoroBackend final : public Backend {
public:
    KokoroBackend(std::shared_ptr<TmodFile> f, const ResourcePlan &) : file_(f), model_(f) {
        info_.sample_rate = model_.cfg().sample_rate;
        info_.samples_per_frame = 600;
        info_.num_codebooks = 1;
        info_.codebook_size = 1;
    }

    std::string name() const override { return "cpu-kokoro"; }
    const ModelInfo &info() const override { return info_; }
    std::string fingerprint() const override {
        const json &m = file_->manifest();
        return "cpu-kokoro-" + hex64(fnv1a64(m.at("model").value("weights_sha256", "") + m.value("precision", json::object()).dump())).substr(0, 12);
    }
    static Capabilities caps(const TmodFile &f) {
        Capabilities c;
        c.named_speakers = true;
        c.speed = true;
        c.streaming_text = true;
        c.text_in = "chunk";
        c.audio_out = "chunks";
        c.stream_exact = true;
        const json &k = f.manifest().at("kokoro");
        c.speakers = k.value("voices", std::vector<std::string>{});
        // the voice Kokoro's own examples speak with (hexgrad/Kokoro-82M), not the alphabetically first af_alloy
        for (const char *d : {"af_heart", "af_bella"})
            if (std::find(c.speakers.begin(), c.speakers.end(), d) != c.speakers.end()) { c.default_speaker = d; break; }
        c.languages = {"en-us", "en-gb"};
        return c;
    }
    Capabilities capabilities() const override { return caps(*file_); }
    SessionCost session_cost(const VoiceData &, const SessionOptions &) const override { return {256u << 20, 0}; }
    json stats() const override {
        return {{"prosody_ms", model_.ms_prosody()}, {"decoder_ms", model_.ms_decoder()}, {"weights_bytes", model_.weight_bytes()}, {"device", "cpu"}};
    }

    // --- not used by a model without codes ---
    std::vector<int32_t> tokenize(const std::string &utf8) override { return model_.phonemes_to_ids(utf8); }
    std::vector<float> speaker_embedding(const float *, size_t, uint32_t) override { no_clone(); return {}; }
    std::vector<int16_t> encode_codes(const float *, size_t, uint32_t, int64_t &) override { no_clone(); return {}; }
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
    void prepare_voice(VoiceData &, const float *, size_t, uint32_t) override { no_clone(); }

    std::unique_ptr<GenState> begin(const VoiceData &voice, const std::string &language) override {
        auto st = std::make_unique<KokoroState>();
        auto it = voice.features.find("feat.kokoro_pack");
        if (it != voice.features.end()) {
            const Tensor &t = it->second;
            if (t.shape.size() != 2 || t.shape[1] != 2 * model_.cfg().style_dim || t.dtype != DType::F32)
                fail(VOX_ERR_ARG, "Kokoro: the voice pack must be F32 [rows][" + std::to_string(2 * model_.cfg().style_dim) + "]");
            st->own_rows = int(t.shape[0]);
            st->own_pack.resize(t.bytes.size() / 4);
            std::memcpy(st->own_pack.data(), t.bytes.data(), t.bytes.size());
        } else {
            if (voice.speaker.empty()) fail(VOX_ERR_ARG, "Kokoro: pick a voice (speaker), e.g. " + (model_.voices().empty() ? std::string("af_heart") : model_.voices()[0]));
            if (!model_.has_voice(voice.speaker)) fail(VOX_ERR_ARG, "Kokoro: unknown voice '" + voice.speaker + "'");
            st->voice = voice.speaker;
        }
        const std::string first = st->voice.empty() ? voice.name : st->voice;
        st->british = !first.empty() && first[0] == 'b';
        if (language == "en-gb" || language == "british") st->british = true;
        if (language == "en-us" || language == "american") st->british = false;
        return st;
    }

    void configure(GenState &g, const SessionOptions &o) override {
        KokoroState &s = static_cast<KokoroState &>(g);
        s.seed = o.seed;
        const json &op = o.options;
        if (op.contains("speed") && op["speed"].is_number()) s.speed = op["speed"].get<float>();
        if (!(s.speed > 0.f && s.speed <= 4.f)) fail(VOX_ERR_ARG, "Kokoro: speed must be in (0, 4]");
        if (op.contains("phonemes") && op["phonemes"].is_boolean()) s.phonemes = op["phonemes"].get<bool>();
        if (op.contains("lang") && op["lang"].is_string()) {
            const std::string l = op["lang"].get<std::string>();
            if (l == "b" || l == "en-gb") s.british = true;
            else if (l == "a" || l == "en-us") s.british = false;
            else fail(VOX_ERR_UNSUPPORTED, "Kokoro: the built-in G2P speaks English (lang a / b); for other languages pass phonemes");
        }
        if (op.contains("noise_file") && op["noise_file"].is_string()) s.injected_noise = safetensors_load(op["noise_file"].get<std::string>()).tensors;
    }

    void direct_text(GenState &g, const std::string &utf8) override { static_cast<KokoroState &>(g).text += utf8; }
    void direct_seal(GenState &g) override { static_cast<KokoroState &>(g).sealed = true; }

    // One call = one piece of audio (or done / waiting for text): the engine treats a call without audio as "waiting for text".
    bool direct_step(GenState &g, bool text_ended, std::vector<float> &pcm) override {
        KokoroState &s = static_cast<KokoroState &>(g);
        for (;;) {
            if (!s.queue.empty()) {
                const std::string ps = s.queue.front();
                s.queue.erase(s.queue.begin());
                synth(s, ps, pcm);
                return false;
            }
            // complete lines: all but the last, unless the text ended / the segment is sealed / the text ends with "\n"
            const bool complete = text_ended || s.sealed || (!s.text.empty() && s.text.back() == '\n');
            const std::vector<std::string> lines = split_lines(s.text);
            const size_t avail = complete ? lines.size() : (lines.empty() ? 0 : lines.size() - 1);
            if (s.next_line < avail) {
                for (; s.next_line < avail; ++s.next_line) {
                    const auto chunks = to_chunks(s, lines[s.next_line]);
                    s.queue.insert(s.queue.end(), chunks.begin(), chunks.end());
                }
                continue;
            }
            if (s.sealed && !text_ended) {   // segment done; the next text starts a new one
                s.text.clear();
                s.next_line = 0;
                s.sealed = false;
                return false;
            }
            return text_ended;
        }
    }

    bool debug_check(const std::string &dump, double tol, std::string &report) override;

    std::string phonemize(const std::string &text, const std::string &lang) override {
        const bool british = lang == "en-gb" || lang == "b";
        if (!lang.empty() && !british && lang != "en-us" && lang != "a" && lang != "en")
            fail(VOX_ERR_UNSUPPORTED, "Kokoro: the built-in G2P speaks English (en-us, en-gb)");
        json chunks = json::array(), tokens = json::array();
        std::string all;
        for (const std::string &line : split_lines(text)) {
            const auto toks = g2p(british).tokens(line);
            for (const auto &t : toks) {
                tokens.push_back({{"text", t.text}, {"phonemes", t.phonemes}, {"whitespace", t.whitespace}});
                all += t.phonemes + t.whitespace;
            }
            for (const auto &c : kokoro_chunks(toks, british)) chunks.push_back(c);
            all += "\n";
        }
        if (!all.empty()) all.pop_back();
        return json{{"phonemes", all}, {"chunks", chunks}, {"tokens", tokens}}.dump();
    }

    std::vector<std::string> to_chunks(const KokoroState &s, const std::string &line) {
        if (s.phonemes) return kokoro_chunks(kokoro_phoneme_tokens(line), true /* no ɾ → T for given phonemes */);
        return kokoro_chunks(g2p(s.british).tokens(line), s.british);
    }

    const KokoroG2P &g2p(bool british) {
        std::lock_guard<std::mutex> lk(g2p_mu_);
        auto &slot = british ? g2p_gb_ : g2p_us_;
        if (!slot) {
            const std::string pre = british ? "gb" : "us";
            const std::string gold = model_.lexicon(pre + "_gold"), silver = model_.lexicon(pre + "_silver");
            if (gold.empty())
                fail(VOX_ERR_UNSUPPORTED, "Kokoro: this .tmod has no English lexicons (convert with misaki installed), "
                                          "so text needs phonemes: pass {\"phonemes\": true} or use the Python frontend");
            slot = std::make_unique<KokoroG2P>(gold, silver, british);
        }
        return *slot;
    }

    void synth(KokoroState &s, const std::string &ps, std::vector<float> &pcm) {
        const uint32_t idx = s.chunk_counter++;
        const std::vector<int32_t> ids = model_.phonemes_to_ids(ps);
        const size_t n = model_.phoneme_count(ps);
        std::vector<float> ref;
        if (!s.own_pack.empty()) {
            const size_t row = std::min<size_t>(n ? n - 1 : 0, size_t(s.own_rows - 1));
            ref.assign(s.own_pack.begin() + long(row * size_t(2 * model_.cfg().style_dim)), s.own_pack.begin() + long((row + 1) * size_t(2 * model_.cfg().style_dim)));
        } else {
            model_.style(s.voice, n, ref);
        }
        KokoroNoise nz;
        const KokoroNoise *inj = nullptr;
        auto a = s.injected_noise.find(std::to_string(idx) + ".rand_ini"), b = s.injected_noise.find(std::to_string(idx) + ".sine_noise");
        if (a != s.injected_noise.end() && b != s.injected_noise.end()) {
            nz.rand_ini.resize(a->second.bytes.size() / 4);
            std::memcpy(nz.rand_ini.data(), a->second.bytes.data(), a->second.bytes.size());
            nz.sine_noise.resize(b->second.bytes.size() / 4);
            std::memcpy(nz.sine_noise.data(), b->second.bytes.data(), b->second.bytes.size());
            inj = &nz;
        }
        std::vector<float> wave;
        model_.synthesize(ids, ref.data(), s.speed, s.seed, idx, inj, wave);
        pcm.insert(pcm.end(), wave.begin(), wave.end());
        s.done_chunks.push_back(ps);
    }

    KokoroModel &model() { return model_; }

private:
    [[noreturn]] void no_codes() const { fail(VOX_ERR_UNSUPPORTED, "Kokoro does not use audio codes"); }
    [[noreturn]] void no_clone() const { fail(VOX_ERR_UNSUPPORTED, "Kokoro has no voice cloning: pick one of its voices (speaker) or load a voice pack"); }
    std::shared_ptr<TmodFile> file_;
    KokoroModel model_;
    ModelInfo info_;
    std::mutex g2p_mu_;
    std::unique_ptr<KokoroG2P> g2p_us_, g2p_gb_;
};

bool KokoroBackend::debug_check(const std::string &dump, double tol, std::string &report) {
    const SafetensorsFile f = safetensors_load(dump);
    auto get = [&](const char *k) -> std::vector<float> {
        auto it = f.tensors.find(k);
        if (it == f.tensors.end()) return {};
        std::vector<float> v(it->second.bytes.size() / 4);
        std::memcpy(v.data(), it->second.bytes.data(), it->second.bytes.size());
        return v;
    };
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
        char line[200];
        std::snprintf(line, sizeof line, "  %-12s max|diff| %.3e  max|ref| %.3e  rel %.3e  %s\n", name, mx, ref, rel, rel <= t ? "ok" : "FAIL");
        out += line;
        if (rel > t) ok = false;
    };
    const std::string ps = meta("phonemes");
    if (ps.empty()) { report = "dump has no phonemes metadata\n"; return false; }
    const auto ids = model_.phonemes_to_ids(ps);
    const auto want_ids = get("input_ids");
    cmp("input ids", std::vector<float>(ids.begin(), ids.end()), want_ids, 0);
    std::vector<float> ref = get("ref_s");
    std::vector<float> mine;
    const std::string voice = meta("voice");
    if (!voice.empty() && model_.has_voice(voice)) {
        model_.style(voice, model_.phoneme_count(ps), mine);
        cmp("style", mine, ref, 0);
    }
    KokoroNoise nz;
    nz.rand_ini = get("rand_ini");
    nz.sine_noise = get("sine_noise");
    std::vector<float> pcm;
    KokoroModel::Trace tr;
    model_.synthesize(ids, ref.data(), 1.0f, 1, 0, nz.sine_noise.empty() ? nullptr : &nz, pcm, &tr);
    cmp("plbert", tr.bert, get("bert"), tol);
    cmp("bert_enc", tr.d_en, get("d_en"), tol);
    cmp("dur encoder", tr.d, get("d"), tol);
    cmp("duration", tr.duration, get("duration"), tol);
    cmp("pred_dur", std::vector<float>(tr.pred_dur.begin(), tr.pred_dur.end()), get("pred_dur"), 0);
    cmp("F0", tr.F0, get("F0"), tol);
    cmp("N", tr.N, get("N"), tol);
    cmp("text enc", tr.t_en, get("t_en"), tol);
    cmp("asr", tr.asr, get("asr"), tol);
    // Decoder on the reference's own F0 / N / alignment. The NSF source integrates F0 over the utterance, so ~1e-6 differences
    // of F0 shift the waveform's phase; and the source STFT has frames whose imaginary part is ~0 (reflection padding makes the
    // edge frames symmetric), where the phase is +π or −π depending on rounding inside the FFT. So: the source signal and its
    // spectrogram (phase compared modulo 2π) are checked, and the generator is checked on the reference spectrogram.
    {
        const auto asr = get("asr"), F0 = get("F0"), N = get("N"), har = get("har");
        KokoroModel::Trace t2;
        std::vector<float> wav;
        model_.decode(asr, int(F0.size() / 2), F0, N, ref.data(), 1, 0, nz.sine_noise.empty() ? nullptr : &nz, wav, &t2);
        cmp("source", t2.har_src, get("har_src"), tol);
        if (!har.empty() && har.size() == t2.har.size()) {
            const int C = 2 * (model_.cfg().n_fft / 2 + 1), NB = C / 2;
            // as complex numbers (magnitude · e^{i·phase}): the phase of a near-zero bin is undefined
            std::vector<float> a, b;
            for (size_t f0 = 0; f0 + size_t(C) <= har.size(); f0 += size_t(C))
                for (int k = 0; k < NB; ++k) {
                    a.push_back(t2.har[f0 + size_t(k)] * std::cos(t2.har[f0 + size_t(NB + k)]));
                    a.push_back(t2.har[f0 + size_t(k)] * std::sin(t2.har[f0 + size_t(NB + k)]));
                    b.push_back(har[f0 + size_t(k)] * std::cos(har[f0 + size_t(NB + k)]));
                    b.push_back(har[f0 + size_t(k)] * std::sin(har[f0 + size_t(NB + k)]));
                }
            cmp("source stft", a, b, std::max(tol, 1e-4));
            KokoroNoise forced;
            forced.har = har;
            std::vector<float> wav2;
            KokoroModel::Trace t3;
            model_.decode(asr, int(F0.size() / 2), F0, N, ref.data(), 1, 0, &forced, wav2, &t3);
            cmp("generator", wav2, get("audio"), std::max(tol, 1e-3));
            tr.post = t3.post;
        }
        tr.har_src = t2.har_src; tr.har = t2.har; tr.dec = t2.dec;
        pcm.swap(wav);
    }
    if (const char *tp = std::getenv("VOX_KOKORO_TRACE")) {   // debugging: the engine's intermediate tensors
        SafetensorsFile sf;
        auto put = [&](const char *k, const std::vector<float> &v) { if (!v.empty()) sf.tensors[k] = Tensor::from(DType::F32, {int64_t(v.size())}, v); };
        put("bert", tr.bert); put("d_en", tr.d_en); put("d", tr.d); put("F0", tr.F0); put("N", tr.N); put("t_en", tr.t_en);
        put("dec", tr.dec); put("har_src", tr.har_src); put("har", tr.har); put("post", tr.post); put("audio", pcm);
        safetensors_save(tp, sf);
    }
    report = out;
    return ok;
}

class KokoroFamily final : public Family {
public:
    std::string name() const override { return "kokoro"; }
    std::string description() const override { return "Kokoro (StyleTTS2 + iSTFTNet): fast phoneme-based TTS with built-in voices"; }
    ModelFootprint footprint(const TmodFile &f) const override {
        const json &m = f.manifest();
        ModelFootprint fp;
        fp.family = name();
        fp.variant = m.at("model").value("variant", "");
        StageDesc pros;
        pros.name = "prosody";
        pros.prefixes = {"bert.", "bert_encoder.", "predictor.", "text_encoder."};
        pros.gpu_capable = false;
        pros.bytes_per_frame = double(blob_bytes(f, pros.prefixes)) / 40.0;
        pros.session_ram_bytes = 64u << 20;
        fp.stages.push_back(pros);
        StageDesc dec;
        dec.name = "decoder";
        dec.prefixes = {"decoder."};
        dec.gpu_capable = false;
        dec.bytes_per_frame = double(blob_bytes(f, dec.prefixes)) / 40.0;
        dec.flops_per_frame = 2.0 * 400e9 / 400.0;
        dec.session_ram_bytes = 192u << 20;
        fp.stages.push_back(dec);
        fp.always_ram_bytes = blob_bytes(f, {"voice.", "g2p."}) + (32u << 20);
        return fp;
    }
    Capabilities capabilities(const TmodFile &f) const override { return KokoroBackend::caps(f); }
    std::unique_ptr<Backend> create(std::shared_ptr<TmodFile> f, const ResourcePlan &plan) const override {
        return std::make_unique<KokoroBackend>(std::move(f), plan);
    }
};

}  // namespace

std::shared_ptr<Family> make_kokoro_family() { return std::make_shared<KokoroFamily>(); }

}  // namespace vox
