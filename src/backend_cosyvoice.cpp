// Family "cosyvoice": CosyVoice2-0.5B (FunAudioLLM) in the layout of build_cosyvoice.py.
// A model without per-frame codes for the engine (the "direct" mode, like F5-TTS and Kokoro): the text is normalized and
// split as CosyVoiceFrontEnd.text_normalize does without a text-normalization package (numbers spelled out with inflect's
// rules, sentences grouped to 60–80 tokens), every piece goes LLM → flow → HiFT at once (CosyVoice2Model.tts, stream=False).
// Voices: from a recording (cloning). With the recording's text → zero-shot (the text and the speech tokens prompt the LLM);
// without it → cross-lingual (only the flow is prompted). An instruction on the voice (a style, an emotion) → instruct2:
// the instruction replaces the prompt text and ends with <|endofprompt|>.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

#include "backend.hpp"
#include "gpu.hpp"
#include "cosyvoice.hpp"
#include "family.hpp"
#include "nn.hpp"
#include "tokenizer.hpp"

namespace vox {
namespace {

#include "unicode_punct.inc"

bool in_ranges(uint32_t cp, const uint32_t (*t)[2], size_t n) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < t[mid][0]) hi = mid;
        else if (cp > t[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

std::vector<uint32_t> u32(const std::string &s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if (i + size_t(len) > s.size()) len = 1;
        uint32_t v = len == 1 ? c : len == 2 ? (c & 0x1Fu) : len == 3 ? (c & 0x0Fu) : (c & 0x07u);
        for (int k = 1; k < len; ++k) v = (v << 6) | (static_cast<unsigned char>(s[i + size_t(k)]) & 0x3Fu);
        out.push_back(v);
        i += size_t(len);
    }
    return out;
}
std::string u8(const std::vector<uint32_t> &v, size_t a = 0, size_t b = std::string::npos) {
    std::string s;
    b = std::min(b, v.size());
    for (size_t i = a; i < b; ++i) {
        const uint32_t c = v[i];
        if (c < 0x80) s += char(c);
        else if (c < 0x800) { s += char(0xC0 | (c >> 6)); s += char(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { s += char(0xE0 | (c >> 12)); s += char(0x80 | ((c >> 6) & 0x3F)); s += char(0x80 | (c & 0x3F)); }
        else { s += char(0xF0 | (c >> 18)); s += char(0x80 | ((c >> 12) & 0x3F)); s += char(0x80 | ((c >> 6) & 0x3F)); s += char(0x80 | (c & 0x3F)); }
    }
    return s;
}
bool contains_chinese(const std::string &s) {
    for (uint32_t c : u32(s))
        if (c >= 0x4E00 && c <= 0x9FFF) return true;
    return false;
}
void replace_all(std::string &s, const std::string &a, const std::string &b) {
    for (size_t p = s.find(a); p != std::string::npos; p = s.find(a, p + b.size())) s.replace(p, a.size(), b);
}
std::string strip(const std::string &s) {   // str.strip(): Unicode white space
    const auto v = u32(s);
    auto ws = [](uint32_t c) { return c == ' ' || (c >= 9 && c <= 13) || c == 0x1C || c == 0x1D || c == 0x1E || c == 0x1F || c == 0x85 ||
                                      c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
                                      c == 0x202F || c == 0x205F || c == 0x3000; };
    size_t a = 0, b = v.size();
    while (a < b && ws(v[a])) ++a;
    while (b > a && ws(v[b - 1])) --b;
    return u8(v, a, b);
}

// inflect.engine().number_to_words for a run of ASCII digits
std::string number_to_words(const std::string &digits) {
    static const char *ones[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                                 "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
    static const char *tens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
    static const char *mill[] = {"", " thousand", " million", " billion", " trillion", " quadrillion", " quintillion", " sextillion",
                                 " septillion", " octillion", " nonillion", " decillion"};
    size_t a = 0;
    while (a + 1 < digits.size() && digits[a] == '0') ++a;
    const std::string d = digits.substr(a);
    if (d == "0") return "zero";
    if ((d.size() + 2) / 3 > 12) {   // inflect raises NumOutOfRangeError: read the digits one by one
        std::string s;
        for (char c : d) s += (s.empty() ? "" : " ") + std::string(ones[c - '0']);
        return s;
    }
    auto under100 = [&](int n) -> std::string {
        if (n < 20) return ones[n];
        return std::string(tens[n / 10]) + (n % 10 ? std::string("-") + ones[n % 10] : "");
    };
    std::vector<std::pair<std::string, int>> parts;   // (words, value of the group) for non-zero groups, highest first
    const size_t ng = (d.size() + 2) / 3;
    for (size_t g = 0; g < ng; ++g) {
        const size_t end = d.size() - (ng - 1 - g) * 3;
        const size_t beg = end >= 3 ? end - 3 : 0;
        const int v = std::stoi(d.substr(beg, end - beg));
        if (!v) continue;
        std::string w;
        if (v >= 100) w = std::string(ones[v / 100]) + " hundred" + (v % 100 ? " and " + under100(v % 100) : "");
        else w = under100(v);
        w += mill[ng - 1 - g];
        parts.push_back({w, (ng - 1 - g) == 0 ? v : 1000});
    }
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out += (i + 1 == parts.size() && parts[i].second < 100) ? " and " : ", ";
        out += parts[i].first;
    }
    return out;
}

std::string spell_out_numbers(const std::string &t) {
    std::string out, run;
    for (char c : t) {
        if (c >= '0' && c <= '9') { run += c; continue; }
        if (!run.empty()) { out += number_to_words(run); run.clear(); }
        out += c;
    }
    if (!run.empty()) out += number_to_words(run);
    return out;
}

bool only_punct(const std::string &s) {
    for (uint32_t c : u32(s))
        if (!in_ranges(c, kPunctSym, kPunctSym_N)) return false;
    return true;
}

struct TextPrep {
    bool zh = false;
    std::string text;   // normalized, not split
};

// CosyVoiceFrontEnd.text_normalize up to the split (text_frontend '' — no ttsfrd / wetext)
TextPrep normalize(const std::string &in) {
    TextPrep r;
    if (in.find("<|") != std::string::npos && in.find("|>") != std::string::npos) { r.text = in; return r; }   // text_frontend off
    std::string t = strip(in);
    if (contains_chinese(t)) {
        r.zh = true;
        replace_all(t, "\n", "");
        {   // replace_blank: a space survives only between two ASCII non-space characters
            const auto v = u32(t);
            std::vector<uint32_t> o;
            for (size_t i = 0; i < v.size(); ++i) {
                if (v[i] == ' ') {
                    const uint32_t nx = i + 1 < v.size() ? v[i + 1] : 0, pv = i > 0 ? v[i - 1] : v.back();
                    if (nx < 0x80 && nx != ' ' && i + 1 < v.size() && pv < 0x80 && pv != ' ') o.push_back(' ');
                } else {
                    o.push_back(v[i]);
                }
            }
            t = u8(o);
        }
        replace_all(t, "²", "平方");
        replace_all(t, "³", "立方");
        replace_all(t, ".", "。");
        replace_all(t, " - ", "，");
        for (const char *b : {"（", "）", "【", "】", "`"}) replace_all(t, b, "");
        replace_all(t, "——", " ");
        {   // re.sub(r'[，,、]+$', '。', text)
            auto v = u32(t);
            size_t e = v.size();
            while (e > 0 && (v[e - 1] == 0xFF0C || v[e - 1] == ',' || v[e - 1] == 0x3001)) --e;
            if (e < v.size()) { v.resize(e); v.push_back(0x3002); t = u8(v); }
        }
    } else {
        t = spell_out_numbers(t);
    }
    r.text = t;
    return r;
}

const std::vector<uint32_t> &puncts(bool zh) {
    static const std::vector<uint32_t> z = {0x3002, 0xFF1F, 0xFF01, 0xFF1B, 0xFF1A, 0x3001, '.', '?', '!', ';'};
    static const std::vector<uint32_t> e = {'.', '?', '!', ';', ':'};
    return zh ? z : e;
}

// split_paragraph, first phase: sentences ending with a punctuation mark (a closing quote right after joins it).
// complete: the text is final (a missing end mark is added); otherwise only finished sentences are returned.
std::vector<std::string> sentences(const std::string &text, bool zh, bool complete) {
    auto v = u32(text);
    const auto &P = puncts(zh);
    auto is_p = [&](uint32_t c) { return std::find(P.begin(), P.end(), c) != P.end(); };
    if (complete && !v.empty() && !is_p(v.back())) v.push_back(zh ? 0x3002 : '.');
    std::vector<std::string> utts;
    size_t st = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        if (!is_p(v[i])) continue;
        if (!complete && i + 1 == v.size()) break;   // the next character may still be a closing quote
        if (i > st) utts.push_back(u8(v, st, i + 1));
        if (i + 1 < v.size() && (v[i + 1] == '"' || v[i + 1] == 0x201D)) {
            if (!utts.empty()) utts.back() += u8(v, i + 1, i + 2);
            st = i + 2;
        } else {
            st = i + 1;
        }
    }
    return utts;
}

struct CosyState : GenState {
    std::vector<int32_t> prompt_tokens, llm_prompt_tokens, prompt_text_tokens;
    std::vector<float> prompt_feat, xvec;
    std::string mode;   // zero_shot | cross_lingual | instruct2
    float speed = 1.0f;
    uint64_t seed = 1;
    std::string text;
    size_t emitted = 0;               // groups of the current segment already queued
    std::vector<std::string> queue;
    uint32_t piece_counter = 0;
    bool sealed = false;
    std::map<std::string, Tensor> injected;   // tests: "<piece>.rand_ini" / ".sine_noise" / ".tokens"
    std::vector<std::string> done;
};

class CosyBackend final : public Backend {
public:
    CosyBackend(std::shared_ptr<TmodFile> f, const ResourcePlan &plan) : file_(f), model_(f), tok_(Tokenizer::from_tmod(*f)) {
        if (plan.uses_gpu() && plan.place_of("flow") == Place::Gpu) {
            gpu_ = model_.enable_gpu();
            if (!gpu_ && plan.strict) fail(VOX_ERR_DEVICE, "GPU strict mode: the flow estimator did not go to the GPU (" + gpu_stats().last_error + ")");
        }
        if (plan.uses_gpu() && plan.place_of("llm") == Place::Gpu && model_.cfg().has_llm) {
            gpu_llm_ = model_.enable_gpu_llm();
            if (!gpu_llm_ && plan.strict) fail(VOX_ERR_DEVICE, "GPU strict mode: the LLM did not go to the GPU (" + gpu_stats().last_error + ")");
        }
        info_.sample_rate = model_.cfg().sample_rate;
        info_.samples_per_frame = 480;
        info_.num_codebooks = 1;
        info_.codebook_size = 1;
    }

    std::string name() const override { return gpu_ || gpu_llm_ ? "vulkan-cosyvoice (" + gpu_device_name() + ")" : "cpu-cosyvoice"; }
    const ModelInfo &info() const override { return info_; }
    std::string fingerprint() const override {
        const json &m = file_->manifest();
        return "cpu-cosyvoice-" + hex64(fnv1a64(m.at("model").value("weights_sha256", "") + m.value("precision", json::object()).dump())).substr(0, 12);
    }
    static Capabilities caps(const TmodFile &f) {
        Capabilities c;
        const json &cv = f.manifest().at("cosyvoice");
        const bool clone = cv.value("campplus", false) && cv.contains("s3tokenizer");
        c.clone_icl = clone;
        c.clone_xvector = clone;
        c.instruct = true;   // instruct2: an instruction together with the cloned voice
        c.speed = true;
        c.streaming_text = true;
        c.text_in = "chunk";
        c.audio_out = "chunks";
        c.stream_exact = true;
        c.tags = {"laugh", "breath", "cough", "sigh"};
        c.languages = {"zh", "en", "ja", "ko", "yue"};
        return c;
    }
    Capabilities capabilities() const override { return caps(*file_); }
    SessionCost session_cost(const VoiceData &, const SessionOptions &) const override { return {512u << 20, 0}; }
    json stats() const override {
        return {{"llm_ms", model_.ms_llm}, {"flow_ms", model_.ms_flow}, {"hift_ms", model_.ms_hift}, {"weights_bytes", model_.weight_bytes()},
                {"device", gpu_ || gpu_llm_ ? "vulkan" : "cpu"}, {"flow_device", gpu_ ? "vulkan" : "cpu"},
                {"llm_device", gpu_llm_ ? "vulkan" : "cpu"}, {"gemm", gpu_ ? gpu_gemm_choice() : ""}};
    }

    std::vector<int32_t> tokenize(const std::string &utf8) override { return tok_.encode(utf8); }
    // What the text frontend makes of a text: the pieces (normalized, grouped) and their tokens — `vox phonemize`.
    std::string phonemize(const std::string &text, const std::string &) override {
        json pieces = json::array(), toks = json::array();
        for (const auto &p : split(native_tags(text), true)) {
            pieces.push_back(p);
            toks.push_back(tok_.encode(p));
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

    // Recording → flow prompt (24 kHz mel), speech tokens and x-vector (16 kHz), aligned as frontend_zero_shot does.
    void prepare_voice(VoiceData &v, const float *pcm, size_t n, uint32_t sr) override {
        if (!model_.can_clone())
            fail(VOX_ERR_UNSUPPORTED, "CosyVoice: this .tmod has no speaker encoder / speech tokenizer: convert it with campplus.onnx and "
                                      "speech_tokenizer_v2.onnx to clone voices");
        if (sr < 16000) fail(VOX_ERR_ARG, "CosyVoice: the recording must be sampled at 16 kHz or more");
        // load_wav: mono, torchaudio's Resample to each rate
        const std::vector<float> a24 = nn::resample_ta(pcm, n, sr, 24000), a16 = nn::resample_ta(pcm, n, sr, 16000);
        if (a16.size() > size_t(30 * 16000)) fail(VOX_ERR_ARG, "CosyVoice: the recording must be at most 30 s long");
        int fm = 0, ff = 0, fw = 0;
        std::vector<float> mel = model_.mel24(a24.data(), a24.size(), fm);
        const std::vector<float> fb = model_.fbank16(a16.data(), a16.size(), ff);
        const std::vector<float> xv = model_.xvector(fb, ff);
        const std::vector<float> wm = model_.whisper_mel(a16.data(), a16.size(), fw);
        std::vector<int32_t> tk = model_.speech_tokens(wm, fw);
        const int len = std::min(fm / 2, int(tk.size()));
        mel.resize(size_t(2 * len) * size_t(model_.cfg().mel_dim));
        tk.resize(size_t(len));
        v.features.clear();
        v.features["feat.cosy_tokens"] = Tensor::from(DType::I32, {int64_t(tk.size())}, tk);
        v.features["feat.cosy_mel"] = Tensor::from(DType::F32, {int64_t(2 * len), int64_t(model_.cfg().mel_dim)}, mel);
        v.features["feat.cosy_xvec"] = Tensor::from(DType::F32, {int64_t(xv.size())}, xv);
    }

    std::unique_ptr<GenState> begin(const VoiceData &voice, const std::string &) override {
        auto st = std::make_unique<CosyState>();
        auto ft = voice.features.find("feat.cosy_tokens"), fm = voice.features.find("feat.cosy_mel"), fx = voice.features.find("feat.cosy_xvec");
        if (ft == voice.features.end() || fm == voice.features.end() || fx == voice.features.end())
            fail(VOX_ERR_ARG, "this voice has no CosyVoice features: make it from a recording with this model (voice clone)");
        st->prompt_tokens.resize(ft->second.bytes.size() / 4);
        std::memcpy(st->prompt_tokens.data(), ft->second.bytes.data(), ft->second.bytes.size());
        st->prompt_feat.resize(fm->second.bytes.size() / 4);
        std::memcpy(st->prompt_feat.data(), fm->second.bytes.data(), fm->second.bytes.size());
        st->xvec.resize(fx->second.bytes.size() / 4);
        std::memcpy(st->xvec.data(), fx->second.bytes.data(), fx->second.bytes.size());
        if (st->prompt_feat.size() != st->prompt_tokens.size() * size_t(2 * model_.cfg().mel_dim))
            fail(VOX_ERR_ARG, "CosyVoice voice features are inconsistent (mel frames != 2 × tokens)");
        if (!voice.instruct.empty()) {
            std::string ins = voice.instruct;
            if (ins.find("<|endofprompt|>") == std::string::npos) ins += "<|endofprompt|>";
            st->prompt_text_tokens = tok_.encode(ins);   // frontend_instruct2: the instruction is not normalized
            st->mode = "instruct2";
        } else if (voice.mode == VOX_VOICE_ICL && !voice.ref_text.empty()) {
            st->prompt_text_tokens = tok_.encode(normalize(voice.ref_text).text);
            st->llm_prompt_tokens = st->prompt_tokens;
            st->mode = "zero_shot";
        } else {
            st->mode = "cross_lingual";
        }
        return st;
    }

    void configure(GenState &g, const SessionOptions &o) override {
        CosyState &s = static_cast<CosyState &>(g);
        s.seed = o.seed;
        const json &op = o.options;
        if (op.contains("speed") && op["speed"].is_number()) s.speed = op["speed"].get<float>();
        if (s.speed <= 0.f) fail(VOX_ERR_ARG, "CosyVoice: speed must be positive");
        if (op.contains("noise_file") && op["noise_file"].is_string()) s.injected = safetensors_load(op["noise_file"].get<std::string>()).tensors;
    }

    void direct_text(GenState &g, const std::string &utf8) override {
        CosyState &s = static_cast<CosyState &>(g);
        s.text += native_tags(utf8);
    }
    void direct_seal(GenState &g) override { static_cast<CosyState &>(g).sealed = true; }

    bool direct_step(GenState &g, bool text_ended, std::vector<float> &pcm) override {
        CosyState &s = static_cast<CosyState &>(g);
        for (;;) {
            if (!s.queue.empty()) {
                const std::string piece = s.queue.front();
                s.queue.erase(s.queue.begin());
                synth(s, piece, pcm);
                return false;
            }
            const bool complete = text_ended || s.sealed;
            const std::vector<std::string> groups = split(s.text, complete);
            if (s.emitted < groups.size()) {
                for (; s.emitted < groups.size(); ++s.emitted) s.queue.push_back(groups[s.emitted]);
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

    // text_normalize(split=True) of the segment's text; when not complete, only the groups that cannot change any more
    std::vector<std::string> split(const std::string &text, bool complete) const {
        const TextPrep tp = normalize(text);
        if (tp.text.empty()) return {};
        if (text.find("<|") != std::string::npos && text.find("|>") != std::string::npos)   // text_frontend off: one piece
            return complete ? std::vector<std::string>{text} : std::vector<std::string>{};
        const std::vector<std::string> utts = sentences(tp.text, tp.zh, complete);
        auto len = [&](const std::string &x) { return tp.zh ? u32(x).size() : tok_.encode(x).size(); };
        std::vector<std::string> fin;
        std::string cur;
        for (const auto &u : utts) {
            if (len(cur + u) > 80 && len(cur) > 60) { fin.push_back(cur); cur.clear(); }
            cur += u;
        }
        if (complete) {
            if (!cur.empty()) {
                if (len(cur) < 20 && !fin.empty()) fin.back() += cur;
                else fin.push_back(cur);
            }
        } else if (!fin.empty()) {
            fin.pop_back();   // the last flushed group can still take the final short remainder
        }
        std::vector<std::string> out;
        for (auto &x : fin)
            if (!only_punct(x)) out.push_back(x);
        return out;
    }

    // engine tags → the spellings CosyVoice2's tokenizer knows
    static std::string native_tags(std::string t) {
        replace_all(t, "[laugh]", "[laughter]");
        return t;
    }

    void synth(CosyState &s, const std::string &piece, std::vector<float> &pcm) {
        const uint32_t idx = s.piece_counter++;
        const std::vector<int32_t> text = tok_.encode(piece);
        std::vector<int32_t> gen;
        auto it = s.injected.find(std::to_string(idx) + ".tokens");
        if (it != s.injected.end()) {
            const Tensor &t = it->second;
            if (t.dtype != DType::I32) fail(VOX_ERR_ARG, "CosyVoice: injected tokens must be I32");
            gen.resize(t.bytes.size() / 4);
            std::memcpy(gen.data(), t.bytes.data(), t.bytes.size());
        } else {
            gen = model_.lm_generate(s.prompt_text_tokens, text, s.llm_prompt_tokens, s.seed, idx);
        }
        std::vector<float> mel;
        model_.flow(s.prompt_tokens, gen, s.prompt_feat, s.xvec, mel);
        const int M = model_.cfg().mel_dim;
        int T = int(mel.size() / size_t(M));
        if (s.speed != 1.0f && T > 0) {   // F.interpolate(mel, size=int(T / speed), mode='linear')
            const int Tn = std::max(1, int(double(T) / double(s.speed)));
            std::vector<float> m2(size_t(Tn) * M);
            const float sc = float(T) / float(Tn);
            for (int i = 0; i < Tn; ++i) {
                float src = (float(i) + 0.5f) * sc - 0.5f;
                if (src < 0.f) src = 0.f;
                const int i0 = std::min(int(src), T - 1), i1 = std::min(i0 + 1, T - 1);
                const float lam = src - float(i0);
                for (int m = 0; m < M; ++m) m2[size_t(i) * M + m] = (1.0f - lam) * mel[size_t(i0) * M + m] + lam * mel[size_t(i1) * M + m];
            }
            mel.swap(m2);
            T = Tn;
        }
        CosyNoise nz;
        const CosyNoise *inj = nullptr;
        auto a = s.injected.find(std::to_string(idx) + ".rand_ini"), b = s.injected.find(std::to_string(idx) + ".sine_noise");
        if (a != s.injected.end() && b != s.injected.end()) {
            nz.rand_ini.resize(a->second.bytes.size() / 4);
            std::memcpy(nz.rand_ini.data(), a->second.bytes.data(), a->second.bytes.size());
            nz.sine_noise.resize(b->second.bytes.size() / 4);
            std::memcpy(nz.sine_noise.data(), b->second.bytes.data(), b->second.bytes.size());
            inj = &nz;
        }
        std::vector<float> wave;
        model_.hift(mel, T, s.seed, idx, inj, wave);
        pcm.insert(pcm.end(), wave.begin(), wave.end());
        s.done.push_back(piece);
        if (const char *pe = std::getenv("VOX_PROFILE"))
            if (std::strstr(pe, "cosy"))
                std::fprintf(stderr, "[cosyvoice] piece %u: %zu text / %zu speech tokens, %.2f s audio; total so far: llm %.0f ms, flow %.0f ms, hift %.0f ms\n",
                             idx, text.size(), gen.size(), double(wave.size()) / 24000.0, model_.ms_llm, model_.ms_flow, model_.ms_hift);
    }

    json state_info(const GenState &g) const override {
        const CosyState &s = static_cast<const CosyState &>(g);
        return {{"mode", s.mode}, {"pieces", s.done}};
    }

    bool debug_check(const std::string &dump, double tol, std::string &report) override;

private:
    [[noreturn]] void no_codes() const { fail(VOX_ERR_UNSUPPORTED, "CosyVoice does not use audio codes"); }
    std::shared_ptr<TmodFile> file_;
    bool gpu_ = false;       // the flow estimator runs on the GPU
    bool gpu_llm_ = false;   // the LLM runs on the GPU
    CosyVoiceModel model_;
    Tokenizer tok_;
    ModelInfo info_;
};

template <typename T>
std::vector<float> as_float(const Tensor &t) {
    std::vector<float> v(t.bytes.size() / sizeof(T));
    for (size_t i = 0; i < v.size(); ++i) { T x; std::memcpy(&x, t.bytes.data() + i * sizeof(T), sizeof(T)); v[i] = float(x); }
    return v;
}

// Every stage against a dump of tools/make_tiny_cosyvoice.py: text tokens, frontend, LLM log-probs on the reference's own
// tokens (teacher forcing), flow, HiFT on the reference mel with its noise.
bool CosyBackend::debug_check(const std::string &dump, double tol, std::string &report) {
    const SafetensorsFile f = safetensors_load(dump);
    auto get = [&](const char *k) -> std::vector<float> {
        auto it = f.tensors.find(k);
        if (it == f.tensors.end()) return {};
        const Tensor &t = it->second;
        if (t.dtype == DType::I32) return as_float<int32_t>(t);
        return as_float<float>(t);
    };
    auto geti = [&](const char *k) {
        const auto v = get(k);
        return std::vector<int32_t>(v.begin(), v.end());
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
    const CosyCfg &c = model_.cfg();
    // ---- text ----
    const std::string text = meta("text");
    cmpi("text tokens", tok_.encode(text), geti("text_tokens"), 1.0);
    const std::string pt = meta("prompt_text");
    if (!pt.empty()) cmpi("prompt tokens", tok_.encode(meta("instruct") == "1" ? pt : normalize(pt).text), geti("prompt_text_tokens"), 1.0);
    // ---- frontend ----
    const auto p24 = get("prompt24"), p16 = get("prompt16");
    if (!p24.empty() && !p16.empty()) {
        int fm = 0, ff = 0, fw = 0;
        cmp("mel 24k", model_.mel24(p24.data(), p24.size(), fm), get("prompt_feat"), tol);
        const auto fb = model_.fbank16(p16.data(), p16.size(), ff);
        cmp("fbank", fb, get("fbank"), std::max(tol, 1e-4));
        if (c.has_campplus) {
            const auto ref_fb = get("fbank");
            cmp("x-vector", model_.xvector(ref_fb, int(ref_fb.size() / 80)), get("xvec"), std::max(tol, 1e-3));
        }
        const auto wm = model_.whisper_mel(p16.data(), p16.size(), fw);
        cmp("whisper mel", wm, get("whisper_mel"), std::max(tol, 1e-4));
        if (c.has_s3) {
            const auto ref_wm = get("whisper_mel");
            cmpi("speech tokens", model_.speech_tokens(ref_wm, int(ref_wm.size() / size_t(c.s3_mels))), geti("speech_tokens"), 0.98);
        }
    }
    // ---- LLM: log-probs of every step on the reference's tokens ----
    const auto ptxt = geti("prompt_text_tokens"), ttxt = geti("text_tokens"), lpt = geti("llm_prompt_tokens"), gen = geti("gen_tokens");
    {
        auto st = model_.lm_begin(ptxt, ttxt, lpt);
        const auto ref = get("llm_logp");
        const size_t V = size_t(c.speech_tokens + 3);
        std::vector<float> mine, lp;
        for (size_t i = 0; i <= gen.size(); ++i) {
            model_.lm_logp(*st, lp);
            mine.insert(mine.end(), lp.begin(), lp.end());
            if (i < gen.size()) model_.lm_accept(*st, gen[i]);
        }
        // compare where it matters: log-probs above -20 (the far tail is dominated by float rounding)
        std::vector<float> a, b;
        for (size_t i = 0; i < std::min(mine.size(), ref.size()); ++i)
            if (ref[i] > -20.f) { a.push_back(mine[i]); b.push_back(ref[i]); }
        if (mine.size() != ref.size()) { a = mine; b = ref; }
        cmp("llm logp", a, b, std::max(tol, 1e-3));
        size_t argmax_same = 0;
        for (size_t r = 0; r * V < std::min(mine.size(), ref.size()); ++r) {
            const auto ma = std::max_element(mine.begin() + long(r * V), mine.begin() + long((r + 1) * V)) - mine.begin();
            const auto mb = std::max_element(ref.begin() + long(r * V), ref.begin() + long((r + 1) * V)) - ref.begin();
            argmax_same += ma == mb;
        }
        out += "  llm argmax       " + std::to_string(argmax_same) + " / " + std::to_string(ref.size() / V) + " steps agree\n";
    }
    // ---- flow on the reference tokens ----
    {
        const auto feat = get("prompt_feat"), xv = get("xvec");
        const std::vector<int32_t> stk = geti("speech_tokens");
        const size_t ntok = std::min(stk.size(), feat.size() / size_t(2 * c.mel_dim));
        const std::vector<int32_t> ptok = lpt.empty() ? std::vector<int32_t>(stk.begin(), stk.begin() + long(ntok)) : lpt;
        std::vector<float> pfeat(feat.begin(), feat.begin() + long(ptok.size() * size_t(2 * c.mel_dim)));
        CosyVoiceModel::FlowTrace tr;
        std::vector<float> mel;
        model_.flow(ptok, gen, pfeat, xv, mel, &tr);
        cmp("flow spk", tr.spk, get("flow_spk"), tol);
        cmp("flow mu", tr.mu, get("flow_mu"), std::max(tol, 1e-3));
        cmp("flow mel", mel, get("flow_mel"), std::max(tol, 2e-3));
    }
    // ---- HiFT on the reference mel and noise ----
    {
        const auto mel = get("flow_mel");
        CosyNoise nz;
        nz.rand_ini = get("rand_ini");
        nz.sine_noise = get("sine_noise");
        CosyVoiceModel::HiftTrace tr;
        std::vector<float> pcm;
        model_.hift(mel, int(mel.size() / size_t(c.mel_dim)), 1, 0, nz.sine_noise.empty() ? nullptr : &nz, pcm, &tr);
        cmp("hift f0", tr.f0, get("hift_f0"), std::max(tol, 1e-4));
        cmp("hift audio", pcm, get("audio"), std::max(tol, 2e-3));
    }
    report = out;
    return ok;
}

class CosyFamily final : public Family {
public:
    std::string name() const override { return "cosyvoice"; }
    std::string description() const override {
        return "CosyVoice2 (Qwen2 speech-token LM + flow matching + HiFT): zero-shot voice cloning, instructions, 24 kHz";
    }
    ModelFootprint footprint(const TmodFile &f) const override {
        const json &m = f.manifest();
        ModelFootprint fp;
        fp.family = name();
        fp.variant = m.at("model").value("variant", "");
        StageDesc lm;
        lm.name = "llm";
        lm.prefixes = {"llm."};
        lm.gpu_capable = true;   // the GPU transformer (gpu_vulkan.cpp), KV cache in video memory
        lm.session_vram_bytes = 64u << 20;
        lm.bytes_per_frame = double(blob_bytes(f, lm.prefixes));
        lm.session_ram_bytes = 128u << 20;
        fp.stages.push_back(lm);
        StageDesc fl;
        fl.name = "flow";
        fl.prefixes = {"flow."};
        fl.gpu_capable = true;   // the estimator (gpu_f5.cpp: gpu_cosy_*); the encoder runs on the CPU from the same weights
        fl.bytes_per_frame = double(blob_bytes(f, fl.prefixes)) * 20.0 / 50.0;   // 10 steps × 2 passes read the estimator
        fl.session_ram_bytes = 384u << 20;
        fl.session_vram_bytes = 96u << 20;   // activations of both passes for ~1500 frames
        fp.stages.push_back(fl);
        StageDesc hf;
        hf.name = "hift";
        hf.prefixes = {"hift."};
        hf.gpu_capable = false;
        hf.bytes_per_frame = double(blob_bytes(f, hf.prefixes)) / 50.0;
        fp.stages.push_back(hf);
        fp.always_ram_bytes = blob_bytes(f, {"camp.", "s3.", "front.", "tokenizer."}) + (32u << 20);
        return fp;
    }
    Capabilities capabilities(const TmodFile &f) const override { return CosyBackend::caps(f); }
    std::unique_ptr<Backend> create(std::shared_ptr<TmodFile> f, const ResourcePlan &plan) const override {
        return std::make_unique<CosyBackend>(std::move(f), plan);
    }
};

}  // namespace

std::shared_ptr<Family> make_cosyvoice_family() { return std::make_shared<CosyFamily>(); }

}  // namespace vox
