// Семейство "f5-tts": F5-TTS (DiT + CFM) с вокодером Vocos. Модель без покадровых кодов — работает через «прямой» режим
// Backend (direct_*): текст режется на куски как в официальном инференсе F5, каждый кусок озвучивается целиком,
// куски склеиваются кроссфейдом. Сессия отдаёт звук по мере готовности кусков.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <regex>

#include "backend.hpp"
#include "f5.hpp"
#include "family.hpp"
#include "gpu.hpp"
#include "qwen3_voice.hpp"

namespace vox {
namespace {

bool ends_with(const std::string &s, const std::string &suf) { return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0; }

// Разбиение как f5_tts.infer.utils_infer.chunk_text: предложения по ;:,.!? + пробелы и по полноширинным знакам,
// накопление до max_chars байт UTF-8.
std::vector<std::string> split_sentences(const std::string &text) {
    std::vector<std::string> out;
    std::string cur;
    size_t i = 0;
    const auto starts = [&](size_t pos, const char *s) { return text.compare(pos, std::strlen(s), s) == 0; };
    static const char *kFull[] = {"；", "：", "，", "。", "！", "？"};
    while (i < text.size()) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        bool boundary = false;
        size_t len = 1;
        if (c < 0x80) {
            cur += char(c);
            if (c == ';' || c == ':' || c == ',' || c == '.' || c == '!' || c == '?') {
                // (?<=[;:,.!?])\s+  — граница только если дальше пробелы; сами пробелы выбрасываются
                size_t j = i + 1;
                while (j < text.size() && (text[j] == ' ' || text[j] == '\t' || text[j] == '\n' || text[j] == '\r' || text[j] == '\f' || text[j] == '\v')) ++j;
                if (j > i + 1) { boundary = true; len = j - i; }
            }
        } else {
            size_t ulen = (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
            ulen = std::min(ulen, text.size() - i);
            cur.append(text, i, ulen);
            for (const char *f : kFull)
                if (starts(i, f)) boundary = true;
            len = ulen;
        }
        i += len;
        if (boundary) { out.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

std::string strip(const std::string &s) {
    static const std::string ws = " \t\n\r\f\v";
    const size_t a = s.find_first_not_of(ws);
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(ws) - a + 1);
}

std::vector<std::string> chunk_text(const std::string &text, int64_t max_chars) {
    std::vector<std::string> chunks;
    std::string cur;
    for (const std::string &sent : split_sentences(text)) {
        if (sent.empty()) continue;
        const std::string add = (sent.back() & 0x80) == 0 ? sent + " " : sent;  // len(sentence[-1].encode()) == 1
        if (int64_t(cur.size() + sent.size()) <= max_chars) cur += add;
        else {
            if (!cur.empty()) chunks.push_back(strip(cur));
            cur = add;
        }
    }
    if (!cur.empty()) chunks.push_back(strip(cur));
    return chunks;
}

struct F5State : GenState {
    std::vector<float> cond_mel;
    int cond_frames = 0, ref_audio_len = 0;
    float rms = 0.1f, ref_sec = 0;
    std::string ref_text;          // после нормировки (кончается ". " или "。")
    std::string text;              // текущий отрезок текста (после запечатывания — следующий)
    size_t next_chunk = 0;         // следующий кусок внутри отрезка
    uint32_t chunk_counter = 0;    // сквозной номер куска: поток шума и признак первого куска
    bool sealed = false;           // отрезок законченный: последний кусок можно озвучивать, не ожидая продолжения
    bool fade_in_next = false;     // предыдущий отрезок закончился затуханием хвоста: следующий кусок начинаем с нарастания
    std::vector<float> tail;       // хвост для кроссфейда (до cross_fade отсчётов)
    bool flushed = false;
    uint64_t seed = 1;
    F5Sampler smp;
    float speed = 1.0f, cross_fade = 0.15f, target_rms = 0.1f;
    int64_t max_chars = 0;
    std::map<std::string, Tensor> injected_noise;   // тестовый хук: noise.<кусок>
    std::string dump_path;                          // тестовый хук: куда писать mel и звук каждого куска
    std::map<std::string, Tensor> dump;
};

class F5Backend final : public Backend {
public:
    F5Backend(std::shared_ptr<TmodFile> f, const ResourcePlan &plan)
        : file_(f), gpu_(plan.uses_gpu() && plan.place_of("dit") == Place::Gpu), model_(f, gpu_ && gpu_active()) {
        const json &m = f->manifest();
        info_.sample_rate = model_.cfg().sample_rate;
        info_.samples_per_frame = uint32_t(model_.cfg().mel.hop);
        info_.num_codebooks = 1;
        info_.codebook_size = 1;
        gen_ = m.value("generation", json::object());
        if (gpu_ && model_.gpu_mats() < model_.linear_mats()) {
            const std::string msg = std::to_string(model_.linear_mats() - model_.gpu_mats()) + " of " + std::to_string(model_.linear_mats()) +
                                    " matrices did not fit on the GPU (" + gpu_stats().last_error + ")";
            if (plan.strict) fail(VOX_ERR_DEVICE, "GPU strict mode: " + msg);
            std::fprintf(stderr, "vox: GPU: %s; they run on the CPU\n", msg.c_str());
        }
    }

    std::string name() const override { return gpu_ ? "vulkan-f5-tts (" + gpu_device_name() + ")" : "cpu-f5-tts"; }
    const ModelInfo &info() const override { return info_; }
    std::string fingerprint() const override {
        const json &m = file_->manifest();
        const std::string base = m.at("model").value("weights_sha256", "") + m.value("precision", json::object()).dump() + (gpu_ ? "vk" : "cpu");
        return std::string(gpu_ ? "vulkan-f5-" : "cpu-f5-") + hex64(fnv1a64(base)).substr(0, 12);
    }
    Capabilities capabilities() const override { return caps(); }
    static Capabilities caps() {
        Capabilities c;
        c.clone_icl = true;
        c.streaming_text = true;
        c.text_in = "chunk";      // куски озвучиваются целиком: частичное предложение ждёт его конца
        c.audio_out = "chunks";
        c.stream_exact = true;    // при commit=auto результат совпадает с синтезом всего текста сразу
        return c;
    }
    SessionCost session_cost(const VoiceData &, const SessionOptions &) const override {
        SessionCost c;
        c.ram = 192u << 20;
        c.vram = gpu_ ? (64u << 20) : 0;
        return c;
    }
    json stats() const override {
        return {{"dit_ms", model_.ms_dit()}, {"vocos_ms", model_.ms_vocos()}, {"weights_bytes", model_.weight_bytes()}, {"device", gpu_ ? "vulkan" : "cpu"}, {"gpu_matrices", model_.gpu_mats()}, {"matrices", model_.linear_mats()}};
    }

    // --- интерфейс, не используемый моделью без кодов ---
    std::vector<int32_t> tokenize(const std::string &utf8) override { return model_.tokenizer().encode(utf8); }
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

    // --- прямой режим ---
    bool direct() const override { return true; }

    void prepare_voice(VoiceData &v, const float *pcm, size_t n, uint32_t sr) override {
        const F5Cfg &c = model_.cfg();
        std::vector<float> a = resample_sinc(pcm, n, sr, c.sample_rate);
        const size_t maxn = size_t(12) * c.sample_rate;  // как в preprocess_ref_audio_text: не больше 12 с образца
        if (a.size() > maxn) a.resize(maxn);
        if (a.size() < size_t(c.mel.n_fft)) fail(VOX_ERR_ARG, "F5-TTS: reference audio is too short");
        double ss = 0;
        for (float x : a) ss += double(x) * double(x);
        const float rms = float(std::sqrt(ss / double(a.size())));
        const float target = gen_.value("target_rms", 0.1f);
        if (rms < target && rms > 0.f) {
            const float g = target / rms;
            for (auto &x : a) x *= g;
        }
        std::vector<float> mel;
        int frames = 0;
        model_.ref_mel(a.data(), a.size(), mel, frames);
        v.features["feat.mel"] = Tensor::from(DType::F32, {frames, c.mel_dim}, mel);
        const std::vector<float> meta = {rms, float(a.size()), float(c.sample_rate)};
        v.features["feat.meta"] = Tensor::from(DType::F32, {3}, meta);
    }

    std::unique_ptr<GenState> begin(const VoiceData &voice, const std::string &) override {
        auto st = std::make_unique<F5State>();
        auto itm = voice.features.find("feat.mel"), itt = voice.features.find("feat.meta");
        if (itm == voice.features.end() || itt == voice.features.end()) fail(VOX_ERR_ARG, "this voice has no F5-TTS features: create it with this model (voice clone)");
        const Tensor &mel = itm->second, &meta = itt->second;
        if (mel.shape.size() != 2 || mel.shape[1] != model_.cfg().mel_dim) fail(VOX_ERR_ARG, "F5-TTS voice features have a wrong shape");
        st->cond_frames = int(mel.shape[0]);
        st->cond_mel.resize(size_t(mel.shape[0]) * size_t(mel.shape[1]));
        std::memcpy(st->cond_mel.data(), mel.bytes.data(), st->cond_mel.size() * 4);
        float m3[3];
        std::memcpy(m3, meta.bytes.data(), sizeof m3);
        st->rms = m3[0];
        const int64_t n24 = int64_t(m3[1]);
        st->ref_audio_len = int(n24 / model_.cfg().mel.hop);
        st->ref_sec = float(double(n24) / double(model_.cfg().sample_rate));
        std::string rt = voice.ref_text;
        if (!ends_with(rt, ". ") && !ends_with(rt, "\xE3\x80\x82")) rt += ends_with(rt, ".") ? " " : ". ";
        st->ref_text = rt;
        return st;
    }

    void configure(GenState &g, const SessionOptions &o) override {
        F5State &s = static_cast<F5State &>(g);
        // имена параметров — как у официального инференса F5 (nfe_step, cfg_strength, sway_sampling_coef, cross_fade_duration…);
        // порядок: параметры сессии (extra_json) → значения голоса → manifest "generation" → умолчание
        auto num = [&](const char *k, double d) {
            for (const json *j : std::initializer_list<const json *>{&o.options, &gen_})
                if (j->contains(k) && (*j)[k].is_number()) return (*j)[k].get<double>();
                else if (j->contains(k) && (*j)[k].is_boolean()) return (*j)[k].get<bool>() ? 1.0 : 0.0;
            return d;
        };
        if (o.options.contains("fix_duration") && !o.options["fix_duration"].is_null())
            fail(VOX_ERR_UNSUPPORTED, "F5-TTS: fix_duration is not supported yet (use speed)");
        s.seed = o.seed;
        s.smp.nfe = std::max(1, int(num("nfe_step", 32)));
        s.smp.cfg = float(num("cfg_strength", 2.0));
        s.smp.sway = float(num("sway_sampling_coef", -1.0));
        s.smp.epss = num("use_epss", 1.0) != 0.0;
        s.speed = float(num("speed", 1.0));
        s.cross_fade = float(num("cross_fade_duration", 0.15));
        s.target_rms = float(num("target_rms", 0.1));
        if (s.speed <= 0.f) fail(VOX_ERR_ARG, "F5-TTS: speed must be positive");
        const double mc = num("max_chars", 0);
        if (mc > 0) s.max_chars = int64_t(mc);
        else {
            s.max_chars = int64_t(double(s.ref_text.size()) / double(s.ref_sec) * (22.0 - double(s.ref_sec)) * double(s.speed));
            if (s.max_chars < 1) s.max_chars = 1;
        }
        if (o.options.contains("dump_file") && o.options["dump_file"].is_string()) s.dump_path = o.options["dump_file"].get<std::string>();
        if (o.options.contains("noise_file") && o.options["noise_file"].is_string()) {  // хук для сверки с эталоном
            s.injected_noise = safetensors_load(o.options["noise_file"].get<std::string>()).tensors;
        }
    }

    void direct_text(GenState &g, const std::string &utf8) override { static_cast<F5State &>(g).text += utf8; }
    void direct_seal(GenState &g) override { static_cast<F5State &>(g).sealed = true; }

    bool direct_step(GenState &g, bool text_ended, std::vector<float> &pcm) override {
        F5State &s = static_cast<F5State &>(g);
        const auto chunks = chunk_text(s.text, s.max_chars);
        // пока текст не закончился, последний кусок может ещё вырасти — озвучиваем только законченные
        // (если отрезок запечатан — flush или конец предложения — последний кусок тоже законченный)
        const size_t avail = (text_ended || s.sealed) ? chunks.size() : (chunks.empty() ? 0 : chunks.size() - 1);
        if (s.next_chunk < avail) {
            synth_chunk(s, chunks[s.next_chunk], s.chunk_counter, pcm);
            ++s.next_chunk;
            ++s.chunk_counter;
            return false;
        }
        if (s.sealed && !text_ended) {
            // отрезок озвучен целиком, продолжения пока нет: отдаём хвост (с затуханием), чтобы последние 150 мс не повисли
            const size_t n = s.tail.size();
            for (size_t i = 0; i < n; ++i) s.tail[i] *= float(1.0 - double(i + 1) / double(n));
            pcm.insert(pcm.end(), s.tail.begin(), s.tail.end());
            s.fade_in_next = n > 0;
            s.tail.clear();
            s.text.clear();
            s.next_chunk = 0;
            s.sealed = false;
            return false;
        }
        if (!text_ended) return false;
        if (!s.flushed) {
            pcm.insert(pcm.end(), s.tail.begin(), s.tail.end());
            s.tail.clear();
            s.flushed = true;
        }
        return true;
    }

    bool debug_check(const std::string &dump, double tol, std::string &report) override;

private:
    [[noreturn]] void no_codes() const { fail(VOX_ERR_UNSUPPORTED, "F5-TTS does not use audio codes"); }

public:
    // Один кусок: mel → звук → кроссфейд с хвостом предыдущего. Возвращает озвученный кусок (до кроссфейда) через wave.
    void synth_chunk(F5State &s, const std::string &gen_text, uint32_t idx, std::vector<float> &pcm) {
        std::vector<float> wave;
        synth_wave(s, gen_text, idx, wave);
        const size_t cf_max = size_t(std::max(0.f, s.cross_fade) * float(model_.cfg().sample_rate));
        if (s.fade_in_next) {  // после паузы между отрезками: мягкое нарастание вместо кроссфейда с предыдущим
            const size_t n = std::min(cf_max, wave.size());
            for (size_t i = 0; i < n; ++i) wave[i] *= float(double(i) / double(n));
            s.fade_in_next = false;
        }
        std::vector<float> buf;
        if (cf_max == 0) {
            pcm.insert(pcm.end(), wave.begin(), wave.end());
            return;
        }
        if (s.tail.empty() && idx == 0) buf = wave;
        else {
            const size_t cfs = std::min({cf_max, s.tail.size(), wave.size()});
            if (cfs == 0) {
                buf = s.tail;
                buf.insert(buf.end(), wave.begin(), wave.end());
            } else {
                buf.assign(s.tail.begin(), s.tail.end() - long(cfs));
                for (size_t i = 0; i < cfs; ++i) {
                    const double fo = cfs == 1 ? 1.0 : 1.0 - double(i) / double(cfs - 1), fi = cfs == 1 ? 0.0 : double(i) / double(cfs - 1);
                    buf.push_back(float(double(s.tail[s.tail.size() - cfs + i]) * fo + double(wave[i]) * fi));
                }
                buf.insert(buf.end(), wave.begin() + long(cfs), wave.end());
            }
        }
        // в хвост уходят последние cf_max отсчётов, остальное можно отдавать
        const size_t keep = std::min(cf_max, buf.size());
        pcm.insert(pcm.end(), buf.begin(), buf.end() - long(keep));
        s.tail.assign(buf.end() - long(keep), buf.end());
    }

    void synth_wave(F5State &s, const std::string &gen_text, uint32_t idx, std::vector<float> &wave) {
        std::string ref = s.ref_text;
        if ((ref.back() & 0x80) == 0) ref += " ";  // infer_batch_process: len(ref_text[-1].encode()) == 1
        const std::vector<int32_t> ids = model_.tokenizer().encode(ref + gen_text);
        const F5Cfg &c = model_.cfg();
        float local_speed = s.speed;
        if (gen_text.size() < 10) local_speed = 0.3f;
        int64_t dur = int64_t(s.ref_audio_len) +
                      int64_t(double(s.ref_audio_len) / double(ref.size()) * double(gen_text.size()) / double(local_speed));
        dur = std::max<int64_t>(dur, int64_t(std::max<size_t>(ids.size(), size_t(s.cond_frames))) + 1);
        dur = std::min<int64_t>(dur, 65536);
        std::vector<float> noise(size_t(dur) * size_t(c.mel_dim));
        auto inj = s.injected_noise.find("noise." + std::to_string(idx));
        if (inj != s.injected_noise.end()) {
            if (inj->second.bytes.size() != noise.size() * 4) fail(VOX_ERR_ARG, "injected noise has a wrong size for chunk " + std::to_string(idx));
            std::memcpy(noise.data(), inj->second.bytes.data(), noise.size() * 4);
        } else {
            F5Model::noise(s.seed, idx, noise.size(), noise.data());
        }
        std::vector<float> mel;
        model_.sample(s.cond_mel, s.cond_frames, ids, int(dur), s.smp, noise.data(), mel);
        const int from = s.ref_audio_len, frames = int(dur) - from;
        model_.vocode(mel.data() + size_t(from) * size_t(c.mel_dim), frames, wave);
        if (s.rms < s.target_rms && s.rms > 0.f) {
            const float g = s.rms / s.target_rms;
            for (auto &x : wave) x *= g;
        }
        if (!s.dump_path.empty()) {
            s.dump["mel." + std::to_string(idx)] = Tensor::from(DType::F32, {frames, c.mel_dim}, std::vector<float>(mel.begin() + long(size_t(from) * size_t(c.mel_dim)), mel.end()));
            s.dump["ids." + std::to_string(idx)] = Tensor::from(DType::F32, {int64_t(ids.size())}, std::vector<float>(ids.begin(), ids.end()));
            s.dump["wave." + std::to_string(idx)] = Tensor::from(DType::F32, {int64_t(wave.size())}, wave);
            SafetensorsFile sf;
            sf.tensors = s.dump;
            safetensors_save(s.dump_path, sf);
        }
    }

    F5Model &model() { return model_; }

private:
    std::shared_ptr<TmodFile> file_;
    bool gpu_;
    F5Model model_;
    ModelInfo info_;
    json gen_;
};

bool F5Backend::debug_check(const std::string &dump, double tol, std::string &report) {
    const SafetensorsFile f = safetensors_load(dump);
    const F5Cfg &c = model_.cfg();
    auto get = [&](const char *k) -> std::vector<float> {
        auto it = f.tensors.find(k);
        if (it == f.tensors.end()) return {};
        if (it->second.dtype != DType::F32) fail(VOX_ERR_FORMAT, std::string("dump tensor must be F32: ") + k);
        std::vector<float> v(it->second.bytes.size() / 4);
        std::memcpy(v.data(), it->second.bytes.data(), it->second.bytes.size());
        return v;
    };
    auto meta = [&](const char *k, const std::string &d) { auto it = f.metadata.find(k); return it == f.metadata.end() ? d : it->second; };
    bool ok = true;
    std::string out;
    auto cmp = [&](const char *name, const std::vector<float> &a, const std::vector<float> &b) {
        if (a.size() != b.size()) {
            out += std::string("  ") + name + ": size mismatch " + std::to_string(a.size()) + " vs " + std::to_string(b.size()) + "  FAIL\n";
            ok = false;
            return;
        }
        double mx = 0, ref = 0;
        for (size_t i = 0; i < a.size(); ++i) { mx = std::max(mx, std::fabs(double(a[i]) - double(b[i]))); ref = std::max(ref, std::fabs(double(b[i]))); }
        const double rel = mx / (ref + 1e-6);
        char line[200];
        std::snprintf(line, sizeof line, "  %-14s max|diff| %.3e  max|ref| %.3e  rel %.3e  %s\n", name, mx, ref, rel, rel <= tol ? "ok" : "FAIL");
        out += line;
        if (rel > tol) ok = false;
    };
    // токенизация: основной случай (text/symbols/ids) и дополнительные tokN.text / tokN.symbols
    auto check_tokens = [&](const std::string &text, const std::string &symbols, const std::string &label) {
        const auto want = json::parse(symbols).get<std::vector<std::string>>();
        const auto got = model_.tokenizer().symbols(text);
        const bool same = want == got;
        out += "  tokenizer " + label + (same ? "ok" : "FAIL") + "\n";
        if (!same) {
            ok = false;
            out += "    text: " + text + "\n    ref: " + json(want).dump() + "\n    got: " + json(got).dump() + "\n";
        }
    };
    const std::string text = meta("text", "");
    if (!text.empty()) {
        check_tokens(text, meta("symbols", "[]"), "      ");
        const auto ids = get("ids");
        if (!ids.empty()) {
            const auto got_ids = model_.tokenizer().encode(text);
            std::vector<float> gi(got_ids.begin(), got_ids.end());
            cmp("token ids", gi, ids);
        }
    }
    for (int i = 0; f.metadata.count("tok" + std::to_string(i) + ".text"); ++i)
        check_tokens(meta(("tok" + std::to_string(i) + ".text").c_str(), ""), meta(("tok" + std::to_string(i) + ".symbols").c_str(), "[]"),
                     "case " + std::to_string(i) + " ");
    // mel образца
    const auto audio = get("audio");
    if (!audio.empty()) {
        std::vector<float> mel;
        int frames = 0;
        model_.ref_mel(audio.data(), audio.size(), mel, frames);
        cmp("ref mel", mel, get("mel"));
    }
    // один проход DiT с обеими ветками CFG
    const auto x = get("x"), cond = get("cond"), tt = get("t");
    if (!x.empty() && !cond.empty() && !tt.empty()) {
        const int N = int(x.size() / size_t(c.mel_dim));
        std::vector<float> ids_f = get("ids_full");
        std::vector<int32_t> ids(ids_f.begin(), ids_f.end());
        std::vector<float> tc, tu;
        model_.text_embed(ids, N, false, tc);
        model_.text_embed(ids, N, true, tu);
        cmp("text embed", tc, get("temb_c"));
        cmp("text embed(0)", tu, get("temb_u"));
        std::vector<float> x2(x), c2(cond), t2(tc), pred(x.size() * 2);
        x2.insert(x2.end(), x.begin(), x.end());
        c2.insert(c2.end(), x.size(), 0.f);
        t2.insert(t2.end(), tu.begin(), tu.end());
        model_.dit_forward(2, N, x2.data(), c2.data(), t2.data(), tt[0], pred.data());
        cmp("dit cond", std::vector<float>(pred.begin(), pred.begin() + long(x.size())), get("pred_c"));
        cmp("dit uncond", std::vector<float>(pred.begin() + long(x.size()), pred.end()), get("pred_u"));
    }
    // Vocos
    const auto vmel = get("vmel");
    if (!vmel.empty()) {
        std::vector<float> w;
        model_.vocode(vmel.data(), int(vmel.size() / size_t(c.mel_dim)), w);
        cmp("vocos", w, get("vwave"));
    }
    // полный цикл CFM с подставленным шумом
    const auto noise = get("noise"), sampled = get("sampled"), cm = get("cond_mel"), sids = get("sample_ids");
    if (!noise.empty() && !sampled.empty()) {
        F5Sampler sm;
        sm.nfe = std::stoi(meta("nfe", "8"));
        sm.cfg = std::stof(meta("cfg", "2"));
        sm.sway = std::stof(meta("sway", "-1"));
        sm.epss = meta("epss", "0") == "1";
        const int N = int(noise.size() / size_t(c.mel_dim)), Cf = int(cm.size() / size_t(c.mel_dim));
        std::vector<int32_t> ids(sids.begin(), sids.end());
        std::vector<float> o;
        model_.sample(cm, Cf, ids, N, sm, noise.data(), o);
        cmp("cfm sample", o, sampled);
    }
    report = out;
    return ok;
}

class F5Family final : public Family {
public:
    std::string name() const override { return "f5-tts"; }
    std::string description() const override { return "F5-TTS: DiT + flow matching, Vocos vocoder (zero-shot voice cloning from a reference clip and its text)"; }

    ModelFootprint footprint(const TmodFile &f) const override {
        const json &m = f.manifest();
        const json &c = m.at("f5");
        ModelFootprint fp;
        fp.family = name();
        fp.variant = m.at("model").value("variant", "");
        const double D = c.at("dim"), L = c.at("depth"), mult = c.value("ff_mult", 2), N = 1500;
        const double params = L * (4 * D * D + 2 * mult * D * D + 6 * D * D);
        StageDesc dit;
        dit.name = "dit";
        dit.prefixes = {"dit."};
        dit.bytes_per_frame = double(blob_bytes(f, {"dit."})) * 64.0 / N;       // 2 ветки CFG × 32 шага читают веса целиком
        dit.flops_per_frame = 2.0 * params * 2.0 * 32.0;
        dit.session_vram_bytes = 64u << 20;
        dit.session_ram_bytes = 192u << 20;
        fp.stages.push_back(dit);
        StageDesc voc;
        voc.name = "vocos";
        voc.prefixes = {"vocos."};
        voc.gpu_capable = false;  // ConvNeXt + iSTFT пока считаются на CPU
        voc.bytes_per_frame = double(blob_bytes(f, {"vocos."})) / 50.0;
        voc.flops_per_frame = 2.0 * 8.0 * 3.0 * 512.0 * 1536.0;
        fp.stages.push_back(voc);
        fp.always_ram_bytes = blob_bytes(f, {"mel."}) + (32u << 20);
        return fp;
    }

    Capabilities capabilities(const TmodFile &) const override { return F5Backend::caps(); }

    std::unique_ptr<Backend> create(std::shared_ptr<TmodFile> f, const ResourcePlan &plan) const override {
        return std::make_unique<F5Backend>(std::move(f), plan);
    }
};

}  // namespace

std::shared_ptr<Family> make_f5_family() { return std::make_shared<F5Family>(); }

}  // namespace vox
