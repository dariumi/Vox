// Бэкенд Qwen3-TTS (CPU, float32 вычисления; веса F32/F16/BF16/Q8 из mmap).
//
// Повторяет генерацию оригинала (qwen_tts/core/models/modeling_qwen3_tts.py, режим потокового
// текста, non_streaming_mode=False):
//   префикс = роль(3) + кодек-префикс + [ICL: (текст референса+текст) ⊕ (bos + коды референса)]
//   каждый кадр: talker → код 0; code predictor (скрытое состояние talker + эмбеддинг кода 0)
//   → коды 1..Q-1; следующий вход talker = сумма эмбеддингов кадра + очередной текстовый эмбеддинг
//   (или tts_pad, когда текст кончился).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <unordered_map>

#include "backend.hpp"
#include "family.hpp"
#include "gpu.hpp"
#include "qwen3.hpp"
#include "qwen3_voice.hpp"
#include "tokenizer.hpp"

namespace vox {
namespace {

Capabilities qwen3_capabilities(const TmodFile &f);

std::string normalize_language(std::string l) {
    std::transform(l.begin(), l.end(), l.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    static const std::map<std::string, std::string> iso = {
        {"zh", "chinese"}, {"en", "english"}, {"ru", "russian"}, {"de", "german"}, {"fr", "french"},
        {"es", "spanish"}, {"it", "italian"}, {"pt", "portuguese"}, {"ja", "japanese"}, {"ko", "korean"}};
    auto it = iso.find(l);
    return it == iso.end() ? l : it->second;
}

// Накопленная статистика бэкенда; делится через shared_ptr, чтобы пережить бэкенд, если сессию освободят позже.
struct Totals {
    std::mutex mu;
    int64_t sessions = 0, frames = 0, prefill_rows = 0, decode_frames = 0;
    double prefill = 0, talker = 0, predictor = 0, warmup = 0, decoder = 0;
};

struct QState : GenState {
    // подготовленный префикс
    std::vector<float> head;        // [head_rows][D]: роль + кодек-строки
    int64_t head_rows = 0;
    std::vector<float> codec_part;  // [codec_lens][D]
    int64_t codec_lens = 0;
    std::vector<float> pad, bos, eos;  // tts_pad/bos/eos после text_projection, по D
    // текст
    std::vector<float> text_stream;    // [text_n][D] (для ICL сначала токены референса)
    int64_t text_n = 0;
    // генерация
    KVCache kv, pkv;
    bool prefilled = false;
    int64_t frames_done = 0;
    std::vector<float> pending, past_hidden, pred_h;
    std::vector<int32_t> codes;
    // декодер кодека
    CodecState cs;
    bool warmed = false;
    std::vector<int16_t> ref_codes;
    int64_t ref_frames = 0;
    uint32_t context_frames = 0, max_frames = 2048;
    // режим VOX_DECODER_REFERENCE: нарезка как в qwen_tts chunked_decode (кусок 300, левый контекст 25)
    bool ref_decode = false;
    int64_t abs_frame = 0, reset_done_at = -1;  // абсолютный номер кадра в последовательности «референс + речь»
    std::vector<int16_t> hist;                 // последние kDecCtx кадров (для переинициализации на границе куска)
    // профилирование (VOX_PROFILE=1)
    double t_prefill = 0, t_talker = 0, t_pred = 0, t_warm = 0, t_decode = 0;
    int64_t prefill_rows = 0, decode_frames = 0;
    std::function<void(const QState &)> on_close;   // сводит статистику сессии в статистику бэкенда
    ~QState() override {
        gpu_seq_free(kv.gpu_seq);
        gpu_seq_free(pkv.gpu_seq);
        if (on_close) on_close(*this);
        if (!std::getenv("VOX_PROFILE")) return;
        std::fprintf(stderr,
                     "[vox profile] frames=%lld  prefill(%lld tok)=%.2fs  talker=%.2fs (%.0f ms/frame)  predictor=%.2fs (%.0f ms/frame)  "
                     "decoder warm-up=%.2fs  decoder=%.2fs (%.0f ms/frame)\n",
                     (long long)frames_done, (long long)prefill_rows, t_prefill, t_talker,
                     frames_done ? 1000 * t_talker / double(frames_done) : 0.0, t_pred,
                     frames_done ? 1000 * t_pred / double(frames_done) : 0.0, t_warm, t_decode,
                     decode_frames ? 1000 * t_decode / double(decode_frames) : 0.0);
    }
};

using Clock = std::chrono::steady_clock;
static double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

class Qwen3Backend final : public Backend {
public:
    Qwen3Backend(std::shared_ptr<TmodFile> f, const ResourcePlan &plan)
        : file_(f), w_(f), tok_(Tokenizer::from_tmod(*f)), plan_(plan), gpu_(plan.uses_gpu()), codec_on_gpu_(plan.place_of("codec") == Place::Gpu) {
        if (gpu_) {
            w_.upload_gpu(plan);
            const GpuStats st = gpu_stats();
            const bool verbose = std::getenv("VOX_VERBOSE") != nullptr;
            if (st.mats_failed > 0 || verbose)
                std::fprintf(stderr, "vox: GPU [%s]: %lld matrices (%.2f GiB) in VRAM, %lld on CPU (%.2f GiB)%s%s; codec decoder on %s\n",
                             gpu_device_name().c_str(), (long long)st.mats_ok, double(st.bytes) / (1u << 30), (long long)st.mats_failed,
                             double(st.bytes_failed) / (1u << 30), st.last_error.empty() ? "" : ", last error: ", st.last_error.c_str(),
                             w_.codec_gpu >= 0 ? "GPU" : "CPU");
            const bool want_t = plan.place_of("talker") == Place::Gpu, want_p = plan.place_of("predictor") == Place::Gpu;
            if (st.mats_failed > 0 && plan.strict)
                fail(VOX_ERR_DEVICE, "GPU strict mode: " + std::to_string(st.mats_failed) + " matrices did not fit in VRAM (" + st.last_error +
                                         "); free VRAM, use a q8 model, or drop 'strict'");
            if (codec_on_gpu_ && w_.codec_gpu < 0 && plan.strict)
                fail(VOX_ERR_DEVICE, "GPU strict mode: codec decoder could not be placed on the GPU (" + gpu_stats().last_error + ")");
            if (verbose) std::fprintf(stderr, "vox: talker on GPU: %s, predictor on GPU: %s, overlapped codec decode: %s\n", w_.talker.gpu_id >= 0 ? "yes" : "no",
                                      w_.pred.gpu_id >= 0 ? "yes" : "no", decode_overlappable() ? "on" : "off");
            if ((want_t && w_.talker.gpu_id < 0) || (want_p && w_.pred.gpu_id < 0))
                if (plan.strict) fail(VOX_ERR_DEVICE, "GPU strict mode: transformer could not be placed fully on the GPU");
        }
        const json &m = f->manifest();
        info_.sample_rate = m.at("audio").value("sample_rate", 24000u);
        info_.samples_per_frame = uint32_t(w_.cc.samples_per_frame());
        info_.num_codebooks = uint32_t(w_.tc.groups);
        info_.codebook_size = uint32_t(w_.pc.vocab);
        info_.eos_code = w_.tc.eos;
        info_.suppress_from = std::max(0, w_.tc.vocab - 1024);
        info_.suppress_to = w_.tc.vocab;
        info_.min_eos_frames = 2;
        if (w_.cc.quantizers != w_.tc.groups) fail(VOX_ERR_FORMAT, "codec quantizers != talker code groups");
        if (info_.samples_per_frame != m.at("audio").value("samples_per_frame", info_.samples_per_frame))
            fail(VOX_ERR_FORMAT, "samples_per_frame in manifest does not match codec configuration");
        role_ = tok_.encode("<|im_start|>assistant\n");
        if (role_.size() != 3) fail(VOX_ERR_FORMAT, "chat role prefix must tokenize to 3 tokens");
    }

    std::string name() const override { return gpu_ ? "vulkan-qwen3-tts (" + gpu_device_name() + ")" : "cpu-qwen3-tts"; }
    const ModelInfo &info() const override { return info_; }
    std::string fingerprint() const override {
        const json &m = file_->manifest();
        std::string base = m.at("model").value("weights_sha256", "") + m.value("precision", json::object()).dump() + "cpu";
        return std::string(gpu_ ? "vulkan-qwen3-" : "cpu-qwen3-") + hex64(fnv1a64(base)).substr(0, 12);
    }

    Capabilities capabilities() const override { return qwen3_capabilities(*file_); }
    SessionCost session_cost(const VoiceData &v, const SessionOptions &o) const override {
        const uint64_t rows = 10 + uint64_t(v.ref_frames), frames = o.max_frames ? o.max_frames : 2048;
        const uint64_t kvdim = uint64_t(w_.tc.kv_heads) * uint64_t(w_.tc.head_dim);
        const uint64_t per_pos = uint64_t(w_.tc.layers) * 2 * kvdim * 4;
        SessionCost c;
        const uint64_t cap = rows + frames + 16;
        if (w_.talker.gpu_id >= 0 && cap <= 3072) {
            const uint64_t k = w_.talker.gpu_layers < 0 ? uint64_t(w_.tc.layers) : uint64_t(w_.talker.gpu_layers), L = uint64_t(w_.tc.layers);
            c.vram += per_pos * cap * k / L;                        // слои на GPU держат KV в видеопамяти (гибрид — первые k)
            c.ram += per_pos * (rows + frames) * (L - k) / L;       // остальные — в RAM
        } else c.ram += per_pos * (rows + frames);
        const uint64_t pkv = uint64_t(w_.pc.layers) * 2 * uint64_t(w_.pc.kv_heads) * uint64_t(w_.pc.head_dim) * 4 * 40;
        (w_.pred.gpu_id >= 0 ? c.vram : c.ram) += pkv;
        (w_.codec_gpu >= 0 ? c.vram : c.ram) += 24u << 20;  // состояние декодера кодека (истории свёрток, кольцевой KV, хвосты)
        c.ram += 4u << 20;  // эмбеддинги префикса, буферы чанков
        return c;
    }
    json stats() const override {
        std::lock_guard<std::mutex> lk(totals_->mu);
        const Totals &t = *totals_;
        auto ms = [](double s, int64_t n) { return n ? 1000.0 * s / double(n) : 0.0; };
        return {{"completed_sessions", t.sessions}, {"frames", t.frames}, {"prefill_rows", t.prefill_rows},
                {"seconds", {{"prefill", t.prefill}, {"talker", t.talker}, {"predictor", t.predictor}, {"codec_warmup", t.warmup}, {"codec", t.decoder}}},
                {"ms_per_frame", {{"talker", ms(t.talker, t.frames)}, {"predictor", ms(t.predictor, t.frames)}, {"codec", ms(t.decoder, t.decode_frames)}}},
                {"talker_on_gpu", w_.talker.gpu_id >= 0}, {"predictor_on_gpu", w_.pred.gpu_id >= 0}, {"codec_on_gpu", w_.codec_gpu >= 0},
                {"overlapped_codec", decode_overlappable()}};
    }

    std::vector<int32_t> tokenize(const std::string &s) override { return tok_.encode(s); }
    std::vector<int32_t> tokenize_ref(const std::string &s) override {
        return tok_.encode("<|im_start|>assistant\n" + s + "<|im_end|>\n");
    }
    size_t text_safe_cut(const std::string &pending) const override { return tok_.safe_cut(pending); }

    // ---------------- создание голоса (нативно, без torch) ----------------
    const Qwen3VoiceEncoder &voice_encoder() {
        std::lock_guard<std::mutex> lk(mu_);
        if (!venc_) venc_ = std::make_unique<Qwen3VoiceEncoder>(file_);
        return *venc_;
    }
    std::vector<float> speaker_embedding(const float *pcm, size_t n, uint32_t sr) override {
        const Qwen3VoiceEncoder &ve = voice_encoder();
        if (!ve.has_speaker_encoder())
            fail(VOX_ERR_UNSUPPORTED, "this .tmod has no speaker-encoder weights: rebuild it with `python -m vox build` (vox >= 0.6) "
                                      "or create the voice with tools/export_voice.py");
        std::vector<float> a = resample_sinc(pcm, n, sr, 24000);
        std::vector<float> e = ve.speaker_embedding(a.data(), a.size());
        if (int(e.size()) != w_.tc.hidden) fail(VOX_ERR_FORMAT, "speaker embedding size does not match talker hidden size");
        return e;
    }
    std::vector<int16_t> encode_codes(const float *pcm, size_t n, uint32_t sr, int64_t &frames) override {
        const Qwen3VoiceEncoder &ve = voice_encoder();
        if (!ve.has_codec_encoder())
            fail(VOX_ERR_UNSUPPORTED, "this .tmod has no speech-tokenizer encoder weights: rebuild it with `python -m vox build` (vox >= 0.6)");
        std::vector<float> a = resample_sinc(pcm, n, sr, 24000);
        int cb = 0;
        std::vector<int16_t> codes = ve.encode(a.data(), a.size(), frames, cb);
        if (cb != w_.tc.groups) fail(VOX_ERR_FORMAT, "speech tokenizer encoder yields " + std::to_string(cb) + " codebooks, talker expects " +
                                                         std::to_string(w_.tc.groups));
        return codes;
    }

    // ---------------- префикс ----------------
    std::unique_ptr<GenState> begin(const VoiceData &v, const std::string &language) override {
        const TalkerCfg &tc = w_.tc;
        const int D = tc.hidden;
        auto st = std::make_unique<QState>();
        {
            std::shared_ptr<Totals> tot = totals_;
            st->on_close = [tot](const QState &q) {
                std::lock_guard<std::mutex> lk(tot->mu);
                tot->sessions += 1; tot->frames += q.frames_done; tot->prefill_rows += q.prefill_rows; tot->decode_frames += q.decode_frames;
                tot->prefill += q.t_prefill; tot->talker += q.t_talker; tot->predictor += q.t_pred; tot->warmup += q.t_warm; tot->decoder += q.t_decode;
            };
        }
        st->kv.init(tc.layers);
        st->pkv.init(w_.pc.layers);
        if (gpu_ && w_.pred.gpu_id >= 0) st->pkv.gpu_seq = gpu_seq_create(w_.pred.gpu_id, 40);
        st->pred_h.assign(size_t(w_.pc.hidden), 0.f);
        st->past_hidden.assign(size_t(D), 0.f);
        st->codes.assign(size_t(tc.groups), 0);

        const bool named = v.mode == VOX_VOICE_SPEAKER;
        if (!named && v.speaker_emb.size() != size_t(D))
            fail(VOX_ERR_FORMAT, "voice speaker embedding has dim " + std::to_string(v.speaker_emb.size()) +
                                     ", model expects " + std::to_string(D));
        // tts_bos / eos / pad
        std::vector<float> sp(3 * size_t(D));
        text_project({w_.ids.tts_bos, w_.ids.tts_eos, w_.ids.tts_pad}, sp.data());
        st->bos.assign(sp.begin(), sp.begin() + D);
        st->eos.assign(sp.begin() + D, sp.begin() + 2 * D);
        st->pad.assign(sp.begin() + 2 * D, sp.end());

        // язык
        int lang_id = -1;
        std::string lang = normalize_language(language.empty() ? "auto" : language);
        if (lang != "auto") {
            auto it = tc.lang.find(lang);
            if (it == tc.lang.end()) fail(VOX_ERR_ARG, "language '" + language + "' is not supported by this model");
            lang_id = it->second;
        }
        // готовый диктор: эмбеддинг — строка codec_embedding; диалектные дикторы подменяют chinese/auto своим диалектом
        int spk_row_id = -1;
        std::string spk_lower;
        if (named && !v.speaker.empty()) {
            spk_lower = v.speaker;
            for (auto &ch : spk_lower) ch = char(std::tolower(static_cast<unsigned char>(ch)));
            auto it = tc.spk_id.find(spk_lower);
            if (it == tc.spk_id.end()) fail(VOX_ERR_ARG, "speaker '" + v.speaker + "' is not in this model");
            spk_row_id = it->second;
            auto dl = tc.spk_dialect.find(spk_lower);
            if ((lang == "chinese" || lang == "auto") && dl != tc.spk_dialect.end()) {
                auto lit = tc.lang.find(dl->second);
                if (lit == tc.lang.end()) fail(VOX_ERR_FORMAT, "dialect '" + dl->second + "' of speaker '" + v.speaker + "' has no language id");
                lang_id = lit->second;
            }
        }
        std::vector<int> prefix = lang_id < 0 ? std::vector<int>{tc.nothink, tc.think_bos, tc.think_eos}
                                              : std::vector<int>{tc.think, tc.think_bos, lang_id, tc.think_eos};
        // codec_input_embedding: [prefix..., speaker, pad, bos]
        const bool has_spk = !named || spk_row_id >= 0;     // VoiceDesign: строки диктора нет
        const int64_t C = int64_t(prefix.size()) + (has_spk ? 3 : 2);
        std::vector<float> cie(size_t(C) * size_t(D));
        int64_t r = 0;
        for (int id : prefix) gather_codec(id, &cie[size_t(r++) * size_t(D)]);
        if (named) {
            if (spk_row_id >= 0) { check_id(spk_row_id, w_.codec_emb.rows, "speaker id"); gather_codec(spk_row_id, &cie[size_t(r++) * size_t(D)]); }
        } else {
            std::copy(v.speaker_emb.begin(), v.speaker_emb.end(), cie.begin() + r++ * D);
        }
        gather_codec(tc.pad, &cie[size_t(r++) * size_t(D)]);
        gather_codec(tc.bos, &cie[size_t(r++) * size_t(D)]);

        // инструкция: <|im_start|>user\n{instruct}<|im_end|>\n → строки text_projection перед ролью, без добавки кодека
        std::vector<int32_t> ins;
        if (named && !v.instruct.empty()) ins = tok_.encode("<|im_start|>user\n" + v.instruct + "<|im_end|>\n");
        const int64_t I = int64_t(ins.size());
        // head = [инструкция(I)] + роль(3) + (C-1) строк: (pad…, bos) ⊕ cie[:-1]
        st->head_rows = I + 3 + (C - 1);
        st->head.resize(size_t(st->head_rows) * size_t(D));
        if (I) text_project(ins, st->head.data());
        text_project(role_, st->head.data() + size_t(I) * size_t(D));
        for (int64_t i = 0; i < C - 1; ++i) {
            const float *t = (i < C - 2) ? st->pad.data() : st->bos.data();
            float *dst = &st->head[size_t(I + 3 + i) * size_t(D)];
            for (int d = 0; d < D; ++d) dst[d] = t[d] + cie[size_t(i) * size_t(D) + size_t(d)];
        }

        const bool icl = v.mode == VOX_VOICE_ICL;
        if (icl) {
            if (v.ref_codes.empty() || v.ref_codebooks != tc.groups)
                fail(VOX_ERR_FORMAT, "ICL voice needs ref_codes with " + std::to_string(tc.groups) + " codebooks");
            if (v.ref_text_tokens.size() < 6) fail(VOX_ERR_FORMAT, "voice has no reference text tokens");
            st->ref_codes = v.ref_codes;
            st->ref_frames = v.ref_frames;
            st->codec_lens = 1 + v.ref_frames;
            st->codec_part.assign(size_t(st->codec_lens) * size_t(D), 0.f);
            gather_codec(tc.bos, st->codec_part.data());
            std::vector<float> tmp(static_cast<size_t>(D));
            for (int64_t f = 0; f < v.ref_frames; ++f) {
                float *dst = &st->codec_part[size_t(1 + f) * size_t(D)];
                for (int q = 0; q < tc.groups; ++q) {
                    int id = v.ref_codes[size_t(f * v.ref_codebooks + q)];
                    if (q == 0) { check_id(id, w_.codec_emb.rows, "ref code"); mat_gather(w_.codec_emb, id, tmp.data()); }
                    else { check_id(id, w_.pred_emb[size_t(q - 1)].rows, "ref code"); mat_gather(w_.pred_emb[size_t(q - 1)], id, tmp.data()); }
                    for (int d = 0; d < D; ++d) dst[d] += tmp[size_t(d)];
                }
            }
            // текст референса: токены без роли (3) и без <|im_end|>\n (2)
            std::vector<int32_t> ref(v.ref_text_tokens.begin() + 3, v.ref_text_tokens.end() - 2);
            append_text(*st, ref);
        } else {
            st->codec_lens = 1;
            st->codec_part.assign(size_t(D), 0.f);
            gather_codec(tc.bos, st->codec_part.data());
        }
        return st;
    }

    void configure(GenState &g, const SessionOptions &o) override {
        auto &st = static_cast<QState &>(g);
        if (o.max_frames) st.max_frames = o.max_frames;
        st.context_frames = o.decoder_context_frames ? o.decoder_context_frames
                                                     : uint32_t(file_->manifest().value("generation", json::object()).value("decoder_context_frames", 50));
        st.ref_decode = o.decoder_mode == VOX_DECODER_REFERENCE;
    }

    void append_text(GenState &g, const std::vector<int32_t> &tokens) override {
        auto &st = static_cast<QState &>(g);
        if (tokens.empty()) return;
        const int D = w_.tc.hidden;
        const size_t old = st.text_stream.size();
        st.text_stream.resize(old + tokens.size() * size_t(D));
        text_project(tokens, st.text_stream.data() + old);
        st.text_n += int64_t(tokens.size());
    }

    bool needs_text(const GenState &g, bool ended) const override {
        auto &st = static_cast<const QState &>(g);
        if (ended) return false;
        if (!st.prefilled) return st.text_n < st.codec_lens;
        return st.text_n < st.codec_lens + st.frames_done;  // нужен элемент codec_lens + frames_done - 1
    }

    // ---------------- talker ----------------
    void talker_logits(GenState &g, bool ended, std::vector<float> &logits) override {
        auto &st = static_cast<QState &>(g);
        const int D = w_.tc.hidden;
        const auto t0 = Clock::now();
        const bool was_prefill = !st.prefilled;
        if (!st.prefilled) {
            if (ended && st.codec_lens == 1 && st.text_n == 0) fail(VOX_ERR_STATE, "empty text");
            const int64_t rows = st.head_rows + st.codec_lens;
            std::vector<float> x(size_t(rows) * size_t(D));
            std::copy(st.head.begin(), st.head.end(), x.begin());
            for (int64_t j = 0; j < st.codec_lens; ++j) {
                const float *t = elem(st, j, ended), *c = &st.codec_part[size_t(j) * size_t(D)];
                float *dst = &x[size_t(st.head_rows + j) * size_t(D)];
                for (int d = 0; d < D; ++d) dst[d] = t[d] + c[d];
            }
            if (gpu_ && w_.talker.gpu_id >= 0 && st.kv.gpu_seq < 0) {
                const int64_t cap = rows + int64_t(st.max_frames) + 16;
                if (cap <= 3072) st.kv.gpu_seq = gpu_seq_create(w_.talker.gpu_id, cap);  // иначе внимание на CPU
                if (plan_.strict && st.kv.gpu_seq < 0)
                    fail(VOX_ERR_OOM, "GPU strict mode: the talker KV cache (" + std::to_string(cap) + " positions) does not fit in VRAM (" +
                                          gpu_stats().last_error + "); lower max_frames or free VRAM");
            }
            transformer_forward(w_.talker, st.kv, x.data(), rows, st.past_hidden.data());
            st.prefilled = true;
            st.prefill_rows = rows;
        } else {
            const float *t = elem(st, st.codec_lens + st.frames_done - 1, ended);
            std::vector<float> x(static_cast<size_t>(D));
            for (int d = 0; d < D; ++d) x[size_t(d)] = st.pending[size_t(d)] + t[d];
            logits.resize(size_t(w_.tc.vocab));
            if (transformer_step_fused(w_.talker, st.kv, x.data(), 1, nullptr, &w_.codec_head, st.past_hidden.data(), logits.data())) {
                st.t_talker += since(t0);
                return;
            }
            transformer_forward(w_.talker, st.kv, x.data(), 1, st.past_hidden.data());
        }
        logits.resize(size_t(w_.tc.vocab));
        matmul(w_.codec_head, st.past_hidden.data(), 1, logits.data());
        (was_prefill ? st.t_prefill : st.t_talker) += since(t0);
    }

    void accept_code0(GenState &g, int32_t code) override {
        auto &st = static_cast<QState &>(g);
        check_id(code, w_.codec_emb.rows, "code");
        st.codes[0] = code;
    }

    // ---------------- code predictor ----------------
    void predictor_logits(GenState &g, uint32_t cb, std::vector<float> &logits) override {
        auto &st = static_cast<QState &>(g);
        const int D = w_.tc.hidden, PH = w_.pc.hidden;
        if (cb < 1 || int(cb) >= w_.tc.groups) fail(VOX_ERR_ARG, "bad codebook index");
        const auto t0 = Clock::now();
        const Mat &head = w_.pred_head[size_t(cb - 1)];
        logits.resize(size_t(w_.pc.vocab));
        const int64_t m = cb == 1 ? 2 : 1;
        std::vector<float> a(size_t(m) * size_t(D));
        if (cb == 1) {
            std::copy(st.past_hidden.begin(), st.past_hidden.end(), a.begin());
            mat_gather(w_.codec_emb, st.codes[0], a.data() + D);
            st.pkv.reset();
        } else {
            const Mat &E = w_.pred_emb[size_t(cb - 2)];
            check_id(st.codes[cb - 1], E.rows, "code");
            mat_gather(E, st.codes[cb - 1], a.data());
        }
        // GPU: проекция → 5 слоёв → норма → голова одной отправкой (раньше три отправки с ожиданием на проход)
        const bool fused = (w_.has_small || w_.pc.hidden == D) &&
                           transformer_step_fused(w_.pred, st.pkv, a.data(), m, w_.has_small ? &w_.small_proj : nullptr, &head,
                                                  st.pred_h.data(), logits.data());
        if (!fused) {
            std::vector<float> in(size_t(m) * size_t(PH));
            project_small(a.data(), m, in.data());
            transformer_forward(w_.pred, st.pkv, in.data(), m, st.pred_h.data());
            matmul(head, st.pred_h.data(), 1, logits.data());
        }
        st.t_pred += since(t0);
    }

    bool predict_frame(GenState &g, const SampleParams &sp, uint32_t key0, uint32_t key1, uint64_t frame, int32_t *codes) override {
        auto &st = static_cast<QState &>(g);
        static const bool off = [] { const char *e = std::getenv("VOX_GPU_SAMPLING"); return e && e[0] == '0'; }();
        if (!gpu_ || off || w_.pred.gpu_id < 0 || st.pkv.gpu_seq < 0 || !(w_.has_small || w_.pc.hidden == w_.tc.hidden)) return false;
        const auto t0 = Clock::now();
        const int D = w_.tc.hidden;
        std::vector<float> in2(2 * size_t(D));
        std::copy(st.past_hidden.begin(), st.past_hidden.end(), in2.begin());
        mat_gather(w_.codec_emb, st.codes[0], in2.data() + D);
        GpuSampleCfg c;
        c.temperature = sp.temperature; c.top_k = sp.top_k; c.top_p = sp.top_p; c.key0 = key0; c.key1 = key1; c.frame = frame;
        codes[0] = st.codes[0];
        st.pkv.reset();
        if (!gpu_pred_frame(w_.pred, st.pkv.gpu_seq, in2.data(), w_.has_small ? &w_.small_proj : nullptr, w_.pred_head, w_.pred_emb, c, codes))
            return false;
        st.pkv.len = w_.tc.groups;
        for (int cb = 1; cb < w_.tc.groups; ++cb) accept_code(g, uint32_t(cb), codes[cb]);
        st.t_pred += since(t0);
        return true;
    }

    void accept_code(GenState &g, uint32_t cb, int32_t code) override {
        auto &st = static_cast<QState &>(g);
        check_id(code, w_.pc.vocab, "code");
        st.codes[cb] = code;
    }

    void end_frame(GenState &g) override {
        auto &st = static_cast<QState &>(g);
        const int D = w_.tc.hidden;
        st.pending.assign(size_t(D), 0.f);
        std::vector<float> tmp(static_cast<size_t>(D));
        mat_gather(w_.codec_emb, st.codes[0], tmp.data());
        for (int d = 0; d < D; ++d) st.pending[size_t(d)] += tmp[size_t(d)];
        for (int q = 1; q < w_.tc.groups; ++q) {
            mat_gather(w_.pred_emb[size_t(q - 1)], st.codes[size_t(q)], tmp.data());
            for (int d = 0; d < D; ++d) st.pending[size_t(d)] += tmp[size_t(d)];
        }
        ++st.frames_done;
    }

    // ---------------- декодер кодека ----------------
    void decode(GenState &g, const int16_t *codes, size_t n_frames, std::vector<float> &pcm) override {
        auto &st = static_cast<QState &>(g);
        if (!st.warmed) { const auto tw = Clock::now(); warm(st); st.t_warm += since(tw); }
        const auto t0 = Clock::now();
        if (st.ref_decode) feed_chunked(st, codes, int64_t(n_frames), &pcm);
        else w_.codec_forward(st.cs, codes, int(n_frames), pcm);
        st.t_decode += since(t0);
        st.decode_frames += int64_t(n_frames);
    }
    void flush(GenState &, std::vector<float> &) override {}
    bool decode_overlappable() const override {
        return gpu_ && w_.codec_gpu < 0 && w_.talker.gpu_id >= 0 && w_.pred.gpu_id >= 0 && !std::getenv("VOX_NO_OVERLAP");
    }

private:
    void check_id(int64_t id, int64_t n, const char *what) const {
        if (id < 0 || id >= n) fail(VOX_ERR_ARG, std::string(what) + " id " + std::to_string(id) + " out of range");
    }
    void gather_codec(int id, float *out) const {
        check_id(id, w_.codec_emb.rows, "codec token");
        mat_gather(w_.codec_emb, id, out);
    }
    // text_embedding → text_projection (fc1 + SiLU + fc2)
    void text_project(const std::vector<int32_t> &toks, float *out) const {
        const int th = w_.tc.text_hidden;
        const int64_t n = int64_t(toks.size());
        std::vector<float> e(size_t(n) * size_t(th)), h(size_t(n) * size_t(th));
        for (int64_t i = 0; i < n; ++i) {
            check_id(toks[size_t(i)], w_.text_emb.rows, "text token");
            mat_gather(w_.text_emb, toks[size_t(i)], &e[size_t(i) * size_t(th)]);
        }
        matmul(w_.tp1, e.data(), n, h.data());
        for (auto &x : h) x = silu(x);
        matmul(w_.tp2, h.data(), n, out);
    }
    void project_small(const float *x, int64_t m, float *out) const {
        if (w_.has_small) matmul(w_.small_proj, x, m, out);
        else std::copy(x, x + m * w_.pc.hidden, out);
    }
    const float *elem(const QState &st, int64_t idx, bool ended) const {
        const int D = w_.tc.hidden;
        if (idx < st.text_n) return &st.text_stream[size_t(idx) * size_t(D)];
        if (ended && idx == st.text_n) return st.eos.data();
        return st.pad.data();
    }

    static constexpr int64_t kDecChunk = 300, kDecCtx = 25;  // qwen_tts: chunked_decode(chunk_size=300, left_context_size=25)

    // Кадры в режиме VOX_DECODER_REFERENCE: на каждой границе куска (абсолютный кадр кратен 300) состояние
    // декодера сбрасывается и заново прогревается 25 предыдущими кадрами — ровно как отдельный вызов декодера
    // на codes[start-25 : end] в оригинале (позиции RoPE тоже начинаются с нуля). out == nullptr → выход отбрасывается.
    void feed_chunked(QState &st, const int16_t *codes, int64_t n, std::vector<float> *out) {
        const int64_t Q = w_.tc.groups;
        std::vector<float> sink;
        int64_t i = 0;
        while (i < n) {
            if (st.abs_frame > 0 && st.abs_frame % kDecChunk == 0 && st.reset_done_at != st.abs_frame) {
                st.cs = w_.new_codec_state();
                const int64_t hn = int64_t(st.hist.size()) / Q;
                if (hn) { sink.clear(); w_.codec_forward(st.cs, st.hist.data(), int(hn), sink); }
                st.reset_done_at = st.abs_frame;
            }
            const int64_t next = (st.abs_frame / kDecChunk + 1) * kDecChunk;
            const int64_t m = std::min(n - i, next - st.abs_frame);
            if (out) w_.codec_forward(st.cs, codes + i * Q, int(m), *out);
            else { sink.clear(); w_.codec_forward(st.cs, codes + i * Q, int(m), sink); }
            st.hist.insert(st.hist.end(), codes + i * Q, codes + (i + m) * Q);
            if (int64_t(st.hist.size()) > kDecCtx * Q) st.hist.erase(st.hist.begin(), st.hist.end() - kDecCtx * Q);
            st.abs_frame += m;
            i += m;
        }
    }

    // Прогрев в режиме VOX_DECODER_REFERENCE: от начала куска, в котором лежит первый кадр речи.
    void warm_reference(QState &st) {
        const int64_t Q = w_.tc.groups;
        st.cs = w_.new_codec_state();
        st.abs_frame = 0; st.reset_done_at = -1; st.hist.clear();
        if (st.ref_frames == 0) return;
        const int64_t start = (st.ref_frames / kDecChunk) * kDecChunk;
        const int64_t h0 = std::max<int64_t>(0, start - kDecCtx);
        std::vector<int16_t> head(st.ref_codes.begin() + h0 * Q, st.ref_codes.begin() + start * Q);
        uint64_t key = fnv1a64(st.ref_codes.data() + size_t(h0 * Q), size_t((st.ref_frames - h0) * Q) * 2) ^ 0x5245464445434F44ull;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = warm_cache_.find(key);
            if (it != warm_cache_.end()) {
                st.cs = w_.clone_codec_state(*it->second);
                st.abs_frame = st.ref_frames; st.reset_done_at = start > 0 ? start : -1;
                const int64_t t0 = std::max<int64_t>(0, st.ref_frames - kDecCtx);
                st.hist.assign(st.ref_codes.begin() + t0 * Q, st.ref_codes.begin() + st.ref_frames * Q);
                return;
            }
        }
        if (start > 0) {  // граница куска уже пройдена внутри референса: свежий декодер + 25 кадров контекста
            std::vector<float> sink;
            w_.codec_forward(st.cs, head.data(), int(start - h0), sink);
            st.reset_done_at = start;
        }
        st.abs_frame = start;
        st.hist = head;
        for (int64_t f = start; f < st.ref_frames; f += 16)
            feed_chunked(st, st.ref_codes.data() + f * Q, std::min<int64_t>(16, st.ref_frames - f), nullptr);
        std::lock_guard<std::mutex> lk(mu_);
        if (warm_cache_.size() > 16) warm_cache_.clear();
        warm_cache_[key] = std::make_shared<CodecState>(w_.clone_codec_state(st.cs));
    }

    // Прогрев декодера кодека кодами референса (в оригинале они декодируются вместе с речью, а затем
    // обрезаются). Результат одинаков для одного голоса, поэтому кэшируется.
    void warm(QState &st) {
        st.warmed = true;
        if (st.ref_decode) { warm_reference(st); return; }
        st.cs = w_.new_codec_state();
        if (st.ref_frames == 0) return;
        const int Q0 = w_.tc.groups;
        const int64_t ctx = (st.context_frames == 0xFFFFFFFFu) ? st.ref_frames : std::min<int64_t>(st.ref_frames, st.context_frames ? st.context_frames : st.ref_frames);
        const int64_t first = st.ref_frames - ctx;  // прогреваем только последние ctx кадров референса
        const uint64_t key = fnv1a64(st.ref_codes.data() + size_t(first) * size_t(Q0), size_t(ctx) * size_t(Q0) * 2);
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = warm_cache_.find(key);
            if (it != warm_cache_.end()) { st.cs = w_.clone_codec_state(*it->second); return; }
        }
        std::vector<float> sink;
        const int Q = w_.tc.groups;
        for (int64_t f = first; f < st.ref_frames; f += 16) {
            const int n = int(std::min<int64_t>(16, st.ref_frames - f));
            sink.clear();
            w_.codec_forward(st.cs, st.ref_codes.data() + size_t(f) * size_t(Q), n, sink);
        }
        std::lock_guard<std::mutex> lk(mu_);
        if (warm_cache_.size() > 16) warm_cache_.clear();
        warm_cache_[key] = std::make_shared<CodecState>(w_.clone_codec_state(st.cs));
    }

    std::shared_ptr<TmodFile> file_;
    Qwen3Weights w_;
    Tokenizer tok_;
    ResourcePlan plan_;
    bool gpu_ = false, codec_on_gpu_ = false;
    ModelInfo info_;
    std::vector<int32_t> role_;
    std::mutex mu_;
    std::shared_ptr<Totals> totals_ = std::make_shared<Totals>();
    std::unordered_map<uint64_t, std::shared_ptr<CodecState>> warm_cache_;
    std::unique_ptr<Qwen3VoiceEncoder> venc_;
};

// ---------------------------------------------------------------- семейство
Capabilities qwen3_capabilities(const TmodFile &f) {
    const json &m = f.manifest();
    const std::string type = m.at("model").value("type", "base");
    const json &t = m.at("qwen3").at("talker");
    Capabilities c;
    if (t.contains("spk_id") && t["spk_id"].is_object())
        for (auto it = t["spk_id"].begin(); it != t["spk_id"].end(); ++it) c.speakers.push_back(it.key());
    std::sort(c.speakers.begin(), c.speakers.end());
    const bool can_encode = f.has("spk.mel_basis") && f.has("enc.encoder.layers.0.conv.weight");
    c.clone_icl = c.clone_xvector = type == "base" && can_encode;
    c.named_speakers = type == "custom_voice" && !c.speakers.empty();
    const std::string size = m.at("model").value("variant", "");
    const bool small = size.find("0b6") != std::string::npos || size.find("0.6b") != std::string::npos;
    c.instruct = (type == "custom_voice" && !small) || type == "voice_design";
    c.requires_instruct = type == "voice_design";
    if (t.contains("codec_language_id") && t["codec_language_id"].is_object())
        for (auto it = t["codec_language_id"].begin(); it != t["codec_language_id"].end(); ++it) c.languages.push_back(it.key());
    std::sort(c.languages.begin(), c.languages.end());
    c.languages.insert(c.languages.begin(), "auto");
    return c;
}

double xfmr_layer_params(double H, double heads, double kv, double hd, double I) { return H * heads * hd + 2 * H * kv * hd + heads * hd * H + 3 * H * I; }

// Оценка FLOPs декодера кодека на кадр по конфигурации (транспонированные свёртки, ConvNeXt, остаточные блоки).
double codec_flops_per_frame(const json &c) {
    const double cb = c.at("codebook_dim"), lat = c.at("latent_dim"), H = c.at("hidden_size"), I = c.at("intermediate_size"), dec = c.at("decoder_dim");
    const double layers = c.at("num_hidden_layers");
    const auto rates = c.at("upsample_rates").get<std::vector<int>>(), ratios = c.at("upsampling_ratios").get<std::vector<int>>();
    double fl = 2 * 3 * cb * lat;                                  // pre_conv
    fl += layers * 2 * (4 * H * H + 3 * H * I) + 4 * lat * H;      // pre-трансформер
    double pos = 1;
    for (int r : ratios) {                                         // апсэмплинг + ConvNeXt
        fl += 2 * lat * lat * r * pos;
        pos *= r;
        fl += pos * (2 * 7 * lat + 2 * 2 * 4 * lat * lat);
    }
    fl += pos * 2 * 7 * lat * dec;                                 // conv0
    double ch = dec;
    for (size_t b = 0; b < rates.size(); ++b) {
        const double out = ch / 2;
        fl += pos * 2 * ch * out * 2 * rates[b];                   // транспонированная свёртка
        pos *= rates[b];
        fl += pos * 3 * (2 * 7 * out * out + 2 * out * out);       // три остаточных блока
        ch = out;
    }
    fl += pos * 2 * 7 * ch;
    return fl;
}

class Qwen3Family final : public Family {
public:
    std::string name() const override { return "qwen3-tts"; }
    std::string description() const override {
        return "Qwen3-TTS-12Hz: talker + code predictor + RVQ codec decoder (Base, CustomVoice, VoiceDesign; 0.6B and 1.7B)";
    }

    ModelFootprint footprint(const TmodFile &f) const override {
        const json &m = f.manifest();
        const json &q = m.at("qwen3"), &t = q.at("talker"), &p = q.at("predictor"), &c = q.at("codec");
        ModelFootprint fp;
        fp.family = name();
        fp.variant = m.at("model").value("variant", "");
        const double H = t.at("hidden_size"), L = t.at("num_hidden_layers"), heads = t.at("num_attention_heads"), kv = t.at("num_key_value_heads");
        const double hd = t.value("head_dim", int(H / heads)), I = t.at("intermediate_size"), V = t.at("vocab_size");
        const double PH = p.at("hidden_size"), PL = p.at("num_hidden_layers"), pheads = p.at("num_attention_heads"), pkv = p.at("num_key_value_heads");
        const double phd = p.value("head_dim", int(PH / pheads)), PI = p.at("intermediate_size"), PV = p.at("vocab_size");
        const int groups = t.at("num_code_groups");
        const uint64_t max_frames = m.value("generation", json::object()).value("max_frames", 2048u);
        const uint64_t cap = std::min<uint64_t>(3072, 512 + max_frames + 16);

        StageDesc tk;
        tk.name = "talker";
        tk.prefixes = {"talker.model.layers.", "talker.model.norm.", "talker.codec_head.", "talker.text_projection."};
        tk.layers = int(L);
        tk.splittable = true;   // первые k слоёв и их KV — на GPU, остальные на CPU (kernels.cpp: transformer_forward)
        tk.layer_bytes = blob_bytes(f, {"talker.model.layers."}) / uint64_t(L);
        tk.bytes_per_frame = double(blob_bytes(f, tk.prefixes));
        tk.flops_per_frame = 2 * (L * xfmr_layer_params(H, heads, kv, hd, I) + H * V);
        tk.session_vram_bytes = uint64_t(L * 2 * kv * hd * 4) * cap;
        tk.session_ram_bytes = tk.session_vram_bytes;
        fp.stages.push_back(tk);

        StageDesc pr;
        pr.name = "predictor";
        pr.prefixes = {"talker.code_predictor."};
        pr.bytes_per_frame = double(groups - 1) * double(blob_bytes(f, {"talker.code_predictor.model.layers."}) + blob_bytes(f, {"talker.code_predictor.model.norm."}));
        pr.flops_per_frame = 2 * double(groups - 1) * (PL * xfmr_layer_params(PH, pheads, pkv, phd, PI) + PH * PV);
        pr.session_vram_bytes = uint64_t(PL * 2 * pkv * phd * 4) * 40;
        pr.session_ram_bytes = pr.session_vram_bytes;
        fp.stages.push_back(pr);

        StageDesc cd;
        cd.name = "codec";
        cd.prefixes = {"codec."};
        cd.exclude = {"codec.quantizer."};
        cd.bytes_per_frame = double(blob_bytes(f, cd.prefixes, cd.exclude)) / 4.0;  // веса читаются раз на чанк из 4 кадров
        cd.flops_per_frame = codec_flops_per_frame(c);
        cd.session_vram_bytes = 24u << 20;
        cd.session_ram_bytes = 24u << 20;
        fp.stages.push_back(cd);

        fp.always_ram_bytes = blob_bytes(f, {"spk.", "enc."}) + (64u << 20);
        return fp;
    }

    Capabilities capabilities(const TmodFile &f) const override { return qwen3_capabilities(f); }

    std::unique_ptr<Backend> create(std::shared_ptr<TmodFile> f, const ResourcePlan &plan) const override {
        return std::make_unique<Qwen3Backend>(std::move(f), plan);
    }
};

}  // namespace

std::shared_ptr<Family> make_qwen3_family() { return std::make_shared<Qwen3Family>(); }

}  // namespace vox
