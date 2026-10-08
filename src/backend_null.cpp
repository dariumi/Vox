// Тестовый бэкенд "vox-test".
//
// Не нейросеть: детерминированная имитация TTS-пайплайна с теми же стадиями и
// контрактами, что у настоящей модели (текст по частям, кодбуки, стриминговый
// декодер, клонирование голоса). Нужен, чтобы рантайм, C ABI, CLI и Python
// разрабатывались и тестировались независимо от весов и железа.
#include <algorithm>
#include <cmath>

#include "backend.hpp"
#include "family.hpp"
#include "gpu.hpp"

namespace vox {
namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<int32_t> utf8_codepoints(const std::string &s) {
    std::vector<int32_t> out;
    for (size_t i = 0; i < s.size();) {
        auto c = static_cast<unsigned char>(s[i]);
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        int32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
        for (int k = 1; k < len && i + size_t(k) < s.size(); ++k)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + size_t(k)]) & 0x3F);
        out.push_back(cp);
        i += size_t(len);
    }
    return out;
}

struct NullState : GenState {
    std::vector<int32_t> text;
    uint64_t frame = 0;
    int32_t code0 = 0;
    double phase = 0, base_hz = 160;
};

class NullBackend final : public Backend {
public:
    NullBackend(std::shared_ptr<TmodFile> tmod) : tmod_(std::move(tmod)) {
        const json &m = tmod_->manifest();
        info_.sample_rate = m.at("audio").value("sample_rate", 24000u);
        info_.samples_per_frame = m.at("audio").value("samples_per_frame", 1920u);
        info_.num_codebooks = m.at("audio").value("num_codebooks", 4u);
        info_.codebook_size = m.at("audio").value("codebook_size", 64u);
        info_.eos_code = int32_t(info_.codebook_size);  // talker-словарь = коды + EOS
        // Проверка, что веса реально читаются из mmap.
        const TensorView &w = tmod_->get("test.weights");
        weights_hash_ = fnv1a64(w.data, w.nbytes);
    }

    std::string name() const override { return "null-cpu"; }
    const ModelInfo &info() const override { return info_; }
    std::string fingerprint() const override {
        std::string prec = tmod_->manifest().value("precision", json::object()).dump();
        return "cpu-null-" + hex64(fnv1a64(prec, weights_hash_)).substr(0, 12);
    }

    std::vector<int32_t> tokenize(const std::string &utf8) override { return utf8_codepoints(utf8); }

    std::vector<float> speaker_embedding(const float *pcm, size_t n, uint32_t) override {
        // 16 значений: RMS 16 равных сегментов, нормированные по максимуму.
        std::vector<float> e(16, 0.f);
        if (n == 0) return e;
        for (size_t s = 0; s < 16; ++s) {
            size_t a = n * s / 16, b = n * (s + 1) / 16;
            double acc = 0;
            for (size_t i = a; i < b; ++i) acc += double(pcm[i]) * pcm[i];
            e[s] = float(std::sqrt(acc / double(std::max<size_t>(1, b - a))));
        }
        float mx = *std::max_element(e.begin(), e.end());
        if (mx > 0) for (auto &v : e) v /= mx;
        return e;
    }

    std::vector<int16_t> encode_codes(const float *pcm, size_t n, uint32_t sr, int64_t &frames) override {
        double secs = double(n) / sr;
        frames = std::max<int64_t>(1, int64_t(std::ceil(secs * info_.sample_rate / info_.samples_per_frame)));
        std::vector<int16_t> codes(size_t(frames) * info_.num_codebooks);
        for (int64_t f = 0; f < frames; ++f) {
            size_t a = size_t(double(n) * f / frames), b = size_t(double(n) * (f + 1) / frames);
            double acc = 0;
            for (size_t i = a; i < b; ++i) acc += std::fabs(pcm[i]);
            int base = int(acc * 1000.0 / double(std::max<size_t>(1, b - a)));
            for (uint32_t c = 0; c < info_.num_codebooks; ++c)
                codes[size_t(f) * info_.num_codebooks + c] = int16_t((base + int(c) * 7) % int(info_.codebook_size));
        }
        return codes;
    }

    std::unique_ptr<GenState> begin(const VoiceData &v, const std::string &) override {
        auto st = std::make_unique<NullState>();
        if (!v.speaker_emb.empty()) {
            double m = 0;
            for (float x : v.speaker_emb) m += x;
            st->base_hz = 110.0 + 120.0 * m / double(v.speaker_emb.size());
        }
        return st;
    }

    void append_text(GenState &s, const std::vector<int32_t> &t) override {
        auto &st = static_cast<NullState &>(s);
        st.text.insert(st.text.end(), t.begin(), t.end());
    }

    // Модель «читает» один символ на 2 кадра (≈ 160 мс), как talker с опережением текста.
    bool needs_text(const GenState &s, bool text_ended) const override {
        auto &st = static_cast<const NullState &>(s);
        return !text_ended && st.frame / 2 >= st.text.size();
    }

    void talker_logits(GenState &s, bool text_ended, std::vector<float> &logits) override {
        auto &st = static_cast<NullState &>(s);
        const uint32_t n = info_.codebook_size;
        logits.assign(n + 1, 0.f);
        for (uint32_t i = 0; i < n; ++i)
            logits[i] = float(std::sin(double(i) * 1.7 + double(st.frame) * 0.31));  // «шум» для сэмплинга
        size_t t = st.frame / 2;
        if (text_ended && t >= st.text.size()) {
            logits[n] = 20.f;  // EOS
            return;
        }
        int32_t target = int32_t((uint64_t(st.text[t]) * 31u + st.frame) % n);
        logits[size_t(target)] = 6.f;
        logits[n] = -20.f;
    }

    void accept_code0(GenState &s, int32_t code) override { static_cast<NullState &>(s).code0 = code; }

    void predictor_logits(GenState &s, uint32_t cb, std::vector<float> &logits) override {
        auto &st = static_cast<NullState &>(s);
        const uint32_t n = info_.codebook_size;
        logits.assign(n, 0.f);
        for (uint32_t i = 0; i < n; ++i) logits[i] = float(std::cos(double(i) * 0.9 + cb));
        logits[size_t((uint32_t(st.code0) + cb * 13u) % n)] = 6.f;
    }

    void accept_code(GenState &, uint32_t, int32_t) override {}
    void end_frame(GenState &s) override { ++static_cast<NullState &>(s).frame; }

    void decode(GenState &s, const int16_t *codes, size_t n_frames, std::vector<float> &pcm) override {
        auto &st = static_cast<NullState &>(s);
        const uint32_t spf = info_.samples_per_frame, nc = info_.num_codebooks;
        for (size_t f = 0; f < n_frames; ++f) {
            double hz = st.base_hz + 3.0 * codes[f * nc];
            double amp = 0.15 + 0.05 * (codes[f * nc + (nc > 1 ? 1 : 0)] % 4) / 3.0;
            for (uint32_t i = 0; i < spf; ++i) {
                st.phase += 2 * kPi * hz / info_.sample_rate;
                if (st.phase > 2 * kPi) st.phase -= 2 * kPi;
                pcm.push_back(float(amp * std::sin(st.phase)));
            }
        }
    }

    void flush(GenState &, std::vector<float> &) override {}

private:
    std::shared_ptr<TmodFile> tmod_;
    ModelInfo info_;
    uint64_t weights_hash_ = 0;
};

}  // namespace

// Модель тестового бэкенда (vox make-test-model, юнит-тесты).
void write_test_model(const std::string &out) {
    json m = {
        {"format", "1.0"},
        {"model", {{"family", "vox-test"}, {"variant", "null-4cb"}, {"weights_sha256", "test"}}},
        {"target", {{"backend", "cpu"}, {"arch", "any"}, {"features", json::array()}}},
        {"audio", {{"sample_rate", 24000}, {"samples_per_frame", 1920}, {"frame_rate", 12.5},
                   {"num_codebooks", 4}, {"codebook_size", 64}}},
        {"stages", json::array({
            {{"name", "speaker_encoder"}, {"kind", "encoder"}, {"input", "ref_audio"}, {"output", "speaker_emb"}},
            {{"name", "talker"}, {"kind", "autoregressive"}, {"kv_cache", true},
             {"inputs", {"text_tokens", "speaker_emb", "ref_codes"}}, {"output", "code_0"}},
            {{"name", "code_predictor"}, {"kind", "subloop"}, {"per", "frame"}, {"input", "code_0"}, {"output", "codes_rest"}},
            {{"name", "decoder"}, {"kind", "streaming_conv"}, {"context", "left_only"}, {"input", "codes_all"}, {"output", "pcm"}},
        })},
        {"precision", {{"talker", "fp32"}, {"code_predictor", "fp32"}, {"decoder", "fp32"}}},
        {"generation", {{"temperature", 0.9}, {"top_k", 50}, {"top_p", 1.0}, {"repetition_penalty", 1.05},
                        {"chunk_frames", 4}, {"max_frames", 2048}}},
    };
    std::vector<float> w(4096);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float(i % 97) / 97.0f;
    std::map<std::string, Tensor> blobs;
    blobs["test.weights"] = Tensor::from(DType::F32, {64, 64}, w);
    tmod_write(out, m, blobs);
}


namespace {
class TestFamily final : public Family {
public:
    std::string name() const override { return "vox-test"; }
    std::string description() const override { return "synthetic deterministic backend for testing the runtime without weights"; }
    ModelFootprint footprint(const TmodFile &f) const override {
        ModelFootprint fp;
        fp.family = name();
        fp.variant = f.manifest().at("model").value("variant", "");
        StageDesc s;
        s.name = "model";
        s.gpu_capable = false;
        s.weight_bytes = blob_bytes(f, {"test."});
        s.session_ram_bytes = 1u << 16;
        fp.stages.push_back(s);
        return fp;
    }
    Capabilities capabilities(const TmodFile &) const override {
        Capabilities c;
        c.clone_icl = c.clone_xvector = true;
        c.languages = {"auto"};
        return c;
    }
    std::unique_ptr<Backend> create(std::shared_ptr<TmodFile> f, const ResourcePlan &) const override {
        return std::make_unique<NullBackend>(std::move(f));
    }
};
}  // namespace

std::shared_ptr<Family> make_test_family() { return std::make_shared<TestFamily>(); }

}  // namespace vox
