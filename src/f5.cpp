#include "f5.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "gpu.hpp"
#include "kernels.hpp"
#include "sampling.hpp"

namespace vox {
namespace {

constexpr double kPi = 3.14159265358979323846;

// ---------- загрузка тензоров ----------
const float *vec_f32(const TmodFile &f, const std::string &name) {
    const TensorView &t = f.get(name);
    if (t.dtype != DType::F32) fail(VOX_ERR_FORMAT, "tensor must be F32: " + name);
    return reinterpret_cast<const float *>(t.data);
}

Mat get_mat(const TmodFile &f, const std::string &name, bool with_bias) {
    const TensorView &t = f.get(name);
    Mat m;
    m.data = t.data;
    m.rows = t.shape.at(0);
    m.cols = t.shape.size() > 1 ? t.shape[1] : 1;
    switch (t.dtype) {
        case DType::F32: m.type = WType::F32; break;
        case DType::F16: m.type = WType::F16; break;
        case DType::BF16: m.type = WType::BF16; break;
        case DType::I8:
            m.type = WType::Q8;
            m.scale = vec_f32(f, name + ".scale");
            break;
        case DType::U8:
            m.type = WType::Q4;
            m.cols *= 2;
            m.scale = vec_f32(f, name + ".scale");
            break;
        default: fail(VOX_ERR_FORMAT, "unsupported weight dtype for " + name);
    }
    if (with_bias) {
        const std::string b = name.substr(0, name.size() - 6) + "bias";
        if (f.has(b)) m.bias = vec_f32(f, b);
    }
    return m;
}

uint64_t mat_bytes(const Mat &m) {
    return uint64_t(m.rows) * mat_row_bytes(m.type, m.cols) + (m.scale ? uint64_t(m.rows) * uint64_t(m.cols / 32) * 4 : 0);
}

// ---------- поэлементные функции ----------
inline float silu_f(float x) { return x / (1.0f + std::exp(-x)); }
inline float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}
inline float mish_f(float x) {
    const float sp = x > 20.f ? x : std::log1p(std::exp(x));
    return x * std::tanh(sp);
}

void ln_rows(const float *x, int64_t R, int64_t D, const float *w, const float *b, float eps, float *y) {
    parallel_for(R, 16, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) layernorm(x + r * D, w, b, D, eps, y + r * D);
    });
}

}  // namespace

// ============================================================================
// Токенизатор
// ============================================================================
namespace {

uint32_t next_cp(const std::string &s, size_t &i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    uint32_t cp = 0xFFFD;
    size_t len = 1;
    if (c < 0x80) cp = c;
    else if ((c >> 5) == 6 && i + 1 < s.size()) { cp = (c & 0x1Fu) << 6 | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu); len = 2; }
    else if ((c >> 4) == 14 && i + 2 < s.size()) {
        cp = (c & 0x0Fu) << 12 | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6 | (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
        len = 3;
    } else if ((c >> 3) == 30 && i + 3 < s.size()) {
        cp = (c & 0x07u) << 18 | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 12 | (static_cast<unsigned char>(s[i + 2]) & 0x3Fu) << 6 |
             (static_cast<unsigned char>(s[i + 3]) & 0x3Fu);
        len = 4;
    }
    i += len;
    return cp;
}

std::string cp_to_utf8(uint32_t cp) {
    std::string o;
    if (cp < 0x80) o += char(cp);
    else if (cp < 0x800) { o += char(0xC0 | (cp >> 6)); o += char(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { o += char(0xE0 | (cp >> 12)); o += char(0x80 | ((cp >> 6) & 0x3F)); o += char(0x80 | (cp & 0x3F)); }
    else { o += char(0xF0 | (cp >> 18)); o += char(0x80 | ((cp >> 12) & 0x3F)); o += char(0x80 | ((cp >> 6) & 0x3F)); o += char(0x80 | (cp & 0x3F)); }
    return o;
}

bool is_alnum_ascii(uint32_t c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool is_digit_ascii(uint32_t c) { return c >= '0' && c <= '9'; }

}  // namespace

void F5Tokenizer::load(const json &tok) {
    pinyin_ = tok.value("type", "pinyin") == "pinyin";
    vocab_.clear();
    pinyin_map_.clear();
    const json &v = tok.at("vocab");
    for (size_t i = 0; i < v.size(); ++i) vocab_.emplace(v[i].get<std::string>(), int32_t(i));
    if (tok.contains("pinyin"))
        for (auto it = tok["pinyin"].begin(); it != tok["pinyin"].end(); ++it) {
            size_t k = 0;
            const std::string key = it.key();
            const uint32_t cp = next_cp(key, k);
            pinyin_map_[cp] = it.value().get<std::string>();
        }
}

std::vector<std::string> F5Tokenizer::symbols(const std::string &text_in) const {
    std::vector<std::string> out;
    if (!pinyin_) {  // посимвольно
        size_t i = 0;
        while (i < text_in.size()) { const size_t s = i; next_cp(text_in, i); out.push_back(text_in.substr(s, i - s)); }
        return out;
    }
    // таблица замен из convert_char_to_pinyin: ; → ,  “” → "  ‘’ → '
    std::vector<uint32_t> cps;
    for (size_t i = 0; i < text_in.size();) {
        uint32_t c = next_cp(text_in, i);
        if (c == ';') c = ',';
        else if (c == 0x201C || c == 0x201D) c = '"';
        else if (c == 0x2018 || c == 0x2019) c = '\'';
        cps.push_back(c);
    }
    const size_t n = cps.size();
    size_t i = 0;
    while (i < n) {
        const uint32_t c = cps[i];
        if (is_alnum_ascii(c)) {  // сегмент [A-Za-z0-9]+(\.\d+)?%? — одно «слово» для rjieba
            size_t j = i;
            while (j < n && is_alnum_ascii(cps[j])) ++j;
            if (j + 1 < n && cps[j] == '.' && is_digit_ascii(cps[j + 1])) {
                j += 1;
                while (j < n && is_digit_ascii(cps[j])) ++j;
            }
            if (j < n && cps[j] == '%') ++j;
            const size_t len = j - i;
            if (len > 1 && !out.empty()) {
                const std::string &last = out.back();
                if (!(last == " " || last == ":" || last == "'" || last == "\"")) out.push_back(" ");
            }
            for (size_t k = i; k < j; ++k) out.push_back(std::string(1, char(cps[k])));
            i = j;
            continue;
        }
        if (c >= 0x3100 && c <= 0x9FFF) {  // китайский иероглиф: пробел + слог
            out.push_back(" ");
            auto it = pinyin_map_.find(c);
            out.push_back(it == pinyin_map_.end() ? cp_to_utf8(c) : it->second);
        } else {
            out.push_back(cp_to_utf8(c));
        }
        ++i;
    }
    return out;
}

std::vector<int32_t> F5Tokenizer::encode(const std::string &utf8) const {
    const auto syms = symbols(utf8);
    std::vector<int32_t> ids;
    ids.reserve(syms.size());
    for (const auto &s : syms) {
        auto it = vocab_.find(s);
        ids.push_back(it == vocab_.end() ? 0 : it->second);
    }
    return ids;
}

// ============================================================================
// Веса и вычисления
// ============================================================================
namespace {

struct CnxV2 {  // ConvNeXtV2Block текстового энкодера
    Conv1d dw;
    const float *ln_w = nullptr, *ln_b = nullptr, *grn_g = nullptr, *grn_b = nullptr;
    Mat pw1, pw2;
};
struct DitBlock {
    Mat q, k, v, o, ada, ff1, ff2;
};
struct VocosBlock {
    Conv1d dw;
    const float *ln_w = nullptr, *ln_b = nullptr, *gamma = nullptr;
    Mat pw1, pw2;
};

}  // namespace

struct F5Model::Impl {
    Mat time1, time2, text_emb, in_proj, norm_out, proj_out;
    std::vector<CnxV2> text_blocks;
    Conv1d pos1, pos2;
    std::vector<DitBlock> blocks;
    // Vocos
    Conv1d v_embed;
    const float *v_ln_w = nullptr, *v_ln_b = nullptr, *v_fln_w = nullptr, *v_fln_b = nullptr;
    std::vector<VocosBlock> v_blocks;
    Mat v_head;
    std::vector<Mat *> linears;  // все линейные слои — для подъёма на GPU
};

F5Model::F5Model(std::shared_ptr<TmodFile> f, bool gpu) : p_(new Impl), file_(f) {
    const json &m = f->manifest();
    const json &c = m.at("f5");
    cfg_.dim = c.at("dim"); cfg_.depth = c.at("depth"); cfg_.heads = c.at("heads");
    cfg_.dim_head = c.value("dim_head", 64); cfg_.ff_mult = c.value("ff_mult", 2);
    cfg_.text_dim = c.at("text_dim"); cfg_.text_rows = c.at("text_rows"); cfg_.mel_dim = c.value("mel_dim", 100);
    cfg_.conv_layers = c.value("conv_layers", 4); cfg_.conv_mult = c.value("conv_mult", 2);
    cfg_.mask_padding = c.value("text_mask_padding", true);
    if (!c.value("qk_norm", json()).is_null()) fail(VOX_ERR_UNSUPPORTED, "F5-TTS: qk_norm is not supported");
    if (c.value("long_skip_connection", false)) fail(VOX_ERR_UNSUPPORTED, "F5-TTS: long_skip_connection is not supported");
    if (c.value("text_embedding_average_upsampling", false)) fail(VOX_ERR_UNSUPPORTED, "F5-TTS: average upsampling is not supported");
    const json &ml = c.at("mel");
    cfg_.mel.sample_rate = ml.value("sample_rate", 24000); cfg_.mel.n_fft = ml.value("n_fft", 1024); cfg_.mel.hop = ml.value("hop", 256);
    cfg_.mel.win = ml.value("win", 1024); cfg_.mel.n_mels = ml.value("n_mels", 100);
    cfg_.sample_rate = uint32_t(cfg_.mel.sample_rate);
    if (cfg_.mel.n_mels != cfg_.mel_dim) fail(VOX_ERR_FORMAT, "F5-TTS: n_mels != mel_dim");
    if (cfg_.dim % 16 || cfg_.heads * cfg_.dim_head < 1) fail(VOX_ERR_FORMAT, "F5-TTS: bad dimensions");
    const json &vc = c.at("vocos");
    cfg_.vocos_dim = vc.at("dim"); cfg_.vocos_inter = vc.at("intermediate"); cfg_.vocos_layers = vc.at("layers");
    cfg_.vocos_n_fft = vc.value("n_fft", 1024); cfg_.vocos_hop = vc.value("hop", 256);
    tok_.load(c.at("tokenizer"));

    const TensorView &fb = f->get("mel.fb");
    if (fb.dtype != DType::F32 || fb.numel() != int64_t(cfg_.mel.n_mels) * (cfg_.mel.n_fft / 2 + 1)) fail(VOX_ERR_FORMAT, "F5-TTS: bad mel filterbank");
    mel_fb_.assign(reinterpret_cast<const float *>(fb.data), reinterpret_cast<const float *>(fb.data) + fb.numel());

    Impl &P = *p_;
    const std::string d = "dit.";
    P.time1 = get_mat(*f, d + "time_embed.time_mlp.0.weight", true);
    P.time2 = get_mat(*f, d + "time_embed.time_mlp.2.weight", true);
    P.text_emb = get_mat(*f, d + "text_embed.text_embed.weight", false);
    P.in_proj = get_mat(*f, d + "input_embed.proj.weight", true);
    auto load_conv = [&](Conv1d &cv, const std::string &name, int cin, int cout, int K, int groups) {
        const TensorView &w = f->get(name + ".weight");
        if (w.dtype != DType::F32) fail(VOX_ERR_FORMAT, "conv weights must be F32: " + name);
        if (w.numel() != int64_t(cout) * (cin / groups) * K) fail(VOX_ERR_FORMAT, "bad conv shape: " + name);
        cv.load(reinterpret_cast<const float *>(w.data), cin, cout, K, groups, f->has(name + ".bias") ? vec_f32(*f, name + ".bias") : nullptr);
    };
    load_conv(P.pos1, d + "input_embed.conv_pos_embed.conv1d.0", cfg_.dim, cfg_.dim, 31, 16);
    load_conv(P.pos2, d + "input_embed.conv_pos_embed.conv1d.2", cfg_.dim, cfg_.dim, 31, 16);
    P.text_blocks.resize(size_t(cfg_.conv_layers));
    for (int i = 0; i < cfg_.conv_layers; ++i) {
        const std::string p = d + "text_embed.text_blocks." + std::to_string(i) + ".";
        CnxV2 &b = P.text_blocks[size_t(i)];
        load_conv(b.dw, p + "dwconv", cfg_.text_dim, cfg_.text_dim, 7, cfg_.text_dim);
        b.ln_w = vec_f32(*f, p + "norm.weight"); b.ln_b = vec_f32(*f, p + "norm.bias");
        b.pw1 = get_mat(*f, p + "pwconv1.weight", true);
        b.pw2 = get_mat(*f, p + "pwconv2.weight", true);
        b.grn_g = vec_f32(*f, p + "grn.gamma"); b.grn_b = vec_f32(*f, p + "grn.beta");
    }
    P.blocks.resize(size_t(cfg_.depth));
    for (int i = 0; i < cfg_.depth; ++i) {
        const std::string p = d + "transformer_blocks." + std::to_string(i) + ".";
        DitBlock &b = P.blocks[size_t(i)];
        b.ada = get_mat(*f, p + "attn_norm.linear.weight", true);
        b.q = get_mat(*f, p + "attn.to_q.weight", true);
        b.k = get_mat(*f, p + "attn.to_k.weight", true);
        b.v = get_mat(*f, p + "attn.to_v.weight", true);
        b.o = get_mat(*f, p + "attn.to_out.0.weight", true);
        b.ff1 = get_mat(*f, p + "ff.ff.0.0.weight", true);
        b.ff2 = get_mat(*f, p + "ff.ff.2.weight", true);
    }
    P.norm_out = get_mat(*f, d + "norm_out.linear.weight", true);
    P.proj_out = get_mat(*f, d + "proj_out.weight", true);

    const std::string v = "vocos.";
    load_conv(P.v_embed, v + "backbone.embed", cfg_.mel_dim, cfg_.vocos_dim, 7, 1);
    P.v_ln_w = vec_f32(*f, v + "backbone.norm.weight"); P.v_ln_b = vec_f32(*f, v + "backbone.norm.bias");
    P.v_fln_w = vec_f32(*f, v + "backbone.final_layer_norm.weight"); P.v_fln_b = vec_f32(*f, v + "backbone.final_layer_norm.bias");
    P.v_blocks.resize(size_t(cfg_.vocos_layers));
    for (int i = 0; i < cfg_.vocos_layers; ++i) {
        const std::string p = v + "backbone.convnext." + std::to_string(i) + ".";
        VocosBlock &b = P.v_blocks[size_t(i)];
        load_conv(b.dw, p + "dwconv", cfg_.vocos_dim, cfg_.vocos_dim, 7, cfg_.vocos_dim);
        b.ln_w = vec_f32(*f, p + "norm.weight"); b.ln_b = vec_f32(*f, p + "norm.bias");
        b.pw1 = get_mat(*f, p + "pwconv1.weight", true);
        b.pw2 = get_mat(*f, p + "pwconv2.weight", true);
        b.gamma = vec_f32(*f, p + "gamma");
    }
    P.v_head = get_mat(*f, v + "head.out.weight", true);

    // линейные слои DiT и текстового энкодера
    P.linears = {&P.time1, &P.time2, &P.in_proj, &P.norm_out, &P.proj_out};
    for (auto &b : P.text_blocks) { P.linears.push_back(&b.pw1); P.linears.push_back(&b.pw2); }
    for (auto &b : P.blocks) for (Mat *x : {&b.q, &b.k, &b.v, &b.o, &b.ada, &b.ff1, &b.ff2}) P.linears.push_back(x);
    for (auto &b : P.v_blocks) { P.linears.push_back(&b.pw1); P.linears.push_back(&b.pw2); }
    P.linears.push_back(&P.v_head);
    for (Mat *x : P.linears) weight_bytes_ += mat_bytes(*x);
    weight_bytes_ += mat_bytes(P.text_emb);
    linear_mats_ = int(P.linears.size());
    if (gpu && gpu_active())
        for (Mat *x : P.linears)
            if (x->cols % 32 == 0 || x->type == WType::F32 || x->type == WType::F16 || x->type == WType::BF16) {
                x->gpu = gpu_upload(*x);
                if (x->gpu >= 0) ++gpu_mats_;
            }
}

F5Model::~F5Model() = default;

void F5Model::ref_mel(const float *pcm, size_t n, std::vector<float> &mel, int &frames) const {
    mel_log_spectrogram(cfg_.mel, mel_fb_.data(), pcm, n, mel, frames);
}

void F5Model::noise(uint64_t seed, uint32_t chunk, size_t n, float *out) {
    const DetRng rng(seed);
    const uint32_t stream = 0xF5000000u + chunk;
    parallel_for(int64_t((n + 1) / 2), 4096, [&](int64_t p0, int64_t p1) {
        for (int64_t p = p0; p < p1; ++p) {
            const double u1 = double(rng.uniform(uint64_t(p), stream, 0)), u2 = double(rng.uniform(uint64_t(p), stream, 1));
            const double r = std::sqrt(-2.0 * std::log(1.0 - u1));  // 1 - u1 ∈ (0, 1]
            const double a = 2.0 * kPi * u2;
            out[size_t(p) * 2] = float(r * std::cos(a));
            if (size_t(p) * 2 + 1 < n) out[size_t(p) * 2 + 1] = float(r * std::sin(a));
        }
    });
}

std::vector<float> F5Model::timesteps(const F5Sampler &s) {
    const int n = std::max(1, s.nfe);
    std::vector<double> t(size_t(n) + 1);
    static const std::map<int, std::vector<int>> epss = {
        {5, {0, 2, 4, 8, 16, 32}},
        {6, {0, 2, 4, 6, 8, 16, 32}},
        {7, {0, 2, 4, 6, 8, 16, 24, 32}},
        {10, {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32}},
        {12, {0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32}},
        {16, {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32}}};
    auto it = epss.find(n);
    if (s.epss && it != epss.end()) {
        t.resize(it->second.size());
        for (size_t i = 0; i < t.size(); ++i) t[i] = it->second[i] / 32.0;
    } else {
        for (int i = 0; i <= n; ++i) t[size_t(i)] = double(i) / double(n);
    }
    std::vector<float> o(t.size());
    for (size_t i = 0; i < t.size(); ++i) {
        double v = t[i];
        v = v + double(s.sway) * (std::cos(kPi / 2.0 * v) - 1.0 + v);
        o[i] = float(v);
    }
    return o;
}

// ---------------------------------------------------------------------------
// текстовый энкодер
// ---------------------------------------------------------------------------
void F5Model::text_embed(const std::vector<int32_t> &ids, int N, bool drop, std::vector<float> &out) const {
    const Impl &P = *p_;
    const int D = cfg_.text_dim;
    std::vector<int32_t> row(size_t(N), 0);
    std::vector<uint8_t> pad(size_t(N), 1);  // text == 0 → заполнитель
    for (int i = 0; i < N && i < int(ids.size()); ++i) {
        row[size_t(i)] = ids[size_t(i)] + 1;
        pad[size_t(i)] = 0;
    }
    if (drop) std::fill(row.begin(), row.end(), 0);
    out.assign(size_t(N) * size_t(D), 0.f);
    std::vector<float> tmp(size_t(P.text_emb.cols));
    for (int i = 0; i < N; ++i) {
        if (row[size_t(i)] >= P.text_emb.rows) fail(VOX_ERR_FORMAT, "F5-TTS: token id outside of the embedding table");
        const float *r = mat_row(P.text_emb, row[size_t(i)], tmp.data());
        std::memcpy(&out[size_t(i) * size_t(D)], r, size_t(D) * 4);
    }
    if (cfg_.conv_layers <= 0) return;
    // абсолютная позиция: [cos | sin] с theta = 10000 на всю размерность
    const int half = D / 2;
    std::vector<double> inv((size_t)half);
    for (int j = 0; j < half; ++j) inv[size_t(j)] = 1.0 / std::pow(10000.0, double(2 * j) / double(D));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < half; ++j) {
            const float a = float(double(i) * inv[size_t(j)]);
            out[size_t(i) * size_t(D) + size_t(j)] += std::cos(a);
            out[size_t(i) * size_t(D) + size_t(half) + size_t(j)] += std::sin(a);
        }
    auto zero_pad = [&]() {
        if (!cfg_.mask_padding) return;
        for (int i = 0; i < N; ++i)
            if (pad[size_t(i)]) std::fill_n(&out[size_t(i) * size_t(D)], D, 0.f);
    };
    zero_pad();
    const int I = D * cfg_.conv_mult;
    std::vector<float> y(size_t(N) * size_t(D)), h(size_t(N) * size_t(I)), z(size_t(N) * size_t(D));
    for (const CnxV2 &b : P.text_blocks) {
        b.dw.forward(out.data(), N, y.data());
        ln_rows(y.data(), N, D, b.ln_w, b.ln_b, 1e-6f, y.data());
        matmul(b.pw1, y.data(), N, h.data());
        parallel_for(int64_t(N) * I, 8192, [&](int64_t a, int64_t e) { for (int64_t i = a; i < e; ++i) h[size_t(i)] = gelu_erf(h[size_t(i)]); });
        // GRN: норма по времени для каждого канала, нормировка средним по каналам
        std::vector<double> gx(size_t(I), 0.0);
        for (int i = 0; i < N; ++i)
            for (int c = 0; c < I; ++c) { const double v = h[size_t(i) * size_t(I) + size_t(c)]; gx[size_t(c)] += v * v; }
        double mean = 0;
        for (int c = 0; c < I; ++c) { gx[size_t(c)] = std::sqrt(gx[size_t(c)]); mean += gx[size_t(c)]; }
        mean /= double(I);
        std::vector<float> nx((size_t)I);
        for (int c = 0; c < I; ++c) nx[size_t(c)] = float(gx[size_t(c)] / (mean + 1e-6));
        for (int i = 0; i < N; ++i)
            for (int c = 0; c < I; ++c) {
                float &v = h[size_t(i) * size_t(I) + size_t(c)];
                v = b.grn_g[c] * (v * nx[size_t(c)]) + b.grn_b[c] + v;
            }
        matmul(b.pw2, h.data(), N, z.data());
        for (size_t i = 0; i < out.size(); ++i) out[i] += z[i];
        zero_pad();
    }
}

// ---------------------------------------------------------------------------
// DiT
// ---------------------------------------------------------------------------
void F5Model::dit_forward(int B, int N, const float *x, const float *cond, const float *temb, float t, float *out) const {
    const Impl &P = *p_;
    const int D = cfg_.dim, M = cfg_.mel_dim, TD = cfg_.text_dim, H = cfg_.heads, HD = cfg_.dim_head, INNER = H * HD;
    const int64_t R = int64_t(B) * N;
    const auto t0 = std::chrono::steady_clock::now();

    // время: синус (256) → Linear → SiLU → Linear
    std::vector<float> te(256), tm((size_t)D), th((size_t)D);
    {
        const int half = 128;
        const double e = std::log(10000.0) / double(half - 1);
        for (int i = 0; i < half; ++i) {
            const float a = float(1000.0 * double(t) * std::exp(double(i) * -e));
            te[size_t(i)] = std::sin(a);
            te[size_t(half + i)] = std::cos(a);
        }
        matmul(P.time1, te.data(), 1, th.data());
        for (auto &v : th) v = silu_f(v);
        matmul(P.time2, th.data(), 1, tm.data());
    }
    std::vector<float> ts((size_t)D);
    for (int i = 0; i < D; ++i) ts[size_t(i)] = silu_f(tm[size_t(i)]);

    // вход: Linear(cat(x, cond, text)) + свёрточное позиционное кодирование
    std::vector<float> h(size_t(R) * size_t(D)), cat(size_t(R) * size_t(2 * M + TD));
    parallel_for(R, 64, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            float *d = &cat[size_t(r) * size_t(2 * M + TD)];
            std::memcpy(d, x + size_t(r) * size_t(M), size_t(M) * 4);
            std::memcpy(d + M, cond + size_t(r) * size_t(M), size_t(M) * 4);
            std::memcpy(d + 2 * M, temb + size_t(r) * size_t(TD), size_t(TD) * 4);
        }
    });
    matmul(P.in_proj, cat.data(), R, h.data());
    {
        std::vector<float> a(size_t(N) * size_t(D)), b(size_t(N) * size_t(D));
        for (int s = 0; s < B; ++s) {
            float *hs = &h[size_t(s) * size_t(N) * size_t(D)];
            P.pos1.forward(hs, N, a.data());
            for (auto &v : a) v = mish_f(v);
            P.pos2.forward(a.data(), N, b.data());
            for (size_t i = 0; i < b.size(); ++i) hs[i] += mish_f(b[i]);
        }
    }

    // RoPE: пары (2i, 2i+1), угол = позиция * theta^(-2i/HD)
    const int HH = HD / 2;
    std::vector<float> rcos(size_t(N) * size_t(HH)), rsin(size_t(N) * size_t(HH));
    for (int j = 0; j < HH; ++j) {
        const float inv = float(1.0 / std::pow(10000.0, double(2 * j) / double(HD)));
        for (int i = 0; i < N; ++i) {
            const float a = float(i) * inv;
            rcos[size_t(i) * size_t(HH) + size_t(j)] = std::cos(a);
            rsin[size_t(i) * size_t(HH) + size_t(j)] = std::sin(a);
        }
    }

    std::vector<float> ada(size_t(6 * D)), nrm(size_t(R) * size_t(D)), q(size_t(R) * size_t(INNER)), k(q.size()), v(q.size()), at(q.size()),
        o(size_t(R) * size_t(D)), ff(size_t(R) * size_t(D) * size_t(cfg_.ff_mult)), ff2(size_t(R) * size_t(D));
    const float scale = 1.0f / std::sqrt(float(HD));

    // LN без параметров + модуляция
    auto modulate = [&](const float *src, const float *sc, const float *sh, float *dst) {
        parallel_for(R, 16, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r) {
                const float *xr = src + size_t(r) * size_t(D);
                double mean = 0;
                for (int i = 0; i < D; ++i) mean += xr[i];
                mean /= D;
                double var = 0;
                for (int i = 0; i < D; ++i) { const double dd = xr[i] - mean; var += dd * dd; }
                var /= D;
                const float inv = float(1.0 / std::sqrt(var + 1e-6));
                float *yr = dst + size_t(r) * size_t(D);
                for (int i = 0; i < D; ++i) yr[i] = (xr[i] - float(mean)) * inv * (1.0f + sc[i]) + sh[i];
            }
        });
    };

    for (const DitBlock &L : P.blocks) {
        matmul(L.ada, ts.data(), 1, ada.data());
        const float *shift_msa = &ada[0], *scale_msa = &ada[size_t(D)], *gate_msa = &ada[size_t(2 * D)];
        const float *shift_mlp = &ada[size_t(3 * D)], *scale_mlp = &ada[size_t(4 * D)], *gate_mlp = &ada[size_t(5 * D)];
        modulate(h.data(), scale_msa, shift_msa, nrm.data());
        matmul(L.q, nrm.data(), R, q.data());
        matmul(L.k, nrm.data(), R, k.data());
        matmul(L.v, nrm.data(), R, v.data());
        parallel_for(R * H, 16, [&](int64_t a0, int64_t a1) {
            for (int64_t a = a0; a < a1; ++a) {
                const int64_t r = a / H, hh = a % H;
                const int pos = int(r % N);
                for (float *base : {&q[size_t(r) * size_t(INNER) + size_t(hh) * size_t(HD)], &k[size_t(r) * size_t(INNER) + size_t(hh) * size_t(HD)]})
                    for (int j = 0; j < HH; ++j) {
                        const float c = rcos[size_t(pos) * size_t(HH) + size_t(j)], s = rsin[size_t(pos) * size_t(HH) + size_t(j)];
                        const float x0 = base[2 * j], x1 = base[2 * j + 1];
                        base[2 * j] = x0 * c - x1 * s;
                        base[2 * j + 1] = x1 * c + x0 * s;
                    }
            }
        });
        // внимание без маски, без причинности
        parallel_for(R * H, 4, [&](int64_t a0, int64_t a1) {
            std::vector<float> sc((size_t)N);
            for (int64_t a = a0; a < a1; ++a) {
                const int64_t r = a / H, hh = a % H, seq = r / N;
                const float *qi = &q[size_t(r) * size_t(INNER) + size_t(hh) * size_t(HD)];
                float mx = -INFINITY;
                for (int j = 0; j < N; ++j) {
                    const float *kj = &k[size_t(seq * N + j) * size_t(INNER) + size_t(hh) * size_t(HD)];
                    float acc = 0;
                    for (int d = 0; d < HD; ++d) acc += qi[d] * kj[d];
                    sc[size_t(j)] = acc * scale;
                    mx = std::max(mx, sc[size_t(j)]);
                }
                float sum = 0;
                for (int j = 0; j < N; ++j) { sc[size_t(j)] = std::exp(sc[size_t(j)] - mx); sum += sc[size_t(j)]; }
                const float inv = 1.0f / sum;
                float *oi = &at[size_t(r) * size_t(INNER) + size_t(hh) * size_t(HD)];
                std::fill_n(oi, HD, 0.f);
                for (int j = 0; j < N; ++j) {
                    const float w = sc[size_t(j)] * inv;
                    const float *vj = &v[size_t(seq * N + j) * size_t(INNER) + size_t(hh) * size_t(HD)];
                    for (int d = 0; d < HD; ++d) oi[d] += w * vj[d];
                }
            }
        });
        matmul(L.o, at.data(), R, o.data());
        parallel_for(R, 32, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r)
                for (int i = 0; i < D; ++i) h[size_t(r) * size_t(D) + size_t(i)] += gate_msa[i] * o[size_t(r) * size_t(D) + size_t(i)];
        });
        modulate(h.data(), scale_mlp, shift_mlp, nrm.data());
        matmul(L.ff1, nrm.data(), R, ff.data());
        parallel_for(int64_t(ff.size()), 8192, [&](int64_t a, int64_t e) { for (int64_t i = a; i < e; ++i) ff[size_t(i)] = gelu_tanh(ff[size_t(i)]); });
        matmul(L.ff2, ff.data(), R, ff2.data());
        parallel_for(R, 32, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r)
                for (int i = 0; i < D; ++i) h[size_t(r) * size_t(D) + size_t(i)] += gate_mlp[i] * ff2[size_t(r) * size_t(D) + size_t(i)];
        });
    }

    // финальная модуляция: порядок (scale, shift)
    std::vector<float> fin(size_t(2 * D));
    matmul(P.norm_out, ts.data(), 1, fin.data());
    modulate(h.data(), &fin[0], &fin[size_t(D)], nrm.data());
    matmul(P.proj_out, nrm.data(), R, out);
    ms_dit_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void F5Model::sample(const std::vector<float> &cond_mel, int cond_frames, const std::vector<int32_t> &ids, int duration, const F5Sampler &s,
                     const float *noise_in, std::vector<float> &out, const std::function<void(int, int)> &progress) const {
    const int M = cfg_.mel_dim, N = duration;
    if (N < cond_frames) fail(VOX_ERR_ARG, "F5-TTS: duration is shorter than the reference");
    const bool use_cfg = s.cfg >= 1e-5f;
    // условие: mel образца, дополненный нулями; безусловная ветка — нули
    std::vector<float> cond(size_t(use_cfg ? 2 : 1) * size_t(N) * size_t(M), 0.f);
    std::memcpy(cond.data(), cond_mel.data(), size_t(cond_frames) * size_t(M) * 4);
    std::vector<float> temb, tu;
    text_embed(ids, N, false, temb);
    if (use_cfg) {
        text_embed(ids, N, true, tu);
        temb.insert(temb.end(), tu.begin(), tu.end());
    }
    const std::vector<float> ts = timesteps(s);
    std::vector<float> y(noise_in, noise_in + size_t(N) * size_t(M)), x2, pred(size_t(use_cfg ? 2 : 1) * size_t(N) * size_t(M));
    const int B = use_cfg ? 2 : 1;
    const size_t per = size_t(N) * size_t(M);
    for (size_t i = 0; i + 1 < ts.size(); ++i) {
        x2.resize(per * size_t(B));
        for (int b = 0; b < B; ++b) std::memcpy(&x2[per * size_t(b)], y.data(), per * 4);
        dit_forward(B, N, x2.data(), cond.data(), temb.data(), ts[i], pred.data());
        const float dt = ts[i + 1] - ts[i];
        if (use_cfg) {
            for (size_t j = 0; j < per; ++j) {
                const float p = pred[j], nu = pred[per + j];
                y[j] += dt * (p + (p - nu) * s.cfg);
            }
        } else {
            for (size_t j = 0; j < per; ++j) y[j] += dt * pred[j];
        }
        if (progress) progress(int(i) + 1, int(ts.size()) - 1);
    }
    out = std::move(y);
    std::memcpy(out.data(), cond_mel.data(), size_t(cond_frames) * size_t(M) * 4);
}

// ---------------------------------------------------------------------------
// Vocos
// ---------------------------------------------------------------------------
void F5Model::vocode(const float *mel, int T, std::vector<float> &pcm) const {
    const Impl &P = *p_;
    const int D = cfg_.vocos_dim, I = cfg_.vocos_inter;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> x(size_t(T) * size_t(D)), y(x.size()), h(size_t(T) * size_t(I)), z(x.size());
    P.v_embed.forward(mel, T, x.data());
    ln_rows(x.data(), T, D, P.v_ln_w, P.v_ln_b, 1e-6f, x.data());
    for (const VocosBlock &b : P.v_blocks) {
        b.dw.forward(x.data(), T, y.data());
        ln_rows(y.data(), T, D, b.ln_w, b.ln_b, 1e-6f, y.data());
        matmul(b.pw1, y.data(), T, h.data());
        parallel_for(int64_t(h.size()), 8192, [&](int64_t a, int64_t e) { for (int64_t i = a; i < e; ++i) h[size_t(i)] = gelu_erf(h[size_t(i)]); });
        matmul(b.pw2, h.data(), T, z.data());
        parallel_for(T, 32, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r)
                for (int c = 0; c < D; ++c) x[size_t(r) * size_t(D) + size_t(c)] += b.gamma[c] * z[size_t(r) * size_t(D) + size_t(c)];
        });
    }
    ln_rows(x.data(), T, D, P.v_fln_w, P.v_fln_b, 1e-6f, x.data());
    const int nb = cfg_.vocos_n_fft / 2 + 1;
    std::vector<float> head(size_t(T) * size_t(2 * nb)), mag(size_t(T) * size_t(nb)), ph(mag.size());
    matmul(P.v_head, x.data(), T, head.data());
    for (int t = 0; t < T; ++t) {
        std::memcpy(&mag[size_t(t) * size_t(nb)], &head[size_t(t) * size_t(2 * nb)], size_t(nb) * 4);
        std::memcpy(&ph[size_t(t) * size_t(nb)], &head[size_t(t) * size_t(2 * nb) + size_t(nb)], size_t(nb) * 4);
    }
    istft_same(mag.data(), ph.data(), T, cfg_.vocos_n_fft, cfg_.vocos_hop, pcm);
    ms_vocos_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace vox
