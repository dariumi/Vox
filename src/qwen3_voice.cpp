// Native voice creation for Qwen3-TTS-12Hz (see qwen3_voice.hpp).
//
// Reference implementation this file mirrors:
//   * qwen_tts/core/models/modeling_qwen3_tts.py — mel_spectrogram(), Qwen3TTSSpeakerEncoder (ECAPA-TDNN)
//   * transformers/models/mimi/modeling_mimi.py   — MimiEncoder (SEANet), MimiTransformerModel, downsample,
//                                                    MimiSplitResidualVectorQuantizer.encode
//   * qwen_tts/core/tokenizer_12hz/...v2.py        — Qwen3TTSTokenizerV2Model.encode (first N quantizers, trim)
// Activations are time-major [T][C] so that every convolution is an im2col + matmul over time rows.
#include "qwen3_voice.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>

namespace vox {
namespace {

const float *vecf(const TmodFile &f, const std::string &name) {
    const TensorView &t = f.get(name);
    if (t.dtype != DType::F32) fail(VOX_ERR_FORMAT, "tensor must be F32: " + name);
    return reinterpret_cast<const float *>(t.data);
}

Mat matf(const TmodFile &f, const std::string &name) {
    const TensorView &t = f.get(name);
    if (t.dtype != DType::F32) fail(VOX_ERR_FORMAT, "voice encoder weights must be F32: " + name);
    Mat m;
    m.type = WType::F32;
    m.data = t.data;
    m.rows = t.shape.at(0);
    m.cols = t.shape.size() > 1 ? t.shape[1] : 1;
    return m;
}

enum class Pad { Zero, Reflect, Replicate };

struct Conv {
    Mat w;  // [cout][K*cin], k-major
    int K = 1, cin = 0, cout = 0, dil = 1, stride = 1;
};

Conv load_conv(const TmodFile &f, const std::string &p, int K, int dil = 1, int stride = 1, bool bias = true) {
    Conv c;
    c.w = matf(f, p + ".weight");
    if (bias && f.has(p + ".bias")) c.w.bias = vecf(f, p + ".bias");
    c.K = K; c.dil = dil; c.stride = stride;
    c.cout = int(c.w.rows);
    if (c.w.cols % K) fail(VOX_ERR_FORMAT, "conv weight shape mismatch: " + p);
    c.cin = int(c.w.cols / K);
    return c;
}

// Conv1d over x [T][cin] with explicit left/right padding; y [Tout][cout]. im2col in time blocks.
std::vector<float> conv1d(const Conv &c, const float *x, int64_t T, int64_t L, int64_t R, Pad mode, int64_t &Tout) {
    const int64_t keff = int64_t(c.K - 1) * c.dil + 1;
    const int64_t Tp = T + L + R;
    Tout = Tp >= keff ? (Tp - keff) / c.stride + 1 : 0;
    std::vector<float> y(size_t(std::max<int64_t>(Tout, 0)) * size_t(c.cout));
    if (Tout <= 0) return y;
    const int64_t cin = c.cin, KC = int64_t(c.K) * cin;
    auto src = [&](int64_t i) -> const float * {  // row of the padded input, nullptr = zero row
        int64_t j = i - L;
        if (j >= 0 && j < T) return x + j * cin;
        switch (mode) {
            case Pad::Zero: return nullptr;
            case Pad::Replicate: return x + std::clamp<int64_t>(j, 0, T - 1) * cin;
            case Pad::Reflect: {
                if (T == 1) return x;
                const int64_t period = 2 * (T - 1);
                int64_t m = j % period;
                if (m < 0) m += period;
                if (m >= T) m = period - m;
                return x + m * cin;
            }
        }
        return nullptr;
    };
    if (c.K == 1 && c.stride == 1 && L == 0 && R == 0) {
        matmul(c.w, x, T, y.data());
        return y;
    }
    const int64_t block = std::max<int64_t>(1, std::min<int64_t>(Tout, (int64_t(1) << 22) / std::max<int64_t>(KC, 1)));
    std::vector<float> col(size_t(block) * size_t(KC));
    for (int64_t t0 = 0; t0 < Tout; t0 += block) {
        const int64_t nb = std::min(block, Tout - t0);
        parallel_for(nb, 64, [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; ++t)
                for (int k = 0; k < c.K; ++k) {
                    const float *s = src((t0 + t) * c.stride + int64_t(k) * c.dil);
                    float *d = &col[size_t(t * KC + int64_t(k) * cin)];
                    if (s) std::memcpy(d, s, size_t(cin) * 4);
                    else std::memset(d, 0, size_t(cin) * 4);
                }
        });
        matmul(c.w, col.data(), nb, y.data() + size_t(t0) * size_t(c.cout));
    }
    return y;
}

// PyTorch Conv1d(padding="same", padding_mode="reflect"): total = dil*(K-1), left = total/2.
std::vector<float> conv_same_reflect(const Conv &c, const float *x, int64_t T) {
    const int64_t total = int64_t(c.dil) * (c.K - 1), L = total / 2, R = total - L;
    int64_t To;
    return conv1d(c, x, T, L, R, Pad::Reflect, To);
}

// MimiConv1d (causal): left pad = keff - stride, right pad = extra so that the last window is complete.
std::vector<float> conv_mimi(const Conv &c, const float *x, int64_t T, Pad mode, int64_t &Tout) {
    const int64_t keff = int64_t(c.K - 1) * c.dil + 1, s = c.stride, pt = keff - s;
    const int64_t xlen = T - keff + pt;  // = T - s
    int64_t nfr = xlen >= 0 ? (xlen + s - 1) / s : -((-xlen) / s);  // ceil(xlen / s)
    const int64_t ideal = nfr * s + keff - pt;
    const int64_t extra = std::max<int64_t>(0, ideal - T);
    return conv1d(c, x, T, pt, extra, mode, Tout);
}

inline void relu(std::vector<float> &v) { for (auto &x : v) x = x > 0 ? x : 0; }
inline void elu(std::vector<float> &v) { for (auto &x : v) x = x > 0 ? x : std::expm1(x); }

// ---------------- FFT (radix-2, double) ----------------
void fft_inplace(std::vector<std::complex<double>> &a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * M_PI / double(len);
        const std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0);
            for (size_t j = 0; j < len / 2; ++j) {
                const std::complex<double> u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

double bessel_i0(double x) {
    double s = 1, t = 1;
    for (int k = 1; k < 60; ++k) {
        t *= (x / (2 * k)) * (x / (2 * k));
        s += t;
        if (t < 1e-17 * s) break;
    }
    return s;
}

}  // namespace

std::vector<float> resample_sinc(const float *x, size_t n, uint32_t sr_from, uint32_t sr_to) {
    if (sr_from == sr_to || n == 0) return std::vector<float>(x, x + n);
    const double ratio = double(sr_to) / double(sr_from);
    const double cutoff = std::min(1.0, ratio) * 0.97;  // anti-aliasing just below the lower Nyquist
    const int zc = 32;                                  // zero crossings per side
    const double half = zc / cutoff;                    // half-width in input samples
    const double beta = 9.0, i0b = bessel_i0(beta);
    const size_t nout = size_t(std::ceil(double(n) * ratio));
    std::vector<float> y(nout);
    parallel_for(int64_t(nout), 256, [&](int64_t a, int64_t b) {
        for (int64_t m = a; m < b; ++m) {
            const double t = double(m) / ratio;
            const int64_t lo = int64_t(std::ceil(t - half)), hi = int64_t(std::floor(t + half));
            double acc = 0, wsum = 0;
            for (int64_t k = lo; k <= hi; ++k) {
                const double d = t - double(k);
                const double u = d / half;
                if (u <= -1.0 || u >= 1.0) continue;
                const double win = bessel_i0(beta * std::sqrt(1.0 - u * u)) / i0b;
                const double arg = M_PI * d * cutoff;
                const double s = std::abs(arg) < 1e-12 ? 1.0 : std::sin(arg) / arg;
                const double h = cutoff * s * win;
                wsum += h;
                if (k >= 0 && k < int64_t(n)) acc += h * double(x[k]);
            }
            y[size_t(m)] = float(wsum != 0 ? acc / wsum * 1.0 : 0.0);
        }
    });
    return y;
}

// ======================================================================================
struct Qwen3VoiceEncoder::Impl {
    std::shared_ptr<TmodFile> f;
    bool spk = false, enc = false;

    // ---- speaker encoder (ECAPA-TDNN) ----
    struct SERes2Net {
        Conv tdnn1, tdnn2, se1, se2;
        std::vector<Conv> res;  // scale-1 convs
        int scale = 8;
    };
    const float *mel_basis = nullptr;  // [128][513]
    int n_mels = 128, enc_dim = 0;
    Conv blk0, mfa, asp_tdnn, asp_conv, fc;
    std::vector<SERes2Net> blocks;

    // ---- Mimi encoder ----
    struct ResBlk { Conv c1, c2; };
    struct Stage { std::vector<ResBlk> res; Conv down; };
    Conv conv0, conv_last, downsample;
    std::vector<Stage> stages;
    struct MLayer {
        const float *ln1w, *ln1b, *ln2w, *ln2b, *s_attn, *s_mlp;
        Mat q, k, v, o, fc1, fc2;
    };
    std::vector<MLayer> xl;
    int H = 0, heads = 0, kv_heads = 0, hd = 0;
    float norm_eps = 1e-5f;
    std::vector<float> inv_freq;
    bool sem_proj = false, ac_proj = false;
    Mat sem_in, ac_in;
    std::vector<Mat> sem_cb, ac_cb;  // [codebook_size][dim]
    int valid_q = 16, down_rate = 1920;

    explicit Impl(std::shared_ptr<TmodFile> file) : f(std::move(file)) {
        const json &q3 = f->manifest().at("qwen3");
        if (q3.contains("speaker_encoder") && f->has("spk.mel_basis")) load_spk(q3.at("speaker_encoder"));
        if (q3.contains("encoder") && f->has("enc.encoder.layers.0.conv.weight")) load_enc(q3.at("encoder"));
    }

    void load_spk(const json &c) {
        const TmodFile &F = *f;
        mel_basis = vecf(F, "spk.mel_basis");
        n_mels = c.value("mel_dim", 128);
        enc_dim = c.value("enc_dim", 1024);
        auto ch = c.at("enc_channels").get<std::vector<int>>();
        auto ks = c.at("enc_kernel_sizes").get<std::vector<int>>();
        auto ds = c.at("enc_dilations").get<std::vector<int>>();
        const int scale = c.value("enc_res2net_scale", 8);
        blk0 = load_conv(F, "spk.blocks.0.conv", ks[0], ds[0]);
        for (size_t i = 1; i + 1 < ch.size(); ++i) {
            const std::string p = "spk.blocks." + std::to_string(i) + ".";
            SERes2Net b;
            b.scale = scale;
            b.tdnn1 = load_conv(F, p + "tdnn1.conv", 1);
            b.tdnn2 = load_conv(F, p + "tdnn2.conv", 1);
            b.se1 = load_conv(F, p + "se_block.conv1", 1);
            b.se2 = load_conv(F, p + "se_block.conv2", 1);
            for (int j = 0; j < scale - 1; ++j)
                b.res.push_back(load_conv(F, p + "res2net_block.blocks." + std::to_string(j) + ".conv", ks[i], ds[i]));
            blocks.push_back(std::move(b));
        }
        mfa = load_conv(F, "spk.mfa.conv", ks.back(), ds.back());
        asp_tdnn = load_conv(F, "spk.asp.tdnn.conv", 1);
        asp_conv = load_conv(F, "spk.asp.conv", 1);
        fc = load_conv(F, "spk.fc", 1);
        spk = true;
    }

    void load_enc(const json &c) {
        const TmodFile &F = *f;
        const auto ratios = c.at("upsampling_ratios").get<std::vector<int>>();
        const int nres = c.value("num_residual_layers", 1), growth = c.value("dilation_growth_rate", 2);
        const int K = c.value("kernel_size", 7), lastK = c.value("last_kernel_size", 3), rk = c.value("residual_kernel_size", 3);
        if (!c.value("use_causal_conv", true)) fail(VOX_ERR_UNSUPPORTED, "Mimi encoder: only causal convolutions are supported");
        if (c.value("use_conv_shortcut", false)) fail(VOX_ERR_UNSUPPORTED, "Mimi encoder: conv shortcut is not supported");
        if (c.value("pad_mode", std::string("constant")) != "constant") fail(VOX_ERR_UNSUPPORTED, "Mimi encoder: pad_mode must be constant");
        if (c.value("hidden_act", std::string("gelu")) != "gelu") fail(VOX_ERR_UNSUPPORTED, "Mimi encoder: hidden_act must be gelu");
        int idx = 0;
        conv0 = load_conv(F, "enc.encoder.layers.0.conv", K);
        idx = 1;
        for (auto it = ratios.rbegin(); it != ratios.rend(); ++it) {
            Stage st;
            for (int j = 0; j < nres; ++j) {
                const std::string p = "enc.encoder.layers." + std::to_string(idx) + ".block.";
                int dil = 1;
                for (int q = 0; q < j; ++q) dil *= growth;
                ResBlk r;
                r.c1 = load_conv(F, p + "1.conv", rk, dil);
                r.c2 = load_conv(F, p + "3.conv", 1);
                st.res.push_back(r);
                ++idx;
            }
            ++idx;  // ELU
            st.down = load_conv(F, "enc.encoder.layers." + std::to_string(idx) + ".conv", 2 * *it, 1, *it);
            ++idx;
            stages.push_back(std::move(st));
        }
        ++idx;  // ELU
        conv_last = load_conv(F, "enc.encoder.layers." + std::to_string(idx) + ".conv", lastK);
        H = c.at("hidden_size");
        heads = c.at("num_attention_heads");
        kv_heads = c.value("num_key_value_heads", heads);
        hd = c.value("head_dim", H / heads);
        norm_eps = c.value("norm_eps", 1e-5f);
        inv_freq = rope_inv_freq(hd, c.value("rope_theta", 10000.0));
        const int L = c.at("num_hidden_layers");
        for (int i = 0; i < L; ++i) {
            const std::string p = "enc.encoder_transformer.layers." + std::to_string(i) + ".";
            MLayer l;
            l.ln1w = vecf(F, p + "input_layernorm.weight"); l.ln1b = vecf(F, p + "input_layernorm.bias");
            l.ln2w = vecf(F, p + "post_attention_layernorm.weight"); l.ln2b = vecf(F, p + "post_attention_layernorm.bias");
            l.s_attn = vecf(F, p + "self_attn_layer_scale.scale"); l.s_mlp = vecf(F, p + "mlp_layer_scale.scale");
            l.q = matf(F, p + "self_attn.q_proj.weight"); l.k = matf(F, p + "self_attn.k_proj.weight");
            l.v = matf(F, p + "self_attn.v_proj.weight"); l.o = matf(F, p + "self_attn.o_proj.weight");
            l.fc1 = matf(F, p + "mlp.fc1.weight"); l.fc2 = matf(F, p + "mlp.fc2.weight");
            xl.push_back(l);
        }
        // downsample: kernel 2*(encodec_frame_rate/frame_rate), stride 2, replicate padding, no bias
        int prod = 1;
        for (int r : ratios) prod *= r;
        const double encodec_fr = double(c.value("sampling_rate", 24000)) / prod;
        const int dk = 2 * int(encodec_fr / c.value("frame_rate", 12.5));
        downsample = load_conv(F, "enc.downsample.conv", dk, 1, 2, false);
        const std::string sq = "enc.quantizer.semantic_residual_vector_quantizer.";
        const std::string aq = "enc.quantizer.acoustic_residual_vector_quantizer.";
        if (F.has(sq + "input_proj.weight")) { sem_in = matf(F, sq + "input_proj.weight"); sem_proj = true; }
        if (F.has(aq + "input_proj.weight")) { ac_in = matf(F, aq + "input_proj.weight"); ac_proj = true; }
        const int nsem = c.value("num_semantic_quantizers", 1);
        const int nq = c.at("num_quantizers");
        for (int i = 0; i < nsem; ++i) sem_cb.push_back(matf(F, sq + "layers." + std::to_string(i) + ".codebook.embed"));
        for (int i = 0; i < nq - nsem; ++i) ac_cb.push_back(matf(F, aq + "layers." + std::to_string(i) + ".codebook.embed"));
        valid_q = c.value("valid_quantizers", 16);
        down_rate = c.value("downsample_rate", 1920);
        if (valid_q < nsem || valid_q > nq) fail(VOX_ERR_FORMAT, "Mimi encoder: bad valid_quantizers");
        enc = true;
    }

    // ---------------- mel (qwen_tts mel_spectrogram, n_fft 1024, hop 256, win 1024, center=False) ----------------
    std::vector<float> mel(const float *y, size_t n, int64_t &frames) const {
        const int nfft = 1024, hop = 256, nb = nfft / 2 + 1;
        const int64_t pad = (nfft - hop) / 2;
        const int64_t N = int64_t(n);
        if (N <= pad) fail(VOX_ERR_ARG, "reference audio is too short");
        std::vector<double> yp(size_t(N + 2 * pad));
        for (int64_t i = 0; i < N + 2 * pad; ++i) {  // reflect padding (edge excluded)
            int64_t j = i - pad;
            if (j < 0) j = -j;
            if (j >= N) j = 2 * (N - 1) - j;
            yp[size_t(i)] = y[j];
        }
        frames = (int64_t(yp.size()) - nfft) / hop + 1;
        std::vector<double> win(nfft);
        for (int i = 0; i < nfft; ++i) win[size_t(i)] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / nfft);  // periodic Hann
        std::vector<float> out(size_t(frames) * size_t(n_mels));
        parallel_for(frames, 8, [&](int64_t a, int64_t b) {
            std::vector<std::complex<double>> buf(nfft);
            std::vector<double> mag(nb);
            for (int64_t t = a; t < b; ++t) {
                for (int i = 0; i < nfft; ++i) buf[size_t(i)] = yp[size_t(t * hop + i)] * win[size_t(i)];
                fft_inplace(buf);
                for (int k = 0; k < nb; ++k) mag[size_t(k)] = std::sqrt(std::norm(buf[size_t(k)]) + 1e-9);
                for (int m = 0; m < n_mels; ++m) {
                    double s = 0;
                    const float *row = mel_basis + size_t(m) * nb;
                    for (int k = 0; k < nb; ++k) s += double(row[k]) * mag[size_t(k)];
                    out[size_t(t) * size_t(n_mels) + size_t(m)] = float(std::log(std::max(s, 1e-5)));
                }
            }
        });
        return out;
    }

    // ---------------- ECAPA-TDNN ----------------
    std::vector<float> se_res2net(const SERes2Net &b, const std::vector<float> &x, int64_t T) const {
        std::vector<float> h = conv_same_reflect(b.tdnn1, x.data(), T);
        relu(h);
        const int C = b.tdnn1.cout, cs = C / b.scale;
        std::vector<float> outp(size_t(T) * size_t(C));
        std::vector<float> part(size_t(T) * size_t(cs)), prev;
        for (int i = 0; i < b.scale; ++i) {
            for (int64_t t = 0; t < T; ++t)
                std::memcpy(&part[size_t(t * cs)], &h[size_t(t * C + i * cs)], size_t(cs) * 4);
            std::vector<float> o;
            if (i == 0) o = part;
            else {
                if (i >= 2) for (size_t k = 0; k < part.size(); ++k) part[k] += prev[k];
                o = conv_same_reflect(b.res[size_t(i - 1)], part.data(), T);
                relu(o);
            }
            for (int64_t t = 0; t < T; ++t)
                std::memcpy(&outp[size_t(t * C + i * cs)], &o[size_t(t * cs)], size_t(cs) * 4);
            prev = std::move(o);
        }
        std::vector<float> h2 = conv_same_reflect(b.tdnn2, outp.data(), T);
        relu(h2);
        // squeeze-excitation: mean over time → conv1 → relu → conv2 → sigmoid → scale channels
        std::vector<float> mean(size_t(C), 0.f);
        for (int64_t t = 0; t < T; ++t) for (int c = 0; c < C; ++c) mean[size_t(c)] += h2[size_t(t * C + c)];
        for (auto &m : mean) m /= float(T);
        std::vector<float> s1(size_t(b.se1.cout)), s2(size_t(b.se2.cout));
        matmul(b.se1.w, mean.data(), 1, s1.data());
        relu(s1);
        matmul(b.se2.w, s1.data(), 1, s2.data());
        for (auto &v : s2) v = 1.0f / (1.0f + std::exp(-v));
        for (int64_t t = 0; t < T; ++t)
            for (int c = 0; c < C; ++c) h2[size_t(t * C + c)] = h2[size_t(t * C + c)] * s2[size_t(c)] + x[size_t(t * C + c)];
        return h2;
    }

    std::vector<float> speaker_embedding(const float *pcm, size_t n) const {
        if (!spk) fail(VOX_ERR_UNSUPPORTED, "this .tmod has no speaker encoder weights (rebuild it with vox >= 0.6)");
        int64_t T;
        std::vector<float> x = mel(pcm, n, T);
        std::vector<float> h = conv_same_reflect(blk0, x.data(), T);
        relu(h);
        std::vector<std::vector<float>> outs;
        for (const auto &b : blocks) {
            h = se_res2net(b, h, T);
            outs.push_back(h);
        }
        // multi-layer feature aggregation: concat SE-Res2Net outputs along channels
        int Ccat = 0;
        for (const auto &b : blocks) Ccat += b.tdnn2.cout;
        std::vector<float> cat(size_t(T) * size_t(Ccat));
        int off = 0;
        for (size_t i = 0; i < outs.size(); ++i) {
            const int C = blocks[i].tdnn2.cout;
            for (int64_t t = 0; t < T; ++t) std::memcpy(&cat[size_t(t * Ccat + off)], &outs[i][size_t(t * C)], size_t(C) * 4);
            off += C;
        }
        std::vector<float> m = conv_same_reflect(mfa, cat.data(), T);
        relu(m);
        const int C = mfa.cout;
        // attentive statistics pooling
        auto stats = [&](const float *w, std::vector<double> &mean, std::vector<double> &sd) {
            mean.assign(size_t(C), 0.0); sd.assign(size_t(C), 0.0);
            for (int64_t t = 0; t < T; ++t)
                for (int c = 0; c < C; ++c) mean[size_t(c)] += double(w ? w[size_t(t * C + c)] : 1.0 / double(T)) * m[size_t(t * C + c)];
            for (int64_t t = 0; t < T; ++t)
                for (int c = 0; c < C; ++c) {
                    const double d = double(m[size_t(t * C + c)]) - mean[size_t(c)];
                    sd[size_t(c)] += double(w ? w[size_t(t * C + c)] : 1.0 / double(T)) * d * d;
                }
            for (auto &s : sd) s = std::sqrt(std::max(s, 1e-12));
        };
        std::vector<double> mu, sd;
        stats(nullptr, mu, sd);
        std::vector<float> att_in(size_t(T) * size_t(3 * C));
        for (int64_t t = 0; t < T; ++t)
            for (int c = 0; c < C; ++c) {
                att_in[size_t(t * 3 * C + c)] = m[size_t(t * C + c)];
                att_in[size_t(t * 3 * C + C + c)] = float(mu[size_t(c)]);
                att_in[size_t(t * 3 * C + 2 * C + c)] = float(sd[size_t(c)]);
            }
        std::vector<float> a1 = conv_same_reflect(asp_tdnn, att_in.data(), T);
        relu(a1);
        for (auto &v : a1) v = std::tanh(v);
        std::vector<float> att = conv_same_reflect(asp_conv, a1.data(), T);
        for (int c = 0; c < C; ++c) {  // softmax over time per channel
            float mx = -INFINITY;
            for (int64_t t = 0; t < T; ++t) mx = std::max(mx, att[size_t(t * C + c)]);
            double s = 0;
            for (int64_t t = 0; t < T; ++t) { float e = std::exp(att[size_t(t * C + c)] - mx); att[size_t(t * C + c)] = e; s += e; }
            for (int64_t t = 0; t < T; ++t) att[size_t(t * C + c)] = float(att[size_t(t * C + c)] / s);
        }
        stats(att.data(), mu, sd);
        std::vector<float> pooled(size_t(2 * C));
        for (int c = 0; c < C; ++c) { pooled[size_t(c)] = float(mu[size_t(c)]); pooled[size_t(C + c)] = float(sd[size_t(c)]); }
        std::vector<float> emb(size_t(fc.cout));
        matmul(fc.w, pooled.data(), 1, emb.data());
        return emb;
    }

    // ---------------- Mimi encoder ----------------
    void transformer(std::vector<float> &x, int64_t T) const {
        const int q_dim = heads * hd, kv_dim = kv_heads * hd, rep = heads / kv_heads;
        std::vector<float> h(size_t(T) * size_t(H)), q(size_t(T) * size_t(q_dim)), k(size_t(T) * size_t(kv_dim)),
            v(size_t(T) * size_t(kv_dim)), att(size_t(T) * size_t(q_dim)), o(size_t(T) * size_t(H));
        const float scl = 1.0f / std::sqrt(float(hd));
        for (const MLayer &l : xl) {
            for (int64_t t = 0; t < T; ++t) layernorm(&x[size_t(t * H)], l.ln1w, l.ln1b, H, norm_eps, &h[size_t(t * H)]);
            matmul(l.q, h.data(), T, q.data());
            matmul(l.k, h.data(), T, k.data());
            matmul(l.v, h.data(), T, v.data());
            for (int64_t t = 0; t < T; ++t) {
                rope_apply(&q[size_t(t * q_dim)], heads, hd, inv_freq.data(), t);
                rope_apply(&k[size_t(t * kv_dim)], kv_heads, hd, inv_freq.data(), t);
            }
            parallel_for(T * heads, 4, [&](int64_t a, int64_t b) {  // full causal attention (sliding_window is not applied by HF)
                std::vector<float> sc(static_cast<size_t>(T));
                for (int64_t i = a; i < b; ++i) {
                    const int64_t t = i / heads, hh = i % heads, kh = hh / rep;
                    const float *qp = &q[size_t(t * q_dim + hh * hd)];
                    for (int64_t s = 0; s <= t; ++s) sc[size_t(s)] = dot(qp, &k[size_t(s * kv_dim + kh * hd)], hd) * scl;
                    softmax_inplace(sc.data(), t + 1);
                    float *op = &att[size_t(t * q_dim + hh * hd)];
                    std::fill(op, op + hd, 0.f);
                    for (int64_t s = 0; s <= t; ++s) {
                        const float w = sc[size_t(s)], *vp = &v[size_t(s * kv_dim + kh * hd)];
                        for (int d = 0; d < hd; ++d) op[d] += w * vp[d];
                    }
                }
            });
            matmul(l.o, att.data(), T, o.data());
            for (int64_t t = 0; t < T; ++t)
                for (int c = 0; c < H; ++c) x[size_t(t * H + c)] += l.s_attn[c] * o[size_t(t * H + c)];
            for (int64_t t = 0; t < T; ++t) layernorm(&x[size_t(t * H)], l.ln2w, l.ln2b, H, norm_eps, &h[size_t(t * H)]);
            const int I = int(l.fc1.rows);
            std::vector<float> m1(size_t(T) * size_t(I));
            matmul(l.fc1, h.data(), T, m1.data());
            for (auto &z : m1) z = 0.5f * z * (1.0f + std::erf(z * float(M_SQRT1_2)));  // exact GELU
            matmul(l.fc2, m1.data(), T, o.data());
            for (int64_t t = 0; t < T; ++t)
                for (int c = 0; c < H; ++c) x[size_t(t * H + c)] += l.s_mlp[c] * o[size_t(t * H + c)];
        }
    }

    static int nearest(const Mat &cb, const float *r) {
        const int64_t n = cb.rows, d = cb.cols;
        const float *e = reinterpret_cast<const float *>(cb.data);
        int best = 0;
        double bd = INFINITY;
        for (int64_t i = 0; i < n; ++i) {
            double s = 0;
            for (int64_t j = 0; j < d; ++j) { const double z = double(r[j]) - double(e[i * d + j]); s += z * z; }
            if (s < bd) { bd = s; best = int(i); }
        }
        return best;
    }

    std::vector<int16_t> encode(const float *pcm, size_t n, int64_t &frames, int &codebooks) const {
        if (!enc) fail(VOX_ERR_UNSUPPORTED, "this .tmod has no speech-tokenizer encoder weights (rebuild it with vox >= 0.6)");
        int64_t T = int64_t(n), To;
        std::vector<float> x = conv_mimi(conv0, pcm, T, Pad::Zero, To);
        T = To;
        for (const Stage &st : stages) {
            for (const ResBlk &r : st.res) {
                std::vector<float> a = x;
                elu(a);
                std::vector<float> b = conv_mimi(r.c1, a.data(), T, Pad::Zero, To);
                elu(b);
                std::vector<float> c = conv_mimi(r.c2, b.data(), T, Pad::Zero, To);
                for (size_t i = 0; i < x.size(); ++i) x[i] += c[i];
            }
            elu(x);
            x = conv_mimi(st.down, x.data(), T, Pad::Zero, To);
            T = To;
        }
        elu(x);
        x = conv_mimi(conv_last, x.data(), T, Pad::Zero, To);
        T = To;
        transformer(x, T);
        x = conv_mimi(downsample, x.data(), T, Pad::Replicate, To);
        T = To;
        // split RVQ: semantic and acoustic quantizers each project the same embedding
        std::vector<float> sem = x, ac = x;
        int64_t sdim = H, adim = H;
        if (sem_proj) { sdim = sem_in.rows; sem.resize(size_t(T * sdim)); matmul(sem_in, x.data(), T, sem.data()); }
        if (ac_proj) { adim = ac_in.rows; ac.resize(size_t(T * adim)); matmul(ac_in, x.data(), T, ac.data()); }
        const int nsem = int(sem_cb.size()), nac = valid_q - nsem;
        const int64_t keep = std::min<int64_t>(T, (int64_t(n) + down_rate - 1) / down_rate);
        std::vector<int16_t> codes(size_t(keep) * size_t(valid_q));
        parallel_for(keep, 1, [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; ++t) {
                std::vector<float> r(sem.begin() + t * sdim, sem.begin() + (t + 1) * sdim);
                for (int i = 0; i < nsem; ++i) {
                    const int id = nearest(sem_cb[size_t(i)], r.data());
                    codes[size_t(t * valid_q + i)] = int16_t(id);
                    const float *e = reinterpret_cast<const float *>(sem_cb[size_t(i)].data) + int64_t(id) * sdim;
                    for (int64_t j = 0; j < sdim; ++j) r[size_t(j)] -= e[j];
                }
                std::vector<float> ra(ac.begin() + t * adim, ac.begin() + (t + 1) * adim);
                for (int i = 0; i < nac; ++i) {
                    const int id = nearest(ac_cb[size_t(i)], ra.data());
                    codes[size_t(t * valid_q + nsem + i)] = int16_t(id);
                    const float *e = reinterpret_cast<const float *>(ac_cb[size_t(i)].data) + int64_t(id) * adim;
                    for (int64_t j = 0; j < adim; ++j) ra[size_t(j)] -= e[j];
                }
            }
        });
        frames = keep;
        codebooks = valid_q;
        return codes;
    }
};

Qwen3VoiceEncoder::Qwen3VoiceEncoder(std::shared_ptr<TmodFile> f) : p_(std::make_unique<Impl>(std::move(f))) {}
Qwen3VoiceEncoder::~Qwen3VoiceEncoder() = default;
bool Qwen3VoiceEncoder::has_speaker_encoder() const { return p_->spk; }
bool Qwen3VoiceEncoder::has_codec_encoder() const { return p_->enc; }
std::vector<float> Qwen3VoiceEncoder::speaker_embedding(const float *pcm, size_t n) const { return p_->speaker_embedding(pcm, n); }
std::vector<float> Qwen3VoiceEncoder::mel(const float *pcm, size_t n, int64_t &frames) const { return p_->mel(pcm, n, frames); }
std::vector<int16_t> Qwen3VoiceEncoder::encode(const float *pcm, size_t n, int64_t &frames, int &codebooks) const {
    return p_->encode(pcm, n, frames, codebooks);
}

}  // namespace vox
