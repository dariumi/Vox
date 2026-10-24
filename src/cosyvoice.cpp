// CosyVoice2: Qwen2 speech-token LM, flow matching (Conformer encoder + U-Net estimator), HiFT vocoder.
// Weights come from python/vox/build_cosyvoice.py (weight_norm folded, names of the official state dicts under
// llm. / flow. / hift.). The frontend nets (CAM++, S3 tokenizer, mel / fbank) are in cosyvoice_front.cpp.
#include "cosyvoice.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>

#include "nn.hpp"
#include "sampling.hpp"

namespace vox {
using namespace nn;

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
// VOX_PROFILE=cosy: time inside the flow by part
// (diagnostics only: one session at a time is measured meaningfully)
const bool g_prof_on = [] { const char *e = std::getenv("VOX_PROFILE"); return e && std::strstr(e, "cosy"); }();
double g_prof[8] = {0};
struct PScope {
    int i; Clock::time_point t0;
    explicit PScope(int k) : i(k) { if (g_prof_on) t0 = Clock::now(); }
    ~PScope() { if (g_prof_on) g_prof[i] += ms_since(t0); }
};

std::vector<float> vecf(const TmodFile &f, const std::string &n) { return to_float(f.get(n)); }

// ---------------------------------------------------------------- flow encoder
// ConformerEncoderLayer without macaron / conv module, normalize_before, RelPositionMultiHeadedAttention
struct RelLayer {
    Linear q, k, v, o, pos, w1, w2;
    std::vector<float> pbu, pbv, ln1w, ln1b, ln2w, ln2b;
    void load(const TmodFile &f, const std::string &p) {
        q.load(f, p + "self_attn.linear_q"); k.load(f, p + "self_attn.linear_k"); v.load(f, p + "self_attn.linear_v");
        o.load(f, p + "self_attn.linear_out"); pos.load(f, p + "self_attn.linear_pos", false);
        pbu = vecf(f, p + "self_attn.pos_bias_u"); pbv = vecf(f, p + "self_attn.pos_bias_v");
        ln1w = vecf(f, p + "norm_mha.weight"); ln1b = vecf(f, p + "norm_mha.bias");
        ln2w = vecf(f, p + "norm_ff.weight"); ln2b = vecf(f, p + "norm_ff.bias");
        w1.load(f, p + "feed_forward.w_1"); w2.load(f, p + "feed_forward.w_2");
    }
    // x [T][D] in place; pe [2T-1][D] (relative positions T-1 … -(T-1))
    void forward(std::vector<float> &x, int T, const std::vector<float> &pe, int H) const {
        const int D = int(q.out), dk = D / H;
        std::vector<float> xn = x;
        layer_norm(xn.data(), T, D, ln1w.data(), ln1b.data(), 1e-12f);
        std::vector<float> Q(size_t(T) * D), K(size_t(T) * D), V(size_t(T) * D), P(size_t(2 * T - 1) * D), att(size_t(T) * D);
        q.forward(xn.data(), T, Q.data());
        k.forward(xn.data(), T, K.data());
        v.forward(xn.data(), T, V.data());
        pos.forward(pe.data(), 2 * T - 1, P.data());
        std::vector<float> QU(Q.size()), QV(Q.size());
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < D; ++c) {
                QU[size_t(t) * D + c] = Q[size_t(t) * D + c] + pbu[size_t(c)];
                QV[size_t(t) * D + c] = Q[size_t(t) * D + c] + pbv[size_t(c)];
            }
        const float scale = 1.0f / std::sqrt(float(dk));
        std::vector<float> qh(size_t(T) * dk), ph(size_t(2 * T - 1) * dk), bd(size_t(T) * size_t(2 * T - 1));
        attention(QU.data(), D, K.data(), D, V.data(), D, T, T, H, dk, scale, att.data(), [&](int h, float *S) {
            // matrix_bd = (q + pos_bias_v) · pᵀ, then rel_shift: column j of row i is relative position i - j
            for (int t = 0; t < T; ++t) std::memcpy(&qh[size_t(t) * dk], &QV[size_t(t) * D + size_t(h) * dk], size_t(dk) * 4);
            for (int r = 0; r < 2 * T - 1; ++r) std::memcpy(&ph[size_t(r) * dk], &P[size_t(r) * D + size_t(h) * dk], size_t(dk) * 4);
            matmul(f32_mat(ph.data(), 2 * T - 1, dk), qh.data(), T, bd.data());
            parallel_for(T, 16, [&](int64_t a, int64_t e) {
                for (int64_t i = a; i < e; ++i)
                    for (int j = 0; j < T; ++j) S[size_t(i) * T + j] += bd[size_t(i) * (2 * T - 1) + size_t(T - 1 - i + j)] * scale;
            });
        });
        std::vector<float> y(size_t(T) * D);
        o.forward(att.data(), T, y.data());
        for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];
        xn = x;
        layer_norm(xn.data(), T, D, ln2w.data(), ln2b.data(), 1e-12f);
        std::vector<float> hdn(size_t(T) * size_t(w1.out));
        w1.forward(xn.data(), T, hdn.data());
        for (auto &z : hdn) z = z / (1.0f + std::exp(-z));   // Swish
        w2.forward(hdn.data(), T, y.data());
        for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];
    }
};

// EspnetRelPositionalEncoding: rows for relative positions T-1 … -(T-1), in float32 as torch computes them
std::vector<float> rel_pe(int T, int D) {
    std::vector<float> div(static_cast<size_t>(D / 2));
    const float lg = float(-(std::log(10000.0) / double(D)));
    for (int i = 0; i < D / 2; ++i) div[size_t(i)] = std::exp(float(2 * i) * lg);
    std::vector<float> pe(size_t(2 * T - 1) * D);
    for (int r = 0; r < 2 * T - 1; ++r) {
        const int rel = T - 1 - r;
        const float pos = float(rel < 0 ? -rel : rel), sgn = rel < 0 ? -1.0f : 1.0f;
        for (int i = 0; i < D / 2; ++i) {
            const float a = (sgn * pos) * div[size_t(i)];   // (-1 * position) * div_term for the negative half
            pe[size_t(r) * D + 2 * i] = std::sin(a);
            pe[size_t(r) * D + 2 * i + 1] = std::cos(a);
        }
    }
    return pe;
}

struct LinearEmbed {   // LinearNoSubsampling: Linear → LayerNorm(1e-5) → × sqrt(D)
    Linear lin;
    std::vector<float> lnw, lnb;
    void load(const TmodFile &f, const std::string &p) {
        lin.load(f, p + "out.0");
        lnw = vecf(f, p + "out.1.weight"); lnb = vecf(f, p + "out.1.bias");
    }
    std::vector<float> forward(const std::vector<float> &x, int T) const {
        const int D = int(lin.out);
        std::vector<float> y(size_t(T) * D);
        lin.forward(x.data(), T, y.data());
        layer_norm(y.data(), T, D, lnw.data(), lnb.data(), 1e-5f);
        const float xs = float(std::sqrt(double(D)));
        for (auto &z : y) z *= xs;
        return y;
    }
};

// ---------------------------------------------------------------- estimator (CausalConditionalDecoder)
void causal_conv(const Conv &c, const std::vector<float> &x, int T, std::vector<float> &y) {   // left pad K-1 (pad = 0 in Conv)
    std::vector<float> xp(size_t(T + c.K - 1) * size_t(c.cin), 0.f);
    std::memcpy(&xp[size_t(c.K - 1) * size_t(c.cin)], x.data(), x.size() * 4);
    c.forward(xp.data(), T + c.K - 1, y);
}

// B sequences of T rows stacked: a causal convolution per sequence
void causal_conv_b(const Conv &c, const std::vector<float> &x, int T, int B, std::vector<float> &y) {
    if (B == 1) { causal_conv(c, x, T, y); return; }
    y.resize(size_t(B) * size_t(T) * size_t(c.cout));
    std::vector<float> xs, ys;
    for (int b = 0; b < B; ++b) {
        xs.assign(x.begin() + long(size_t(b) * T * c.cin), x.begin() + long(size_t(b + 1) * T * c.cin));
        causal_conv(c, xs, T, ys);
        std::memcpy(&y[size_t(b) * T * c.cout], ys.data(), ys.size() * 4);
    }
}

// The estimator runs the conditional and the unconditional pass together: B sequences of T rows stacked; row-wise layers see
// B·T rows, convolutions and attention work per sequence.
struct CBlock {   // CausalBlock1D: CausalConv1d(k3) → LayerNorm → Mish
    Conv conv;
    std::vector<float> lnw, lnb;
    void load(const TmodFile &f, const std::string &p) {
        conv.load(f, p + "block.0", 1, 0);
        lnw = vecf(f, p + "block.2.weight"); lnb = vecf(f, p + "block.2.bias");
    }
    void forward(const std::vector<float> &x, int T, int B, std::vector<float> &y) const {
        causal_conv_b(conv, x, T, B, y);
        layer_norm(y.data(), B * T, conv.cout, lnw.data(), lnb.data(), 1e-5f);
        map_inplace(y.data(), int64_t(y.size()), mish);
    }
};

struct CResnet {   // CausalResnetBlock1D
    CBlock b1, b2;
    Linear mlp;
    Conv res;
    void load(const TmodFile &f, const std::string &p) {
        b1.load(f, p + "block1."); b2.load(f, p + "block2.");
        mlp.load(f, p + "mlp.1");
        res.load(f, p + "res_conv", 1, 0);
    }
    // temb_mish = Mish(t embedding) (shared by every block and both passes: the same t)
    void forward(const std::vector<float> &x, int T, int B, const std::vector<float> &temb_mish, std::vector<float> &out) const {
        PScope ps(0);
        std::vector<float> h, h2, r;
        b1.forward(x, T, B, h);
        const int C = b1.conv.cout;
        std::vector<float> tb(static_cast<size_t>(C));
        mlp.forward(temb_mish.data(), 1, tb.data());
        parallel_for(int64_t(B) * T, 256, [&](int64_t a, int64_t e) {
            for (int64_t t = a; t < e; ++t)
                for (int c = 0; c < C; ++c) h[size_t(t) * C + c] += tb[size_t(c)];
        });
        b2.forward(h, T, B, h2);
        res.forward(x.data(), B * T, r);
        out.resize(h2.size());
        for (size_t i = 0; i < out.size(); ++i) out[i] = h2[i] + r[i];
    }
};

struct XBlock {   // diffusers-style BasicTransformerBlock: self-attention + GELU feed-forward, pre-LayerNorm
    std::vector<float> n1w, n1b, n3w, n3b;
    Linear q, k, v, o, ff1, ff2;
    void load(const TmodFile &f, const std::string &p) {
        n1w = vecf(f, p + "norm1.weight"); n1b = vecf(f, p + "norm1.bias");
        n3w = vecf(f, p + "norm3.weight"); n3b = vecf(f, p + "norm3.bias");
        q.load(f, p + "attn1.to_q", false); k.load(f, p + "attn1.to_k", false); v.load(f, p + "attn1.to_v", false);
        o.load(f, p + "attn1.to_out.0");
        ff1.load(f, p + "ff.net.0.proj"); ff2.load(f, p + "ff.net.2");
    }
    void forward(std::vector<float> &x, int T, int B, int H) const {
        PScope ps(1);
        const int C = int(o.out), inner = int(q.out), dh = inner / H, R = B * T;
        std::vector<float> xn = x;
        layer_norm(xn.data(), R, C, n1w.data(), n1b.data(), 1e-5f);
        std::vector<float> Q(size_t(R) * inner), K(Q.size()), V(Q.size()), A(Q.size()), y(size_t(R) * C);
        { PScope pq(3); q.forward(xn.data(), R, Q.data()); k.forward(xn.data(), R, K.data()); v.forward(xn.data(), R, V.data()); }
        {
            PScope pa(2);
            for (int b = 0; b < B; ++b) {
                const size_t off = size_t(b) * T * inner;
                attention(Q.data() + off, inner, K.data() + off, inner, V.data() + off, inner, T, T, H, dh, 1.0f / std::sqrt(float(dh)), A.data() + off);
            }
        }
        o.forward(A.data(), R, y.data());
        for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];
        xn = x;
        layer_norm(xn.data(), R, C, n3w.data(), n3b.data(), 1e-5f);
        std::vector<float> hd(size_t(R) * size_t(ff1.out));
        { PScope pf(4); ff1.forward(xn.data(), R, hd.data()); }
        { PScope pg(5); gelu_inplace(hd.data(), int64_t(hd.size())); }
        { PScope pf(4); ff2.forward(hd.data(), R, y.data()); }
        for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];
    }
};

// ---------------------------------------------------------------- HiFT
struct SnakeRes {   // ResBlock with Snake activations (alpha_logscale = False)
    std::vector<Conv> c1, c2;
    std::vector<std::vector<float>> a1, a2;
    void load(const TmodFile &f, const std::string &p, int K, const std::vector<int> &dils) {
        for (size_t j = 0; j < dils.size(); ++j) {
            Conv x, y;
            x.load(f, p + "convs1." + std::to_string(j), 1, (K * dils[j] - dils[j]) / 2, dils[j]);
            y.load(f, p + "convs2." + std::to_string(j), 1, (K - 1) / 2, 1);
            c1.push_back(std::move(x)); c2.push_back(std::move(y));
            a1.push_back(vecf(f, p + "activations1." + std::to_string(j) + ".alpha"));
            a2.push_back(vecf(f, p + "activations2." + std::to_string(j) + ".alpha"));
        }
    }
    static void snake(std::vector<float> &x, int C, const std::vector<float> &a) {
        const int64_t T = int64_t(x.size()) / C;
        parallel_for(T, 256, [&](int64_t s, int64_t e) {
            for (int64_t t = s; t < e; ++t)
                for (int c = 0; c < C; ++c) {
                    float &z = x[size_t(t) * C + c];
                    const float al = a[size_t(c)], sn = std::sin(al * z);
                    z = z + (1.0f / (al + 1e-9f)) * (sn * sn);
                }
        });
    }
    void forward(std::vector<float> &x, int T) const {
        const int C = c1[0].cin;
        std::vector<float> xt, y;
        for (size_t j = 0; j < c1.size(); ++j) {
            xt = x;
            snake(xt, C, a1[j]);
            c1[j].forward(xt.data(), T, y);
            snake(y, C, a2[j]);
            c2[j].forward(y.data(), T, xt);
            for (size_t i = 0; i < x.size(); ++i) x[i] += xt[i];
        }
    }
};

// torch.linspace(0, 1, n + 1) on the CPU (float32): the first half counts up from start, the second down from end
std::vector<float> linspace01(int steps) {
    std::vector<float> t(static_cast<size_t>(steps));
    const float step = (1.0f - 0.0f) / float(steps - 1);
    const int half = steps / 2;
    for (int i = 0; i < steps; ++i) t[size_t(i)] = i < half ? 0.0f + step * float(i) : 1.0f - step * float(steps - i - 1);
    return t;
}

}  // namespace

// ======================================================================================================
struct CosyVoiceModel::Impl {
    // LLM
    TransformerW lm;
    Mat embed, llm_emb, speech_emb, dec;
    std::vector<float> dec_b;
    // flow
    Mat in_emb;
    Linear spk_aff, enc_proj;
    LinearEmbed emb, up_emb;
    Conv pl1, pl2, up_conv;
    std::vector<RelLayer> enc, up_enc;
    std::vector<float> after_w, after_b, rand_noise;   // rand_noise [80][N]
    int noise_len = 0;
    Linear t1, t2;
    CResnet down_res, up_res;
    std::vector<XBlock> down_x, up_x;
    std::vector<CResnet> mid_res;
    std::vector<std::vector<XBlock>> mid_x;
    Conv down_conv, up_out;
    CBlock final_block;
    Conv final_proj;
    // HiFT
    std::vector<Conv> f0_conv;
    Linear f0_cls, src_lin;
    Conv conv_pre, conv_post;
    std::vector<ConvT> ups;
    std::vector<Conv> src_down;
    std::vector<SnakeRes> src_res, res;
};


namespace {
void load_cfg(const json &c, CosyCfg &g) {
    auto gi = [&](const char *k, int &v) { if (c.contains(k)) v = c.at(k).get<int>(); };
    auto gf = [&](const char *k, float &v) { if (c.contains(k)) v = c.at(k).get<float>(); };
    if (c.contains("llm")) {
        const json &l = c.at("llm");
        g.llm_hidden = l.value("hidden_size", g.llm_hidden); g.llm_layers = l.value("num_hidden_layers", g.llm_layers);
        g.llm_heads = l.value("num_attention_heads", g.llm_heads); g.llm_kv_heads = l.value("num_key_value_heads", g.llm_kv_heads);
        g.llm_head_dim = l.value("head_dim", g.llm_hidden / g.llm_heads); g.llm_inter = l.value("intermediate_size", g.llm_inter);
        g.llm_eps = l.value("rms_norm_eps", g.llm_eps); g.llm_theta = l.value("rope_theta", g.llm_theta);
        g.speech_tokens = l.value("speech_token_size", g.speech_tokens);
        g.top_p = l.value("top_p", g.top_p); g.top_k = l.value("top_k", g.top_k); g.win_size = l.value("win_size", g.win_size);
        g.tau_r = l.value("tau_r", g.tau_r);
        g.min_ratio = l.value("min_token_text_ratio", g.min_ratio); g.max_ratio = l.value("max_token_text_ratio", g.max_ratio);
    }
    if (c.contains("flow")) {
        const json &f = c.at("flow");
        g.enc_dim = f.value("encoder_dim", g.enc_dim); g.enc_heads = f.value("encoder_heads", g.enc_heads);
        g.enc_units = f.value("encoder_units", g.enc_units); g.enc_blocks = f.value("encoder_blocks", g.enc_blocks);
        g.enc_up_blocks = f.value("encoder_up_blocks", g.enc_up_blocks); g.pre_lookahead = f.value("pre_lookahead_len", g.pre_lookahead);
        g.token_mel_ratio = f.value("token_mel_ratio", g.token_mel_ratio); g.mel_dim = f.value("mel_dim", g.mel_dim);
        g.spk_dim = f.value("spk_embed_dim", g.spk_dim);
        g.est_ch = f.value("estimator_channels", g.est_ch); g.est_heads = f.value("estimator_heads", g.est_heads);
        g.est_head_dim = f.value("estimator_head_dim", g.est_head_dim); g.est_blocks = f.value("estimator_blocks", g.est_blocks);
        g.est_mid = f.value("estimator_mid_blocks", g.est_mid); g.n_timesteps = f.value("n_timesteps", g.n_timesteps);
        g.cfg_rate = f.value("inference_cfg_rate", g.cfg_rate); g.cosine = f.value("t_scheduler", std::string("cosine")) == "cosine";
    }
    if (c.contains("hift")) {
        const json &h = c.at("hift");
        g.hift_ch = h.value("base_channels", g.hift_ch); g.nb_harmonics = h.value("nb_harmonics", g.nb_harmonics);
        g.n_fft = h.value("n_fft", g.n_fft); g.hop = h.value("hop_len", g.hop);
        g.up_rates = h.value("upsample_rates", g.up_rates); g.up_kernels = h.value("upsample_kernel_sizes", g.up_kernels);
        g.rb_kernels = h.value("resblock_kernel_sizes", g.rb_kernels); g.rb_dils = h.value("resblock_dilation_sizes", g.rb_dils);
        g.src_rb_kernels = h.value("source_resblock_kernel_sizes", g.src_rb_kernels);
        g.src_rb_dils = h.value("source_resblock_dilation_sizes", g.src_rb_dils);
        g.lrelu = h.value("lrelu_slope", g.lrelu); g.audio_limit = h.value("audio_limit", g.audio_limit);
    }
    gi("mel_n_fft", g.mel_n_fft); gi("mel_hop", g.mel_hop);
    g.sample_rate = c.value("sample_rate", 24000u);
    if (c.contains("s3tokenizer")) {
        const json &s = c.at("s3tokenizer");
        g.has_s3 = true;
        g.s3_dim = s.value("n_audio_state", g.s3_dim); g.s3_heads = s.value("n_audio_head", g.s3_heads);
        g.s3_layers = s.value("n_audio_layer", g.s3_layers); g.s3_mels = s.value("n_mels", g.s3_mels);
    }
    g.has_campplus = c.value("campplus", false);
    g.campplus_pool_window = c.value("campplus_pool", std::string("count")) == "window";
    if (c.contains("flow")) g.dt_diff = c.at("flow").value("dt", std::string("accumulate")) == "diff";
    if (c.contains("hift")) g.sinegen = c.at("hift").value("sinegen", 2);
    (void)gf;
}
}  // namespace

CosyVoiceModel::CosyVoiceModel(std::shared_ptr<TmodFile> f) : p_(new Impl), file_(std::move(f)) {
    const TmodFile &F = *file_;
    const json &m = F.manifest();
    if (!m.contains("cosyvoice")) fail(VOX_ERR_FORMAT, "not a CosyVoice .tmod (no \"cosyvoice\" section)");
    load_cfg(m.at("cosyvoice"), cfg_);
    cfg_.has_llm = F.has("llm.embed_tokens.weight");
    cfg_.fixed_noise = F.has("flow.rand_noise");
    Impl &P = *p_;
    const CosyCfg &c = cfg_;
    for (const auto &kv : F.blobs()) weight_bytes_ += kv.second.nbytes;

    // ---- LLM (Qwen2: q/k/v biases, no q/k norms) ----
    if (c.has_llm) {
        TransformerW &w = P.lm;
        w.hidden = c.llm_hidden; w.heads = c.llm_heads; w.kv_heads = c.llm_kv_heads; w.head_dim = c.llm_head_dim;
        w.inter = c.llm_inter; w.eps = c.llm_eps; w.theta = c.llm_theta;
        w.inv_freq = rope_inv_freq(c.llm_head_dim, c.llm_theta);
        w.layers.resize(size_t(c.llm_layers));
        auto mat_b = [&](const std::string &n, bool bias) {
            Mat x = mat_named(F, n + ".weight");
            if (bias && F.has(n + ".bias")) {
                const TensorView &b = F.get(n + ".bias");
                if (b.dtype != DType::F32) fail(VOX_ERR_FORMAT, "CosyVoice: biases must be F32: " + n);
                x.bias = reinterpret_cast<const float *>(b.data);
            }
            return x;
        };
        auto vec_ptr = [&](const std::string &n) {
            const TensorView &t = F.get(n);
            if (t.dtype != DType::F32) fail(VOX_ERR_FORMAT, "CosyVoice: norms must be F32: " + n);
            return reinterpret_cast<const float *>(t.data);
        };
        for (int i = 0; i < c.llm_layers; ++i) {
            const std::string p = "llm.layers." + std::to_string(i) + ".";
            LayerW &L = w.layers[size_t(i)];
            L.q = mat_b(p + "self_attn.q_proj", true); L.k = mat_b(p + "self_attn.k_proj", true);
            L.v = mat_b(p + "self_attn.v_proj", true); L.o = mat_b(p + "self_attn.o_proj", false);
            L.gate = mat_b(p + "mlp.gate_proj", false); L.up = mat_b(p + "mlp.up_proj", false); L.down = mat_b(p + "mlp.down_proj", false);
            L.ln1 = vec_ptr(p + "input_layernorm.weight"); L.ln2 = vec_ptr(p + "post_attention_layernorm.weight");
        }
        w.final_norm = vec_ptr("llm.norm.weight");
        P.embed = mat_of(F.get("llm.embed_tokens.weight"));
        P.llm_emb = mat_of(F.get("llm.llm_embedding.weight"));
        P.speech_emb = mat_of(F.get("llm.speech_embedding.weight"));
        P.dec = mat_of(F.get("llm.llm_decoder.weight"));
        P.dec_b = vecf(F, "llm.llm_decoder.bias");
    }
    // ---- flow ----
    {
        const std::string e = "flow.encoder.";
        P.in_emb = mat_of(F.get("flow.input_embedding.weight"));
        P.spk_aff.load(F, "flow.spk_embed_affine_layer");
        P.enc_proj.load(F, "flow.encoder_proj");
        P.emb.load(F, e + "embed.");
        P.up_emb.load(F, e + "up_embed.");
        P.pl1.load(F, e + "pre_lookahead_layer.conv1", 1, 0);
        P.pl2.load(F, e + "pre_lookahead_layer.conv2", 1, 0);
        P.up_conv.load(F, e + "up_layer.conv", 1, 0);
        for (int i = 0; i < c.enc_blocks; ++i) { RelLayer L; L.load(F, e + "encoders." + std::to_string(i) + "."); P.enc.push_back(std::move(L)); }
        for (int i = 0; i < c.enc_up_blocks; ++i) { RelLayer L; L.load(F, e + "up_encoders." + std::to_string(i) + "."); P.up_enc.push_back(std::move(L)); }
        P.after_w = vecf(F, e + "after_norm.weight"); P.after_b = vecf(F, e + "after_norm.bias");
        if (c.fixed_noise) {
            const TensorView &rn = F.get("flow.rand_noise");
            if (rn.shape.size() != 2 || rn.shape[0] != c.mel_dim) fail(VOX_ERR_FORMAT, "CosyVoice: flow.rand_noise must be [80][N]");
            P.rand_noise = to_float(rn);
            P.noise_len = int(rn.shape[1]);
        }
        const std::string d = "flow.decoder.estimator.";
        P.t1.load(F, d + "time_mlp.linear_1"); P.t2.load(F, d + "time_mlp.linear_2");
        P.down_res.load(F, d + "down_blocks.0.0.");
        for (int j = 0; j < c.est_blocks; ++j) { XBlock x; x.load(F, d + "down_blocks.0.1." + std::to_string(j) + "."); P.down_x.push_back(std::move(x)); }
        P.down_conv.load(F, d + "down_blocks.0.2", 1, 0);
        for (int i = 0; i < c.est_mid; ++i) {
            CResnet r; r.load(F, d + "mid_blocks." + std::to_string(i) + ".0."); P.mid_res.push_back(std::move(r));
            std::vector<XBlock> xs;
            for (int j = 0; j < c.est_blocks; ++j) { XBlock x; x.load(F, d + "mid_blocks." + std::to_string(i) + ".1." + std::to_string(j) + "."); xs.push_back(std::move(x)); }
            P.mid_x.push_back(std::move(xs));
        }
        P.up_res.load(F, d + "up_blocks.0.0.");
        for (int j = 0; j < c.est_blocks; ++j) { XBlock x; x.load(F, d + "up_blocks.0.1." + std::to_string(j) + "."); P.up_x.push_back(std::move(x)); }
        P.up_out.load(F, d + "up_blocks.0.2", 1, 0);
        P.final_block.load(F, d + "final_block.");
        P.final_proj.load(F, d + "final_proj", 1, 0);
        if (F.has(d + "down_blocks.1.0.block1.block.0.weight"))
            fail(VOX_ERR_UNSUPPORTED, "CosyVoice: estimators with more than one channel level are not supported (CosyVoice2 has channels [256])");
    }
    // ---- HiFT ----
    {
        const std::string h = "hift.";
        for (int i = 0; i < 5; ++i) { Conv cv; cv.load(F, h + "f0_predictor.condnet." + std::to_string(2 * i), 1, 1); P.f0_conv.push_back(std::move(cv)); }
        P.f0_cls.load(F, h + "f0_predictor.classifier");
        P.src_lin.load(F, h + "m_source.l_linear");
        P.conv_pre.load(F, h + "conv_pre", 1, 3);
        P.conv_post.load(F, h + "conv_post", 1, 3);
        const size_t nu = c.up_rates.size();
        for (size_t i = 0; i < nu; ++i) {
            ConvT u;
            u.load(F, h + "ups." + std::to_string(i), c.up_rates[i], (c.up_kernels[i] - c.up_rates[i]) / 2);
            P.ups.push_back(std::move(u));
        }
        // source down-sampling rates: cumprod([1] + upsample_rates[::-1][:-1]) reversed
        std::vector<int> dr{1};
        for (size_t i = nu; i-- > 1;) dr.push_back(dr.back() * c.up_rates[i]);
        std::reverse(dr.begin(), dr.end());
        for (size_t i = 0; i < nu; ++i) {
            Conv sd;
            if (dr[i] == 1) sd.load(F, h + "source_downs." + std::to_string(i), 1, 0);
            else sd.load(F, h + "source_downs." + std::to_string(i), dr[i], dr[i] / 2);
            P.src_down.push_back(std::move(sd));
            SnakeRes r;
            r.load(F, h + "source_resblocks." + std::to_string(i) + ".", c.src_rb_kernels[i], c.src_rb_dils[i]);
            P.src_res.push_back(std::move(r));
            for (size_t j = 0; j < c.rb_kernels.size(); ++j) {
                SnakeRes rr;
                rr.load(F, h + "resblocks." + std::to_string(i * c.rb_kernels.size() + j) + ".", c.rb_kernels[j], c.rb_dils[j]);
                P.res.push_back(std::move(rr));
            }
        }
    }
    load_front();
}

CosyVoiceModel::~CosyVoiceModel() = default;

int CosyVoiceModel::text_vocab() const { return int(p_->embed.rows); }

// ---------------------------------------------------------------- LLM
std::unique_ptr<CosyVoiceModel::LmState> CosyVoiceModel::lm_begin(const std::vector<int32_t> &prompt_text, const std::vector<int32_t> &text,
                                                                  const std::vector<int32_t> &prompt_tokens) const {
    if (!cfg_.has_llm) fail(VOX_ERR_UNSUPPORTED, "this .tmod has no CosyVoice LLM");
    const Impl &P = *p_;
    const int H = cfg_.llm_hidden;
    auto st = std::make_unique<LmState>();
    st->kv.init(cfg_.llm_layers);
    st->text_len = int64_t(text.size());
    const size_t n = 1 + prompt_text.size() + text.size() + 1 + prompt_tokens.size();
    std::vector<float> x(n * size_t(H));
    size_t r = 0;
    mat_gather(P.llm_emb, 0, &x[r++ * H]);
    for (int32_t t : prompt_text) {
        if (t < 0 || t >= P.embed.rows) fail(VOX_ERR_ARG, "CosyVoice: text token out of range");
        mat_gather(P.embed, t, &x[r++ * H]);
    }
    for (int32_t t : text) {
        if (t < 0 || t >= P.embed.rows) fail(VOX_ERR_ARG, "CosyVoice: text token out of range");
        mat_gather(P.embed, t, &x[r++ * H]);
    }
    mat_gather(P.llm_emb, 1, &x[r++ * H]);
    for (int32_t t : prompt_tokens) {
        if (t < 0 || t >= P.speech_emb.rows) fail(VOX_ERR_ARG, "CosyVoice: speech token out of range");
        mat_gather(P.speech_emb, t, &x[r++ * H]);
    }
    st->h.resize(size_t(H));
    transformer_forward(P.lm, st->kv, x.data(), int64_t(n), st->h.data());
    return st;
}

void CosyVoiceModel::lm_logp(LmState &s, std::vector<float> &logp) const {
    const Impl &P = *p_;
    logp.resize(size_t(P.dec.rows));
    matmul(P.dec, s.h.data(), 1, logp.data());
    float mx = -INFINITY;
    for (size_t i = 0; i < logp.size(); ++i) { logp[i] += P.dec_b[i]; mx = std::max(mx, logp[i]); }
    double sum = 0;
    for (float v : logp) sum += std::exp(double(v - mx));
    const float lse = mx + float(std::log(sum));
    for (auto &v : logp) v -= lse;
}

void CosyVoiceModel::lm_accept(LmState &s, int32_t token) const {
    const Impl &P = *p_;
    std::vector<float> x(static_cast<size_t>(cfg_.llm_hidden));
    mat_gather(P.speech_emb, token, x.data());
    transformer_forward(P.lm, s.kv, x.data(), 1, s.h.data());
}

int32_t CosyVoiceModel::ras_sample(std::vector<float> &logp, const std::vector<int32_t> &decoded, float u0, float u1) const {
    const int n = int(logp.size());
    // nucleus_sampling: softmax, stable descending sort, keep while cum < top_p and fewer than top_k
    float mx = -INFINITY;
    for (float v : logp) mx = std::max(mx, v);
    std::vector<double> pr(static_cast<size_t>(n));
    double sum = 0;
    for (int i = 0; i < n; ++i) { pr[size_t(i)] = std::exp(double(logp[size_t(i)] - mx)); sum += pr[size_t(i)]; }
    for (auto &p : pr) p /= sum;
    std::vector<int> idx(static_cast<size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return pr[size_t(a)] > pr[size_t(b)]; });
    std::vector<int> keep;
    double cum = 0, ksum = 0;
    for (int i = 0; i < n; ++i) {
        if (cum < cfg_.top_p && int(keep.size()) < cfg_.top_k) {
            cum += pr[size_t(idx[size_t(i)])];
            keep.push_back(idx[size_t(i)]);
        } else {
            break;
        }
    }
    for (int k : keep) ksum += pr[size_t(k)];
    int32_t top = keep.back();
    double acc = 0;
    const double target = double(u0) * ksum;
    for (int k : keep) { acc += pr[size_t(k)]; if (target < acc) { top = k; break; } }
    // repetition-aware: the token already occurs in the window → sample from the full distribution without it
    int rep = 0;
    const size_t from = decoded.size() > size_t(cfg_.win_size) ? decoded.size() - size_t(cfg_.win_size) : 0;
    for (size_t i = from; i < decoded.size(); ++i) rep += decoded[i] == top;
    if (double(rep) >= double(cfg_.win_size) * double(cfg_.tau_r)) {
        logp[size_t(top)] = -INFINITY;
        double s2 = 0;
        for (int i = 0; i < n; ++i) s2 += i == top ? 0.0 : pr[size_t(i)];
        const double t2 = double(u1) * s2;
        double a2 = 0;
        int32_t last = top == 0 ? 1 : 0;
        for (int i = 0; i < n; ++i) {
            if (i == top) continue;
            last = i;
            a2 += pr[size_t(i)];
            if (t2 < a2) return i;
        }
        return last;
    }
    return top;
}

std::vector<int32_t> CosyVoiceModel::lm_generate(const std::vector<int32_t> &prompt_text, const std::vector<int32_t> &text,
                                                 const std::vector<int32_t> &prompt_tokens, uint64_t seed, uint32_t piece) const {
    const auto t0 = Clock::now();
    auto st = lm_begin(prompt_text, text, prompt_tokens);
    const int min_len = int(double(text.size()) * double(cfg_.min_ratio)), max_len = int(double(text.size()) * double(cfg_.max_ratio));
    const DetRng rng(seed);
    const uint32_t stream = 0x43560000u + piece;   // "CV" + piece
    std::vector<int32_t> out;
    std::vector<float> logp;
    for (int i = 0; i < max_len; ++i) {
        lm_logp(*st, logp);
        int32_t top = 0;
        for (int trial = 0;; ++trial) {
            const float u0 = rng.uniform(uint64_t(i), stream, uint32_t(2 * trial)), u1 = rng.uniform(uint64_t(i), stream, uint32_t(2 * trial + 1));
            top = ras_sample(logp, out, u0, u1);
            if (i >= min_len || top < cfg_.speech_tokens) break;
            if (trial > 100) fail(VOX_ERR_STATE, "CosyVoice: sampling reaches max_trials and still gets eos when ignore_eos is True");
        }
        if (top >= cfg_.speech_tokens) break;
        out.push_back(top);
        lm_accept(*st, top);
    }
    ms_llm += ms_since(t0);
    return out;
}

// ---------------------------------------------------------------- flow
void CosyVoiceModel::flow(const std::vector<int32_t> &prompt_tokens, const std::vector<int32_t> &tokens, const std::vector<float> &prompt_feat,
                          const std::vector<float> &xvec, std::vector<float> &mel, FlowTrace *tr, uint64_t seed, uint32_t piece,
                          const std::vector<float> *z0) const {
    const auto t0 = Clock::now();
    const Impl &P = *p_;
    const CosyCfg &c = cfg_;
    const int D = c.enc_dim, M = c.mel_dim;
    const int T = int(prompt_tokens.size() + tokens.size());
    if (tokens.empty()) { mel.clear(); return; }
    const int mel1 = int(prompt_feat.size() / size_t(M));
    if (int(xvec.size()) != c.spk_dim) fail(VOX_ERR_ARG, "CosyVoice: the speaker x-vector must have " + std::to_string(c.spk_dim) + " values");
    // speaker: F.normalize → affine
    std::vector<float> spk(static_cast<size_t>(M));
    {
        double n2 = 0;
        for (float v : xvec) n2 += double(v) * double(v);
        const float inv = 1.0f / std::max(float(std::sqrt(n2)), 1e-12f);
        std::vector<float> e(xvec.size());
        for (size_t i = 0; i < e.size(); ++i) e[i] = xvec[i] * inv;
        P.spk_aff.forward(e.data(), 1, spk.data());
    }
    // token embedding
    std::vector<float> x(size_t(T) * D);
    for (int t = 0; t < T; ++t) {
        const int32_t id = std::max<int32_t>(0, t < int(prompt_tokens.size()) ? prompt_tokens[size_t(t)] : tokens[size_t(t) - prompt_tokens.size()]);
        if (id >= P.in_emb.rows) fail(VOX_ERR_ARG, "CosyVoice: speech token out of range");
        mat_gather(P.in_emb, id, &x[size_t(t) * D]);
    }
    // encoder
    std::vector<float> h = P.emb.forward(x, T);
    {
        // pre-lookahead: pad right by 3, conv1 (k 4) → leaky ReLU → pad left 2, conv2 (k 3) → + input
        const int la = c.pre_lookahead;
        std::vector<float> xp(size_t(T + la) * D, 0.f), y1, y1p, y2;
        std::memcpy(xp.data(), h.data(), h.size() * 4);
        P.pl1.forward(xp.data(), T + la, y1);
        for (auto &z : y1) z = z > 0.f ? z : 0.01f * z;
        causal_conv(P.pl2, y1, T, y2);
        for (size_t i = 0; i < h.size(); ++i) h[i] += y2[i];
    }
    {
        const std::vector<float> pe = rel_pe(T, D);
        for (const auto &L : P.enc) L.forward(h, T, pe, c.enc_heads);
    }
    const int T2 = 2 * T;
    {
        // Upsample1D: nearest ×2, pad left 4, conv k 5
        std::vector<float> up(size_t(T2 + 4) * D, 0.f), y;
        for (int t = 0; t < T2; ++t) std::memcpy(&up[size_t(t + 4) * D], &h[size_t(t / 2) * D], size_t(D) * 4);
        P.up_conv.forward(up.data(), T2 + 4, y);
        h = P.up_emb.forward(y, T2);
        const std::vector<float> pe = rel_pe(T2, D);
        for (const auto &L : P.up_enc) L.forward(h, T2, pe, c.enc_heads);
        layer_norm(h.data(), T2, D, P.after_w.data(), P.after_b.data(), 1e-5f);
    }
    std::vector<float> mu(size_t(T2) * M);
    P.enc_proj.forward(h.data(), T2, mu.data());
    if (mel1 >= T2) {   // no frames left after the prompt (a piece without speech tokens): nothing to say
        mel.clear();
        if (tr) { tr->mu.clear(); tr->mel.clear(); }
        return;
    }
    std::vector<float> cond(size_t(T2) * M, 0.f);
    std::memcpy(cond.data(), prompt_feat.data(), prompt_feat.size() * 4);
    if (!z0 && c.fixed_noise && T2 > P.noise_len)
        fail(VOX_ERR_ARG, "CosyVoice: utterance too long for the flow noise (" + std::to_string(T2) + " > " + std::to_string(P.noise_len) + " frames)");
    if (tr) { tr->spk = spk; tr->mu = mu; }

    // ---- CFM: Euler over a cosine schedule, conditional and unconditional estimator passes ----
    std::vector<float> z(size_t(T2) * M);
    if (z0) {
        if (z0->size() != z.size()) fail(VOX_ERR_ARG, "CosyVoice: injected flow noise has a wrong size");
        for (int t = 0; t < T2; ++t)
            for (int m = 0; m < M; ++m) z[size_t(t) * M + m] = (*z0)[size_t(m) * size_t(T2) + size_t(t)];
    } else if (c.fixed_noise) {
        for (int t = 0; t < T2; ++t)
            for (int m = 0; m < M; ++m) z[size_t(t) * M + m] = P.rand_noise[size_t(m) * size_t(P.noise_len) + size_t(t)];
    } else {   // torch.randn_like(mu) [1][80][T]: Philox Gaussians in that order
        const DetRng rng(seed);
        const uint32_t st = 0x464C0000u + piece;   // "FL"
        const size_t n = z.size(), pairs = (n + 1) / 2;
        std::vector<float> g(n);
        for (size_t p = 0; p < pairs; ++p) {
            const double u1 = double(rng.uniform(uint64_t(p), st, 0)), u2 = double(rng.uniform(uint64_t(p), st, 1));
            const double r = std::sqrt(-2.0 * std::log(1.0 - u1)), a = 2.0 * kPi * u2;
            g[2 * p] = float(r * std::cos(a));
            if (2 * p + 1 < n) g[2 * p + 1] = float(r * std::sin(a));
        }
        for (int t = 0; t < T2; ++t)
            for (int m = 0; m < M; ++m) z[size_t(t) * M + m] = g[size_t(m) * size_t(T2) + size_t(t)];
    }
    std::vector<float> ts = linspace01(c.n_timesteps + 1);
    if (c.cosine)
        for (auto &t : ts) t = 1.0f - std::cos((t * 0.5f) * float(kPi));
    const int C = c.est_ch;
    const int temb_in = int(P.t1.in);
    // one call = the conditional (sequence 0) and the unconditional (sequence 1: mu, spks, cond zero) estimator passes
    auto estimator = [&](const std::vector<float> &xx, float t, std::vector<float> &out) {
        // time embedding: SinusoidalPosEmb(320) (scale 1000) → linear → SiLU → linear; Mish of it feeds every resnet
        std::vector<float> te(static_cast<size_t>(temb_in)), th(size_t(P.t1.out)), temb(size_t(P.t2.out));
        const int half = temb_in / 2;
        const float ef = float(std::log(10000.0) / double(half - 1));
        for (int i = 0; i < half; ++i) {
            const float e = std::exp(float(i) * -ef);
            const float a = (1000.0f * t) * e;
            te[size_t(i)] = std::sin(a);
            te[size_t(half + i)] = std::cos(a);
        }
        P.t1.forward(te.data(), 1, th.data());
        for (auto &v : th) v = silu(v);
        P.t2.forward(th.data(), 1, temb.data());
        for (auto &v : temb) v = mish(v);
        const int B = 2;
        // input: [x | mu | spks | cond]
        std::vector<float> in(size_t(B) * size_t(T2) * size_t(4 * M), 0.f);
        for (int b = 0; b < B; ++b)
            for (int tt = 0; tt < T2; ++tt) {
                float *r = &in[(size_t(b) * T2 + size_t(tt)) * size_t(4 * M)];
                std::memcpy(r, &xx[size_t(tt) * M], size_t(M) * 4);
                if (b == 0) {
                    std::memcpy(r + M, &mu[size_t(tt) * M], size_t(M) * 4);
                    std::memcpy(r + 2 * M, spk.data(), size_t(M) * 4);
                    std::memcpy(r + 3 * M, &cond[size_t(tt) * M], size_t(M) * 4);
                }
            }
        std::vector<float> a, b;
        P.down_res.forward(in, T2, B, temb, a);
        for (const auto &xb : P.down_x) xb.forward(a, T2, B, c.est_heads);
        const std::vector<float> skip = a;
        causal_conv_b(P.down_conv, a, T2, B, b);
        for (size_t i = 0; i < P.mid_res.size(); ++i) {
            P.mid_res[i].forward(b, T2, B, temb, a);
            for (const auto &xb : P.mid_x[i]) xb.forward(a, T2, B, c.est_heads);
            b.swap(a);
        }
        std::vector<float> cat(size_t(B) * size_t(T2) * size_t(2 * C));
        for (int r = 0; r < B * T2; ++r) {
            std::memcpy(&cat[size_t(r) * size_t(2 * C)], &b[size_t(r) * C], size_t(C) * 4);
            std::memcpy(&cat[size_t(r) * size_t(2 * C) + C], &skip[size_t(r) * C], size_t(C) * 4);
        }
        P.up_res.forward(cat, T2, B, temb, a);
        for (const auto &xb : P.up_x) xb.forward(a, T2, B, c.est_heads);
        causal_conv_b(P.up_out, a, T2, B, b);
        P.final_block.forward(b, T2, B, a);
        P.final_proj.forward(a.data(), B * T2, out);
    };
    float t = ts[0], dt = ts[1] - ts[0];
    std::vector<float> dd;
    for (int step = 1; step <= c.n_timesteps; ++step) {
        estimator(z, t, dd);
        const float *d1 = dd.data(), *d2 = dd.data() + z.size();
        for (size_t i = 0; i < z.size(); ++i) {
            const float d = (1.0f + c.cfg_rate) * d1[i] - c.cfg_rate * d2[i];
            z[i] = z[i] + dt * d;
        }
        if (c.dt_diff) {   // Chatterbox: for t, r in zip(t_span[:-1], t_span[1:]): dt = r - t
            t = ts[size_t(step)];
            if (step < c.n_timesteps) dt = ts[size_t(step + 1)] - t;
        } else {
            t = t + dt;
            if (step < c.n_timesteps) dt = ts[size_t(step + 1)] - t;
        }
    }
    mel.assign(z.begin() + long(size_t(mel1) * M), z.end());
    if (tr) tr->mel = mel;
    ms_flow += ms_since(t0);
    if (g_prof_on)
            std::fprintf(stderr, "[cosyvoice] flow %d frames: resnets %.0f ms, transformer blocks %.0f ms (attention %.0f, qkv %.0f, ff %.0f, gelu %.0f ms)\n", T2, g_prof[0], g_prof[1], g_prof[2], g_prof[3], g_prof[4], g_prof[5]);
}

// ---------------------------------------------------------------- HiFT
void CosyVoiceModel::make_noise(uint64_t seed, uint32_t piece, size_t L, CosyNoise &n) {
    const DetRng rng(seed);
    const uint32_t s_ini = 0x48460000u + 2 * piece, s_noise = 0x48460000u + 2 * piece + 1;   // "HF"
    n.rand_ini.assign(9, 0.f);
    for (int h = 1; h < 9; ++h) n.rand_ini[size_t(h)] = rng.uniform(uint64_t(h), s_ini, 0);
    n.sine_noise.resize(L * 9);
    const size_t pairs = (n.sine_noise.size() + 1) / 2;
    parallel_for(int64_t(pairs), 4096, [&](int64_t p0, int64_t p1) {
        for (int64_t p = p0; p < p1; ++p) {
            const double u1 = double(rng.uniform(uint64_t(p), s_noise, 0)), u2 = double(rng.uniform(uint64_t(p), s_noise, 1));
            const double r = std::sqrt(-2.0 * std::log(1.0 - u1)), a = 2.0 * kPi * u2;
            n.sine_noise[size_t(p) * 2] = float(r * std::cos(a));
            if (size_t(p) * 2 + 1 < n.sine_noise.size()) n.sine_noise[size_t(p) * 2 + 1] = float(r * std::sin(a));
        }
    });
}

void CosyVoiceModel::hift(const std::vector<float> &mel, int T, uint64_t seed, uint32_t piece, const CosyNoise *inject,
                          std::vector<float> &pcm, HiftTrace *tr) const {
    const auto t0 = Clock::now();
    const Impl &P = *p_;
    const CosyCfg &c = cfg_;
    const int H = c.nb_harmonics + 1;
    if (T <= 0) { pcm.clear(); return; }
    // F0: 5 × (conv k3 → ELU) → linear → |·|
    std::vector<float> a = mel, b;
    for (const auto &cv : P.f0_conv) {
        cv.forward(a.data(), T, b);
        for (auto &z : b) z = elu(z);
        a.swap(b);
    }
    std::vector<float> f0(static_cast<size_t>(T));
    P.f0_cls.forward(a.data(), T, f0.data());
    for (auto &z : f0) z = std::fabs(z);
    if (tr) tr->f0 = f0;
    // source: F0 up-sampled to the sample rate (nearest), SineGen2, merge
    int up = c.hop;
    for (int r : c.up_rates) up *= r;
    const size_t L = size_t(T) * size_t(up);
    CosyNoise own;
    const CosyNoise *nz = inject;
    if (!nz) { make_noise(seed, piece, L, own); nz = &own; }
    if (int(nz->rand_ini.size()) != H || nz->sine_noise.size() != L * size_t(H)) fail(VOX_ERR_ARG, "CosyVoice: source noise has a wrong size");
    std::vector<float> src;
    {
        std::vector<float> tmpw(static_cast<size_t>(H));
        const float *wrow = mat_row(P.src_lin.w, 0, tmpw.data());
        if (c.sinegen == 1)
            nsf_source_v1(f0.data(), T, up, double(c.sample_rate), H, nz->rand_ini.data(), nz->sine_noise.data(), wrow,
                          P.src_lin.b.empty() ? 0.f : P.src_lin.b[0], src);
        else
            nsf_source(f0.data(), T, up, double(c.sample_rate), H, nz->rand_ini.data(), nz->sine_noise.data(), wrow,
                       P.src_lin.b.empty() ? 0.f : P.src_lin.b[0], src);
    }
    if (tr) tr->source = src;
    std::vector<float> s_stft;
    stft_center(src, c.n_fft, c.hop, false, s_stft);   // [frames][re | im]
    const int frames = int(L / size_t(c.hop)) + 1;
    // generator
    std::vector<float> x;
    P.conv_pre.forward(mel.data(), T, x);
    int Tc = T;
    const size_t nk = c.rb_kernels.size();
    for (size_t i = 0; i < P.ups.size(); ++i) {
        for (auto &z : x) z = z > 0.f ? z : c.lrelu * z;
        std::vector<float> u;
        P.ups[i].forward(x.data(), Tc, u);
        const int Cc = P.ups[i].cout;
        int Tu = int(u.size() / size_t(Cc));
        if (i + 1 == P.ups.size()) {   // ReflectionPad1d((1, 0))
            std::vector<float> pd(size_t(Tu + 1) * size_t(Cc));
            std::memcpy(pd.data(), &u[size_t(Cc)], size_t(Cc) * 4);
            std::memcpy(pd.data() + Cc, u.data(), u.size() * 4);
            u.swap(pd);
            ++Tu;
        }
        std::vector<float> si;
        P.src_down[i].forward(s_stft.data(), frames, si);
        const int Ts = int(si.size() / size_t(P.src_down[i].cout));
        P.src_res[i].forward(si, Ts);
        if (Ts != Tu) fail(VOX_ERR_FORMAT, "CosyVoice HiFT: length mismatch (" + std::to_string(Tu) + " vs " + std::to_string(Ts) + ")");
        for (size_t z = 0; z < u.size(); ++z) u[z] += si[z];
        std::vector<float> acc(u.size(), 0.f);
        for (size_t j = 0; j < nk; ++j) {
            std::vector<float> r = u;
            P.res[i * nk + j].forward(r, Tu);
            for (size_t z = 0; z < acc.size(); ++z) acc[z] += r[z];
        }
        for (auto &z : acc) z /= float(nk);
        x.swap(acc);
        Tc = Tu;
    }
    for (auto &z : x) z = z > 0.f ? z : 0.01f * z;
    std::vector<float> post;
    P.conv_post.forward(x.data(), Tc, post);
    if (Tc != frames) fail(VOX_ERR_FORMAT, "CosyVoice HiFT: spectrogram frames mismatch");
    istft_center(post, frames, c.n_fft, c.hop, 100.0f, pcm);
    for (auto &z : pcm) z = std::min(std::max(z, -c.audio_limit), c.audio_limit);
    ms_hift += ms_since(t0);
}

}  // namespace vox
