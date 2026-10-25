#include "kokoro.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "nn.hpp"
#include "sampling.hpp"

namespace vox {
using namespace nn;
namespace {

using Clock = std::chrono::steady_clock;

// VOX_PROFILE=kokoro: time per stage, printed at exit
struct KProf {
    bool on = false;
    std::vector<std::pair<std::string, double>> acc;
    KProf() { const char *e = std::getenv("VOX_PROFILE"); on = e && std::strstr(e, "kokoro"); }
    ~KProf() {
        if (!on || acc.empty()) return;
        double tot = 0;
        for (auto &a : acc) if (a.first[0] != ' ') tot += a.second;   // indented names are parts of the generator, already in its total
        std::fprintf(stderr, "vox Kokoro profile (%.0f ms):\n", tot);
        for (auto &a : acc) std::fprintf(stderr, "  %-16s %9.1f ms %5.1f%%\n", a.first.c_str(), a.second, 100.0 * a.second / tot);
    }
    void add(const char *n, double ms) {
        for (auto &a : acc) if (a.first == n) { a.second += ms; return; }
        acc.push_back({n, ms});
    }
};
KProf g_kprof;
struct KScope {
    const char *n;
    Clock::time_point t0;
    explicit KScope(const char *name) : n(name), t0(Clock::now()) {}
    ~KScope() { if (g_kprof.on) g_kprof.add(n, std::chrono::duration<double, std::milli>(Clock::now() - t0).count()); }
};




// Depthwise ConvTranspose1d (k=3, stride 2, padding 1, output_padding 1): T → 2T (the "pool" of upsampling AdaIN blocks)
struct DwConvT {
    int C = 0, K = 3;
    std::vector<float> w, b;   // [C][K]
    void load(const TmodFile &f, const std::string &n) {
        const TensorView &t = f.get(n + ".weight");
        C = int(t.shape[0]); K = int(t.shape[2]);
        if (t.shape[1] != 1 || K != 3) fail(VOX_ERR_FORMAT, "Kokoro: unexpected upsampling pool shape");
        w = to_float(t);
        if (f.has(n + ".bias")) b = to_float(f.get(n + ".bias"));
    }
    void forward(const float *x, int T, std::vector<float> &y) const {
        const int To = 2 * T;   // (T-1)*2 - 2 + 3 + 1
        y.assign(size_t(To) * size_t(C), 0.f);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < K; ++k) {
                const int to = t * 2 + k - 1;
                if (to < 0 || to >= To) continue;
                for (int c = 0; c < C; ++c) y[size_t(to) * C + c] += x[size_t(t) * C + c] * w[size_t(c) * K + k];
            }
        if (!b.empty())
            for (int t = 0; t < To; ++t)
                for (int c = 0; c < C; ++c) y[size_t(t) * C + c] += b[size_t(c)];
    }
};

// Bidirectional single-layer LSTM (PyTorch gate order i, f, g, o); x [T][in] → y [T][2H]
struct BiLSTM {
    int in = 0, H = 0;
    Mat wih[2];
    std::vector<float> whh[2], bias[2];   // whh float [4H][H]; bias = b_ih + b_hh
    void load(const TmodFile &f, const std::string &n) {
        for (int d = 0; d < 2; ++d) {
            const std::string suf = d ? "_l0_reverse" : "_l0";
            wih[d] = mat_of(f.get(n + ".weight_ih" + suf));
            whh[d] = to_float(f.get(n + ".weight_hh" + suf));
            const auto bi = to_float(f.get(n + ".bias_ih" + suf)), bh = to_float(f.get(n + ".bias_hh" + suf));
            bias[d].resize(bi.size());
            for (size_t i = 0; i < bi.size(); ++i) bias[d][i] = bi[i] + bh[i];
        }
        H = int(f.get(n + ".weight_hh_l0").shape[1]);
        in = int(wih[0].cols);
    }
    void forward(const float *x, int T, std::vector<float> &y) const {
        y.assign(size_t(T) * size_t(2 * H), 0.f);
        const int G = 4 * H;
        for (int d = 0; d < 2; ++d) {
            std::vector<float> gx(size_t(T) * size_t(G));
            matmul(wih[d], x, T, gx.data());
            std::vector<float> h(size_t(H), 0.f), c(size_t(H), 0.f), g(static_cast<size_t>(G));
            for (int s = 0; s < T; ++s) {
                const int t = d ? T - 1 - s : s;
                const float *gr = &gx[size_t(t) * size_t(G)];
                const float *W = whh[d].data();
                const float *bb = bias[d].data();
                parallel_for(G, 64, [&](int64_t a, int64_t e) {
                    for (int64_t r = a; r < e; ++r) {
                        const float *wr = W + r * H;
                        float acc = 0.f;
                        for (int k = 0; k < H; ++k) acc += wr[k] * h[size_t(k)];
                        g[size_t(r)] = gr[r] + bb[r] + acc;
                    }
                });
                for (int j = 0; j < H; ++j) {
                    const float ig = sigmoidf(g[size_t(j)]), fg = sigmoidf(g[size_t(H + j)]), gg = std::tanh(g[size_t(2 * H + j)]),
                                og = sigmoidf(g[size_t(3 * H + j)]);
                    c[size_t(j)] = fg * c[size_t(j)] + ig * gg;
                    h[size_t(j)] = og * std::tanh(c[size_t(j)]);
                }
                std::memcpy(&y[size_t(t) * size_t(2 * H) + size_t(d * H)], h.data(), size_t(H) * 4);
            }
        }
    }
};


// AdaIN1d: InstanceNorm over time per channel (no affine, eps 1e-5), then (1 + gamma) * x + beta, [gamma; beta] = fc(s)
struct AdaIN {
    Linear fc;
    int C = 0;
    void load(const TmodFile &f, const std::string &n) { fc.load(f, n + ".fc"); C = int(fc.out / 2); }
    void forward(std::vector<float> &x, int T, const float *s) const {
        std::vector<float> h(size_t(2 * C));
        fc.forward(s, 1, h.data());
        // per-channel statistics over time, row by row (cache friendly): blocks of time summed in parallel, two passes
        const int nb = std::max(1, std::min(64, T / 256));
        std::vector<double> part(size_t(nb) * size_t(C), 0.0);
        parallel_for(nb, 1, [&](int64_t b0, int64_t b1) {
            for (int64_t bi = b0; bi < b1; ++bi) {
                double *acc = &part[size_t(bi) * C];
                for (int64_t t = bi * T / nb; t < (bi + 1) * T / nb; ++t) {
                    const float *r = &x[size_t(t) * C];
                    for (int c = 0; c < C; ++c) acc[c] += r[c];
                }
            }
        });
        std::vector<double> mean(static_cast<size_t>(C), 0.0), var(static_cast<size_t>(C), 0.0);
        for (int bi = 0; bi < nb; ++bi) for (int c = 0; c < C; ++c) mean[size_t(c)] += part[size_t(bi) * C + c];
        for (auto &m : mean) m /= T;
        std::fill(part.begin(), part.end(), 0.0);
        parallel_for(nb, 1, [&](int64_t b0, int64_t b1) {
            for (int64_t bi = b0; bi < b1; ++bi) {
                double *acc = &part[size_t(bi) * C];
                for (int64_t t = bi * T / nb; t < (bi + 1) * T / nb; ++t) {
                    const float *r = &x[size_t(t) * C];
                    for (int c = 0; c < C; ++c) { const double d = r[c] - mean[size_t(c)]; acc[c] += d * d; }
                }
            }
        });
        for (int bi = 0; bi < nb; ++bi) for (int c = 0; c < C; ++c) var[size_t(c)] += part[size_t(bi) * C + c];
        std::vector<float> sc(static_cast<size_t>(C)), sh(static_cast<size_t>(C));
        for (int c = 0; c < C; ++c) {
            const float inv = float(1.0 / std::sqrt(var[size_t(c)] / T + 1e-5));
            sc[size_t(c)] = (1.0f + h[size_t(c)]) * inv;
            sh[size_t(c)] = h[size_t(C + c)] - (1.0f + h[size_t(c)]) * inv * float(mean[size_t(c)]);
        }
        parallel_for(T, 256, [&](int64_t a, int64_t e) {
            for (int64_t t = a; t < e; ++t) {
                float *r = &x[size_t(t) * C];
                for (int c = 0; c < C; ++c) r[c] = r[c] * sc[size_t(c)] + sh[size_t(c)];
            }
        });
    }
};

// AdainResBlk1d (prosody F0/N branches and the decoder)
struct AdainResBlk {
    int din = 0, dout = 0;
    bool up = false, learned_sc = false;
    AdaIN n1, n2;
    Conv c1, c2, sc;
    DwConvT pool;
    void load(const TmodFile &f, const std::string &n, bool upsample) {
        up = upsample;
        n1.load(f, n + ".norm1"); n2.load(f, n + ".norm2");
        c1.load(f, n + ".conv1"); c2.load(f, n + ".conv2");
        din = c1.cin; dout = c1.cout;
        learned_sc = f.has(n + ".conv1x1.weight");
        if (learned_sc) sc.load(f, n + ".conv1x1");
        if (up) pool.load(f, n + ".pool");
    }
    // x [T][din] → y [T'][dout] (T' = 2T if upsample)
    void forward(const std::vector<float> &x, int T, const float *s, std::vector<float> &y) const {
        std::vector<float> h(x), t1, t2;
        n1.forward(h, T, s);
        for (auto &v : h) v = leaky(v, 0.2f);
        int Tn = T;
        if (up) { pool.forward(h.data(), T, t1); h.swap(t1); Tn = 2 * T; }
        c1.forward(h.data(), Tn, t1);
        n2.forward(t1, Tn, s);
        for (auto &v : t1) v = leaky(v, 0.2f);
        c2.forward(t1.data(), Tn, y);
        // shortcut: nearest ×2, 1x1 conv without bias
        std::vector<float> sx;
        const std::vector<float> *src = &x;
        if (up) {
            sx.resize(size_t(Tn) * size_t(din));
            for (int t = 0; t < Tn; ++t) std::memcpy(&sx[size_t(t) * din], &x[size_t(t / 2) * din], size_t(din) * 4);
            src = &sx;
        }
        const float k = float(1.0 / std::sqrt(2.0));
        if (learned_sc) {
            sc.forward(src->data(), Tn, t2);
            for (size_t i = 0; i < y.size(); ++i) y[i] = (y[i] + t2[i]) * k;
        } else {
            for (size_t i = 0; i < y.size(); ++i) y[i] = (y[i] + (*src)[i]) * k;
        }
    }
};

// AdaINResBlock1 of the generator: 3 x (AdaIN → Snake → dilated conv → AdaIN → Snake → conv) with residuals
struct AdaINResBlock1 {
    int C = 0;
    AdaIN a1[3], a2[3];
    Conv c1[3], c2[3];
    std::vector<float> al1[3], al2[3];
    void load(const TmodFile &f, const std::string &n, const std::vector<int> &dils) {
        for (int j = 0; j < 3; ++j) {
            const std::string i = std::to_string(j);
            c1[j].load(f, n + ".convs1." + i, 1, -1, dils[size_t(j)]);
            c2[j].load(f, n + ".convs2." + i, 1, -1, 1);
            a1[j].load(f, n + ".adain1." + i); a2[j].load(f, n + ".adain2." + i);
            al1[j] = to_float(f.get(n + ".alpha1." + i)); al2[j] = to_float(f.get(n + ".alpha2." + i));
        }
        C = c1[0].cin;
    }
    static void snake(std::vector<float> &x, int C, const std::vector<float> &al) {
        parallel_for(int64_t(x.size() / size_t(C)), 256, [&](int64_t a, int64_t e) {
            for (int64_t t = a; t < e; ++t)
                for (int c = 0; c < C; ++c) {
                    float &v = x[size_t(t) * C + c];
                    const float s = fast_sin(al[size_t(c)] * v);
                    v = v + (1.0f / al[size_t(c)]) * (s * s);
                }
        });
    }
    void forward(std::vector<float> &x, int T, const float *s) const {
        std::vector<float> xt, t2;
        for (int j = 0; j < 3; ++j) {
            { KScope k("  gen.misc"); xt = x; }
            { KScope k("  gen.adain"); a1[j].forward(xt, T, s); }
            { KScope k("  gen.snake"); snake(xt, C, al1[j]); }
            { KScope k("  gen.conv"); c1[j].forward(xt.data(), T, t2); }
            { KScope k("  gen.adain"); a2[j].forward(t2, T, s); }
            { KScope k("  gen.snake"); snake(t2, C, al2[j]); }
            { KScope k("  gen.conv"); c2[j].forward(t2.data(), T, xt); }
            { KScope k("  gen.misc"); for (size_t i = 0; i < x.size(); ++i) x[i] += xt[i]; }
        }
    }
};


}  // namespace

// ============================================================================================================
struct KokoroModel::Impl {
    // PL-BERT
    Mat word_emb, pos_emb, type_emb;
    std::vector<float> emb_ln_w, emb_ln_b;
    Linear map_in, q, k, v, dense, ffn, ffn_out, bert_enc;
    std::vector<float> attn_ln_w, attn_ln_b, full_ln_w, full_ln_b;
    // prosody predictor
    std::vector<BiLSTM> dur_lstm;
    std::vector<Linear> dur_ada;     // AdaLayerNorm fc
    BiLSTM pred_lstm, shared_lstm;
    Linear dur_proj;
    AdainResBlk F0b[3], Nb[3];
    Conv F0_proj, N_proj;
    // text encoder
    Mat te_emb;
    std::vector<Conv> te_cnn;
    std::vector<std::vector<float>> te_ln_g, te_ln_b;
    BiLSTM te_lstm;
    // decoder
    Conv F0_conv, N_conv, asr_res;
    AdainResBlk encode, decode[4];
    // generator
    Linear src_linear;
    std::vector<ConvT> ups;
    std::vector<Conv> noise_convs;
    std::vector<AdaINResBlock1> noise_res, resblocks;
    Conv conv_post;
    // voices
    std::map<std::string, const float *> packs;
    int pack_rows = 0;
};

KokoroModel::KokoroModel(std::shared_ptr<TmodFile> f) : p_(new Impl), file_(std::move(f)) {
    const TmodFile &F = *file_;
    const json &c = F.manifest().at("kokoro");
    cfg_.n_token = c.value("n_token", 178);
    cfg_.hidden = c.value("hidden_dim", 512);
    cfg_.style_dim = c.value("style_dim", 128);
    cfg_.n_layer = c.value("n_layer", 3);
    cfg_.max_dur = c.value("max_dur", 50);
    cfg_.text_kernel = c.value("text_encoder_kernel_size", 5);
    cfg_.context = c.value("context_length", 512);
    const json &pb = c.at("plbert");
    cfg_.bert_hidden = pb.at("hidden_size"); cfg_.bert_heads = pb.at("num_attention_heads"); cfg_.bert_inter = pb.at("intermediate_size");
    cfg_.bert_layers = pb.at("num_hidden_layers"); cfg_.bert_emb = pb.value("embedding_size", 128); cfg_.bert_max_pos = pb.value("max_position_embeddings", 512);
    const json &is = c.at("istftnet");
    cfg_.up_rates = is.at("upsample_rates").get<std::vector<int>>();
    cfg_.up_kernels = is.at("upsample_kernel_sizes").get<std::vector<int>>();
    cfg_.rb_kernels = is.at("resblock_kernel_sizes").get<std::vector<int>>();
    cfg_.rb_dils = is.at("resblock_dilation_sizes").get<std::vector<std::vector<int>>>();
    cfg_.up_initial = is.at("upsample_initial_channel");
    cfg_.n_fft = is.at("gen_istft_n_fft"); cfg_.hop = is.at("gen_istft_hop_size");
    if (cfg_.hidden != 512) fail(VOX_ERR_UNSUPPORTED, "Kokoro: hidden_dim must be 512 (the decoder widths are fixed)");
    if (cfg_.up_rates.size() != 2 || cfg_.rb_kernels.size() != 3) fail(VOX_ERR_UNSUPPORTED, "Kokoro: unsupported iSTFTNet layout");
    for (const auto &kv : c.at("vocab").items()) vocab_[kv.key()] = kv.value().get<int32_t>();

    Impl &P = *p_;
    // ---- PL-BERT (ALBERT; one shared layer)
    const std::string b = "bert.";
    P.word_emb = mat_of(F.get(b + "embeddings.word_embeddings.weight"));
    P.pos_emb = mat_of(F.get(b + "embeddings.position_embeddings.weight"));
    P.type_emb = mat_of(F.get(b + "embeddings.token_type_embeddings.weight"));
    P.emb_ln_w = to_float(F.get(b + "embeddings.LayerNorm.weight")); P.emb_ln_b = to_float(F.get(b + "embeddings.LayerNorm.bias"));
    P.map_in.load(F, b + "encoder.embedding_hidden_mapping_in");
    const std::string L = b + "encoder.albert_layer_groups.0.albert_layers.0.";
    P.q.load(F, L + "attention.query"); P.k.load(F, L + "attention.key"); P.v.load(F, L + "attention.value");
    P.dense.load(F, L + "attention.dense");
    P.attn_ln_w = to_float(F.get(L + "attention.LayerNorm.weight")); P.attn_ln_b = to_float(F.get(L + "attention.LayerNorm.bias"));
    P.ffn.load(F, L + "ffn"); P.ffn_out.load(F, L + "ffn_output");
    P.full_ln_w = to_float(F.get(L + "full_layer_layer_norm.weight")); P.full_ln_b = to_float(F.get(L + "full_layer_layer_norm.bias"));
    P.bert_enc.load(F, "bert_encoder");
    // ---- prosody predictor
    const std::string pr = "predictor.";
    for (int i = 0; i < cfg_.n_layer; ++i) {
        BiLSTM l;
        l.load(F, pr + "text_encoder.lstms." + std::to_string(2 * i));
        P.dur_lstm.push_back(std::move(l));
        Linear a;
        a.load(F, pr + "text_encoder.lstms." + std::to_string(2 * i + 1) + ".fc");
        P.dur_ada.push_back(std::move(a));
    }
    P.pred_lstm.load(F, pr + "lstm");
    P.dur_proj.load(F, pr + "duration_proj.linear_layer");
    P.shared_lstm.load(F, pr + "shared");
    for (int i = 0; i < 3; ++i) {
        P.F0b[i].load(F, pr + "F0." + std::to_string(i), i == 1);
        P.Nb[i].load(F, pr + "N." + std::to_string(i), i == 1);
    }
    P.F0_proj.load(F, pr + "F0_proj"); P.N_proj.load(F, pr + "N_proj");
    // ---- text encoder
    P.te_emb = mat_of(F.get("text_encoder.embedding.weight"));
    for (int i = 0; i < cfg_.n_layer; ++i) {
        const std::string n = "text_encoder.cnn." + std::to_string(i);
        Conv cv;
        cv.load(F, n + ".0");
        P.te_cnn.push_back(std::move(cv));
        P.te_ln_g.push_back(to_float(F.get(n + ".1.gamma")));
        P.te_ln_b.push_back(to_float(F.get(n + ".1.beta")));
    }
    P.te_lstm.load(F, "text_encoder.lstm");
    // ---- decoder
    const std::string d = "decoder.";
    P.F0_conv.load(F, d + "F0_conv", 2, 1); P.N_conv.load(F, d + "N_conv", 2, 1);
    P.asr_res.load(F, d + "asr_res.0");
    P.encode.load(F, d + "encode", false);
    for (int i = 0; i < 4; ++i) P.decode[i].load(F, d + "decode." + std::to_string(i), i == 3);
    // ---- generator
    const std::string g = d + "generator.";
    P.src_linear.load(F, g + "m_source.l_linear");
    const size_t nu = cfg_.up_rates.size();
    for (size_t i = 0; i < nu; ++i) {
        const int u = cfg_.up_rates[i], k = cfg_.up_kernels[i];
        ConvT t;
        t.load(F, g + "ups." + std::to_string(i), u, (k - u) / 2);
        P.ups.push_back(std::move(t));
        Conv nc;
        if (i + 1 < nu) {
            int sf = 1;
            for (size_t j = i + 1; j < nu; ++j) sf *= cfg_.up_rates[j];
            nc.load(F, g + "noise_convs." + std::to_string(i), sf, (sf + 1) / 2);
        } else {
            nc.load(F, g + "noise_convs." + std::to_string(i), 1, 0);
        }
        P.noise_convs.push_back(std::move(nc));
        AdaINResBlock1 nr;
        nr.load(F, g + "noise_res." + std::to_string(i), {1, 3, 5});
        P.noise_res.push_back(std::move(nr));
        for (size_t j = 0; j < cfg_.rb_kernels.size(); ++j) {
            AdaINResBlock1 rb;
            rb.load(F, g + "resblocks." + std::to_string(i * cfg_.rb_kernels.size() + j), cfg_.rb_dils[j]);
            P.resblocks.push_back(std::move(rb));
        }
    }
    P.conv_post.load(F, g + "conv_post");
    // ---- voices
    for (const auto &kv : F.blobs())
        if (kv.first.rfind("voice.", 0) == 0) {
            if (kv.second.dtype != DType::F32 || kv.second.shape.size() != 2 || kv.second.shape[1] != 2 * cfg_.style_dim)
                fail(VOX_ERR_FORMAT, "Kokoro: bad voice pack " + kv.first);
            if (kv.second.shape[0] <= 0 || kv.second.nbytes < size_t(kv.second.shape[0]) * size_t(kv.second.shape[1]) * sizeof(float) ||
                (P.pack_rows != 0 && P.pack_rows != int(kv.second.shape[0])))
                fail(VOX_ERR_FORMAT, "Kokoro: voice pack " + kv.first + " has a size that does not match its shape");
            P.packs[kv.first.substr(6)] = reinterpret_cast<const float *>(kv.second.data);
            P.pack_rows = int(kv.second.shape[0]);
            voices_.push_back(kv.first.substr(6));
        }
    for (const auto &kv : F.blobs()) weight_bytes_ += kv.second.nbytes;
}

KokoroModel::~KokoroModel() = default;

static std::vector<std::string> utf8_chars(const std::string &s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        n = std::min(n, s.size() - i);
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

std::vector<int32_t> KokoroModel::phonemes_to_ids(const std::string &ps) const {
    std::vector<int32_t> ids;
    for (const auto &ch : utf8_chars(ps)) {
        auto it = vocab_.find(ch);
        if (it != vocab_.end()) ids.push_back(it->second);
    }
    return ids;
}
size_t KokoroModel::phoneme_count(const std::string &ps) const { return utf8_chars(ps).size(); }

bool KokoroModel::has_voice(const std::string &name) const {
    size_t a = 0;
    bool any = false;
    while (a <= name.size()) {
        size_t b = name.find(',', a);
        if (b == std::string::npos) b = name.size();
        std::string one = name.substr(a, b - a);
        if (!one.empty()) {
            if (!p_->packs.count(one)) return false;
            any = true;
        }
        a = b + 1;
    }
    return any;
}

void KokoroModel::style(const std::string &voice, size_t n, std::vector<float> &ref_s) const {
    const int S = 2 * cfg_.style_dim;
    ref_s.assign(size_t(S), 0.f);
    std::vector<const float *> packs;
    size_t a = 0;
    while (a <= voice.size()) {
        size_t b = voice.find(',', a);
        if (b == std::string::npos) b = voice.size();
        const std::string one = voice.substr(a, b - a);
        if (!one.empty()) {
            auto it = p_->packs.find(one);
            if (it == p_->packs.end()) fail(VOX_ERR_ARG, "Kokoro: unknown voice '" + one + "'");
            packs.push_back(it->second);
        }
        a = b + 1;
    }
    if (packs.empty()) fail(VOX_ERR_ARG, "Kokoro: no voice given");
    if (n == 0) n = 1;
    const size_t row = std::min<size_t>(n - 1, size_t(p_->pack_rows - 1));
    // torch.mean(torch.stack(packs), dim=0): sum in float, divide once
    for (const float *p : packs)
        for (int i = 0; i < S; ++i) ref_s[size_t(i)] += p[row * size_t(S) + size_t(i)];
    if (packs.size() > 1)
        for (auto &x : ref_s) x /= float(packs.size());
}

std::string KokoroModel::lexicon(const std::string &name) const {
    const std::string k = "g2p." + name;
    if (!file_->has(k)) return {};
    const TensorView &t = file_->get(k);
    return std::string(reinterpret_cast<const char *>(t.data), t.nbytes);
}

void KokoroModel::make_noise(uint64_t seed, uint32_t chunk, size_t L, KokoroNoise &n) {
    const DetRng rng(seed);
    const uint32_t s_ini = 0x4B000000u + 2 * chunk, s_noise = 0x4B000000u + 2 * chunk + 1;
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

void KokoroModel::synthesize(const std::vector<int32_t> &ids_in, const float *ref_s, float speed, uint64_t seed, uint32_t chunk,
                             const KokoroNoise *inject, std::vector<float> &pcm, Trace *tr) const {
    const Impl &P = *p_;
    const auto t_start = Clock::now();
    const int H = cfg_.hidden, SD = cfg_.style_dim, BH = cfg_.bert_hidden, BE = cfg_.bert_emb;
    std::vector<int32_t> ids;
    ids.push_back(0);
    ids.insert(ids.end(), ids_in.begin(), ids_in.end());
    ids.push_back(0);
    const int T = int(ids.size());
    if (T > cfg_.context) fail(VOX_ERR_ARG, "Kokoro: too many phonemes in one piece (" + std::to_string(T - 2) + " > " + std::to_string(cfg_.context - 2) + ")");
    for (int32_t id : ids)
        if (id < 0 || id >= cfg_.n_token) fail(VOX_ERR_ARG, "Kokoro: token id out of range");
    const float *s_pred = ref_s + SD;

    // ---------------- PL-BERT ----------------
    std::unique_ptr<KScope> ks(new KScope("plbert"));
    std::vector<float> e(size_t(T) * size_t(BE)), tmp(size_t(std::max<int64_t>({P.word_emb.cols, P.pos_emb.cols, P.type_emb.cols})));
    for (int t = 0; t < T; ++t) {
        float *r = &e[size_t(t) * BE];
        mat_gather(P.word_emb, ids[size_t(t)], r);
        mat_gather(P.pos_emb, t, tmp.data());
        for (int i = 0; i < BE; ++i) r[i] += tmp[size_t(i)];
        mat_gather(P.type_emb, 0, tmp.data());
        for (int i = 0; i < BE; ++i) r[i] += tmp[size_t(i)];
    }
    layer_norm(e.data(), T, BE, P.emb_ln_w.data(), P.emb_ln_b.data(), 1e-12f);
    std::vector<float> h(size_t(T) * size_t(BH)), q(h.size()), k(h.size()), v(h.size()), ctx(h.size()), ao(h.size()),
        ff(size_t(T) * size_t(cfg_.bert_inter)), fo(h.size());
    P.map_in.forward(e.data(), T, h.data());
    const int NH = cfg_.bert_heads, HD = BH / NH;
    const float scale = 1.0f / std::sqrt(float(HD));
    for (int l = 0; l < cfg_.bert_layers; ++l) {
        P.q.forward(h.data(), T, q.data());
        P.k.forward(h.data(), T, k.data());
        P.v.forward(h.data(), T, v.data());
        parallel_for(int64_t(T) * NH, 4, [&](int64_t a0, int64_t a1) {
            std::vector<float> sc(static_cast<size_t>(T));
            for (int64_t a = a0; a < a1; ++a) {
                const int i = int(a / NH), hh = int(a % NH);
                const float *qi = &q[size_t(i) * BH + size_t(hh) * HD];
                float mx = -INFINITY;
                for (int j = 0; j < T; ++j) {
                    const float *kj = &k[size_t(j) * BH + size_t(hh) * HD];
                    float acc = 0;
                    for (int dd = 0; dd < HD; ++dd) acc += qi[dd] * kj[dd];
                    sc[size_t(j)] = acc * scale;
                    mx = std::max(mx, sc[size_t(j)]);
                }
                float sum = 0;
                for (int j = 0; j < T; ++j) { sc[size_t(j)] = std::exp(sc[size_t(j)] - mx); sum += sc[size_t(j)]; }
                float *o = &ctx[size_t(i) * BH + size_t(hh) * HD];
                std::fill_n(o, HD, 0.f);
                for (int j = 0; j < T; ++j) {
                    const float w = sc[size_t(j)] / sum;
                    const float *vj = &v[size_t(j) * BH + size_t(hh) * HD];
                    for (int dd = 0; dd < HD; ++dd) o[dd] += w * vj[dd];
                }
            }
        });
        P.dense.forward(ctx.data(), T, ao.data());
        for (size_t i = 0; i < ao.size(); ++i) ao[i] += h[i];
        layer_norm(ao.data(), T, BH, P.attn_ln_w.data(), P.attn_ln_b.data(), 1e-12f);
        P.ffn.forward(ao.data(), T, ff.data());
        for (auto &x : ff) {   // gelu_new (tanh approximation)
            const float c = 0.7978845608028654f;
            x = 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
        }
        P.ffn_out.forward(ff.data(), T, fo.data());
        for (size_t i = 0; i < fo.size(); ++i) fo[i] += ao[i];
        layer_norm(fo.data(), T, BH, P.full_ln_w.data(), P.full_ln_b.data(), 1e-12f);
        h.swap(fo);
    }
    std::vector<float> d_en(size_t(T) * size_t(H));
    P.bert_enc.forward(h.data(), T, d_en.data());
    if (tr) { tr->bert = h; tr->d_en = d_en; }

    // ---------------- prosody: duration encoder ----------------
    ks.reset(new KScope("durations"));
    const int DS = H + SD;
    std::vector<float> x(size_t(T) * size_t(DS)), y, ada(size_t(2 * H));
    auto cat_style = [&](const std::vector<float> &src, int C) {
        x.resize(size_t(T) * size_t(C + SD));
        for (int t = 0; t < T; ++t) {
            std::memcpy(&x[size_t(t) * (C + SD)], &src[size_t(t) * C], size_t(C) * 4);
            std::memcpy(&x[size_t(t) * (C + SD) + C], s_pred, size_t(SD) * 4);
        }
    };
    cat_style(d_en, H);
    for (int l = 0; l < cfg_.n_layer; ++l) {
        P.dur_lstm[size_t(l)].forward(x.data(), T, y);   // [T][H]
        P.dur_ada[size_t(l)].forward(s_pred, 1, ada.data());
        layer_norm(y.data(), T, H, ada.data(), ada.data() + H, 1e-5f, true);
        cat_style(y, H);
    }
    std::vector<float> d = x;   // [T][H + SD]
    if (tr) tr->d = d;
    std::vector<float> lx;
    P.pred_lstm.forward(d.data(), T, lx);
    std::vector<float> dl(size_t(T) * size_t(cfg_.max_dur));
    P.dur_proj.forward(lx.data(), T, dl.data());
    std::vector<int32_t> dur(static_cast<size_t>(T));
    std::vector<float> durf(static_cast<size_t>(T));
    int F = 0;
    for (int t = 0; t < T; ++t) {
        float sum = 0.f;
        for (int j = 0; j < cfg_.max_dur; ++j) sum += sigmoidf(dl[size_t(t) * cfg_.max_dur + j]);
        sum /= speed;
        durf[size_t(t)] = sum;
        const float r = std::nearbyint(sum);   // torch.round: half to even
        dur[size_t(t)] = std::max(1, int(r));
        F += dur[size_t(t)];
    }
    if (tr) { tr->duration = durf; tr->pred_dur = dur; }
    std::vector<int> frame_tok(static_cast<size_t>(F));
    for (int t = 0, f = 0; t < T; ++t)
        for (int r = 0; r < dur[size_t(t)]; ++r) frame_tok[size_t(f++)] = t;

    // ---------------- F0 / N ----------------
    ks.reset(new KScope("F0/N"));
    std::vector<float> en(size_t(F) * size_t(DS));
    for (int f = 0; f < F; ++f) std::memcpy(&en[size_t(f) * DS], &d[size_t(frame_tok[size_t(f)]) * DS], size_t(DS) * 4);
    std::vector<float> sh;
    P.shared_lstm.forward(en.data(), F, sh);   // [F][H]
    auto branch = [&](const AdainResBlk *blk, const Conv &proj, std::vector<float> &out) {
        std::vector<float> a = sh, bv;
        int Tc = F;
        for (int i = 0; i < 3; ++i) {
            blk[i].forward(a, Tc, s_pred, bv);
            if (blk[i].up) Tc *= 2;
            a.swap(bv);
        }
        proj.forward(a.data(), Tc, out);   // [2F][1]
    };
    std::vector<float> F0, Nn;
    branch(P.F0b, P.F0_proj, F0);
    branch(P.Nb, P.N_proj, Nn);
    if (tr) { tr->F0 = F0; tr->N = Nn; }
    ms_prosody_ += std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
    const auto t_dec = Clock::now();

    // ---------------- text encoder ----------------
    ks.reset(new KScope("text encoder"));
    std::vector<float> te(size_t(T) * size_t(H));
    for (int t = 0; t < T; ++t) mat_gather(P.te_emb, ids[size_t(t)], &te[size_t(t) * H]);
    for (size_t i = 0; i < P.te_cnn.size(); ++i) {
        std::vector<float> o;
        P.te_cnn[i].forward(te.data(), T, o);
        layer_norm(o.data(), T, H, P.te_ln_g[i].data(), P.te_ln_b[i].data(), 1e-5f);
        for (auto &z : o) z = leaky(z, 0.2f);
        te.swap(o);
    }
    std::vector<float> t_en;
    P.te_lstm.forward(te.data(), T, t_en);
    if (tr) tr->t_en = t_en;
    std::vector<float> asr(size_t(F) * size_t(H));
    for (int f = 0; f < F; ++f) std::memcpy(&asr[size_t(f) * H], &t_en[size_t(frame_tok[size_t(f)]) * H], size_t(H) * 4);
    if (tr) tr->asr = asr;

    ks.reset();
    decode(asr, F, F0, Nn, ref_s, seed, chunk, inject, pcm, tr);
    ms_decoder_ += std::chrono::duration<double, std::milli>(Clock::now() - t_dec).count();
}

void KokoroModel::decode(const std::vector<float> &asr, int F, const std::vector<float> &F0, const std::vector<float> &Nn, const float *ref_s,
                         uint64_t seed, uint32_t chunk, const KokoroNoise *inject, std::vector<float> &pcm, Trace *tr) const {
    const Impl &P = *p_;
    const int H = cfg_.hidden;
    const float *s_dec = ref_s;
    if (int(asr.size()) != F * H || int(F0.size()) != 2 * F || int(Nn.size()) != 2 * F) fail(VOX_ERR_ARG, "Kokoro decode: bad input sizes");
    // ---------------- decoder ----------------
    std::unique_ptr<KScope> ks(new KScope("decoder"));
    std::vector<float> f0c, nc;
    P.F0_conv.forward(F0.data(), 2 * F, f0c);   // [F][1]
    P.N_conv.forward(Nn.data(), 2 * F, nc);
    if (int(f0c.size()) != F || int(nc.size()) != F) fail(VOX_ERR_FORMAT, "Kokoro: F0/N length mismatch");
    std::vector<float> xin(size_t(F) * size_t(H + 2));
    for (int f = 0; f < F; ++f) {
        std::memcpy(&xin[size_t(f) * (H + 2)], &asr[size_t(f) * H], size_t(H) * 4);
        xin[size_t(f) * (H + 2) + H] = f0c[size_t(f)];
        xin[size_t(f) * (H + 2) + H + 1] = nc[size_t(f)];
    }
    std::vector<float> xd, rres, xc;
    P.encode.forward(xin, F, s_dec, xd);   // [F][1024]
    P.asr_res.forward(asr.data(), F, rres);   // [F][64]
    int Tc = F;
    bool res = true;
    for (int i = 0; i < 4; ++i) {
        const AdainResBlk &blk = P.decode[i];
        if (res) {
            const int C0 = int(xd.size() / size_t(Tc)), C = C0 + 64 + 2;
            xc.resize(size_t(Tc) * size_t(C));
            for (int f = 0; f < Tc; ++f) {
                float *r = &xc[size_t(f) * C];
                std::memcpy(r, &xd[size_t(f) * C0], size_t(C0) * 4);
                std::memcpy(r + C0, &rres[size_t(f) * 64], 64 * 4);
                r[C0 + 64] = f0c[size_t(f)];
                r[C0 + 65] = nc[size_t(f)];
            }
        } else {
            xc = xd;
        }
        std::vector<float> o;
        blk.forward(xc, Tc, s_dec, o);
        xd.swap(o);
        if (blk.up) { Tc *= 2; res = false; }
    }
    // xd: [2F][512]
    if (tr) tr->dec = xd;

    // ---------------- generator: harmonic source ----------------
    ks.reset(new KScope("source"));
    int up_total = cfg_.hop;
    for (int r : cfg_.up_rates) up_total *= r;   // 300
    const int Tg = Tc;                            // 2F
    const size_t Ls = size_t(Tg) * size_t(up_total);
    KokoroNoise own;
    const KokoroNoise *nz = inject;
    if (!nz) { make_noise(seed, chunk, Ls, own); nz = &own; }
    const bool given_har = !nz->har.empty();
    if (!given_har && (nz->rand_ini.size() != 9 || nz->sine_noise.size() != Ls * 9)) fail(VOX_ERR_ARG, "Kokoro: source noise has a wrong size");
    const double SR = double(cfg_.sample_rate);
    const int NF = cfg_.n_fft, HOP = cfg_.hop, NB = NF / 2 + 1;
    const int frames = int(Ls / size_t(HOP)) + 1;
    std::vector<float> har(size_t(frames) * size_t(2 * NB));
    if (given_har) {
        if (nz->har.size() != har.size()) fail(VOX_ERR_ARG, "Kokoro: injected source spectrogram has a wrong size");
        har = nz->har;
    } else {
    std::vector<float> har_src;
    {
        std::vector<float> tmpw(9);
        const float *wrow = mat_row(P.src_linear.w, 0, tmpw.data());
        nsf_source(F0.data(), Tg, up_total, SR, 9, nz->rand_ini.data(), nz->sine_noise.data(), wrow,
                   P.src_linear.b.empty() ? 0.f : P.src_linear.b[0], har_src);
    }
    if (tr) tr->har_src = har_src;
    stft_center(har_src, NF, HOP, true, har);
    }
    if (tr) tr->har = har;
    // ---------------- generator: upsampling ----------------
    ks.reset(new KScope("generator"));
    std::vector<float> gx = xd;
    int Tcur = Tg;
    const int nk = int(cfg_.rb_kernels.size());
    for (size_t i = 0; i < P.ups.size(); ++i) {
        for (auto &z : gx) z = leaky(z, 0.1f);
        std::vector<float> xs;
        P.noise_convs[i].forward(har.data(), frames, xs);
        const int Ts = int(xs.size() / size_t(P.noise_convs[i].cout));
        P.noise_res[i].forward(xs, Ts, s_dec);
        std::vector<float> u;
        { KScope k("  gen.ups"); P.ups[i].forward(gx.data(), Tcur, u); }
        int Tu = int(u.size() / size_t(P.ups[i].cout));
        const int C = P.ups[i].cout;
        if (i + 1 == P.ups.size()) {   // ReflectionPad1d((1, 0))
            std::vector<float> pd(size_t(Tu + 1) * size_t(C));
            std::memcpy(pd.data(), &u[size_t(1) * C], size_t(C) * 4);
            std::memcpy(pd.data() + C, u.data(), u.size() * 4);
            u.swap(pd);
            ++Tu;
        }
        if (Tu != Ts) fail(VOX_ERR_FORMAT, "Kokoro: generator length mismatch (" + std::to_string(Tu) + " vs " + std::to_string(Ts) + ")");
        for (size_t z = 0; z < u.size(); ++z) u[z] += xs[z];
        std::vector<float> acc(u.size(), 0.f);
        for (int j = 0; j < nk; ++j) {
            std::vector<float> r = u;
            P.resblocks[i * size_t(nk) + size_t(j)].forward(r, Tu, s_dec);
            for (size_t z = 0; z < acc.size(); ++z) acc[z] += r[z];
        }
        for (auto &z : acc) z /= float(nk);
        gx.swap(acc);
        Tcur = Tu;
    }
    ks.reset(new KScope("post+istft"));
    for (auto &z : gx) z = leaky(z, 0.01f);
    std::vector<float> post;
    P.conv_post.forward(gx.data(), Tcur, post);   // [frames][n_fft + 2]
    if (Tcur != frames) fail(VOX_ERR_FORMAT, "Kokoro: spectrogram frames mismatch");
    if (tr) tr->post = post;
    istft_center(post, frames, NF, HOP, 0.f, pcm);
}


}  // namespace vox
