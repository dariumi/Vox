// CosyVoice2 frontend: what CosyVoiceFrontEnd computes from a recording.
//   _extract_speech_feat:  matcha mel_spectrogram (n_fft 1920, hop 480, 80 bands, log) of the 24 kHz audio
//   _extract_spk_embedding: Kaldi fbank (80 bands, torchaudio.compliance.kaldi defaults, mean removed) of the 16 kHz audio
//                           → CAM++ (3D-Speaker; the PyTorch layout of chatterbox.models.s3gen.xvector.CAMPPlus)
//   _extract_speech_token: Whisper log-mel (128 bands) → S3TokenizerV2 (s3tokenizer.model_v2): 2 strided convolutions,
//                           transformer blocks with rotary attention and an FSMN memory, FSQ (8 digits in base 3)
#include <algorithm>
#include <cmath>
#include <cstring>

#include "cosyvoice.hpp"
#include "nn.hpp"

namespace vox {
using namespace nn;

namespace {
struct BN { std::vector<float> scale, shift; };   // eval-mode BatchNorm folded by the converter: y = x·scale + shift

BN load_bn(const TmodFile &f, const std::string &p) { return {to_float(f.get(p + ".scale")), to_float(f.get(p + ".shift"))}; }

void bn_apply(std::vector<float> &x, int C, const BN &b, bool relu) {   // x [rows][C]
    const int64_t R = int64_t(x.size()) / C;
    parallel_for(R, 256, [&](int64_t s, int64_t e) {
        for (int64_t r = s; r < e; ++r)
            for (int c = 0; c < C; ++c) {
                float &z = x[size_t(r) * C + c];
                z = z * b.scale[size_t(c)] + b.shift[size_t(c)];
                if (relu && z < 0.f) z = 0.f;
            }
    });
}

// Conv2d (k 3×3 or 1×1, padding k/2, stride (sf, 1), no bias) on [C][F][T] activations stored as [F·T][C]
struct Conv2 {
    int cin = 0, cout = 0, k = 3, sf = 1;
    std::vector<float> w, b;   // [cout][cin·k·k] as torch flattens it; bias (an exporter may fold a BatchNorm into it)
    Mat m;
    void load(const TmodFile &f, const std::string &n, int stride_f) {
        const TensorView &t = f.get(n + ".weight");
        if (t.shape.size() != 4) fail(VOX_ERR_FORMAT, "CAM++: conv2d weight must be 4-D: " + n);
        cout = int(t.shape[0]); cin = int(t.shape[1]); k = int(t.shape[2]); sf = stride_f;
        w = to_float(t);
        if (f.has(n + ".bias")) b = to_float(f.get(n + ".bias"));
        m = f32_mat(w.data(), cout, int64_t(cin) * k * k);
        m.bias = b.empty() ? nullptr : b.data();
    }
    // x [F][T][cin] → y [Fo][T][cout]
    void forward(const std::vector<float> &x, int F, int T, std::vector<float> &y, int &Fo) const {
        const int pad = k / 2;
        Fo = (F + 2 * pad - k) / sf + 1;
        const int64_t KC = int64_t(cin) * k * k, P = int64_t(Fo) * T;
        std::vector<float> col(static_cast<size_t>(P * KC));
        parallel_for(P, 64, [&](int64_t a, int64_t e) {
            for (int64_t p = a; p < e; ++p) {
                const int fo = int(p / T), t = int(p % T);
                float *dst = &col[size_t(p * KC)];
                for (int ci = 0; ci < cin; ++ci)
                    for (int kf = 0; kf < k; ++kf)
                        for (int kt = 0; kt < k; ++kt) {
                            const int fi = fo * sf + kf - pad, ti = t + kt - pad;
                            dst[(ci * k + kf) * k + kt] = (fi >= 0 && fi < F && ti >= 0 && ti < T) ? x[(size_t(fi) * T + size_t(ti)) * cin + ci] : 0.f;
                        }
            }
        });
        y.resize(size_t(P) * size_t(cout));
        matmul(m, col.data(), P, y.data());
    }
};

struct ResBlock2 {   // BasicResBlock of the FCM head
    Conv2 c1, c2, sc;
    BN b1, b2, bsc;
    bool has_sc = false;
    void load(const TmodFile &f, const std::string &p, int stride) {
        c1.load(f, p + "conv1", stride); b1 = load_bn(f, p + "bn1");
        c2.load(f, p + "conv2", 1); b2 = load_bn(f, p + "bn2");
        has_sc = f.has(p + "shortcut.0.weight");
        if (has_sc) { sc.load(f, p + "shortcut.0", stride); bsc = load_bn(f, p + "shortcut.1"); }
    }
    void forward(std::vector<float> &x, int &F, int T) const {
        std::vector<float> a, b, s;
        int F1, F2;
        c1.forward(x, F, T, a, F1);
        bn_apply(a, c1.cout, b1, true);
        c2.forward(a, F1, T, b, F2);
        bn_apply(b, c2.cout, b2, false);
        if (has_sc) { int Fs; sc.forward(x, F, T, s, Fs); bn_apply(s, sc.cout, bsc, false); }
        else s = x;
        for (size_t i = 0; i < b.size(); ++i) b[i] = std::max(0.f, b[i] + s[i]);
        x.swap(b);
        F = F2;
    }
};

struct CamDense {   // CAMDenseTDNNLayer
    BN n1, n2;
    Conv lin1, local, l1, l2;
    void load(const TmodFile &f, const std::string &p, int dil) {
        n1 = load_bn(f, p + "nonlinear1.batchnorm");
        lin1.load(f, p + "linear1", 1, 0);
        n2 = load_bn(f, p + "nonlinear2.batchnorm");
        local.load(f, p + "cam_layer.linear_local", 1, dil, dil);
        l1.load(f, p + "cam_layer.linear1", 1, 0);
        l2.load(f, p + "cam_layer.linear2", 1, 0);
    }
    bool pool_window = false;
    std::vector<float> forward(const std::vector<float> &x, int T) const {   // x [T][cin] → [T][32]
        std::vector<float> a = x, b, y;
        bn_apply(a, lin1.cin, n1, true);
        lin1.forward(a.data(), T, b);
        bn_apply(b, lin1.cout, n2, true);
        local.forward(b.data(), T, y);
        // context: mean over time + mean over segments of 100 frames (avg_pool1d, ceil_mode), per frame
        const int C = lin1.cout;
        std::vector<double> mean(static_cast<size_t>(C), 0.0);
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < C; ++c) mean[size_t(c)] += b[size_t(t) * C + c];
        std::vector<float> mf(static_cast<size_t>(C));
        for (int c = 0; c < C; ++c) mf[size_t(c)] = float(mean[size_t(c)] / T);
        const int nseg = (T + 99) / 100;
        std::vector<float> seg(size_t(nseg) * C, 0.f);
        for (int s = 0; s < nseg; ++s) {
            const int t0 = s * 100, t1 = std::min(T, t0 + 100);
            for (int c = 0; c < C; ++c) {
                double acc = 0;
                for (int t = t0; t < t1; ++t) acc += b[size_t(t) * C + c];
                seg[size_t(s) * C + c] = float(acc / (pool_window ? 100 : t1 - t0));
            }
        }
        std::vector<float> ctx(size_t(T) * C);
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < C; ++c) ctx[size_t(t) * C + c] = mf[size_t(c)] + seg[size_t(t / 100) * C + c];
        // the context goes through the 1×1 layers only once per segment (it is constant inside one)
        std::vector<float> h1, m;
        l1.forward(ctx.data(), T, h1);
        for (auto &z : h1) z = std::max(0.f, z);
        l2.forward(h1.data(), T, m);
        for (size_t i = 0; i < y.size(); ++i) y[i] *= 1.0f / (1.0f + std::exp(-m[i]));
        return y;
    }
};

struct S3Block {
    std::vector<float> ln1w, ln1b, ln2w, ln2b, fsmn;   // fsmn [D][31] depthwise
    Linear q, k, v, o, m1, m2;
};
}  // namespace

struct CosyVoiceModel::Front {
    std::vector<float> mel_basis;      // [80][n_fft/2 + 1] (librosa slaney)
    std::vector<float> fb_banks;       // [80][257] (Kaldi, padded window 512)
    std::vector<float> wfilters;       // [128][201] (Whisper)
    std::unique_ptr<RealDFT> dft24, dft512, dft400;
    // CAM++
    Conv2 head_c1, head_c2;
    BN head_b1, head_b2;
    std::vector<ResBlock2> head_l1, head_l2;
    Conv tdnn;
    BN tdnn_bn;
    std::vector<std::vector<CamDense>> blocks;
    std::vector<BN> transit_bn;
    std::vector<Conv> transit;
    BN out_bn, dense_bn;
    Conv dense;
    // S3 tokenizer
    Conv s3c1, s3c2;
    std::vector<S3Block> s3b;
    Linear fsq;
};

void CosyVoiceModel::FrontFree::operator()(Front *f) const { delete f; }

void CosyVoiceModel::load_front() {
    fp_.reset(new Front);
    Front &F = *fp_;
    const TmodFile &f = *file_;
    if (f.has("front.mel_basis")) F.mel_basis = to_float(f.get("front.mel_basis"));
    if (f.has("front.fbank_banks")) F.fb_banks = to_float(f.get("front.fbank_banks"));
    if (f.has("front.whisper_filters")) F.wfilters = to_float(f.get("front.whisper_filters"));
    if (cfg_.has_campplus) {
        const std::string p = "camp.";
        F.head_c1.load(f, p + "head.conv1", 1); F.head_b1 = load_bn(f, p + "head.bn1");
        for (int i = 0; i < 2; ++i) {
            ResBlock2 r; r.load(f, p + "head.layer1." + std::to_string(i) + ".", i == 0 ? 2 : 1); F.head_l1.push_back(std::move(r));
        }
        for (int i = 0; i < 2; ++i) {
            ResBlock2 r; r.load(f, p + "head.layer2." + std::to_string(i) + ".", i == 0 ? 2 : 1); F.head_l2.push_back(std::move(r));
        }
        F.head_c2.load(f, p + "head.conv2", 2); F.head_b2 = load_bn(f, p + "head.bn2");
        F.tdnn.load(f, p + "xvector.tdnn.linear", 2, 2);
        F.tdnn_bn = load_bn(f, p + "xvector.tdnn.nonlinear.batchnorm");
        const int layers[3] = {12, 24, 16}, dils[3] = {1, 2, 2};
        for (int b = 0; b < 3; ++b) {
            std::vector<CamDense> blk;
            for (int i = 0; i < layers[b]; ++i) {
                CamDense d;
                d.load(f, p + "xvector.block" + std::to_string(b + 1) + ".tdnnd" + std::to_string(i + 1) + ".", dils[b]);
                d.pool_window = cfg_.campplus_pool_window;
                blk.push_back(std::move(d));
            }
            F.blocks.push_back(std::move(blk));
            F.transit_bn.push_back(load_bn(f, p + "xvector.transit" + std::to_string(b + 1) + ".nonlinear.batchnorm"));
            Conv t;
            t.load(f, p + "xvector.transit" + std::to_string(b + 1) + ".linear", 1, 0);
            F.transit.push_back(std::move(t));
        }
        F.out_bn = load_bn(f, p + "xvector.out_nonlinear.batchnorm");
        F.dense.load(f, p + "xvector.dense.linear", 1, 0);
        F.dense_bn = load_bn(f, p + "xvector.dense.nonlinear.batchnorm");
    }
    if (cfg_.has_s3) {
        const std::string p = "s3.encoder.";
        F.s3c1.load(f, p + "conv1", 2, 1);
        F.s3c2.load(f, p + "conv2", 2, 1);
        for (int i = 0; i < cfg_.s3_layers; ++i) {
            const std::string b = p + "blocks." + std::to_string(i) + ".";
            S3Block B;
            B.ln1w = to_float(f.get(b + "attn_ln.weight")); B.ln1b = to_float(f.get(b + "attn_ln.bias"));
            B.ln2w = to_float(f.get(b + "mlp_ln.weight")); B.ln2b = to_float(f.get(b + "mlp_ln.bias"));
            B.q.load(f, b + "attn.query"); B.k.load(f, b + "attn.key", false); B.v.load(f, b + "attn.value");
            B.o.load(f, b + "attn.out");
            B.fsmn = to_float(f.get(b + "attn.fsmn_block.weight"));
            B.m1.load(f, b + "mlp.0"); B.m2.load(f, b + "mlp.2");
            F.s3b.push_back(std::move(B));
        }
        F.fsq.load(f, "s3.quantizer._codebook.project_down");
    }
}

// matcha.utils.audio.mel_spectrogram: reflect pad (n_fft - hop) / 2, |STFT| (+1e-9 inside the root), mel, log(clamp 1e-5)
std::vector<float> CosyVoiceModel::mel24(const float *y, size_t n, int &frames) const {
    Front &F = *fp_;
    const int nfft = cfg_.mel_n_fft, hop = cfg_.mel_hop, nb = nfft / 2 + 1, M = cfg_.mel_dim;
    if (F.mel_basis.size() != size_t(M) * size_t(nb)) fail(VOX_ERR_FORMAT, "CosyVoice: this .tmod has no mel filterbank (front.mel_basis)");
    if (!F.dft24) F.dft24 = std::make_unique<RealDFT>(nfft);
    const int64_t pad = (nfft - hop) / 2, N = int64_t(n);
    if (N <= pad) fail(VOX_ERR_ARG, "the recording is too short");
    std::vector<float> yp(static_cast<size_t>(N + 2 * pad));
    for (int64_t i = 0; i < N + 2 * pad; ++i) {
        int64_t j = i - pad;
        if (j < 0) j = -j;
        if (j >= N) j = 2 * (N - 1) - j;
        yp[size_t(i)] = y[j];
    }
    frames = int((int64_t(yp.size()) - nfft) / hop + 1);
    const auto win = hann(nfft);
    std::vector<float> fr(size_t(frames) * size_t(nfft)), spec(size_t(frames) * size_t(2 * nb));
    for (int t = 0; t < frames; ++t)
        for (int i = 0; i < nfft; ++i) fr[size_t(t) * nfft + i] = float(double(yp[size_t(t) * hop + i]) * win[size_t(i)]);
    F.dft24->forward(fr.data(), frames, spec.data());
    std::vector<float> mag(size_t(frames) * size_t(nb)), out(size_t(frames) * size_t(M));
    for (int t = 0; t < frames; ++t)
        for (int k = 0; k < nb; ++k) {
            const float re = spec[size_t(t) * (2 * nb) + k], im = spec[size_t(t) * (2 * nb) + nb + k];
            mag[size_t(t) * nb + k] = std::sqrt(re * re + im * im + 1e-9f);
        }
    matmul(f32_mat(F.mel_basis.data(), M, nb), mag.data(), frames, out.data());
    for (auto &z : out) z = std::log(std::max(z, 1e-5f));
    return out;
}

// torchaudio.compliance.kaldi.fbank(num_mel_bins=80, dither=0): frames of 25 ms every 10 ms (snip edges), DC removed,
// pre-emphasis 0.97, Povey window, |FFT|² on 512 points, mel banks, log(max(·, eps)); then the mean over time is removed.
std::vector<float> CosyVoiceModel::fbank16(const float *x, size_t n, int &frames) const {
    Front &F = *fp_;
    const int W = 400, S = 160, NP = 512, NB = NP / 2 + 1, M = 80;
    if (F.fb_banks.size() != size_t(M) * size_t(NB)) fail(VOX_ERR_FORMAT, "CosyVoice: this .tmod has no fbank banks (front.fbank_banks)");
    if (n < size_t(W)) fail(VOX_ERR_ARG, "the recording is too short");
    if (!F.dft512) F.dft512 = std::make_unique<RealDFT>(NP);
    frames = int(1 + (n - W) / S);
    std::vector<double> win(W);
    for (int i = 0; i < W; ++i) win[size_t(i)] = std::pow(0.5 - 0.5 * std::cos(2.0 * kPi * i / (W - 1)), 0.85);
    std::vector<float> fr(size_t(frames) * NP, 0.f), spec(size_t(frames) * size_t(2 * NB));
    parallel_for(frames, 32, [&](int64_t a, int64_t e) {
        std::vector<float> s(W);
        for (int64_t t = a; t < e; ++t) {
            const float *src = x + t * S;
            float mean = 0.f;
            {
                double acc = 0;
                for (int i = 0; i < W; ++i) acc += src[i];
                mean = float(acc / W);
            }
            for (int i = 0; i < W; ++i) s[size_t(i)] = src[i] - mean;
            float *dst = &fr[size_t(t) * NP];
            for (int i = W - 1; i >= 0; --i) {
                const float prev = s[size_t(i > 0 ? i - 1 : 0)];
                dst[i] = float(double(s[size_t(i)] - 0.97f * prev) * win[size_t(i)]);
            }
        }
    });
    F.dft512->forward(fr.data(), frames, spec.data());
    std::vector<float> pw(size_t(frames) * NB), out(size_t(frames) * M);
    for (int t = 0; t < frames; ++t)
        for (int k = 0; k < NB; ++k) {
            const float re = spec[size_t(t) * (2 * NB) + k], im = spec[size_t(t) * (2 * NB) + NB + k];
            pw[size_t(t) * NB + k] = re * re + im * im;
        }
    matmul(f32_mat(F.fb_banks.data(), M, NB), pw.data(), frames, out.data());
    const float eps = 1.1920928955078125e-07f;
    for (auto &z : out) z = std::log(std::max(z, eps));
    for (int m = 0; m < M; ++m) {
        double acc = 0;
        for (int t = 0; t < frames; ++t) acc += out[size_t(t) * M + m];
        const float mu = float(acc / frames);
        for (int t = 0; t < frames; ++t) out[size_t(t) * M + m] -= mu;
    }
    return out;
}

std::vector<float> CosyVoiceModel::xvector(const std::vector<float> &fb, int T) const {
    if (!cfg_.has_campplus) fail(VOX_ERR_UNSUPPORTED, "CosyVoice: this .tmod has no speaker encoder (CAM++): convert it with campplus.onnx");
    const Front &F = *fp_;
    const int FD = int(fb.size() / size_t(T));
    // FCM head on [1][F][T]: activations [F·T][C]
    std::vector<float> x(size_t(FD) * T);
    for (int t = 0; t < T; ++t)
        for (int fq = 0; fq < FD; ++fq) x[size_t(fq) * T + t] = fb[size_t(t) * FD + fq];
    std::vector<float> y;
    int Fc = FD, Fo;
    F.head_c1.forward(x, Fc, T, y, Fo);
    bn_apply(y, F.head_c1.cout, F.head_b1, true);
    Fc = Fo;
    for (const auto &r : F.head_l1) r.forward(y, Fc, T);
    for (const auto &r : F.head_l2) r.forward(y, Fc, T);
    F.head_c2.forward(y, Fc, T, x, Fo);
    bn_apply(x, F.head_c2.cout, F.head_b2, true);
    Fc = Fo;
    const int C0 = F.head_c2.cout, CH = C0 * Fc;
    std::vector<float> h(size_t(T) * CH);   // reshape (B, C, F, T) → (B, C·F, T)
    for (int fq = 0; fq < Fc; ++fq)
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < C0; ++c) h[size_t(t) * CH + size_t(c) * Fc + fq] = x[(size_t(fq) * T + t) * C0 + c];
    // x-vector trunk
    std::vector<float> a;
    F.tdnn.forward(h.data(), T, a);
    int Tt = int(a.size() / size_t(F.tdnn.cout));
    bn_apply(a, F.tdnn.cout, F.tdnn_bn, true);
    int C = F.tdnn.cout;
    for (size_t b = 0; b < F.blocks.size(); ++b) {
        for (const auto &layer : F.blocks[b]) {
            const std::vector<float> o = layer.forward(a, Tt);
            const int G = int(o.size() / size_t(Tt));
            std::vector<float> cat(size_t(Tt) * size_t(C + G));
            for (int t = 0; t < Tt; ++t) {
                std::memcpy(&cat[size_t(t) * (C + G)], &a[size_t(t) * C], size_t(C) * 4);
                std::memcpy(&cat[size_t(t) * (C + G) + C], &o[size_t(t) * G], size_t(G) * 4);
            }
            a.swap(cat);
            C += G;
        }
        bn_apply(a, C, F.transit_bn[b], true);
        std::vector<float> t2;
        F.transit[b].forward(a.data(), Tt, t2);
        a.swap(t2);
        C = F.transit[b].cout;
    }
    bn_apply(a, C, F.out_bn, true);
    // statistics pooling: mean and unbiased std over time
    std::vector<float> st(static_cast<size_t>(2 * C));
    for (int c = 0; c < C; ++c) {
        double s = 0;
        for (int t = 0; t < Tt; ++t) s += a[size_t(t) * C + c];
        const double mu = s / Tt;
        double v = 0;
        for (int t = 0; t < Tt; ++t) { const double d = a[size_t(t) * C + c] - mu; v += d * d; }
        st[size_t(c)] = float(mu);
        st[size_t(C + c)] = float(std::sqrt(v / std::max(1, Tt - 1)));
    }
    std::vector<float> e;
    F.dense.forward(st.data(), 1, e);
    bn_apply(e, F.dense.cout, F.dense_bn, false);
    return e;
}

// Whisper log-mel: STFT n_fft 400, hop 160 (centered, reflect), the last frame dropped, |·|², mel, log10(clamp 1e-10),
// floor at max − 8, (x + 4) / 4. Output [128][frames].
std::vector<float> CosyVoiceModel::whisper_mel(const float *x, size_t n, int &frames) const {
    Front &F = *fp_;
    const int NF = 400, HOP = 160, NB = NF / 2 + 1, M = cfg_.s3_mels;
    if (F.wfilters.size() != size_t(M) * size_t(NB)) fail(VOX_ERR_FORMAT, "CosyVoice: this .tmod has no Whisper mel filters (front.whisper_filters)");
    if (!F.dft400) F.dft400 = std::make_unique<RealDFT>(NF);
    const int64_t N = int64_t(n), half = NF / 2;
    if (N <= half) fail(VOX_ERR_ARG, "the recording is too short");
    frames = int(N / HOP);   // N / hop + 1 STFT frames, the last one dropped
    const auto win = hann(NF);
    std::vector<float> fr(size_t(frames) * NF), spec(size_t(frames) * size_t(2 * NB));
    for (int t = 0; t < frames; ++t)
        for (int i = 0; i < NF; ++i) {
            int64_t j = int64_t(t) * HOP + i - half;
            if (j < 0) j = -j;
            if (j >= N) j = 2 * (N - 1) - j;
            fr[size_t(t) * NF + i] = float(double(x[j]) * win[size_t(i)]);
        }
    F.dft400->forward(fr.data(), frames, spec.data());
    std::vector<float> pw(size_t(frames) * NB), mel(size_t(frames) * M);
    for (int t = 0; t < frames; ++t)
        for (int k = 0; k < NB; ++k) {
            const float re = spec[size_t(t) * (2 * NB) + k], im = spec[size_t(t) * (2 * NB) + NB + k];
            pw[size_t(t) * NB + k] = re * re + im * im;
        }
    matmul(f32_mat(F.wfilters.data(), M, NB), pw.data(), frames, mel.data());
    float mx = -INFINITY;
    for (auto &z : mel) { z = std::log10(std::max(z, 1e-10f)); mx = std::max(mx, z); }
    std::vector<float> out(size_t(M) * size_t(frames));
    for (int t = 0; t < frames; ++t)
        for (int m = 0; m < M; ++m) out[size_t(m) * frames + t] = (std::max(mel[size_t(t) * M + m], mx - 8.0f) + 4.0f) / 4.0f;
    return out;
}

std::vector<int32_t> CosyVoiceModel::speech_tokens(const std::vector<float> &wmel, int T) const {
    if (!cfg_.has_s3) fail(VOX_ERR_UNSUPPORTED, "CosyVoice: this .tmod has no speech tokenizer: convert it with speech_tokenizer_v2.onnx");
    const Front &F = *fp_;
    const int M = cfg_.s3_mels, D = cfg_.s3_dim, H = cfg_.s3_heads, dh = D / H;
    std::vector<float> x(size_t(T) * M);
    for (int t = 0; t < T; ++t)
        for (int m = 0; m < M; ++m) x[size_t(t) * M + m] = wmel[size_t(m) * T + t];
    std::vector<float> a, b;
    F.s3c1.forward(x.data(), T, a);
    for (auto &z : a) z = gelu_erf(z);
    const int T1 = int(a.size() / size_t(D));
    F.s3c2.forward(a.data(), T1, b);
    for (auto &z : b) z = gelu_erf(z);
    const int T2 = int(b.size() / size_t(D));
    // rotary tables (precompute_freqs_cis(64, ·) duplicated over both halves)
    const int rd = 64;
    if (dh != rd) fail(VOX_ERR_FORMAT, "S3 tokenizer: head size must be 64");
    std::vector<float> cs(size_t(T2) * rd), sn(size_t(T2) * rd);
    for (int t = 0; t < T2; ++t)
        for (int i = 0; i < rd / 2; ++i) {
            const float fr = 1.0f / std::pow(10000.0f, float(2 * i) / float(rd));
            const float ang = float(t) * fr;
            cs[size_t(t) * rd + i] = cs[size_t(t) * rd + i + rd / 2] = std::cos(ang);
            sn[size_t(t) * rd + i] = sn[size_t(t) * rd + i + rd / 2] = std::sin(ang);
        }
    const float scale = std::pow(float(dh), -0.25f);
    std::vector<float> xn, Q(size_t(T2) * D), K(Q.size()), V(Q.size()), A(Q.size()), y(Q.size());
    for (const auto &B : F.s3b) {
        xn = b;
        layer_norm(xn.data(), T2, D, B.ln1w.data(), B.ln1b.data(), 1e-5f);
        B.q.forward(xn.data(), T2, Q.data()); B.k.forward(xn.data(), T2, K.data()); B.v.forward(xn.data(), T2, V.data());
        auto rope = [&](std::vector<float> &z) {
            for (int t = 0; t < T2; ++t)
                for (int h = 0; h < H; ++h) {
                    float *r = &z[size_t(t) * D + size_t(h) * dh];
                    float tmp[64];
                    for (int i = 0; i < dh; ++i) tmp[i] = i < dh / 2 ? -r[i + dh / 2] : r[i - dh / 2];
                    for (int i = 0; i < dh; ++i) r[i] = r[i] * cs[size_t(t) * rd + i] + tmp[i] * sn[size_t(t) * rd + i];
                }
        };
        rope(Q);
        rope(K);
        // FSMN memory: depthwise conv (k 31, pad 15 / 15) over v, + v
        std::vector<float> mem(V.size());
        const int KS = int(B.fsmn.size() / size_t(D)), lp = (KS - 1) / 2;
        parallel_for(T2, 32, [&](int64_t s, int64_t e) {
            for (int64_t t = s; t < e; ++t)
                for (int c = 0; c < D; ++c) {
                    float acc = 0.f;
                    for (int kk = 0; kk < KS; ++kk) {
                        const int64_t ti = t + kk - lp;
                        if (ti >= 0 && ti < T2) acc += B.fsmn[size_t(c) * KS + kk] * V[size_t(ti) * D + c];
                    }
                    mem[size_t(t) * D + c] = acc + V[size_t(t) * D + c];
                }
        });
        for (auto &z : Q) z *= scale;
        for (auto &z : K) z *= scale;
        attention(Q.data(), D, K.data(), D, V.data(), D, T2, T2, H, dh, 1.0f, A.data());
        B.o.forward(A.data(), T2, y.data());
        for (size_t i = 0; i < b.size(); ++i) b[i] += y[i] + mem[i];
        xn = b;
        layer_norm(xn.data(), T2, D, B.ln2w.data(), B.ln2b.data(), 1e-5f);
        std::vector<float> hd(size_t(T2) * size_t(B.m1.out));
        B.m1.forward(xn.data(), T2, hd.data());
        for (auto &z : hd) z = gelu_erf(z);
        B.m2.forward(hd.data(), T2, y.data());
        for (size_t i = 0; i < b.size(); ++i) b[i] += y[i];
    }
    // FSQ: project to 8 → tanh → ×0.999 → round (half to even) + 1 → base-3 number
    std::vector<float> pd(size_t(T2) * 8);
    F.fsq.forward(b.data(), T2, pd.data());
    std::vector<int32_t> tok(static_cast<size_t>(T2));
    for (int t = 0; t < T2; ++t) {
        double mu = 0, pw = 1;
        for (int i = 0; i < 8; ++i) {
            const float h = std::tanh(pd[size_t(t) * 8 + i]) * 0.9990000128746033f;
            mu += double(std::nearbyint(h) + 1.0f) * pw;
            pw *= 3;
        }
        tok[size_t(t)] = int32_t(mu);
    }
    return tok;
}

}  // namespace vox
