#pragma once
// Layers shared by the convolutional vocoders and encoders (Kokoro, CosyVoice2 / Chatterbox S3Gen): time-major activations [T][C],
// weights read from a .tmod (F32 / F16 / BF16), matrix products through kernels.cpp (threaded, GPU when the matrix is there).
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "formats.hpp"
#include "common.hpp"
#include "kernels.hpp"

namespace vox {
namespace nn {

constexpr double kPi = 3.14159265358979323846;

std::vector<float> to_float(const TensorView &t);
Mat mat_of(const TensorView &t);
// mat_of, plus the quantized layouts of the converters (I8 → Q8, U8 → Q4, scales in "<name>.scale")
Mat mat_named(const TmodFile &f, const std::string &name);

inline float leaky(float x, float a) { return x >= 0.f ? x : a * x; }
// sin for Snake activations: Cody–Waite reduction by π (3 parts) and an odd minimax polynomial; |error| < 2e-7 for |x| < 1e5,
// branch-free so the compiler vectorizes the loop (std::sin was a third of the generator's time)
inline float fast_sin(float x) {
    const float q = std::nearbyint(x * 0.31830988618379067f);   // x / π
    float r = x - q * 3.140625f;
    r = r - q * 9.67502593994140625e-4f;
    r = r - q * 1.509957990978376432e-7f;
    const float r2 = r * r;
    float p = -2.39e-8f;
    p = p * r2 + 2.7526e-6f;
    p = p * r2 - 1.98409e-4f;
    p = p * r2 + 8.3333315e-3f;
    p = p * r2 - 1.6666666e-1f;
    const float s = r + r * r2 * p;
    const int qi = int(q);
    return (qi & 1) ? -s : s;
}
inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// y[T][O] = x[T][I] · Wᵀ (+ b)
void linear(const Mat &W, const std::vector<float> *b, const float *x, int64_t T, float *y);

struct Linear {
    Mat w;
    std::vector<float> b;
    int64_t in = 0, out = 0;
    void load(const TmodFile &f, const std::string &n, bool bias = true);
    void forward(const float *x, int64_t T, float *y) const { linear(w, &b, x, T, y); }
};

// 1D convolution, time-major: x [T][cin] → y [Tout][cout]; weights PyTorch [cout][cin][K] re-laid out K-major [cout][K*cin]
struct Conv {
    int cin = 0, cout = 0, K = 1, stride = 1, pad = 0, dil = 1;
    std::vector<float> w, b;
    Mat m;
    void load(const TmodFile &f, const std::string &n, int stride_ = 1, int pad_ = -1, int dil_ = 1);
    int out_len(int T) const { return (T + 2 * pad - dil * (K - 1) - 1) / stride + 1; }
    void forward(const float *x, int T, std::vector<float> &y) const;
};

// ConvTranspose1d (groups = 1): x [T][cin] → y [(T-1)*stride - 2*pad + K + out_pad][cout]; PyTorch weight [cin][cout][K]
struct ConvT {
    int cin = 0, cout = 0, K = 1, stride = 1, pad = 0, out_pad = 0;
    std::vector<float> w, b;   // w: [K*cout][cin]
    Mat m;
    void load(const TmodFile &f, const std::string &n, int stride_, int pad_, int out_pad_ = 0);
    void forward(const float *x, int T, std::vector<float> &y) const;
};

// LayerNorm over channels of [T][C] rows (modulate: (1 + w) · n + b, the AdaLayerNorm form)
void layer_norm(float *x, int T, int C, const float *w, const float *b, float eps, bool modulate = false);
// Hann window, periodic (scipy.signal.get_window('hann', n, fftbins=True))
std::vector<double> hann(int n);

// Harmonic-plus-noise source of NSF vocoders (SineGen2 + Linear(H→1) + tanh, as in iSTFTNet / HiFT):
// f0 per frame [Tg] (nearest-upsampled by `up` to L = Tg·up samples), H = harmonics + 1, rand_ini [H] (initial phases,
// [0] ignored), noise [L][H] standard normal, w [H] / b the merge layer. Voiced above 10 Hz, sine amp 0.1, noise 0.003.
// Mirrors PyTorch's CPU arithmetic (linear interpolations, cumsum in double) → bit-exact on x86.
void nsf_source(const float *f0, int Tg, int up, double sr, int H, const float *rand_ini, const float *noise, const float *w, float b,
                std::vector<float> &out);
// torch.stft(center=True, reflect, periodic Hann, win = n_fft): x → [frames = L/hop + 1][2·(n_fft/2+1)],
// either |X| then angle(X) (polar) or Re then Im
void stft_center(const std::vector<float> &x, int n_fft, int hop, bool polar, std::vector<float> &out);
// torch.istft(center=True) of spec = exp(x[:nb]) (clipped at mag_clip if > 0) · e^{i·sin(x[nb:])}, x = [frames][n_fft + 2]
void istft_center(const std::vector<float> &x, int frames, int n_fft, int hop, float mag_clip, std::vector<float> &pcm);

inline Mat f32_mat(const float *data, int64_t rows, int64_t cols) {
    Mat m;
    m.type = WType::F32;
    m.data = reinterpret_cast<const uint8_t *>(data);
    m.rows = rows;
    m.cols = cols;
    return m;
}
// x · tanh(softplus(x)) with softplus's threshold 20 (torch); tanh(log(1 + e)) = n / (n + 2), n = e (e + 2)
inline float mish(float x) {
    if (x > 20.f) return x;
    const float e = std::exp(x), n = e * (e + 2.0f);
    return x * (n / (n + 2.0f));
}
// fn on every element, in parallel
template <typename F>
void map_inplace(float *x, int64_t n, F fn) {
    parallel_for(n, 4096, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; ++i) x[i] = fn(x[i]); });
}
inline float elu(float x) { return x > 0.f ? x : std::expm1(x); }

// Vectorized elementwise functions (AVX2 when the CPU has it): exp with a degree-6 polynomial (≤ 2 ulp),
// GELU through an erfc with relative error < 1.2e-7 (Numerical Recipes' Chebyshev fit), softmax of one row.
void exp_inplace(float *x, int64_t n);
void gelu_inplace(float *x, int64_t n);          // parallel
void softmax_fast(float *x, int64_t n);

// Multi-head attention over rows: q [Tq][H*dh], k / v [Tk][H*dh] (row strides qs / ks / vs, head h at column h*dh),
// out [Tq][H*dh]. scores = scale · q·kᵀ (+ add(h, scores[Tq][Tk]) when given) → softmax → ·v. Products through matmul.
void attention(const float *q, int64_t qs, const float *k, int64_t ks, const float *v, int64_t vs, int Tq, int Tk, int H, int dh,
               float scale, float *out, const std::function<void(int, float *)> &add = nullptr);

// torchaudio.transforms.Resample (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99): the polyphase kernel in
// float64 cast to float32, the convolution in float32 — what CosyVoice's load_wav does to recordings.
std::vector<float> resample_ta(const float *x, size_t n, uint32_t from, uint32_t to);

// Real DFT of frames as one matrix product: [frames][n] → [frames][nb] re + [frames][nb] im (nb = n/2 + 1), any n.
struct RealDFT {
    int n = 0, nb = 0;
    std::vector<float> table;   // [2·nb][n]: cos rows, then −sin rows
    explicit RealDFT(int n_);
    // out [frames][2·nb]: re[0..nb) | im[0..nb)
    void forward(const float *x, int64_t frames, float *out) const;
};

}  // namespace nn
}  // namespace vox
