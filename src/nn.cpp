#include "nn.hpp"

#include <algorithm>
#include <cstring>

namespace vox {
namespace nn {

#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && defined(__linux__)
#  define NN_MULTIVERSION __attribute__((target_clones("avx2,fma", "default")))
#else
#  define NN_MULTIVERSION
#endif

namespace {
// exp without libm calls, so that loops over it vectorize: x = n·ln2 + r, |r| ≤ ln2/2, e^r by a polynomial, 2^n in the exponent bits
inline float exp_poly(float x) {
    x = std::min(std::max(x, -87.3f), 88.7f);
    const float fn = std::floor(x * 1.44269504088896341f + 0.5f);
    const float r = (x - fn * 0.693359375f) - fn * -2.12194440e-4f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.0f;
    const int32_t e = (int32_t(fn) + 127) << 23;
    return p * __builtin_bit_cast(float, e);
}
inline float gelu_poly(float x) {
    const float z = std::fabs(x) * 0.70710678118654752f;
    const float t = 1.0f / (1.0f + 0.5f * z);
    const float y = t * exp_poly(-z * z - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f +
                    t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f)))))))));   // erfc(z)
    const float erf_abs = 1.0f - y;
    return 0.5f * x * (1.0f + (x < 0.f ? -erf_abs : erf_abs));
}
}  // namespace

#if defined(__GNUC__)
typedef float v8f_ __attribute__((vector_size(32), aligned(4)));
typedef int32_t v8i_ __attribute__((vector_size(32), aligned(4)));
static inline v8f_ bc8(float f) { return v8f_{f, f, f, f, f, f, f, f}; }
// exp_poly on 8 lanes (GCC vector extensions: AVX2 in the avx2 clone)
static inline v8f_ exp8(v8f_ x) {
    x = x < bc8(-87.3f) ? bc8(-87.3f) : x;
    x = x > bc8(88.7f) ? bc8(88.7f) : x;
    const v8f_ y = x * bc8(1.44269504088896341f) + bc8(0.5f);
    v8i_ ni = __builtin_convertvector(y, v8i_);   // toward zero
    v8f_ fn = __builtin_convertvector(ni, v8f_);
    const v8i_ adj = fn > y;                      // floor for negative values
    ni = ni + adj;                                // adj is -1 where true
    fn = __builtin_convertvector(ni, v8f_);
    const v8f_ r = (x - fn * bc8(0.693359375f)) - fn * bc8(-2.12194440e-4f);
    v8f_ p = bc8(1.9875691500e-4f);
    p = p * r + bc8(1.3981999507e-3f);
    p = p * r + bc8(8.3334519073e-3f);
    p = p * r + bc8(4.1665795894e-2f);
    p = p * r + bc8(1.6666665459e-1f);
    p = p * r + bc8(5.0000001201e-1f);
    p = p * r * r + r + bc8(1.0f);
    const v8i_ e = (ni + 127) << 23;
    v8f_ sc;
    __builtin_memcpy(&sc, &e, 32);
    return p * sc;
}
static inline v8f_ gelu8(v8f_ x) {
    const v8f_ ax = x < bc8(0.f) ? -x : x;
    const v8f_ z = ax * bc8(0.70710678118654752f);
    const v8f_ t = bc8(1.0f) / (bc8(1.0f) + bc8(0.5f) * z);
    v8f_ q = bc8(0.17087277f);
    for (float c : {-0.82215223f, 1.48851587f, -1.13520398f, 0.27886807f, -0.18628806f, 0.09678418f, 0.37409196f, 1.00002368f}) q = q * t + bc8(c);
    const v8f_ y = t * exp8(-z * z - bc8(1.26551223f) + t * q);
    const v8f_ ea = bc8(1.0f) - y;
    return bc8(0.5f) * x * (bc8(1.0f) + (x < bc8(0.f) ? -ea : ea));
}
#endif

NN_MULTIVERSION void exp_inplace(float *x, int64_t n) {
    int64_t i = 0;
#if defined(__GNUC__)
    for (; i + 8 <= n; i += 8) { v8f_ v; __builtin_memcpy(&v, x + i, 32); v = exp8(v); __builtin_memcpy(x + i, &v, 32); }
#endif
    for (; i < n; ++i) x[i] = exp_poly(x[i]);
}
NN_MULTIVERSION static void gelu_range(float *x, int64_t n) {
    int64_t i = 0;
#if defined(__GNUC__)
    for (; i + 8 <= n; i += 8) { v8f_ v; __builtin_memcpy(&v, x + i, 32); v = gelu8(v); __builtin_memcpy(x + i, &v, 32); }
#endif
    for (; i < n; ++i) x[i] = gelu_poly(x[i]);
}
void gelu_inplace(float *x, int64_t n) {
    parallel_for(n, 8192, [&](int64_t a, int64_t b) { gelu_range(x + a, b - a); });
}
NN_MULTIVERSION void softmax_fast(float *x, int64_t n) {
    float mx = -INFINITY;
    for (int64_t i = 0; i < n; ++i) mx = std::max(mx, x[i]);
    for (int64_t i = 0; i < n; ++i) x[i] -= mx;
    exp_inplace(x, n);
    float sum = 0.f;
    for (int64_t i = 0; i < n; ++i) sum += x[i];
    const float inv = 1.0f / sum;
    for (int64_t i = 0; i < n; ++i) x[i] *= inv;
}

std::vector<float> to_float(const TensorView &t) {
    std::vector<float> v(size_t(t.numel()));
    switch (t.dtype) {
        case DType::F32: std::memcpy(v.data(), t.data, v.size() * 4); break;
        case DType::F16: { auto p = reinterpret_cast<const uint16_t *>(t.data); for (size_t i = 0; i < v.size(); ++i) v[i] = f16_to_f32(p[i]); break; }
        case DType::BF16: { auto p = reinterpret_cast<const uint16_t *>(t.data); for (size_t i = 0; i < v.size(); ++i) v[i] = bf16_to_f32(p[i]); break; }
        default: fail(VOX_ERR_FORMAT, "nn: unsupported tensor dtype");
    }
    return v;
}
Mat mat_of(const TensorView &t) {
    Mat m;
    m.data = t.data;
    m.rows = t.shape.at(0);
    m.cols = t.numel() / m.rows;
    switch (t.dtype) {
        case DType::F32: m.type = WType::F32; break;
        case DType::F16: m.type = WType::F16; break;
        case DType::BF16: m.type = WType::BF16; break;
        default: fail(VOX_ERR_FORMAT, "nn: unsupported matrix dtype");
    }
    return m;
}

Mat mat_named(const TmodFile &f, const std::string &name) {
    const TensorView &t = f.get(name);
    if (t.dtype != DType::I8 && t.dtype != DType::U8) return mat_of(t);
    Mat m;
    m.data = t.data;
    m.rows = t.shape.at(0);
    m.cols = t.shape.size() > 1 ? t.shape[1] : 1;
    m.type = t.dtype == DType::I8 ? WType::Q8 : WType::Q4;
    if (m.type == WType::Q4) m.cols *= 2;
    const TensorView &sc = f.get(name + ".scale");
    if (sc.dtype != DType::F32) fail(VOX_ERR_FORMAT, "quantization scales must be F32: " + name);
    m.scale = reinterpret_cast<const float *>(sc.data);
    return m;
}

// y[T][O] = x[T][I] · Wᵀ (+ b)
void linear(const Mat &W, const std::vector<float> *b, const float *x, int64_t T, float *y) {
    matmul(W, x, T, y);
    if (b && !b->empty())
        parallel_for(T, 64, [&](int64_t a, int64_t e) {
            for (int64_t t = a; t < e; ++t)
                for (int64_t o = 0; o < W.rows; ++o) y[t * W.rows + o] += (*b)[size_t(o)];
        });
}


void Linear::load(const TmodFile &f, const std::string &n, bool bias) {
    w = mat_of(f.get(n + ".weight"));
    out = w.rows; in = w.cols;
    if (bias && f.has(n + ".bias")) b = to_float(f.get(n + ".bias"));
}

void Conv::load(const TmodFile &f, const std::string &n, int stride_, int pad_, int dil_) {
    const TensorView &t = f.get(n + ".weight");
    if (t.shape.size() != 3) fail(VOX_ERR_FORMAT, "conv weight must be 3-D: " + n);
    cout = int(t.shape[0]); cin = int(t.shape[1]); K = int(t.shape[2]);
    stride = stride_; dil = dil_;
    pad = pad_ < 0 ? (K - 1) * dil / 2 : pad_;
    const std::vector<float> src = to_float(t);
    w.resize(src.size());
    for (int o = 0; o < cout; ++o)
        for (int c = 0; c < cin; ++c)
            for (int k = 0; k < K; ++k) w[(size_t(o) * K + k) * cin + c] = src[(size_t(o) * cin + c) * K + k];
    if (f.has(n + ".bias")) b = to_float(f.get(n + ".bias"));
    m.type = WType::F32; m.data = reinterpret_cast<const uint8_t *>(w.data()); m.rows = cout; m.cols = int64_t(K) * cin;
}

void Conv::forward(const float *x, int T, std::vector<float> &y) const {
    const int To = out_len(T);
    y.assign(size_t(std::max(To, 0)) * size_t(cout), 0.f);
    if (To <= 0) return;
    if (K == 1 && stride == 1 && pad == 0) { linear(m, &b, x, T, y.data()); return; }
    const int64_t KC = int64_t(K) * cin;
    const int64_t blk = std::max<int64_t>(64, (int64_t(4) << 20) / std::max<int64_t>(KC, 1));  // ≤ 16 MB of columns per block
    std::vector<float> col;
    for (int64_t t0 = 0; t0 < To; t0 += blk) {
        const int64_t n = std::min<int64_t>(blk, To - t0);
        col.resize(size_t(n * KC));
        parallel_for(n, 32, [&](int64_t a, int64_t e) {
            for (int64_t i = a; i < e; ++i) {
                const int64_t t = t0 + i;
                for (int k = 0; k < K; ++k) {
                    const int64_t s = t * stride + int64_t(k) * dil - pad;
                    float *dst = &col[size_t(i * KC + int64_t(k) * cin)];
                    if (s >= 0 && s < T) std::memcpy(dst, x + s * cin, size_t(cin) * 4);
                    else std::memset(dst, 0, size_t(cin) * 4);
                }
            }
        });
        linear(m, &b, col.data(), n, y.data() + size_t(t0) * size_t(cout));
    }
}

void ConvT::load(const TmodFile &f, const std::string &n, int stride_, int pad_, int out_pad_) {
    const TensorView &t = f.get(n + ".weight");
    cin = int(t.shape[0]); cout = int(t.shape[1]); K = int(t.shape[2]);
    stride = stride_; pad = pad_; out_pad = out_pad_;
    const std::vector<float> src = to_float(t);
    w.resize(src.size());
    for (int c = 0; c < cin; ++c)
        for (int o = 0; o < cout; ++o)
            for (int k = 0; k < K; ++k) w[(size_t(k) * cout + o) * cin + c] = src[(size_t(c) * cout + o) * K + k];
    if (f.has(n + ".bias")) b = to_float(f.get(n + ".bias"));
    m.type = WType::F32; m.data = reinterpret_cast<const uint8_t *>(w.data()); m.rows = int64_t(K) * cout; m.cols = cin;
}

void ConvT::forward(const float *x, int T, std::vector<float> &y) const {
    const int To = (T - 1) * stride - 2 * pad + K + out_pad;
    y.assign(size_t(To) * size_t(cout), 0.f);
    const int64_t KO = int64_t(K) * cout;
    const int64_t blk = std::max<int64_t>(16, (int64_t(4) << 20) / KO);
    std::vector<float> z;
    for (int64_t t0 = 0; t0 < T; t0 += blk) {
        const int64_t n = std::min<int64_t>(blk, T - t0);
        z.resize(size_t(n * KO));
        matmul(m, x + t0 * cin, n, z.data());
        // overlap-add; parallel over output channels (no write conflicts)
        parallel_for(cout, 8, [&](int64_t o0, int64_t o1) {
            for (int64_t i = 0; i < n; ++i) {
                const int64_t t = t0 + i;
                for (int k = 0; k < K; ++k) {
                    const int64_t to = t * stride + k - pad;
                    if (to < 0 || to >= To) continue;
                    const float *zr = &z[size_t(i * KO + int64_t(k) * cout)];
                    float *yr = &y[size_t(to) * size_t(cout)];
                    for (int64_t o = o0; o < o1; ++o) yr[o] += zr[o];
                }
            }
        });
    }
    if (!b.empty())
        for (int t = 0; t < To; ++t)
            for (int o = 0; o < cout; ++o) y[size_t(t) * size_t(cout) + size_t(o)] += b[size_t(o)];
}

// LayerNorm over channels of [T][C] rows
void layer_norm(float *x, int T, int C, const float *w, const float *b, float eps, bool modulate) {
    parallel_for(T, 16, [&](int64_t a, int64_t e) {
        for (int64_t t = a; t < e; ++t) {
            float *r = x + t * C;
            double mean = 0;
            for (int c = 0; c < C; ++c) mean += r[c];
            mean /= C;
            double var = 0;
            for (int c = 0; c < C; ++c) { const double d = r[c] - mean; var += d * d; }
            var /= C;
            const float inv = float(1.0 / std::sqrt(var + eps));
            for (int c = 0; c < C; ++c) {
                const float n = (r[c] - float(mean)) * inv;
                r[c] = w ? (modulate ? (1.0f + w[c]) * n + b[c] : n * w[c] + (b ? b[c] : 0.f)) : n;
            }
        }
    });
}

// Hann window, periodic (scipy.signal.get_window('hann', n, fftbins=True))
std::vector<double> hann(int n) {
    std::vector<double> w(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) w[size_t(i)] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / n);
    return w;
}

void nsf_source(const float *F0, int Tg, int up_total, double SR, int H, const float *rand_ini, const float *noise, const float *wrow, float lb,
                std::vector<float> &har_src) {
    const size_t Ls = size_t(Tg) * size_t(up_total), HH = size_t(H);
    // rad = (f0 * h / sr) mod 1 at the sample rate; the first sample gets the random initial phase
    std::vector<float> rad(Ls * HH);
    parallel_for(int64_t(Ls), 4096, [&](int64_t a, int64_t e) {
        for (int64_t i = a; i < e; ++i) {
            const float f0 = F0[size_t(i / up_total)];
            for (int hh = 0; hh < H; ++hh) {
                const float fn = f0 * float(hh + 1);
                float r = fn / float(SR);
                r = r - std::floor(r);   // Python-style remainder (result in [0, 1))
                rad[size_t(i) * HH + size_t(hh)] = r;
            }
        }
    });
    for (int hh = 1; hh < H; ++hh) rad[size_t(hh)] += rand_ini[hh];   // rand_ini[:, 0] = 0: the fundamental has no random phase
    // F.interpolate(scale 1/up, linear, align_corners=False): src = up*(i+0.5)-0.5 → mean of samples up*i+up/2-1 and up*i+up/2
    std::vector<double> phase(size_t(Tg) * HH);
    {
        std::vector<float> rd(size_t(Tg) * HH);
        for (int i = 0; i < Tg; ++i) {
            const double src = double(up_total) * (i + 0.5) - 0.5;
            const int64_t i0 = int64_t(std::floor(src));
            const int64_t i1 = std::min<int64_t>(i0 + 1, int64_t(Ls) - 1);
            const float lam = float(src - double(i0));
            for (int hh = 0; hh < H; ++hh)
                rd[size_t(i) * HH + size_t(hh)] = (1.0f - lam) * rad[size_t(i0) * HH + size_t(hh)] + lam * rad[size_t(i1) * HH + size_t(hh)];
        }
        std::vector<double> cs(HH, 0.0);   // torch.cumsum on the CPU accumulates in double, stores float
        for (int i = 0; i < Tg; ++i)
            for (int hh = 0; hh < H; ++hh) {
                cs[size_t(hh)] += double(rd[size_t(i) * HH + size_t(hh)]);
                const float c = float(cs[size_t(hh)]);
                // torch.cumsum(...) * 2 * np.pi, then * upsample_scale — float multiplications in this order
                phase[size_t(i) * HH + size_t(hh)] = double(((c * 2.0f) * float(kPi)) * float(up_total));
            }
    }
    har_src.resize(Ls);
    parallel_for(int64_t(Ls), 4096, [&](int64_t a, int64_t e) {
        for (int64_t j = a; j < e; ++j) {
            // F.interpolate(scale up, linear, align_corners=False) as the CPU kernel does it, in float:
            // src = fma(float(1/up), j + 0.5, -0.5) (clamped at 0), fma(x0, 1 - lam, x1 * lam) — bit-exact with PyTorch on x86
            const float scale_f = float(1.0 / double(up_total));
            float src = std::fma(scale_f, float(j) + 0.5f, -0.5f);   // the CPU kernel's fused multiply-add
            if (src < 0.f) src = 0.f;
            const int64_t i0 = std::min<int64_t>(int64_t(std::floor(src)), Tg - 1);
            const int64_t i1 = i0 + (i0 < Tg - 1 ? 1 : 0);
            const float lam = std::min(std::max(src - float(i0), 0.f), 1.f), lam0 = 1.0f - lam;
            const float f0 = F0[size_t(j / up_total)];
            const float uv = f0 > 10.0f ? 1.0f : 0.0f;
            const float namp = uv * 0.003f + (1.0f - uv) * (0.1f / 3.0f);
            float acc = 0.f;
            for (int hh = 0; hh < H; ++hh) {
                const float ph = std::fma(float(phase[size_t(i0) * HH + size_t(hh)]), lam0, float(phase[size_t(i1) * HH + size_t(hh)]) * lam);
                float sw = std::sin(ph) * 0.1f;
                sw = sw * uv + namp * noise[size_t(j) * HH + size_t(hh)];
                acc += sw * wrow[hh];
            }
            har_src[size_t(j)] = std::tanh(acc + lb);
        }
    });
}

void nsf_source_v1(const float *F0, int Tg, int up, double SR, int H, const float *rand_ini, const float *noise, const float *wrow, float lb,
                   std::vector<float> &out) {
    const size_t L = size_t(Tg) * size_t(up), HH = size_t(H);
    std::vector<float> theta(L * HH);   // [H][L]
    parallel_for(H, 1, [&](int64_t a, int64_t e) {
        for (int64_t h = a; h < e; ++h) {
            double cs = 0.0;   // torch.cumsum on the CPU: double accumulator, float result
            for (size_t j = 0; j < L; ++j) {
                const float fm = (F0[j / size_t(up)] * float(h + 1)) / float(SR);
                cs += double(fm);
                float c = float(cs);
                c = c - std::floor(c);   // % 1
                theta[size_t(h) * L + j] = float(2.0 * kPi) * c;
            }
        }
    });
    std::vector<float> ph(HH, 0.f);
    for (int h = 1; h < H; ++h) ph[size_t(h)] = float(-kPi) + rand_ini[h] * float(2.0 * kPi);   // Uniform(-π, π)
    out.resize(L);
    parallel_for(int64_t(L), 4096, [&](int64_t a, int64_t e) {
        for (int64_t j = a; j < e; ++j) {
            const float f0 = F0[size_t(j) / size_t(up)];
            const float uv = f0 > 10.0f ? 1.0f : 0.0f;
            const float namp = uv * 0.003f + (1.0f - uv) * (0.1f / 3.0f);
            float acc = 0.f;
            for (int h = 0; h < H; ++h) {
                const float sw = 0.1f * std::sin(theta[size_t(h) * L + size_t(j)] + ph[size_t(h)]);
                acc += (sw * uv + namp * noise[size_t(h) * L + size_t(j)]) * wrow[h];
            }
            out[size_t(j)] = std::tanh(acc + lb);
        }
    });
}

void stft_center(const std::vector<float> &x, int NF, int HOP, bool polar, std::vector<float> &har) {
    const int NB = NF / 2 + 1;
    const int64_t Ln = int64_t(x.size());
    const int frames = int(Ln / HOP) + 1;
    har.assign(size_t(frames) * size_t(2 * NB), 0.f);
    const auto win = hann(NF);
    std::vector<double> cosT(size_t(NB) * NF), sinT(size_t(NB) * NF);
    for (int kk = 0; kk < NB; ++kk)
        for (int n = 0; n < NF; ++n) {
            cosT[size_t(kk) * NF + n] = std::cos(2.0 * kPi * kk * n / NF);
            sinT[size_t(kk) * NF + n] = std::sin(2.0 * kPi * kk * n / NF);
        }
    const int64_t half = NF / 2;
    parallel_for(frames, 256, [&](int64_t a, int64_t e) {
        std::vector<double> seg(static_cast<size_t>(NF));
        for (int64_t fr = a; fr < e; ++fr) {
            for (int n = 0; n < NF; ++n) {
                int64_t idx = fr * HOP + n - half;
                if (idx < 0) idx = -idx;
                if (idx >= Ln) idx = 2 * (Ln - 1) - idx;
                seg[size_t(n)] = double(x[size_t(idx)]) * win[size_t(n)];
            }
            for (int kk = 0; kk < NB; ++kk) {
                double re = 0, im = 0;
                for (int n = 0; n < NF; ++n) { re += seg[size_t(n)] * cosT[size_t(kk) * NF + n]; im -= seg[size_t(n)] * sinT[size_t(kk) * NF + n]; }
                if (kk == 0 || kk == NB - 1) im = 0.0;   // real input: DC and Nyquist are exactly real (+0 imaginary, as the FFT gives)
                const float fre = float(re), fim = float(im);
                float *o = &har[size_t(fr) * size_t(2 * NB)];
                if (polar) {
                    o[kk] = std::sqrt(fre * fre + fim * fim);
                    o[NB + kk] = std::atan2(fim, fre);
                } else {
                    o[kk] = fre;
                    o[NB + kk] = fim;
                }
            }
        }
    });
}

void istft_center(const std::vector<float> &post, int frames, int NF, int HOP, float mag_clip, std::vector<float> &pcm) {
    const int NB = NF / 2 + 1;
    const auto win = hann(NF);
    const int64_t total = int64_t(frames - 1) * HOP + NF;
    std::vector<double> yv(static_cast<size_t>(total), 0.0), env(size_t(total), 0.0);
    std::vector<std::vector<double>> fr_out(static_cast<size_t>(frames), std::vector<double>(static_cast<size_t>(NF)));
    parallel_for(frames, 256, [&](int64_t a, int64_t e) {
        for (int64_t fr = a; fr < e; ++fr) {
            const float *r = &post[size_t(fr) * size_t(NF + 2)];
            auto &o = fr_out[size_t(fr)];
            for (int n = 0; n < NF; ++n) {
                double acc = 0;
                for (int kk = 0; kk < NB; ++kk) {
                    float magf = std::exp(r[kk]);
                    if (mag_clip > 0.f && magf > mag_clip) magf = mag_clip;
                    const double mag = double(magf), ph = double(std::sin(r[NB + kk]));
                    double re = mag * std::cos(ph), im = mag * std::sin(ph);
                    if (kk == 0 || kk == NB - 1) im = 0;   // irfft ignores the imaginary part of DC / Nyquist
                    const double w = (kk == 0 || kk == NB - 1) ? 1.0 : 2.0;
                    const double ang = 2.0 * kPi * kk * n / NF;
                    acc += w * (re * std::cos(ang) - im * std::sin(ang));
                }
                o[size_t(n)] = acc / NF;
            }
        }
    });
    for (int fr = 0; fr < frames; ++fr)
        for (int n = 0; n < NF; ++n) {
            yv[size_t(fr) * HOP + n] += fr_out[size_t(fr)][size_t(n)] * win[size_t(n)];
            env[size_t(fr) * HOP + n] += win[size_t(n)] * win[size_t(n)];
        }
    const int64_t start = NF / 2, len = int64_t(frames - 1) * HOP;
    pcm.resize(size_t(len));
    for (int64_t i = 0; i < len; ++i) pcm[size_t(i)] = float(yv[size_t(start + i)] / env[size_t(start + i)]);
}

void attention(const float *q, int64_t qs, const float *k, int64_t ks, const float *v, int64_t vs, int Tq, int Tk, int H, int dh,
               float scale, float *out, const std::function<void(int, float *)> &add) {
    if (add) {   // a score term per head (relative positions): whole score matrices, products through matmul
        std::vector<float> Q(size_t(Tq) * size_t(dh)), K(size_t(Tk) * size_t(dh)), Vt(size_t(dh) * size_t(Tk)), S(size_t(Tq) * size_t(Tk)),
            O(size_t(Tq) * size_t(dh));
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < Tq; ++t) std::memcpy(&Q[size_t(t) * dh], q + t * qs + int64_t(h) * dh, size_t(dh) * 4);
            for (int t = 0; t < Tk; ++t) {
                std::memcpy(&K[size_t(t) * dh], k + t * ks + int64_t(h) * dh, size_t(dh) * 4);
                const float *vr = v + t * vs + int64_t(h) * dh;
                for (int d = 0; d < dh; ++d) Vt[size_t(d) * size_t(Tk) + size_t(t)] = vr[d];
            }
            matmul(f32_mat(K.data(), Tk, dh), Q.data(), Tq, S.data());   // S[i][j] = q_i · k_j
            for (auto &z : S) z *= scale;
            add(h, S.data());
            parallel_for(Tq, 16, [&](int64_t a, int64_t e) {
                for (int64_t i = a; i < e; ++i) softmax_inplace(&S[size_t(i) * size_t(Tk)], Tk);
            });
            matmul(f32_mat(Vt.data(), dh, Tk), S.data(), Tq, O.data());
            for (int t = 0; t < Tq; ++t) std::memcpy(out + int64_t(t) * H * dh + int64_t(h) * dh, &O[size_t(t) * dh], size_t(dh) * 4);
        }
        return;
    }
    // per head: S = Q·Kᵀ and O = P·V as matrix products, softmax rows in parallel
    std::vector<float> Q(size_t(Tq) * size_t(dh)), K(size_t(Tk) * size_t(dh)), Vt(size_t(dh) * size_t(Tk)), S(size_t(Tq) * size_t(Tk)),
        O(size_t(Tq) * size_t(dh));
    for (int h = 0; h < H; ++h) {
        parallel_for(std::max(Tq, Tk), 64, [&](int64_t a, int64_t e) {
            for (int64_t t = a; t < e; ++t) {
                if (t < Tq) {
                    const float *qr = q + t * qs + int64_t(h) * dh;
                    for (int d = 0; d < dh; ++d) Q[size_t(t) * dh + d] = qr[d] * scale;
                }
                if (t < Tk) {
                    std::memcpy(&K[size_t(t) * dh], k + t * ks + int64_t(h) * dh, size_t(dh) * 4);
                    const float *vr = v + t * vs + int64_t(h) * dh;
                    for (int d = 0; d < dh; ++d) Vt[size_t(d) * size_t(Tk) + size_t(t)] = vr[d];
                }
            }
        });
        matmul(f32_mat(K.data(), Tk, dh), Q.data(), Tq, S.data());
        parallel_for(Tq, 8, [&](int64_t a, int64_t e) {
            for (int64_t i = a; i < e; ++i) softmax_fast(&S[size_t(i) * size_t(Tk)], Tk);
        });
        matmul(f32_mat(Vt.data(), dh, Tk), S.data(), Tq, O.data());
        parallel_for(Tq, 64, [&](int64_t a, int64_t e) {
            for (int64_t t = a; t < e; ++t) std::memcpy(out + t * H * dh + int64_t(h) * dh, &O[size_t(t) * dh], size_t(dh) * 4);
        });
    }
}

std::vector<float> resample_ta(const float *x, size_t n, uint32_t from, uint32_t to) {
    if (from == to || n == 0) return std::vector<float>(x, x + n);
    uint32_t a = from, b = to;
    while (b) { const uint32_t t = a % b; a = b; b = t; }
    const int64_t orig = int64_t(from / a), nw = int64_t(to / a);
    const double lw = 6.0, base = double(std::min(orig, nw)) * 0.99;
    const int64_t width = int64_t(std::ceil(lw * double(orig) / base));
    const int64_t K = 2 * width + orig;
    std::vector<float> ker(size_t(nw * K));
    for (int64_t j = 0; j < nw; ++j)
        for (int64_t k = 0; k < K; ++k) {
            const double idx = double(k - width) / double(orig);
            double t = (double(-j) / double(nw) + idx) * base;
            t = std::min(std::max(t, -lw), lw);
            const double w = std::pow(std::cos(t * kPi / lw / 2.0), 2.0);
            t *= kPi;
            const double sc = base / double(orig);
            const double kv = (t == 0.0 ? 1.0 : std::sin(t) / t) * (w * sc);
            ker[size_t(j * K + k)] = float(kv);
        }
    const int64_t L = int64_t(n), Lp = L + 2 * width + orig;
    std::vector<float> xp(size_t(Lp), 0.f);
    std::memcpy(&xp[size_t(width)], x, n * 4);
    const int64_t frames = (Lp - K) / orig + 1;
    const int64_t target = int64_t(std::ceil(double(nw) * double(L) / double(orig)));
    std::vector<float> y(size_t(std::min(frames * nw, target)));
    parallel_for(frames, 256, [&](int64_t s, int64_t e) {
        for (int64_t i = s; i < e; ++i)
            for (int64_t j = 0; j < nw; ++j) {
                const int64_t o = i * nw + j;
                if (o >= int64_t(y.size())) continue;
                float acc = 0.f;
                const float *kr = &ker[size_t(j * K)], *xr = &xp[size_t(i * orig)];
                for (int64_t k = 0; k < K; ++k) acc += kr[k] * xr[k];
                y[size_t(o)] = acc;
            }
    });
    return y;
}

RealDFT::RealDFT(int n_) : n(n_), nb(n_ / 2 + 1), table(size_t(2 * (n_ / 2 + 1)) * size_t(n_)) {
    for (int k = 0; k < nb; ++k)
        for (int i = 0; i < n; ++i) {
            const double a = 2.0 * kPi * double((int64_t(k) * i) % n) / double(n);
            table[size_t(k) * n + i] = float(std::cos(a));
            table[size_t(nb + k) * n + i] = float(-std::sin(a));
        }
}

void RealDFT::forward(const float *x, int64_t frames, float *out) const {
    matmul(f32_mat(table.data(), 2 * nb, n), x, frames, out);
}

}  // namespace nn
}  // namespace vox
