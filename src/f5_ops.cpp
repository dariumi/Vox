#include "f5_ops.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>

namespace vox {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Комплексное БПФ длины n = 2^k на месте (прямое при sign = -1, обратное без нормировки при sign = +1).
void fft_inplace(std::vector<std::complex<double>> &a, int sign) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * kPi / double(len) * double(sign);
        const std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; ++k) {
                const std::complex<double> u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

std::vector<double> hann_periodic(int n) {
    std::vector<double> w((size_t)n);
    for (int i = 0; i < n; ++i) w[size_t(i)] = 0.5 - 0.5 * std::cos(2.0 * kPi * double(i) / double(n));
    return w;
}

}  // namespace

void mel_log_spectrogram(const MelCfg &c, const float *fb, const float *pcm, size_t n, std::vector<float> &out, int &frames) {
    if (!is_pow2(c.n_fft)) fail(VOX_ERR_ARG, "n_fft must be a power of two");
    const int pad = c.n_fft / 2, nb = c.n_fft / 2 + 1;
    if (n <= size_t(pad)) fail(VOX_ERR_ARG, "reference audio is too short for the mel spectrogram");
    frames = 1 + int(n / size_t(c.hop));
    // отражённое дополнение (reflect) с обеих сторон
    std::vector<float> xp(n + 2 * size_t(pad));
    for (int64_t i = -pad; i < int64_t(n) + pad; ++i) {
        int64_t j = i;
        if (j < 0) j = -j;
        if (j >= int64_t(n)) j = 2 * (int64_t(n) - 1) - j;
        xp[size_t(i + pad)] = pcm[j];
    }
    // окно длины win по центру n_fft
    std::vector<double> win((size_t)c.n_fft, 0.0);
    {
        const auto h = hann_periodic(c.win);
        const int off = (c.n_fft - c.win) / 2;
        for (int i = 0; i < c.win; ++i) win[size_t(off + i)] = h[size_t(i)];
    }
    out.assign(size_t(frames) * size_t(c.n_mels), 0.f);
    parallel_for(frames, 8, [&](int64_t t0, int64_t t1) {
        std::vector<std::complex<double>> buf((size_t)c.n_fft);
        std::vector<float> mag((size_t)nb);
        for (int64_t t = t0; t < t1; ++t) {
            const float *fr = &xp[size_t(t) * size_t(c.hop)];
            for (int i = 0; i < c.n_fft; ++i) buf[size_t(i)] = std::complex<double>(double(fr[i]) * win[size_t(i)], 0.0);
            fft_inplace(buf, -1);
            for (int k = 0; k < nb; ++k) mag[size_t(k)] = float(std::abs(buf[size_t(k)]));
            for (int m = 0; m < c.n_mels; ++m) {
                double acc = 0;
                const float *f = fb + size_t(m) * size_t(nb);
                for (int k = 0; k < nb; ++k) acc += double(f[k]) * double(mag[size_t(k)]);
                out[size_t(t) * size_t(c.n_mels) + size_t(m)] = float(std::log(std::max(acc, 1e-5)));
            }
        }
    });
}

void Conv1d::load(const float *src, int cin_, int cout_, int K_, int groups_, const float *bias) {
    cin = cin_; cout = cout_; K = K_; groups = groups_;
    if (K % 2 == 0) fail(VOX_ERR_FORMAT, "Conv1d: even kernel size is not supported");
    if (cin % groups || cout % groups) fail(VOX_ERR_FORMAT, "Conv1d: channels not divisible by groups");
    const int cig = cin / groups, cog = cout / groups;
    b.clear();
    if (bias) b.assign(bias, bias + cout);
    if (depthwise()) {  // [C][K]
        w.assign(src, src + size_t(cout) * size_t(K));
        return;
    }
    w.assign(size_t(groups) * size_t(cog) * size_t(K) * size_t(cig), 0.f);
    for (int g = 0; g < groups; ++g)
        for (int o = 0; o < cog; ++o)
            for (int i = 0; i < cig; ++i)
                for (int k = 0; k < K; ++k)
                    w[((size_t(g) * size_t(cog) + size_t(o)) * size_t(K) + size_t(k)) * size_t(cig) + size_t(i)] =
                        src[(size_t(g * cog + o) * size_t(cig) + size_t(i)) * size_t(K) + size_t(k)];
}

void Conv1d::forward(const float *x, int T, float *y) const {
    const int pad = K / 2;
    if (depthwise()) {
        parallel_for(T, 16, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t)
                for (int c = 0; c < cin; ++c) {
                    float acc = b.empty() ? 0.f : b[size_t(c)];
                    const float *wc = &w[size_t(c) * size_t(K)];
                    for (int k = 0; k < K; ++k) {
                        const int64_t tt = t + k - pad;
                        if (tt >= 0 && tt < T) acc += wc[k] * x[size_t(tt) * size_t(cin) + size_t(c)];
                    }
                    y[size_t(t) * size_t(cout) + size_t(c)] = acc;
                }
        });
        return;
    }
    const int cig = cin / groups, cog = cout / groups;
    std::vector<float> col(size_t(T) * size_t(K) * size_t(cig)), yg(size_t(T) * size_t(cog));
    for (int g = 0; g < groups; ++g) {
        parallel_for(T, 16, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) {
                float *dst = &col[size_t(t) * size_t(K) * size_t(cig)];
                for (int k = 0; k < K; ++k) {
                    const int64_t tt = t + k - pad;
                    if (tt >= 0 && tt < T) std::memcpy(dst + size_t(k) * size_t(cig), x + size_t(tt) * size_t(cin) + size_t(g) * size_t(cig), size_t(cig) * 4);
                    else std::memset(dst + size_t(k) * size_t(cig), 0, size_t(cig) * 4);
                }
            }
        });
        Mat W;
        W.type = WType::F32;
        W.data = reinterpret_cast<const uint8_t *>(&w[size_t(g) * size_t(cog) * size_t(K) * size_t(cig)]);
        W.rows = cog;
        W.cols = int64_t(K) * cig;
        matmul(W, col.data(), T, yg.data());
        for (int64_t t = 0; t < T; ++t)
            for (int o = 0; o < cog; ++o)
                y[size_t(t) * size_t(cout) + size_t(g) * size_t(cog) + size_t(o)] =
                    yg[size_t(t) * size_t(cog) + size_t(o)] + (b.empty() ? 0.f : b[size_t(g * cog + o)]);
    }
}

void istft_same(const float *mag, const float *phase, int T, int n_fft, int hop, std::vector<float> &out, bool center) {
    if (!is_pow2(n_fft)) fail(VOX_ERR_ARG, "n_fft must be a power of two");
    const int nb = n_fft / 2 + 1, pad = center ? n_fft / 2 : (n_fft - hop) / 2;
    const size_t total = size_t(T - 1) * size_t(hop) + size_t(n_fft);
    const auto win = hann_periodic(n_fft);
    std::vector<double> y(total, 0.0), env(total, 0.0);
    // кадры обрабатываются параллельно, перекрытие — после
    std::vector<std::vector<float>> frames((size_t)T);
    parallel_for(T, 4, [&](int64_t t0, int64_t t1) {
        std::vector<std::complex<double>> buf((size_t)n_fft);
        for (int64_t t = t0; t < t1; ++t) {
            for (int k = 0; k < nb; ++k) {
                const double m = std::min(double(std::exp(mag[size_t(t) * size_t(nb) + size_t(k)])), 1e2);  // mag приходит как логарифм (голова Vocos: exp, потолок 1e2)
                const double p = phase[size_t(t) * size_t(nb) + size_t(k)];
                std::complex<double> v(m * std::cos(p), m * std::sin(p));
                if (k == 0 || k == nb - 1) v = std::complex<double>(v.real(), 0.0);  // irfft игнорирует мнимую часть DC и Найквиста
                buf[size_t(k)] = v;
                if (k > 0 && k < nb - 1) buf[size_t(n_fft - k)] = std::conj(v);
            }
            fft_inplace(buf, +1);
            auto &f = frames[size_t(t)];
            f.resize(size_t(n_fft));
            for (int i = 0; i < n_fft; ++i) f[size_t(i)] = float(buf[size_t(i)].real() / double(n_fft));
        }
    });
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < n_fft; ++i) {
            y[size_t(t) * size_t(hop) + size_t(i)] += double(frames[size_t(t)][size_t(i)]) * win[size_t(i)];
            env[size_t(t) * size_t(hop) + size_t(i)] += win[size_t(i)] * win[size_t(i)];
        }
    out.resize(total > 2 * size_t(pad) ? total - 2 * size_t(pad) : 0);
    for (size_t i = 0; i < out.size(); ++i) out[i] = float(y[i + size_t(pad)] / env[i + size_t(pad)]);
}

void mat_to_float(const Mat &W, std::vector<float> &out) {
    out.resize(size_t(W.rows) * size_t(W.cols));
    std::vector<float> tmp(size_t(W.cols));
    for (int64_t r = 0; r < W.rows; ++r) {
        const float *row = mat_row(W, r, tmp.data());
        std::memcpy(&out[size_t(r) * size_t(W.cols)], row, size_t(W.cols) * 4);
    }
}

}  // namespace vox
