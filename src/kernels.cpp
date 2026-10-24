#include "kernels.hpp"
#include "gpu.hpp"

#if defined(__x86_64__) && defined(__GNUC__)
#  include <immintrin.h>
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#if defined(__linux__)
#  include <pthread.h>
#  include <sched.h>
#  include <sys/resource.h>
#  include <sys/syscall.h>
#  include <unistd.h>
#endif

namespace vox {

// ====================== пул потоков ======================
// Рабочие потоки сначала крутятся на атомарном счётчике задач (микросекунды между мелкими областями
// внутри одного кадра), и только потом засыпают на condition_variable. Пулов может быть несколько:
// у каждого движка свой (число потоков, привязка к ядрам, приоритет), активный пул выбирает PoolScope.
namespace {
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}
}  // namespace

class ThreadPool {
public:
    ThreadPool(int count, const CpuTuning &tune) : tune_(tune), spin_(tune.spin ? 4000 : 0) { resize(count); }
    ~ThreadPool() { stop_workers(); }
    int size() const { return n_; }
    const CpuTuning &tuning() const { return tune_; }

    void resize(int count) {
        std::lock_guard<std::mutex> run(run_mu_);  // не меняем число потоков посреди параллельного запуска
        stop_workers();
        stop_.store(false);
        n_ = std::max(1, count);
        for (int i = 1; i < n_; ++i) workers_.emplace_back([this, i] { loop(i); });
    }

    void run(int64_t total, int64_t grain, const std::function<void(int64_t, int64_t)> &fn) {
        if (n_ == 1 || total <= grain) { fn(0, total); return; }
        std::lock_guard<std::mutex> run(run_mu_);  // один параллельный запуск за раз (сессии из разных потоков ждут очереди)
        {
            std::lock_guard<std::mutex> lk(mu_);
            fn_ = &fn;
            total_ = total;
            grain_ = std::max<int64_t>(1, grain);
            next_.store(0);
            active_.store(n_ - 1);
            job_id_.fetch_add(1, std::memory_order_release);
        }
        cv_start_.notify_all();
        work();
        for (int i = 0; i < spin_ && active_.load(std::memory_order_acquire) != 0; ++i) cpu_relax();
        if (active_.load() != 0) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_done_.wait(lk, [&] { return active_.load() == 0; });
        }
    }

private:
    void work() {
        for (;;) {
            int64_t b = next_.fetch_add(grain_);
            if (b >= total_) return;
            (*fn_)(b, std::min(total_, b + grain_));
        }
    }
    void apply_tuning(int index) {
#if defined(__linux__)
        if (!tune_.affinity.empty()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            // рабочий i привязывается к ядру affinity[i % size]: потоки не прыгают между ядрами
            CPU_SET(tune_.affinity[size_t(index) % tune_.affinity.size()], &set);
            pthread_setaffinity_np(pthread_self(), sizeof set, &set);
        }
        if (tune_.nice != 0) setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), tune_.nice);
#else
        (void)index;
#endif
    }
    void loop(int index) {
        apply_tuning(index);
        uint64_t seen = 0;
        for (;;) {
            for (int i = 0; i < spin_ && job_id_.load(std::memory_order_acquire) == seen && !stop_.load(std::memory_order_relaxed); ++i)
                cpu_relax();
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_start_.wait(lk, [&] { return stop_.load() || job_id_.load() != seen; });
                if (stop_.load()) return;
                seen = job_id_.load();
            }
            work();
            if (active_.fetch_sub(1) == 1) {
                std::lock_guard<std::mutex> lk(mu_);
                cv_done_.notify_one();
            }
        }
    }
    void stop_workers() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_.store(true);
        }
        cv_start_.notify_all();
        for (auto &t : workers_) t.join();
        workers_.clear();
    }

    CpuTuning tune_;
    int spin_ = 4000;
    std::vector<std::thread> workers_;
    std::mutex mu_, run_mu_;
    std::condition_variable cv_start_, cv_done_;
    const std::function<void(int64_t, int64_t)> *fn_ = nullptr;
    int64_t total_ = 0, grain_ = 1;
    std::atomic<int64_t> next_{0};
    std::atomic<int> active_{0};
    std::atomic<uint64_t> job_id_{0};
    std::atomic<bool> stop_{false};
    int n_ = 1;
};

namespace {
thread_local ThreadPool *t_pool = nullptr;
ThreadPool &global_pool() {
    static ThreadPool *p = [] {
        unsigned hc = std::thread::hardware_concurrency();
        return new ThreadPool(hc ? int(hc) : 1, CpuTuning{});  // живёт до конца процесса
    }();
    return *p;
}
ThreadPool &cur_pool() { return t_pool ? *t_pool : global_pool(); }
}  // namespace

std::shared_ptr<ThreadPool> make_thread_pool(int threads, const CpuTuning &tune) {
    if (threads <= 0) { unsigned hc = std::thread::hardware_concurrency(); threads = hc ? int(hc) : 1; }
    return std::make_shared<ThreadPool>(threads, tune);
}
PoolScope::PoolScope(ThreadPool *p) : prev_(t_pool) { if (p) t_pool = p; }
PoolScope::~PoolScope() { t_pool = prev_; }

void set_num_threads(int n) {
    if (n <= 0) { unsigned hc = std::thread::hardware_concurrency(); n = hc ? int(hc) : 1; }
    global_pool().resize(n);
}
int num_threads() { return cur_pool().size(); }

void parallel_for(int64_t total, int64_t grain, const std::function<void(int64_t, int64_t)> &fn) {
    if (total <= 0) return;
    cur_pool().run(total, grain, fn);
}

// ====================== базовые ядра ======================
#if defined(__GNUC__) && !defined(_MSC_VER)
#  define VOX_VECTOR_EXT 1
typedef float v8f __attribute__((vector_size(32), aligned(4)));
#  define VOX_INLINE __attribute__((always_inline)) inline
#else
#  define VOX_INLINE inline
#endif
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && defined(__linux__)
#  define VOX_MULTIVERSION __attribute__((target_clones("avx2,fma", "default")))
#else
#  define VOX_MULTIVERSION
#endif

// Загрузчики весов: преобразование в float прямо в регистрах (без отдельного прохода по строке).
template <WType T> struct WL;
#ifdef VOX_VECTOR_EXT
typedef uint16_t v8h __attribute__((vector_size(16)));
typedef uint32_t v8u __attribute__((vector_size(32)));
typedef int32_t v8i32 __attribute__((vector_size(32)));
typedef int8_t v8q __attribute__((vector_size(8)));
template <> struct WL<WType::F32> {
    static VOX_INLINE v8f load8(const uint8_t *row, int64_t k, const float *) { v8f v; std::memcpy(&v, row + k * 4, 32); return v; }
    static VOX_INLINE float at(const uint8_t *row, int64_t k, const float *) { float f; std::memcpy(&f, row + k * 4, 4); return f; }
};
template <> struct WL<WType::BF16> {
    static VOX_INLINE v8f load8(const uint8_t *row, int64_t k, const float *) {
        v8h h; std::memcpy(&h, row + k * 2, 16);
        v8u u = __builtin_convertvector(h, v8u);
        u <<= 16;
        v8f f; std::memcpy(&f, &u, 32); return f;
    }
    static VOX_INLINE float at(const uint8_t *row, int64_t k, const float *) { uint16_t h; std::memcpy(&h, row + k * 2, 2); return bf16_to_f32(h); }
};
template <> struct WL<WType::F16> {
    // Нормальные числа и нули; субнормальные f16 (|x| < 6e-5) обнуляются — для весов это несущественно.
    static VOX_INLINE v8f load8(const uint8_t *row, int64_t k, const float *) {
        v8h hh; std::memcpy(&hh, row + k * 2, 16);
        v8u h = __builtin_convertvector(hh, v8u);
        v8u sign = (h & 0x8000u) << 16;
        v8u mag = ((h & 0x7FFFu) << 13) + 0x38000000u;
        v8i32 nz = (h & 0x7C00u) != 0;
        v8u r = sign | (mag & (v8u)nz);
        v8f f; std::memcpy(&f, &r, 32); return f;
    }
    static VOX_INLINE float at(const uint8_t *row, int64_t k, const float *) { uint16_t h; std::memcpy(&h, row + k * 2, 2); return f16_to_f32(h); }
};
template <> struct WL<WType::Q8> {
    static VOX_INLINE v8f load8(const uint8_t *row, int64_t k, const float *sc) {
        v8q q; std::memcpy(&q, row + k, 8);
        v8i32 qi = __builtin_convertvector(q, v8i32);  // int8 → int32 (vpmovsxbd), затем → float
        v8f f = __builtin_convertvector(qi, v8f);
        return f * sc[k >> 5];
    }
    static VOX_INLINE float at(const uint8_t *row, int64_t k, const float *sc) { return float(reinterpret_cast<const int8_t *>(row)[k]) * sc[k >> 5]; }
};

// Плитка RW строк весов × XI строк входа: каждая загрузка используется несколько раз.
template <WType T, int RW, int XI>
VOX_INLINE void tile(const uint8_t *const *w, const float *const *sc, const float *const *x, int64_t cols, float *out) {
    v8f acc[RW][XI];
    for (int r = 0; r < RW; ++r)
        for (int j = 0; j < XI; ++j) acc[r][j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    int64_t k = 0;
    for (; k + 8 <= cols; k += 8) {
        v8f xv[XI];
        for (int j = 0; j < XI; ++j) std::memcpy(&xv[j], x[j] + k, 32);
        for (int r = 0; r < RW; ++r) {
            const v8f wv = WL<T>::load8(w[r], k, sc[r]);
            for (int j = 0; j < XI; ++j) acc[r][j] += wv * xv[j];
        }
    }
    for (int r = 0; r < RW; ++r)
        for (int j = 0; j < XI; ++j) {
            const v8f v = acc[r][j];
            float s = ((v[0] + v[4]) + (v[1] + v[5])) + ((v[2] + v[6]) + (v[3] + v[7]));
            for (int64_t kk = k; kk < cols; ++kk) s += WL<T>::at(w[r], kk, sc[r]) * x[j][kk];
            out[r * XI + j] = s;
        }
}
#else
template <> struct WL<WType::F32> { static inline float at(const uint8_t *row, int64_t k, const float *) { float f; std::memcpy(&f, row + k * 4, 4); return f; } };
template <> struct WL<WType::BF16> { static inline float at(const uint8_t *row, int64_t k, const float *) { uint16_t h; std::memcpy(&h, row + k * 2, 2); return bf16_to_f32(h); } };
template <> struct WL<WType::F16> { static inline float at(const uint8_t *row, int64_t k, const float *) { uint16_t h; std::memcpy(&h, row + k * 2, 2); return f16_to_f32(h); } };
template <> struct WL<WType::Q8> { static inline float at(const uint8_t *row, int64_t k, const float *sc) { return float(reinterpret_cast<const int8_t *>(row)[k]) * sc[k >> 5]; } };
template <WType T, int RW, int XI>
VOX_INLINE void tile(const uint8_t *const *w, const float *const *sc, const float *const *x, int64_t cols, float *out) {
    for (int r = 0; r < RW; ++r)
        for (int j = 0; j < XI; ++j) {
            float s = 0;
            for (int64_t k = 0; k < cols; ++k) s += WL<T>::at(w[r], k, sc[r]) * x[j][k];
            out[r * XI + j] = s;
        }
}
#endif

float dot(const float *a, const float *b, int64_t n) {
    const uint8_t *w[1] = {reinterpret_cast<const uint8_t *>(a)};
    const float *sc[1] = {nullptr}, *x[1] = {b};
    float o;
    tile<WType::F32, 1, 1>(w, sc, x, n, &o);
    return o;
}

template <WType T, int RW>
VOX_INLINE void rows_block(const uint8_t *const *wr, const float *const *sc, const float *bias_r, const float *x,
                           int64_t m, int64_t cols, float *y, int64_t rows, int64_t r0) {
    // плитка RW × XI аккумуляторов: 4×3 (12 регистров + 3 под вход + 1 под веса = все 16 ymm) или 2×4, 1×4; порядок суммирования
    // у каждого выхода один и тот же при любой форме плитки
    constexpr int XI = RW == 4 ? 3 : 4;
    int64_t i = 0;
    float o[RW * 4];
    for (; i + XI <= m; i += XI) {
        const float *xs[4];
        for (int j = 0; j < XI; ++j) xs[j] = x + (i + j) * cols;
        tile<T, RW, XI>(wr, sc, xs, cols, o);
        for (int r = 0; r < RW; ++r)
            for (int j = 0; j < XI; ++j) y[(i + j) * rows + r0 + r] = o[r * XI + j] + bias_r[r];
    }
    for (; i < m; ++i) {
        const float *xs[1] = {x + i * cols};
        tile<T, RW, 1>(wr, sc, xs, cols, o);
        for (int r = 0; r < RW; ++r) y[i * rows + r0 + r] = o[r] + bias_r[r];
    }
}

template <WType T>
VOX_INLINE void matmul_range_t(const Mat &W, const float *x, int64_t m, float *y, int64_t r0, int64_t r1) {
    const int64_t rows = W.rows, cols = W.cols;
    constexpr int64_t bpe = T == WType::F32 ? 4 : (T == WType::Q8 ? 1 : 2);
    constexpr int RW = 4;   // 4 строки весов на плитку (при m > 1 — плитка 4×3, см. rows_block)
    constexpr int64_t TM = 64;  // плитка по m: блок x остаётся в кэше, пока бежим по строкам весов
    for (int64_t i0 = 0; i0 < m; i0 += TM) {
        const int64_t i1 = std::min(m, i0 + TM), mm = i1 - i0;
        const float *xb = x + i0 * cols;
        float *yb = y + i0 * rows;
        int64_t r = r0;
        for (; r < r1; r += RW) {
            const int n = int(std::min<int64_t>(RW, r1 - r));
            const uint8_t *wr[4];
            const float *sc[4];
            float bz[4];
            for (int q = 0; q < 4; ++q) {
                const int64_t rr = r + (q < n ? q : 0);
                wr[q] = W.data + rr * cols * bpe;
                sc[q] = W.scale ? W.scale + rr * (cols / 32) : nullptr;
                bz[q] = W.bias ? W.bias[rr] : 0.0f;
            }
            if (n == 4) { rows_block<T, 4>(wr, sc, bz, xb, mm, cols, yb, rows, r); continue; }
            int q0 = 0;
            if (n >= 2 && mm > 1) { rows_block<T, 2>(wr, sc, bz, xb, mm, cols, yb, rows, r); q0 = 2; }
            for (int q = q0; q < n; ++q) {
                const uint8_t *w1[1] = {wr[q]};
                const float *s1[1] = {sc[q]};
                rows_block<T, 1>(w1, s1, bz + q, xb, mm, cols, yb, rows, r + q);
            }
        }
    }
}

VOX_MULTIVERSION static void mm_f32(const Mat &W, const float *x, int64_t m, float *y, int64_t a, int64_t b) { matmul_range_t<WType::F32>(W, x, m, y, a, b); }
VOX_MULTIVERSION static void mm_bf16(const Mat &W, const float *x, int64_t m, float *y, int64_t a, int64_t b) { matmul_range_t<WType::BF16>(W, x, m, y, a, b); }
VOX_MULTIVERSION static void mm_f16(const Mat &W, const float *x, int64_t m, float *y, int64_t a, int64_t b) { matmul_range_t<WType::F16>(W, x, m, y, a, b); }
VOX_MULTIVERSION static void mm_q8(const Mat &W, const float *x, int64_t m, float *y, int64_t a, int64_t b) { matmul_range_t<WType::Q8>(W, x, m, y, a, b); }

static void matmul_q4(const Mat &W, const float *x, int64_t m, float *y);
static void matmul_range(const Mat &W, const float *x, int64_t m, float *y, int64_t r0, int64_t r1) {
    switch (W.type) {
        case WType::F32: mm_f32(W, x, m, y, r0, r1); break;
        case WType::BF16: mm_bf16(W, x, m, y, r0, r1); break;
        case WType::F16: mm_f16(W, x, m, y, r0, r1); break;
        case WType::Q8: mm_q8(W, x, m, y, r0, r1); break;
        case WType::Q4: matmul_q4(W, x, m, y); break;  // обычно не доходит сюда: matmul() считает Q4 сам
    }
}

const float *mat_row(const Mat &W, int64_t r, float *tmp) {
    const int64_t c = W.cols;
    switch (W.type) {
        case WType::F32: return reinterpret_cast<const float *>(W.data) + r * c;
        case WType::F16: {
            auto p = reinterpret_cast<const uint16_t *>(W.data) + r * c;
            for (int64_t i = 0; i < c; ++i) tmp[i] = f16_to_f32(p[i]);
            return tmp;
        }
        case WType::BF16: {
            auto p = reinterpret_cast<const uint16_t *>(W.data) + r * c;
            for (int64_t i = 0; i < c; ++i) tmp[i] = bf16_to_f32(p[i]);
            return tmp;
        }
        case WType::Q8: {
            auto p = reinterpret_cast<const int8_t *>(W.data) + r * c;
            const float *s = W.scale + r * (c / 32);
            for (int64_t b = 0; b < c / 32; ++b)
                for (int j = 0; j < 32; ++j) tmp[b * 32 + j] = float(p[b * 32 + j]) * s[b];
            return tmp;
        }
        case WType::Q4: {
            auto p = W.data + size_t(r) * size_t(c / 2);
            const float *s = W.scale + r * (c / 32);
            for (int64_t b = 0; b < c / 32; ++b)
                for (int j = 0; j < 16; ++j) {
                    const uint8_t v = p[b * 16 + j];
                    tmp[b * 32 + j] = float(int(v & 15) - 8) * s[b];
                    tmp[b * 32 + 16 + j] = float(int(v >> 4) - 8) * s[b];
                }
            return tmp;
        }
    }
    return nullptr;
}

void mat_gather(const Mat &W, int64_t r, float *out) {
    const float *p = mat_row(W, r, out);
    if (p != out) std::memcpy(out, p, size_t(W.cols) * sizeof(float));
}


// ---------- Q8: вход квантуется в int8 по блокам из 32, скалярные произведения — целочисленные ----------
static float q8dot_scalar(const int8_t *wq, const float *ws, const int8_t *xq, const float *xs, int64_t nb) {
    float s = 0;
    for (int64_t b = 0; b < nb; ++b) {
        int acc = 0;
        for (int j = 0; j < 32; ++j) acc += int(wq[b * 32 + j]) * int(xq[b * 32 + j]);
        s += float(acc) * ws[b] * xs[b];
    }
    return s;
}
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx2,fma"))) static float q8dot_avx2(const int8_t *wq, const float *ws, const int8_t *xq,
                                                           const float *xs, int64_t nb) {
    __m256 acc = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    for (int64_t b = 0; b < nb; ++b) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wq + b * 32));
        __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq + b * 32));
        __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(a, a), _mm256_sign_epi8(x, a));
        __m256 f = _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones));
        acc = _mm256_fmadd_ps(f, _mm256_set1_ps(ws[b] * xs[b]), acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
static bool cpu_has_avx2() {
    static const bool v = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return v;
}
#else
static bool cpu_has_avx2() { return false; }
static float q8dot_avx2(const int8_t *a, const float *b, const int8_t *c, const float *d, int64_t n) { return q8dot_scalar(a, b, c, d, n); }
#endif

static void matmul_q8(const Mat &W, const float *x, int64_t m, float *y) {
    const int64_t rows = W.rows, cols = W.cols, nb = cols / 32;
    std::vector<int8_t> xq(size_t(m * cols));
    std::vector<float> xs(size_t(m * nb));
    for (int64_t i = 0; i < m; ++i)
        for (int64_t b = 0; b < nb; ++b) {
            const float *p = x + i * cols + b * 32;
            float mx = 0;
            for (int j = 0; j < 32; ++j) mx = std::max(mx, std::fabs(p[j]));
            const float sc = mx / 127.0f, inv = sc > 0 ? 1.0f / sc : 0.0f;
            xs[size_t(i * nb + b)] = sc;
            for (int j = 0; j < 32; ++j) xq[size_t(i * cols + b * 32 + j)] = int8_t(std::lrintf(p[j] * inv));
        }
    const bool avx = cpu_has_avx2();
    const int8_t *wq = reinterpret_cast<const int8_t *>(W.data);
    parallel_for(rows, 8, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const int8_t *wr = wq + r * cols;
            const float *ws = W.scale + r * nb;
            const float bz = W.bias ? W.bias[r] : 0.0f;
            for (int64_t i = 0; i < m; ++i) {
                const int8_t *xr = &xq[size_t(i * cols)];
                const float *xsr = &xs[size_t(i * nb)];
                y[i * rows + r] = (avx ? q8dot_avx2(wr, ws, xr, xsr, nb) : q8dot_scalar(wr, ws, xr, xsr, nb)) + bz;
            }
        }
    });
}

// ---------- Q4: те же целочисленные скалярные произведения, веса распаковываются из полубайтов ----------
static float q4dot_scalar(const uint8_t *wq, const float *ws, const int8_t *xq, const float *xs, int64_t nb) {
    float s = 0;
    for (int64_t b = 0; b < nb; ++b) {
        int acc = 0;
        for (int j = 0; j < 16; ++j) {
            const uint8_t v = wq[b * 16 + j];
            acc += (int(v & 15) - 8) * int(xq[b * 32 + j]) + (int(v >> 4) - 8) * int(xq[b * 32 + 16 + j]);
        }
        s += float(acc) * ws[b] * xs[b];
    }
    return s;
}
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx2,fma"))) static float q4dot_avx2(const uint8_t *wq, const float *ws, const int8_t *xq,
                                                           const float *xs, int64_t nb) {
    __m256 acc = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1), eight = _mm256_set1_epi8(8);
    const __m128i m15 = _mm_set1_epi8(15);
    for (int64_t b = 0; b < nb; ++b) {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(wq + b * 16));
        const __m128i lo = _mm_and_si128(raw, m15), hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m15);
        const __m256i a = _mm256_sub_epi8(_mm256_set_m128i(hi, lo), eight);  // веса -8..7: элементы 0..15 из младших, 16..31 из старших полубайтов
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq + b * 32));
        const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(a, a), _mm256_sign_epi8(x, a));
        const __m256 f = _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones));
        acc = _mm256_fmadd_ps(f, _mm256_set1_ps(ws[b] * xs[b]), acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
#else
static float q4dot_avx2(const uint8_t *a, const float *b, const int8_t *c, const float *d, int64_t n) { return q4dot_scalar(a, b, c, d, n); }
#endif

static void matmul_q4(const Mat &W, const float *x, int64_t m, float *y) {
    const int64_t rows = W.rows, cols = W.cols, nb = cols / 32;
    std::vector<int8_t> xq(size_t(m * cols));
    std::vector<float> xs(size_t(m * nb));
    for (int64_t i = 0; i < m; ++i)
        for (int64_t b = 0; b < nb; ++b) {
            const float *p = x + i * cols + b * 32;
            float mx = 0;
            for (int j = 0; j < 32; ++j) mx = std::max(mx, std::fabs(p[j]));
            const float sc = mx / 127.0f, inv = sc > 0 ? 1.0f / sc : 0.0f;
            xs[size_t(i * nb + b)] = sc;
            for (int j = 0; j < 32; ++j) xq[size_t(i * cols + b * 32 + j)] = int8_t(std::lrintf(p[j] * inv));
        }
    const bool avx = cpu_has_avx2();
    parallel_for(rows, 8, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const uint8_t *wr = W.data + size_t(r) * size_t(cols / 2);
            const float *ws = W.scale + r * nb;
            const float bz = W.bias ? W.bias[r] : 0.0f;
            for (int64_t i = 0; i < m; ++i) {
                const int8_t *xr = &xq[size_t(i * cols)];
                const float *xsr = &xs[size_t(i * nb)];
                y[i * rows + r] = (avx ? q4dot_avx2(wr, ws, xr, xsr, nb) : q4dot_scalar(wr, ws, xr, xsr, nb)) + bz;
            }
        }
    });
}

void quantize_q4(const float *w, int64_t rows, int64_t cols, std::vector<uint8_t> &q, std::vector<float> &scale) {
    q.assign(size_t(rows * cols / 2), 0);
    scale.assign(size_t(rows * (cols / 32)), 0.f);
    // Масштаб блока подбирается по минимуму квадрата ошибки: перебор t ∈ [7, 9] (d = mx / −t), затем уточнение методом
    // наименьших квадратов при найденных кодах. Формат от этого не меняется (код − 8) · d; t = 8 даёт прежнюю схему.
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t b = 0; b < cols / 32; ++b) {
            const float *p = w + r * cols + b * 32;
            float mx = 0;  // значение с наибольшим модулем (со знаком)
            for (int j = 0; j < 32; ++j) if (std::fabs(p[j]) > std::fabs(mx)) mx = p[j];
            float best_d = mx / -8.0f, best_err = INFINITY;
            int8_t best_q[32] = {0};
            if (mx != 0.0f)
                for (int k = 0; k <= 8; ++k) {
                    const float d = mx / -(7.0f + 0.25f * float(k)), inv = 1.0f / d;
                    int8_t qq[32];
                    double err = 0;
                    for (int j = 0; j < 32; ++j) {
                        const float c = std::min(15.0f, std::max(0.0f, std::floor(p[j] * inv + 8.5f)));
                        qq[j] = int8_t(int(c) - 8);
                        const double e = double(p[j]) - double(qq[j]) * double(d);
                        err += e * e;
                    }
                    if (err < best_err) { best_err = err; best_d = d; std::copy(qq, qq + 32, best_q); }
                }
            if (mx != 0.0f) {
                double num = 0, den = 0;
                for (int j = 0; j < 32; ++j) { num += double(p[j]) * best_q[j]; den += double(best_q[j]) * best_q[j]; }
                if (den > 0) {
                    const float d2 = float(num / den);
                    double err = 0;
                    for (int j = 0; j < 32; ++j) { const double e = double(p[j]) - double(best_q[j]) * double(d2); err += e * e; }
                    if (err < best_err) best_d = d2;
                }
            }
            scale[size_t(r * (cols / 32) + b)] = best_d;
            uint8_t *out = &q[size_t(r * cols / 2 + b * 16)];
            for (int j = 0; j < 16; ++j) out[j] = uint8_t(uint8_t(best_q[j] + 8) | (uint8_t(best_q[j + 16] + 8) << 4));
        }
}

constexpr int64_t kQ8IntMaxRows = 8;
static bool mm2d_enabled() {
    static const bool on = [] { const char *e = std::getenv("VOX_MM2D"); return !(e && e[0] == '0'); }();
    return on;
}

void matmul(const Mat &W, const float *x, int64_t m, float *y) {
    const int64_t rows = W.rows, cols = W.cols;
    if ((W.type == WType::Q8 || W.type == WType::Q4) && (cols % 32)) fail(VOX_ERR_FORMAT, "Q8/Q4 matrix needs cols % 32 == 0");
    if (W.gpu >= 0 && gpu_matmul(W, x, m, y)) return;
    // Q8: при малом m (декодирование) — целочисленные скалярные произведения со входом, квантованным в int8;
    // при большом m (префилл, DiT, вокодер) — плиточный путь с деквантованием весов и входом float: он быстрее (блок x держится
    // в кэше) и точнее (квантуются только веса, как в шейдере GPU).
    if (W.type == WType::Q8 && m <= kQ8IntMaxRows) { matmul_q8(W, x, m, y); return; }
    if (W.type == WType::Q8) {   // развернуть веса во float один раз (это ~1/m от работы самого умножения) и считать как f32
        thread_local std::vector<float> wf_buf;
        wf_buf.resize(size_t(rows) * size_t(cols));
        float *wf = wf_buf.data();   // в лямбде нельзя трогать thread_local: она выполняется в потоках пула
        parallel_for(rows, 16, [&](int64_t r0, int64_t r1) {
            const int8_t *q = reinterpret_cast<const int8_t *>(W.data);
            for (int64_t r = r0; r < r1; ++r)
                for (int64_t b = 0; b < cols / 32; ++b) {
                    const float sc = W.scale[r * (cols / 32) + b];
                    const int8_t *src = q + r * cols + b * 32;
                    float *dst = &wf[size_t(r * cols + b * 32)];
                    for (int j = 0; j < 32; ++j) dst[j] = float(src[j]) * sc;
                }
        });
        Mat F = W;
        F.type = WType::F32; F.data = reinterpret_cast<const uint8_t *>(wf); F.scale = nullptr; F.gpu = -1;
        matmul(F, x, m, y);
        return;
    }
    if (W.type == WType::Q4) { matmul_q4(W, x, m, y); return; }
    // Много строк входа (префилл, DiT, свёртки вокодеров): делим работу и по m, и по строкам весов. Деление только по строкам
    // весов заставляет каждый поток заново читать весь вход, и при малом числе строк весов на поток (свёртки Kokoro: 128–256
    // каналов) упирается в память: на 2 ядрах свёртки генератора Kokoro шли в 1,47 раза быстрее одного ядра, с 2D-делением — в 2,08.
    // Задача = 64 строки входа × ≤ 256 строк весов; начала блоков кратны 64 и 256, поэтому плитки и порядок суммирования те же,
    // что в одном потоке, — результат бит в бит. VOX_MM2D=0 отключает (для сравнения).
    const int nt = num_threads();
    if (m >= 128 && nt > 1 && mm2d_enabled()) {
        constexpr int64_t MB = 64, RB = 256;
        const int64_t nm = (m + MB - 1) / MB, nr = (rows + RB - 1) / RB;
        if (nm * nr >= nt) {
            parallel_for(nm * nr, 1, [&](int64_t a, int64_t e) {
                for (int64_t t = a; t < e; ++t) {
                    const int64_t i0 = (t / nr) * MB, r0 = (t % nr) * RB;
                    matmul_range(W, x + i0 * cols, std::min(MB, m - i0), y + i0 * rows, r0, std::min(rows, r0 + RB));
                }
            });
            return;
        }
    }
    // зерно параллелизма: диапазон строк кратен 4, чтобы плитки не рвались между потоками
    const int64_t work_per_row = std::max<int64_t>(1, cols * m);
    int64_t grain = std::max<int64_t>(4, (65536 / work_per_row) * 4);
    grain = (grain + 3) / 4 * 4;
    parallel_for(rows, grain, [&](int64_t r0, int64_t r1) { matmul_range(W, x, m, y, r0, r1); });
}

void rmsnorm(const float *x, const float *w, int64_t n, float eps, float *out) {
    float ss = 0;
    for (int64_t i = 0; i < n; ++i) ss += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(ss / float(n) + eps);
    for (int64_t i = 0; i < n; ++i) out[i] = w[i] * (x[i] * inv);
}

void layernorm(const float *x, const float *w, const float *b, int64_t n, float eps, float *out) {
    double mean = 0;
    for (int64_t i = 0; i < n; ++i) mean += x[i];
    mean /= double(n);
    double var = 0;
    for (int64_t i = 0; i < n; ++i) { double d = x[i] - mean; var += d * d; }
    var /= double(n);
    const float inv = float(1.0 / std::sqrt(var + double(eps)));
    for (int64_t i = 0; i < n; ++i) out[i] = (x[i] - float(mean)) * inv * w[i] + (b ? b[i] : 0.0f);
}

void softmax_inplace(float *x, int64_t n) {
    float mx = -INFINITY;
    for (int64_t i = 0; i < n; ++i) mx = std::max(mx, x[i]);
    float sum = 0;
    for (int64_t i = 0; i < n; ++i) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
    const float inv = 1.0f / sum;
    for (int64_t i = 0; i < n; ++i) x[i] *= inv;
}

std::vector<float> rope_inv_freq(int64_t hd, double theta) {
    std::vector<float> f(size_t(hd / 2));
    for (int64_t i = 0; i < hd / 2; ++i) f[size_t(i)] = float(1.0 / std::pow(theta, double(2 * i) / double(hd)));
    return f;
}

void rope_apply(float *x, int64_t heads, int64_t hd, const float *inv_freq, int64_t pos) {
    const int64_t half = hd / 2;
    thread_local std::vector<float> cs, sn;
    if (int64_t(cs.size()) < half) { cs.resize(size_t(half)); sn.resize(size_t(half)); }
    for (int64_t i = 0; i < half; ++i) {
        float f = inv_freq[i] * float(pos);
        cs[size_t(i)] = std::cos(f);
        sn[size_t(i)] = std::sin(f);
    }
    for (int64_t h = 0; h < heads; ++h) {
        float *p = x + h * hd;
        for (int64_t i = 0; i < half; ++i) {
            float a = p[i], b = p[i + half];
            p[i] = a * cs[size_t(i)] - b * sn[size_t(i)];
            p[i + half] = b * cs[size_t(i)] + a * sn[size_t(i)];
        }
    }
}

// ====================== transformer ======================
void transformer_forward(const TransformerW &w, KVCache &kv, const float *x_in, int64_t m, float *out_last,
                         float *out_all) {
    const int64_t H = w.hidden, nh = w.heads, nkv = w.kv_heads, hd = w.head_dim, I = w.inter;
    const int64_t qd = nh * hd, kd = nkv * hd, group = nh / nkv;
    // гибрид: первые gpu_layers слоёв (и их KV) в видеопамяти, остальные считает CPU со своим KV
    const bool hybrid = w.gpu_id >= 0 && w.gpu_layers >= 0 && w.gpu_layers < int(w.layers.size()) && kv.gpu_seq >= 0;
    if (w.gpu_id >= 0 && kv.gpu_seq >= 0 && !out_all && !hybrid) {
        if (!gpu_xfmr_forward(w, kv.gpu_seq, kv.len, x_in, m, out_last)) fail(VOX_ERR_DEVICE, "GPU transformer forward failed");
        kv.len += m;
        return;
    }
    const size_t first_layer = hybrid ? size_t(w.gpu_layers) : 0;
    if (kv.k.size() != w.layers.size()) kv.init(int(w.layers.size()));
    const int64_t pos0 = kv.len;
    if (w.window > 0) {  // скользящее окно: старые ключи, которые уже никому не нужны, выбрасываем
        const int64_t kd0 = nkv * hd, stored = kv.len - kv.base, keep = w.window - 1;
        if (stored > keep) {
            const int64_t drop = stored - keep;
            for (size_t li = 0; li < kv.k.size(); ++li) {
                kv.k[li].erase(kv.k[li].begin(), kv.k[li].begin() + drop * kd0);
                kv.v[li].erase(kv.v[li].begin(), kv.v[li].begin() + drop * kd0);
            }
            kv.base += drop;
        }
    }
    const float scale = 1.0f / std::sqrt(float(hd));

    std::vector<float> h(x_in, x_in + m * H), xn(size_t(m * H)), q(size_t(m * qd)), k(size_t(m * kd)),
        v(size_t(m * kd)), ao(size_t(m * qd)), o(size_t(m * H)), g(size_t(m * I)), u(size_t(m * I)), d(size_t(m * H));
    if (hybrid && !gpu_xfmr_forward_hidden(w, kv.gpu_seq, pos0, x_in, m, h.data())) fail(VOX_ERR_DEVICE, "GPU transformer forward failed");

    for (size_t li = first_layer; li < w.layers.size(); ++li) {
        const LayerW &L = w.layers[li];
        for (int64_t i = 0; i < m; ++i) rmsnorm(&h[size_t(i * H)], L.ln1, H, w.eps, &xn[size_t(i * H)]);
        matmul(L.q, xn.data(), m, q.data());
        matmul(L.k, xn.data(), m, k.data());
        matmul(L.v, xn.data(), m, v.data());
        for (int64_t i = 0; i < m; ++i) {
            float *qi = &q[size_t(i * qd)], *ki = &k[size_t(i * kd)];
            if (L.qn) for (int64_t hh = 0; hh < nh; ++hh) rmsnorm(qi + hh * hd, L.qn, hd, w.eps, qi + hh * hd);
            if (L.kn) for (int64_t hh = 0; hh < nkv; ++hh) rmsnorm(ki + hh * hd, L.kn, hd, w.eps, ki + hh * hd);
            rope_apply(qi, nh, hd, w.inv_freq.data(), pos0 + i);
            rope_apply(ki, nkv, hd, w.inv_freq.data(), pos0 + i);
        }
        auto &K = kv.k[li], &V = kv.v[li];
        K.insert(K.end(), k.begin(), k.end());
        V.insert(V.end(), v.begin(), v.end());

        const int64_t attn_work = m * nh * (pos0 + m) * hd;
        parallel_for(m * nh, attn_work < 300000 ? m * nh : 1, [&](int64_t t0, int64_t t1) {
            std::vector<float> sc;
            for (int64_t t = t0; t < t1; ++t) {
                const int64_t i = t / nh, hh = t % nh, gk = hh / group;
                const int64_t last = pos0 + i;                       // последний видимый ключ
                const int64_t first = w.window > 0 ? std::max<int64_t>(0, last - w.window + 1) : 0;
                const int64_t n = last - first + 1;
                sc.resize(size_t(n));
                const float *qh = &q[size_t(i * qd + hh * hd)];
                for (int64_t j = 0; j < n; ++j) sc[size_t(j)] = scale * dot(qh, &K[size_t((first + j - kv.base) * kd + gk * hd)], hd);
                softmax_inplace(sc.data(), n);
                float *oh = &ao[size_t(i * qd + hh * hd)];
                for (int64_t e = 0; e < hd; ++e) oh[e] = 0;
                for (int64_t j = 0; j < n; ++j) {
                    const float p = sc[size_t(j)];
                    const float *vj = &V[size_t((first + j - kv.base) * kd + gk * hd)];
                    for (int64_t e = 0; e < hd; ++e) oh[e] += p * vj[e];
                }
            }
        });
        matmul(L.o, ao.data(), m, o.data());
        if (L.attn_scale) for (int64_t i = 0; i < m; ++i) for (int64_t e = 0; e < H; ++e) o[size_t(i * H + e)] *= L.attn_scale[e];
        for (int64_t i = 0; i < m * H; ++i) h[size_t(i)] += o[size_t(i)];

        for (int64_t i = 0; i < m; ++i) rmsnorm(&h[size_t(i * H)], L.ln2, H, w.eps, &xn[size_t(i * H)]);
        matmul(L.gate, xn.data(), m, g.data());
        matmul(L.up, xn.data(), m, u.data());
        for (int64_t i = 0; i < m * I; ++i) g[size_t(i)] = silu(g[size_t(i)]) * u[size_t(i)];
        matmul(L.down, g.data(), m, d.data());
        if (L.mlp_scale) for (int64_t i = 0; i < m; ++i) for (int64_t e = 0; e < H; ++e) d[size_t(i * H + e)] *= L.mlp_scale[e];
        for (int64_t i = 0; i < m * H; ++i) h[size_t(i)] += d[size_t(i)];
    }
    kv.len += m;
    if (out_last) rmsnorm(&h[size_t((m - 1) * H)], w.final_norm, H, w.eps, out_last);
    if (out_all) for (int64_t i = 0; i < m; ++i) rmsnorm(&h[size_t(i * H)], w.final_norm, H, w.eps, out_all + i * H);
}

bool transformer_step_fused(const TransformerW &w, KVCache &kv, const float *x, int64_t m, const Mat *in_proj, const Mat *head,
                            float *out_last, float *head_out) {
    if (w.gpu_id < 0 || kv.gpu_seq < 0) return false;
    if (w.gpu_layers >= 0 && w.gpu_layers < int(w.layers.size())) return false;  // гибрид: проход идёт через transformer_forward
    if ((in_proj && in_proj->gpu < 0) || (head && head->gpu < 0)) return false;
    if (!gpu_xfmr_forward(w, kv.gpu_seq, kv.len, x, m, out_last, in_proj, head, head_out)) fail(VOX_ERR_DEVICE, "GPU transformer forward failed");
    kv.len += m;
    return true;
}

void quantize_q8(const float *w, int64_t rows, int64_t cols, std::vector<int8_t> &q, std::vector<float> &scale) {
    q.assign(size_t(rows * cols), 0);
    scale.assign(size_t(rows * (cols / 32)), 0.f);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t b = 0; b < cols / 32; ++b) {
            float mx = 0;
            for (int j = 0; j < 32; ++j) mx = std::max(mx, std::fabs(w[r * cols + b * 32 + j]));
            const float s = mx / 127.0f;
            scale[size_t(r * (cols / 32) + b)] = s;
            for (int j = 0; j < 32; ++j) q[size_t(r * cols + b * 32 + j)] = s > 0 ? int8_t(std::lround(w[r * cols + b * 32 + j] / s)) : 0;
        }
}

}  // namespace vox
