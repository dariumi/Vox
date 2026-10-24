#pragma once
// Вычислительные ядра (CPU, float32). Веса могут лежать в F32/F16/BF16/Q8 и читаются
// прямо из mmap; вычисления всегда в float32.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "common.hpp"

namespace vox {

// ---------- потоки ----------
void set_num_threads(int n);   // размер общего пула (его используют вызовы вне PoolScope); 0 = все ядра
int num_threads();             // размер активного пула
// Настройка рабочих потоков пула: привязка к ядрам (рабочий i → ядро affinity[i % n]), приоритет (nice, Linux),
// ожидание заданий активным кручением (быстрее) или сном (не греет ядро, когда работы нет).
struct CpuTuning {
    std::vector<int> affinity;
    int nice = 0;
    bool spin = true;
};
class ThreadPool;
std::shared_ptr<ThreadPool> make_thread_pool(int threads, const CpuTuning &tune = {});
// Внутри PoolScope parallel_for этого потока использует указанный пул (nullptr — оставляет текущий).
class PoolScope {
public:
    explicit PoolScope(ThreadPool *p);
    ~PoolScope();
    PoolScope(const PoolScope &) = delete;
    PoolScope &operator=(const PoolScope &) = delete;

private:
    ThreadPool *prev_;
};
// fn(begin, end) вызывается для непересекающихся диапазонов, покрывающих [0, total)
void parallel_for(int64_t total, int64_t grain, const std::function<void(int64_t, int64_t)> &fn);

// ---------- веса ----------
enum class WType { F32, F16, BF16, Q8, Q4 };

// Матрица [rows][cols] (как torch.nn.Linear.weight: rows = out, cols = in).
// Q8: int8 [rows][cols] + масштабы f32 [rows][cols/32] (блоки по 32).
// Q4: 4 бита на вес, блоки по 32 в 16 байтах: байт j = (вес j) | (вес j+16) << 4, значение = (u - 8) * масштаб блока;
//     масштабы f32 [rows][cols/32]; cols кратно 32. Данные: rows * cols / 2 байт.
struct Mat {
    WType type = WType::F32;
    const uint8_t *data = nullptr;
    const float *scale = nullptr;
    int64_t rows = 0, cols = 0;
    const float *bias = nullptr;  // [rows] или nullptr
    int gpu = -1;                 // дескриптор весов на GPU (см. gpu.hpp), -1 = только CPU
    bool valid() const { return data != nullptr; }
};

// Сколько байт занимает строка матрицы в данных (без масштабов).
inline size_t mat_row_bytes(WType t, int64_t cols) {
    return t == WType::F32 ? size_t(cols) * 4 : t == WType::Q8 ? size_t(cols) : t == WType::Q4 ? size_t(cols) / 2 : size_t(cols) * 2;
}

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000u) << 16, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {  // субнормальное
            int e = -1;
            do { ++e; man <<= 1; } while (!(man & 0x400u));
            bits = sign | uint32_t(127 - 15 - e) << 23 | (man & 0x3FFu) << 13;
        }
    } else if (exp == 31) bits = sign | 0x7F800000u | man << 13;
    else bits = sign | (exp + 112) << 23 | man << 13;
    float f; std::memcpy(&f, &bits, 4); return f;
}
inline float bf16_to_f32(uint16_t h) { uint32_t b = uint32_t(h) << 16; float f; std::memcpy(&f, &b, 4); return f; }

// Копирует строку r матрицы в float (для F32 возвращает указатель без копии).
const float *mat_row(const Mat &W, int64_t r, float *tmp);
// Строка embedding-таблицы в float (any WType, без bias).
void mat_gather(const Mat &W, int64_t r, float *out);

// y[m][rows] = x[m][cols] · Wᵀ (+ bias)
void matmul(const Mat &W, const float *x, int64_t m, float *y);

float dot(const float *a, const float *b, int64_t n);

// Реализован в gpu_vulkan.cpp (или заглушкой, если собрано без Vulkan). false → считать на CPU.
bool gpu_matmul(const Mat &W, const float *x, int64_t m, float *y);
// Весь проход трансформера на GPU (нормы, RoPE, внимание, KV-кэш в видеопамяти). false → ошибка.
struct TransformerW;
bool gpu_xfmr_forward(const TransformerW &w, int seq, int64_t pos0, const float *x, int64_t m, float *out_last,
                      const Mat *in_proj = nullptr, const Mat *head = nullptr, float *head_out = nullptr);
// Только слои, загруженные на GPU (первые gpu_layers), без финальной нормы: возвращает скрытые состояния всех m строк
// (hidden_out[m][hidden]) — их дальше считает CPU. Гибридное размещение: часть слоёв в видеопамяти, остальные на процессоре.
bool gpu_xfmr_forward_hidden(const TransformerW &w, int seq, int64_t pos0, const float *x, int64_t m, float *hidden_out);

// ---------- поэлементные ----------
void rmsnorm(const float *x, const float *w, int64_t n, float eps, float *out);       // out может совпадать с x
void layernorm(const float *x, const float *w, const float *b, int64_t n, float eps, float *out);
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float gelu_erf(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f)); }
void softmax_inplace(float *x, int64_t n);

// RoPE (rotate_half): x[head][hd], позиция pos; inv_freq[hd/2]
void rope_apply(float *x, int64_t heads, int64_t hd, const float *inv_freq, int64_t pos);
std::vector<float> rope_inv_freq(int64_t hd, double theta);

// ---------- Transformer (Qwen3-подобный) ----------
struct LayerW {
    Mat q, k, v, o, gate, up, down;
    const float *qn = nullptr, *kn = nullptr;    // RMSNorm по head_dim (может не быть)
    const float *ln1 = nullptr, *ln2 = nullptr;
    const float *attn_scale = nullptr, *mlp_scale = nullptr;  // layer scale декодера кодека (может не быть)
};
struct TransformerW {
    std::vector<LayerW> layers;
    const float *final_norm = nullptr;
    int hidden = 0, heads = 0, kv_heads = 0, head_dim = 0, inter = 0;
    float eps = 1e-6f;
    double theta = 10000.0;
    int window = 0;  // 0 = полное внимание; иначе число видимых ключей (включая текущий)
    std::vector<float> inv_freq;
    int gpu_id = -1;  // дескриптор трансформера на GPU (gpu_xfmr_create), -1 = считать на CPU
    int gpu_layers = -1;  // сколько первых слоёв на GPU (остальные на CPU); -1 = все слои
};
struct KVCache {
    std::vector<std::vector<float>> k, v;  // по слоям: [хранится][kv_heads*head_dim]
    int64_t len = 0;   // сколько токенов уже обработано (абсолютная позиция следующего)
    int gpu_seq = -1;  // последовательность на GPU (KV-кэш в видеопамяти); при >= 0 CPU-кэш не используется
    int64_t base = 0;  // абсолютная позиция первого хранимого ключа (при скользящем окне > 0)
    void init(int layers) { k.assign(size_t(layers), {}); v.assign(size_t(layers), {}); len = base = 0; }
    void reset() { for (auto &a : k) a.clear(); for (auto &a : v) a.clear(); len = base = 0; }
};
// x[m][hidden] — входные эмбеддинги; считает все слои и кладёт KV. Позиции: kv.len .. kv.len+m-1.
// out_last[hidden] — нормированный скрытый вектор последнего токена (может быть nullptr);
// out_all[m][hidden] — нормированные векторы всех токенов (может быть nullptr).
void transformer_forward(const TransformerW &w, KVCache &kv, const float *x, int64_t m, float *out_last,
                         float *out_all = nullptr);
// One token for each of B independent sequences (their own KV caches, CPU only): the weight matrices are read once for all B
// (classifier-free guidance runs its two passes this way). x[B][hidden] → out[B][hidden] (normed), as B calls of
// transformer_forward with m = 1 would give.
void transformer_forward_steps(const TransformerW &w, KVCache *const *kv, int64_t B, const float *x, float *out);
// Один шаг «вход → [in_proj] → трансформер → норма последней строки → [head]» одной отправкой на GPU.
// x: [m][in_proj ? in_proj->cols : hidden]. Возвращает false, если трансформер не на GPU (тогда считать по частям).
bool transformer_step_fused(const TransformerW &w, KVCache &kv, const float *x, int64_t m, const Mat *in_proj, const Mat *head,
                            float *out_last, float *head_out);

// Квантование матрицы в Q4 (раскладка выше): масштаб = (значение с наибольшим |x|) / -8, как у Q4_0 в llama.cpp.
void quantize_q4(const float *w, int64_t rows, int64_t cols, std::vector<uint8_t> &q, std::vector<float> &scale);
// Квантование строки в Q8 (для конвертера и тестов): блоки по 32, масштаб = max|x|/127.
void quantize_q8(const float *w, int64_t rows, int64_t cols, std::vector<int8_t> &q, std::vector<float> &scale);

}  // namespace vox
