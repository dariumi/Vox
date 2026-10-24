#pragma once
// Internals of the Vulkan backend shared by gpu_vulkan.cpp (matmul / transformer) and gpu_codec.cpp (codec decoder).
#include <vulkan/vulkan.h>

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu.hpp"

namespace vox {
namespace vk {

#define VKC(x)                                                                                   \
    do {                                                                                         \
        VkResult r_ = (x);                                                                       \
        if (r_ != VK_SUCCESS) throw std::runtime_error(std::string(#x) + " failed: VkResult " + std::to_string(int(r_))); \
    } while (0)

struct GpuMat {
    VkBuffer w = VK_NULL_HANDLE, s = VK_NULL_HANDLE, b = VK_NULL_HANDLE;
    VkDeviceMemory mw = VK_NULL_HANDLE, ms = VK_NULL_HANDLE, mb = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    WType type = WType::F32;
    int64_t rows = 0, cols = 0;
    bool bias = false;
};

constexpr size_t kScratchFloats = 8u << 20;  // 32 МБ на вход и на выход; большие m режутся на куски
constexpr int64_t kMaxM = 256;               // токенов за один проход трансформера на GPU
constexpr int64_t kMaxCtx = 3072;            // предел контекста внимания (shared-память шейдера)
// 4 матмула (f32/f16/bf16/q8) + rmsnorm, qkrope, attention, silumul + декодер кодека: свёрточный GEMM (f32/f16/bf16),
// snake, depthwise conv, layernorm, overlap-add, история, внимание со скользящим окном, запись в кольцо
// + слитые QKV/SwiGLU, выборка эмбеддинга по индексу из буфера (f32/f16/bf16), сэмплинг на GPU
// + F5-TTS: flash-внимание, групповая свёртка, LayerNorm/AdaLN, поэлементные операции, RoPE
// + большой GEMM с регистровыми плитками 8×8 для многих строк входа (F5: DiT и Vocos), f32/f16/bf16/q8
constexpr int kNumPipes = 36;
// P_MM и P_GEMM — по одному конвейеру на тип весов (f32, f16, bf16, q8, q4)
enum Pipe { P_MM = 0, P_RMS = 5, P_ROPE = 6, P_ATTN = 7, P_SILU = 8, P_GEMM = 9, P_SNAKE = 14, P_DW = 15, P_LN = 16,
            P_OLA = 17, P_HIST = 18, P_ATTNW = 19, P_RING = 20, P_QKV = 21, P_SWIGLU = 22, P_SAMPLE = 23, P_GATHER = 24,
            P_F5ATTN = 27, P_F5CONV = 28, P_F5LN = 29, P_F5EW = 30, P_F5ROPE = 31, P_GEMMB = 32 };

struct GpuXfmr {
    std::string name;
    int layers = 0, H = 0, heads = 0, kv = 0, hd = 0, I = 0;
    float eps = 1e-6f;
    std::vector<float> inv_freq;
    VkBuffer norms = VK_NULL_HANDLE;
    VkDeviceMemory nm = VK_NULL_HANDLE;
    // qkv = rows of q, k, v stacked; gu = rows of gate, up stacked (one GEMV each instead of three / two)
    struct L { int qkv, o, gu, down; size_t ln1, ln2, qn, kn; bool qk = false; };
    std::vector<L> L_;
    size_t final_off = 0;
};
struct GpuSeq {
    std::vector<VkBuffer> k, v;
    std::vector<VkDeviceMemory> km, vm;
    int64_t cap = 0;
    bool used = false;
};

struct Ctx {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue q = VK_NULL_HANDLE;
    uint32_t qf = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkShaderModule sm = VK_NULL_HANDLE, extra_modules[22] = {};
    VkPipeline pipe[kNumPipes] = {};          // compiled on first use (pipe_get)
    VkShaderModule pipe_mod[kNumPipes] = {};  // module and specialization constant (WTYPE; -1 = none) of each pipeline
    int pipe_spec[kNumPipes] = {};
    VkDescriptorPool dyn = VK_NULL_HANDLE;
    VkBuffer csb = VK_NULL_HANDLE;
    VkDeviceMemory csm = VK_NULL_HANDLE;
    float *csp = nullptr;
    // рабочие буферы трансформера (device-local), общие для всех последовательностей
    VkBuffer sh = VK_NULL_HANDLE, sxn = VK_NULL_HANDLE, sq = VK_NULL_HANDLE, sao = VK_NULL_HANDLE, sg = VK_NULL_HANDLE, su = VK_NULL_HANDLE;
    VkDeviceMemory mh = VK_NULL_HANDLE, mxn = VK_NULL_HANDLE, mq = VK_NULL_HANDLE, mao = VK_NULL_HANDLE, mg = VK_NULL_HANDLE, mup = VK_NULL_HANDLE;
    size_t scratch_h = 0, scratch_q = 0, scratch_i = 0;
    std::vector<GpuXfmr> xf;
    std::vector<GpuSeq> seqs;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkBuffer xb = VK_NULL_HANDLE, yb = VK_NULL_HANDLE, dummy = VK_NULL_HANDLE, stage = VK_NULL_HANDLE;
    VkDeviceMemory xm = VK_NULL_HANDLE, ym = VK_NULL_HANDLE, dm = VK_NULL_HANDLE, stm = VK_NULL_HANDLE;
    float *xp = nullptr, *yp = nullptr;
    void *stp = nullptr;
    VkDeviceSize stage_size = 0;
    VkPhysicalDeviceMemoryProperties mp{};
    uint64_t max_range = 0;
    std::vector<GpuMat> mats;
    std::mutex mu;
    std::string name, init_error;
    bool ok = false, warned = false;
    int dev_index = -1;
    // учёт видеопамяти vox: каждое device-local выделение через make_buffer (освобождение — dev_mem_free)
    uint64_t vram_limit = 0, vram_used = 0;
    std::unordered_map<VkDeviceMemory, uint64_t> dev_mem;
    bool subgroups = false;
    uint32_t subgroup_size = 0;
    GpuStats stats;
    // VOX_PROFILE=gpu: метки времени GPU после каждой отправки шейдера → сумма по «стадия/шейдер»
    bool prof = false;
    VkQueryPool qpool = VK_NULL_HANDLE;
    uint32_t qn = 0, qcap = 0;
    double ts_ns = 1.0;
    std::string prof_stage = "-";
    std::vector<std::string> qlabels;
    struct ProfAcc { double ns = 0; uint64_t calls = 0; };
    std::vector<std::pair<std::string, ProfAcc>> prof_acc;
    double submits = 0;
    // code predictor loop on the GPU: codes, logits and RoPE table for positions 0..63
    VkBuffer pcodes = VK_NULL_HANDLE, plog = VK_NULL_HANDLE, pin = VK_NULL_HANDLE, pcs = VK_NULL_HANDLE;
    VkDeviceMemory mpcodes = VK_NULL_HANDLE, mplog = VK_NULL_HANDLE, mpin = VK_NULL_HANDLE, mpcs = VK_NULL_HANDLE;
    int32_t *pcodes_p = nullptr;
    float *pcs_p = nullptr;
    ~Ctx() { shutdown(); }
    void shutdown();
};
extern Ctx g;

uint32_t find_mem(uint32_t bits, VkMemoryPropertyFlags want);
void make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer &buf, VkDeviceMemory &mem);
void dev_mem_free(VkDeviceMemory m);  // вычитает из учёта и освобождает device-local память
void submit_wait(VkCommandBuffer cmd);
void begin_cmd();
void upload_buffer(const void *data, VkDeviceSize bytes, VkBuffer &buf, VkDeviceMemory &mem);
uint32_t vec_flag(const GpuMat &m);
VkDescriptorSet dyn_set(const VkBuffer (&b)[6]);
void barrier_cc();
// y (+)= W x for m rows of the uploaded matrix mat (GEMV for m <= 8, tiled GEMM otherwise); see gpu_vulkan.cpp.
void rec_mm(int mat, VkBuffer in, VkBuffer out, int64_t m, bool accum, int64_t y_off = 0);
void host_barrier();
void disp(int pipe, const VkBuffer (&b)[6], const uint32_t *pc, uint32_t pc_words, uint32_t gx, uint32_t gy);
void disp3(int pipe, const VkBuffer (&b)[6], const uint32_t *pc, uint32_t pc_words, uint32_t gx, uint32_t gy, uint32_t gz);
uint32_t fbits(float f);
VkPipeline pipe_get(int i);   // the pipeline, compiled now if this is its first use (call with g.mu held)
// Метка стадии для профайлера GPU (VOX_PROFILE=gpu); действует до следующего вызова.
inline void prof_stage(const char *s) { if (g.prof) g.prof_stage = s; }

}  // namespace vk
}  // namespace vox
