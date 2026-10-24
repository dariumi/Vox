#pragma once
// GPU-бэкенд (Vulkan): веса лежат в видеопамяти, matmul() для матриц с Mat::gpu >= 0 считается на GPU.
#include <string>
#include <vector>

#include "kernels.hpp"

namespace vox {
// device: "vulkan", "vulkan:N" (индекс устройства, либо переменная VOX_VK_DEVICE). Ошибка → false и текст в err.
bool gpu_init(const std::string &device, std::string &err);
bool gpu_active();
std::string gpu_device_name();
// Загружает матрицу в видеопамять; -1, если не вышло (тогда она считается на CPU).
int gpu_upload(const Mat &W);
// Учёт загрузки: сколько матриц/байт ушло в видеопамять, сколько не влезло и почему.
struct GpuStats { int64_t mats_ok = 0, mats_failed = 0; uint64_t bytes = 0, bytes_failed = 0; std::string last_error; };
GpuStats gpu_stats();
// Список устройств Vulkan (индекс, имя, тип, VRAM) — для `vox devices`.
std::string gpu_list_devices();
// Устройства Vulkan с характеристиками, нужными планировщику ресурсов. vram_free — сколько видеопамяти ещё доступно
// этому процессу (VK_EXT_memory_budget: бюджет за вычетом того, что заняли другие процессы), без расширения = vram_total.
struct GpuDeviceInfo {
    int index = 0;
    std::string name, type;          // type: discrete | integrated | virtual | cpu | other
    bool software = false;           // программный рендерер (llvmpipe/lavapipe): считает на CPU, в «auto» не берётся
    bool unified_memory = false;     // видеопамять — та же оперативная (iGPU, телефоны): расход идёт из RAM
    uint64_t vram_total = 0, vram_free = 0, max_storage_range = 0;
    bool subgroup_arith = false, coop_matrix = false, memory_budget = false;
    uint32_t api_version = 0, vendor_id = 0, driver_version = 0;
};
std::vector<GpuDeviceInfo> gpu_enumerate();
// Лимит видеопамяти, которую vox выделяет на активном устройстве (0 = без лимита). Выделение сверх лимита —
// ошибка (матрица остаётся на CPU, в strict-режиме — ошибка загрузки). Действует после gpu_init.
void gpu_set_vram_limit(uint64_t bytes);
struct GpuMemInfo { uint64_t used = 0, limit = 0, device_free = 0, device_total = 0; int device_index = -1; };
GpuMemInfo gpu_mem_info();
// Трансформер целиком на GPU: все матрицы слоёв уже загружены (Mat::gpu >= 0). -1 → считать на CPU.
// n_layers < 0 — все слои, иначе только первые n_layers (остальные считает CPU). uploaded вызывается для каждой матрицы
// сразу после её загрузки (чтобы вернуть системе страницы mmap и не раздувать пиковый RSS).
int gpu_xfmr_create(const TransformerW &w, const char *name = "xfmr", int n_layers = -1, const std::function<void(const Mat &)> &uploaded = nullptr);
// Последовательность = KV-кэш в видеопамяти на capacity (<= 3072) позиций; -1 при неудаче.
int gpu_seq_create(int xfmr, int64_t capacity);
void gpu_seq_free(int seq);

// Декодер кодека Qwen3-TTS целиком на GPU: все свёртки, транспонированные свёртки, Snake, ConvNeXt и трансформер
// со скользящим окном пишутся в один командный буфер на вызов. -1 → не поддерживается (ошибка в gpu_stats().last_error).
// Весь цикл code predictor одного кадра (кодбуки 1..heads.size()) одной отправкой: проекция → трансформер → голова →
// сэмплинг на GPU (тот же алгоритм и Philox, что vox::sample_logits) → эмбеддинг выбранного кода → следующий проход.
// in2: [2][in_proj ? in_proj->cols : hidden] — скрытое состояние talker и эмбеддинг кода 0. false → считать по проходам.
struct GpuSampleCfg { float temperature = 0.9f, top_p = 1.0f; int32_t top_k = 50; uint32_t key0 = 0, key1 = 0; uint64_t frame = 0; };
bool gpu_pred_frame(const TransformerW &w, int seq, const float *in2, const Mat *in_proj, const std::vector<Mat> &heads,
                    const std::vector<Mat> &embs, const GpuSampleCfg &cfg, int32_t *codes);

class Qwen3Weights;
int gpu_codec_create(const Qwen3Weights &w);
int gpu_codec_state_new(int codec);
int gpu_codec_state_clone(int state);
void gpu_codec_state_free(int state);
// z: [F][codebook_dim] выход квантизатора (считается на CPU); добавляет F * samples_per_frame отсчётов в pcm.
bool gpu_codec_forward(int codec, int state, const float *z, int F, std::vector<float> &pcm);
constexpr int kGpuCodecMaxFrames = 16;

// F5-TTS целиком на GPU: шаг DiT (вход → свёрточное позиционное кодирование → блоки с AdaLN, flash-вниманием, FF → выход)
// одной отправкой и вокодер Vocos (без iSTFT). Все линейные слои уже загружены (Mat::gpu >= 0); свёртки — f32 в раскладке
// [cout][K * cin/groups] (K-мажорно внутри группы, как Conv1d::w).
struct GpuF5Conv { const float *w = nullptr, *b = nullptr; int cin = 0, cout = 0, K = 1, groups = 1; };
struct GpuF5Desc {
    int dim = 0, depth = 0, heads = 0, dim_head = 0, ff_mult = 2, mel_dim = 100, text_dim = 0, pe_attn_head = 0;
    int in_proj = -1, norm_out = -1, proj_out = -1;
    struct Block { int ada = -1, q = -1, k = -1, v = -1, o = -1, ff1 = -1, ff2 = -1; };
    std::vector<Block> blocks;
    GpuF5Conv pos1, pos2;
    // Vocos (vocos_dim == 0 — вокодер на CPU)
    int vocos_dim = 0, vocos_inter = 0, vocos_out = 0;
    GpuF5Conv v_embed;
    const float *v_ln_w = nullptr, *v_ln_b = nullptr, *v_fln_w = nullptr, *v_fln_b = nullptr;
    struct VBlock { GpuF5Conv dw; const float *ln_w = nullptr, *ln_b = nullptr, *gamma = nullptr; int pw1 = -1, pw2 = -1; };
    std::vector<VBlock> v_blocks;
    int v_head = -1;
};
// -1 → не поддерживается (причина в gpu_stats().last_error), считать на CPU.
int gpu_f5_create(const GpuF5Desc &d);
// Начало сэмплирования: B последовательностей длины N; cat0 — [B][N][2*mel + text] вход DiT с нулями на месте x
// (условие и текст не меняются между шагами), rope — [N][dim_head/2][2] (cos, sin).
bool gpu_f5_begin(int id, int B, int N, const float *cat0, const float *rope);
// Шаг: x [N][mel] (одинаков для всех B), ts [dim] — SiLU(эмбеддинг времени). out: [B][N][mel].
bool gpu_f5_step(int id, const float *x, const float *ts, float *out);
// Vocos: mel [T][mel] → выход головы [T][vocos_out] (лог-модуль и фаза; iSTFT считает CPU).
bool gpu_f5_vocos(int id, const float *mel, int T, std::vector<float> &head);
void gpu_f5_end(int id);   // освободить буферы шага (веса остаются)

// Which GEMM the F5 / flow graphs use for many-row linear layers: the 64×64 convgemm tile or the register-blocked gemmb.
// VOX_F5_GEMM=old / new forces one; otherwise both are timed on the model's own shapes the first time a graph is built on a
// device (a software renderer keeps convgemm unless VOX_F5_GEMM=auto) and the faster is kept for the process.
// Returns "convgemm" / "gemmb" (+ " (timed)"), "" before any graph was built.
std::string gpu_gemm_choice();
// A choice measured earlier (vox tune): "gemmb" / "convgemm"; ignored when VOX_F5_GEMM is set or a choice was already made.
void gpu_gemm_preset(const std::string &choice);

// The flow estimator of CosyVoice2 / Chatterbox (CausalConditionalDecoder) on the GPU: one Euler step — the conditional and
// the unconditional pass (B = 2 sequences of T frames) — in ONE command buffer. Convolutions are f32 K-major [cout][K*cin]
// (Conv::w), linear layers uploaded matrices (Mat::gpu with their biases). Mirrors cosyvoice.cpp's estimator.
struct GpuCosyDesc {
    int M = 80, C = 256, heads = 8, head_dim = 64;
    struct Res { GpuF5Conv c1, c2, res; const float *n1w = nullptr, *n1b = nullptr, *n2w = nullptr, *n2b = nullptr; };
    struct X { const float *n1w = nullptr, *n1b = nullptr, *n3w = nullptr, *n3b = nullptr; int q = -1, k = -1, v = -1, o = -1, ff1 = -1, ff2 = -1, ff_dim = 0; };
    Res down_res, up_res;
    std::vector<Res> mid_res;
    std::vector<X> down_x, up_x;
    std::vector<std::vector<X>> mid_x;
    GpuF5Conv down_conv, up_out, final_conv, final_proj;
    const float *fn_w = nullptr, *fn_b = nullptr;
};
int gpu_cosy_create(const GpuCosyDesc &d);   // -1 → not available (reason in gpu_stats().last_error)
// in0 [2][T][4*M]: the estimator input [x | mu | spks | cond] of both passes with the x columns left zero
bool gpu_cosy_begin(int id, int T, const float *in0);
// x [T][M] (the same for both passes), tb [resnets][C] the time-embedding bias of every resnet (down, mid…, up) → out [2][T][M]
bool gpu_cosy_step(int id, const float *x, const float *tb, float *out);
void gpu_cosy_end(int id);
}  // namespace vox
