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
}  // namespace vox
