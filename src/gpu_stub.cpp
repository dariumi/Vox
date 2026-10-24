// Сборка без Vulkan: GPU недоступен, всё считается на CPU.
#include "gpu.hpp"
namespace vox {
bool gpu_matmul(const Mat &, const float *, int64_t, float *) { return false; }
bool gpu_init(const std::string &, std::string &err) { err = "vox was built without Vulkan support (VOX_VULKAN=OFF or Vulkan headers not found)"; return false; }
bool gpu_active() { return false; }
std::string gpu_device_name() { return ""; }
int gpu_upload(const Mat &) { return -1; }
GpuStats gpu_stats() { return {}; }
std::string gpu_list_devices() { return "(vox built without Vulkan)\n"; }
std::vector<GpuDeviceInfo> gpu_enumerate() { return {}; }
void gpu_set_vram_limit(uint64_t) {}
GpuMemInfo gpu_mem_info() { return {}; }
int gpu_xfmr_create(const TransformerW &, const char *, int, const std::function<void(const Mat &)> &) { return -1; }
int gpu_seq_create(int, int64_t) { return -1; }
void gpu_seq_free(int) {}
int gpu_codec_create(const Qwen3Weights &) { return -1; }
bool gpu_pred_frame(const TransformerW &, int, const float *, const Mat *, const std::vector<Mat> &, const std::vector<Mat> &,
                    const GpuSampleCfg &, int32_t *) { return false; }
int gpu_codec_state_new(int) { return -1; }
int gpu_codec_state_clone(int) { return -1; }
void gpu_codec_state_free(int) {}
bool gpu_codec_forward(int, int, const float *, int, std::vector<float> &) { return false; }
bool gpu_xfmr_forward(const TransformerW &, int, int64_t, const float *, int64_t, float *, const Mat *, const Mat *, float *) { return false; }
bool gpu_xfmr_forward_hidden(const TransformerW &, int, int64_t, const float *, int64_t, float *) { return false; }
bool gpu_xfmr_forward_pair(const TransformerW &, int, int, int64_t, const float *, const Mat *, float *, float *) { return false; }
int gpu_f5_create(const GpuF5Desc &) { return -1; }
bool gpu_f5_begin(int, int, int, const float *, const float *) { return false; }
bool gpu_f5_step(int, const float *, const float *, float *) { return false; }
bool gpu_f5_vocos(int, const float *, int, std::vector<float> &) { return false; }
void gpu_f5_end(int) {}
std::string gpu_gemm_choice() { return ""; }
void gpu_gemm_preset(const std::string &) {}
int gpu_cosy_create(const GpuCosyDesc &) { return -1; }
bool gpu_cosy_begin(int, int, const float *) { return false; }
bool gpu_cosy_step(int, const float *, const float *, float *) { return false; }
void gpu_cosy_end(int) {}
}  // namespace vox
