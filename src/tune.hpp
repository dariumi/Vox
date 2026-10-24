#pragma once
// Self-tuning: which device, stage placement, thread count and GPU GEMM this model runs fastest with on this computer.
//
// `vox tune` (or vox.tune() in Python, or the Studio) measures every candidate configuration of a model in a child process
// (one Vulkan device per process), keeps the fastest and writes it to a small cache file. Later, an engine opened with the
// default resources ("auto": no device, threads or placement given) looks the model and the hardware up there and opens
// with the measured configuration. A different computer, another GPU or driver, or a rebuilt model are different keys.
//   cache: $VOX_TUNE_CACHE, else $XDG_CACHE_HOME/vox/tune.json, else ~/.cache/vox/tune.json (%LOCALAPPDATA%\vox on Windows)
//   VOX_TUNE=0 or {"tuned": false} in the resources: ignore the cache.
#include <string>

#include "common.hpp"
#include "formats.hpp"

namespace vox {

std::string tune_cache_path();
// What the hardware is: CPU brand and logical cores, RAM (GiB), every non-software GPU with its driver version.
json tune_hardware();
std::string tune_hw_key();
// The model: weights hash, precision, family, variant, size of its blobs.
std::string tune_model_key(const TmodFile &f);
json tune_entry(const TmodFile &f);              // the stored result for this model on this hardware, or null
void tune_store(const TmodFile &f, json entry);  // adds the keys, the hardware and the time; replaces an older one
bool tune_forget(const TmodFile &f);
// Configurations worth measuring: the CPU (all threads; half of them when there are SMT siblings), every non-software GPU
// with the default placement and with each GPU-capable stage moved to the CPU. [{"label", "device"}].
json tune_candidates(const TmodFile &f);

}  // namespace vox
