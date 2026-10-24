#pragma once
#include <algorithm>

#include "backend.hpp"

namespace vox {
std::string today_iso();
json voice_to_json(const VoiceData &v);
// One built voice inside a safetensors file, its tensors under a name prefix ("" = the 1.x layout, "c3." = a profile cache entry).
void voice_put(SafetensorsFile &f, const VoiceData &v, const std::string &prefix, bool with_ref_audio);
VoiceData voice_get(const SafetensorsFile &f, const json &meta, const std::string &prefix);
// 1.x single-voice file (kept for tools; vox_voice_save writes profiles, see profile.hpp)
void voice_save(const VoiceData &v, const std::string &path);
// Any .tvoice (1.x or a 0.15 profile) → its default voice as built for the model it was made with.
VoiceData voice_load(const std::string &path);
}  // namespace vox
