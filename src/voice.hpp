#pragma once
#include <algorithm>

#include "backend.hpp"

namespace vox {
std::string today_iso();
json voice_to_json(const VoiceData &v);
void voice_save(const VoiceData &v, const std::string &path);
VoiceData voice_load(const std::string &path);
}  // namespace vox
