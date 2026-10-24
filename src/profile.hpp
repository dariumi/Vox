#pragma once
// Voice profiles (.tvoice since 0.15): several voices, each with styles (emotions / intonation), recordings that any cloning model can
// rebuild the voice from, built-in speakers to use on models that cannot clone, and a cache of voices already built for concrete models.
//
// File layout (safetensors, so 1.x readers keep working):
//   metadata "format"  = "1.1" when the profile has a built default voice: "voice" + top-level tensors are that voice in the 1.0 layout,
//                        so vox ≤ 0.14 and vox.js open it as an ordinary voice; otherwise "2.0" (older runtimes refuse it cleanly)
//   metadata "profile" = JSON (see Profile::to_json): voices, styles, recordings, cache index
//   tensors  "ref.<id>"    recordings (WAV bytes); the recording of the legacy view is the top-level "ref_audio"
//            "c<N>.<name>" cache entry N (speaker_emb, ref_codes, feat.*, kv.* …); the legacy entry has no prefix
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "backend.hpp"

namespace vox {

struct StyleDef {
    std::string ref;          // recording id ("" = the voice's default-style recording)
    std::string instruct;     // style instruction for models that take one (Qwen3 CustomVoice 1.7B, VoiceDesign)
    json options = json::object();   // generation defaults of this style: speed, sampling, model extras
    std::string description;
};

struct VoiceDef {
    std::string description;  // what the voice sounds like: the VoiceDesign instruction, and documentation for people
    std::string language = "auto";
    std::string default_style = "default";
    std::map<std::string, StyleDef> styles;
    std::map<std::string, std::string> speakers;   // "kokoro" / "qwen3-tts/custom_voice" → built-in speaker name (Kokoro: "a,b" blend)
    json options = json::object();   // generation defaults for the whole voice
};

struct Recording {
    std::string text;         // what is said in it ("" = unknown: only x-vector style cloning)
    std::string tensor;       // where the WAV bytes live in the file
    std::vector<uint8_t> wav;
    std::string language;
};

struct CacheEntry {
    std::string voice, style;
    std::shared_ptr<const VoiceData> data;   // as built for data->model_family / weights_sha256 (/ variant for speaker voices)
};

struct Profile {
    std::string name = "profile", created, default_voice;
    std::map<std::string, VoiceDef> voices;
    std::map<std::string, Recording> recordings;
    std::vector<CacheEntry> cache;

    json to_json(bool with_cache_details = true) const;
    void apply_json(const json &j);                   // the description part (voices, styles, names), not recordings or cache
    const VoiceDef &voice(const std::string &name) const;
    std::string add_recording(std::vector<uint8_t> wav, const std::string &text, const std::string &language);
    std::string unique_ref_id() const;
    void drop_unused_recordings();
};

// A profile shared by a vox_voice handle and the sessions made from it (sessions build styles lazily, under the mutex).
struct ProfileRef {
    std::mutex mu;
    Profile p;
};

Profile profile_load(const std::string &path);
void profile_save(const Profile &p, const std::string &path);
// One built voice (1.x semantics) → a profile with one voice and one style, the voice as its cache entry.
Profile profile_from_voice(const VoiceData &v, const std::string &voice_name);
// RFC 7386 JSON merge patch on the description (voices / styles / speakers / options / name / default_voice); null removes.
void profile_patch(Profile &p, const json &patch);
std::string profile_key_lower(std::string s);

}  // namespace vox
