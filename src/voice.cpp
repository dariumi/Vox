// .tvoice = safetensors. Метаданные: "format" = "1.0", "voice" = JSON-строка профиля.
#include "voice.hpp"

#include <ctime>

namespace vox {

static constexpr const char *kTvoiceFormat = "1.0";

std::string today_iso() {
    std::time_t t = std::time(nullptr);
    char buf[16];
    std::strftime(buf, sizeof buf, "%Y-%m-%d", std::gmtime(&t));
    return buf;
}

json voice_to_json(const VoiceData &v) {
    json kv = json::array();
    for (auto &[k, _] : v.kv_cache) {
        // "kv.<fingerprint>.<name>" → fingerprint
        auto a = k.find('.'), b = k.rfind('.');
        if (a != std::string::npos && b > a) {
            std::string fp = k.substr(a + 1, b - a - 1);
            if (std::find(kv.begin(), kv.end(), fp) == kv.end()) kv.push_back(fp);
        }
    }
    return {
        {"format", kTvoiceFormat},
        {"name", v.name},
        {"mode", v.mode == VOX_VOICE_ICL ? "icl" : v.mode == VOX_VOICE_SPEAKER ? "speaker" : "xvector_only"},
        {"ref_text", v.ref_text},
        {"speaker", v.speaker},
        {"instruct", v.instruct},
        {"language", v.language},
        {"model", {{"family", v.model_family}, {"variant", v.model_variant}, {"weights_sha256", v.weights_sha256}}},
        {"sampling", v.sampling},
        {"kv_fingerprints", kv},
        {"speaker_emb_dim", v.speaker_emb.size()},
        {"ref_frames", v.ref_frames},
        {"ref_codebooks", v.ref_codebooks},
        {"has_ref_audio", !v.ref_audio.empty()},
        {"features", [&] { json a = json::array(); for (auto &[k, _] : v.features) a.push_back(k); return a; }()},
        {"created", v.created},
    };
}

void voice_save(const VoiceData &v, const std::string &path) {
    SafetensorsFile f;
    f.metadata["format"] = kTvoiceFormat;
    f.metadata["voice"] = voice_to_json(v).dump();
    f.tensors["speaker_emb"] = Tensor::from(DType::F32, {int64_t(v.speaker_emb.size())}, v.speaker_emb);
    if (v.mode == VOX_VOICE_ICL && v.ref_frames > 0) {
        f.tensors["ref_codes"] = Tensor::from(DType::I16, {v.ref_frames, v.ref_codebooks}, v.ref_codes);
        f.tensors["ref_text_tokens"] = Tensor::from(DType::I32, {int64_t(v.ref_text_tokens.size())}, v.ref_text_tokens);
    }
    if (!v.ref_audio.empty())
        f.tensors["ref_audio"] = Tensor::from(DType::U8, {int64_t(v.ref_audio.size())}, v.ref_audio);
    for (auto &[k, t] : v.kv_cache) f.tensors[k] = t;
    for (auto &[k, t] : v.features) f.tensors[k] = t;
    safetensors_save(path, f);
}

VoiceData voice_load(const std::string &path) {
    SafetensorsFile f = safetensors_load(path);
    auto fmt = f.metadata.find("format");
    if (fmt == f.metadata.end() || f.metadata.count("voice") == 0)
        fail(VOX_ERR_FORMAT, "not a .tvoice file (missing metadata): " + path);
    if (fmt->second.empty() || fmt->second[0] != kTvoiceFormat[0])
        fail(VOX_ERR_FORMAT, ".tvoice format " + fmt->second + " is not supported by this runtime");

    json j;
    try { j = json::parse(f.metadata["voice"]); } catch (const std::exception &e) {
        fail(VOX_ERR_FORMAT, std::string("bad .tvoice metadata: ") + e.what());
    }
    VoiceData v;
    v.name = j.value("name", "voice");
    {
        const std::string m = j.value("mode", "icl");
        v.mode = m == "xvector_only" ? VOX_VOICE_XVECTOR_ONLY : m == "speaker" ? VOX_VOICE_SPEAKER : VOX_VOICE_ICL;
    }
    v.speaker = j.value("speaker", "");
    v.instruct = j.value("instruct", "");
    v.ref_text = j.value("ref_text", "");
    v.language = j.value("language", "auto");
    if (j.contains("model")) {
        v.model_family = j["model"].value("family", "");
        v.model_variant = j["model"].value("variant", "");
        v.weights_sha256 = j["model"].value("weights_sha256", "");
    }
    v.sampling = j.value("sampling", json::object());
    v.created = j.value("created", "");

    auto need = [&](const char *name) -> Tensor & {
        auto it = f.tensors.find(name);
        if (it == f.tensors.end()) fail(VOX_ERR_FORMAT, std::string(".tvoice is missing tensor ") + name);
        return it->second;
    };
    Tensor &emb = need("speaker_emb");
    if (emb.dtype != DType::F32) fail(VOX_ERR_FORMAT, "speaker_emb must be F32");
    v.speaker_emb = emb.as<float>();
    const bool has_feat = std::any_of(f.tensors.begin(), f.tensors.end(), [](const auto &kv) { return kv.first.rfind("feat.", 0) == 0; });
    if (v.mode == VOX_VOICE_ICL && !(has_feat && f.tensors.count("ref_codes") == 0)) {  // модели без кодов хранят признаки, а не коды
        Tensor &codes = need("ref_codes");
        if (codes.dtype != DType::I16 || codes.shape.size() != 2) fail(VOX_ERR_FORMAT, "ref_codes must be I16 [frames, codebooks]");
        v.ref_codes = codes.as<int16_t>();
        v.ref_frames = codes.shape[0];
        v.ref_codebooks = codes.shape[1];
        Tensor &tt = need("ref_text_tokens");
        if (tt.dtype != DType::I32) fail(VOX_ERR_FORMAT, "ref_text_tokens must be I32");
        v.ref_text_tokens = tt.as<int32_t>();
    }
    if (auto it = f.tensors.find("ref_audio"); it != f.tensors.end()) v.ref_audio = it->second.bytes;
    for (auto &[k, t] : f.tensors)
        if (k.rfind("kv.", 0) == 0) v.kv_cache[k] = t;
        else if (k.rfind("feat.", 0) == 0) v.features[k] = t;
    return v;
}

}  // namespace vox
