// .tvoice = safetensors. Метаданные: "format", "voice" = JSON одного собранного голоса (вид 1.x) и, с 0.15, "profile" = профиль
// с несколькими голосами, стилями и кэшами под разные модели (см. profile.cpp).
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

void voice_put(SafetensorsFile &f, const VoiceData &v, const std::string &p, bool with_ref_audio) {
    f.tensors[p + "speaker_emb"] = Tensor::from(DType::F32, {int64_t(v.speaker_emb.size())}, v.speaker_emb);
    if (v.mode == VOX_VOICE_ICL && v.ref_frames > 0) {
        f.tensors[p + "ref_codes"] = Tensor::from(DType::I16, {v.ref_frames, v.ref_codebooks}, v.ref_codes);
        f.tensors[p + "ref_text_tokens"] = Tensor::from(DType::I32, {int64_t(v.ref_text_tokens.size())}, v.ref_text_tokens);
    }
    if (with_ref_audio && !v.ref_audio.empty())
        f.tensors[p + "ref_audio"] = Tensor::from(DType::U8, {int64_t(v.ref_audio.size())}, v.ref_audio);
    for (auto &[k, t] : v.kv_cache) f.tensors[p + k] = t;
    for (auto &[k, t] : v.features) f.tensors[p + k] = t;
}

VoiceData voice_get(const SafetensorsFile &f, const json &j, const std::string &p) {
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

    auto need = [&](const char *name) -> const Tensor & {
        auto it = f.tensors.find(p + name);
        if (it == f.tensors.end()) fail(VOX_ERR_FORMAT, std::string(".tvoice is missing tensor ") + p + name);
        return it->second;
    };
    const Tensor &emb = need("speaker_emb");
    if (emb.dtype != DType::F32) fail(VOX_ERR_FORMAT, "speaker_emb must be F32");
    v.speaker_emb = emb.as<float>();
    const std::string fp = p + "feat.";
    bool has_feat = false;
    for (auto it = f.tensors.lower_bound(fp); it != f.tensors.end() && it->first.compare(0, fp.size(), fp) == 0; ++it) has_feat = true;
    if (v.mode == VOX_VOICE_ICL && !(has_feat && f.tensors.count(p + "ref_codes") == 0)) {  // модели без кодов хранят признаки, а не коды
        const Tensor &codes = need("ref_codes");
        if (codes.dtype != DType::I16 || codes.shape.size() != 2) fail(VOX_ERR_FORMAT, "ref_codes must be I16 [frames, codebooks]");
        v.ref_codes = codes.as<int16_t>();
        v.ref_frames = codes.shape[0];
        v.ref_codebooks = codes.shape[1];
        const Tensor &tt = need("ref_text_tokens");
        if (tt.dtype != DType::I32) fail(VOX_ERR_FORMAT, "ref_text_tokens must be I32");
        v.ref_text_tokens = tt.as<int32_t>();
    }
    if (auto it = f.tensors.find(p + "ref_audio"); it != f.tensors.end()) v.ref_audio = it->second.bytes;
    for (auto it = f.tensors.lower_bound(p); it != f.tensors.end() && it->first.compare(0, p.size(), p) == 0; ++it) {
        const std::string k = it->first.substr(p.size());
        if (k.rfind("kv.", 0) == 0) v.kv_cache[k] = it->second;
        else if (k.rfind("feat.", 0) == 0) v.features[k] = it->second;
    }
    return v;
}

void voice_save(const VoiceData &v, const std::string &path) {
    SafetensorsFile f;
    f.metadata["format"] = kTvoiceFormat;
    f.metadata["voice"] = voice_to_json(v).dump();
    voice_put(f, v, "", true);
    safetensors_save(path, f);
}

}  // namespace vox
