#include "profile.hpp"

#include <algorithm>

#include "voice.hpp"
#include "wav.hpp"

namespace vox {

std::string profile_key_lower(std::string s) {
    for (auto &c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

namespace {

void check_name(const std::string &n, const char *what) {
    if (n.empty() || n.size() > 64) fail(VOX_ERR_ARG, std::string(what) + " name must be 1..64 bytes: '" + n + "'");
    for (char c : n)
        if (c == '[' || c == ']' || c == ':' || c == ',' || c == '\n' || c == '\r' || c == '=')
            fail(VOX_ERR_ARG, std::string(what) + " name '" + n + "' must not contain [ ] : , = or line breaks");
}

json style_json(const StyleDef &s) {
    json j = {{"ref", s.ref}, {"instruct", s.instruct}, {"options", s.options}};
    if (!s.description.empty()) j["description"] = s.description;
    return j;
}

json voice_def_json(const VoiceDef &v) {
    json styles = json::object();
    for (const auto &[k, s] : v.styles) styles[k] = style_json(s);
    return {{"description", v.description}, {"language", v.language}, {"default_style", v.default_style},
            {"styles", styles}, {"speakers", v.speakers}, {"options", v.options}};
}

double wav_seconds(const std::vector<uint8_t> &w) {
    if (w.empty()) return 0.0;
    try {
        const Audio a = wav_decode(w.data(), w.size());
        return a.sample_rate ? double(a.pcm.size()) / a.sample_rate : 0.0;
    } catch (...) {
        return 0.0;
    }
}

}  // namespace

const VoiceDef &Profile::voice(const std::string &name) const {
    auto it = voices.find(name);
    if (it == voices.end()) {
        for (auto i = voices.begin(); i != voices.end(); ++i)
            if (profile_key_lower(i->first) == profile_key_lower(name)) return i->second;
        std::string all;
        for (const auto &[k, _] : voices) all += (all.empty() ? "" : ", ") + k;
        fail(VOX_ERR_ARG, "the profile has no voice '" + name + "'" + (all.empty() ? " (it has no voices)" : "; it has: " + all));
    }
    return it->second;
}

std::string Profile::unique_ref_id() const {
    for (int i = 1;; ++i) {
        std::string id = "r" + std::to_string(i);
        if (!recordings.count(id)) return id;
    }
}

std::string Profile::add_recording(std::vector<uint8_t> wav, const std::string &text, const std::string &language) {
    for (const auto &[id, r] : recordings)   // the same recording twice is stored once
        if (r.wav == wav && r.text == text) return id;
    const std::string id = unique_ref_id();
    Recording r;
    r.wav = std::move(wav);
    r.text = text;
    r.language = language;
    recordings[id] = std::move(r);
    return id;
}

void Profile::drop_unused_recordings() {
    for (auto it = recordings.begin(); it != recordings.end();) {
        bool used = false;
        for (const auto &[_, v] : voices)
            for (const auto &[__, s] : v.styles) used = used || s.ref == it->first;
        it = used ? std::next(it) : recordings.erase(it);
    }
}

json Profile::to_json(bool file) const {
    json vs = json::object();
    for (const auto &[k, v] : voices) vs[k] = voice_def_json(v);
    json recs = json::object();
    for (const auto &[id, r] : recordings) {
        json rj = {{"text", r.text}, {"language", r.language}, {"bytes", r.wav.size()}, {"seconds", wav_seconds(r.wav)}};
        if (file) rj["tensor"] = r.tensor;
        recs[id] = rj;
    }
    json cache_j = json::array();
    for (const auto &c : this->cache) {
        const VoiceData &d = *c.data;
        json cj = {{"voice", c.voice}, {"style", c.style},
                   {"model", {{"family", d.model_family}, {"variant", d.model_variant}, {"weights_sha256", d.weights_sha256}}},
                   {"mode", d.mode == VOX_VOICE_ICL ? "icl" : d.mode == VOX_VOICE_SPEAKER ? "speaker" : "xvector_only"},
                   {"speaker", d.speaker}, {"instruct", d.instruct}};
        cache_j.push_back(cj);
    }
    return {{"format", "2.0"}, {"name", name}, {"created", created}, {"default_voice", default_voice},
            {"voices", vs}, {"recordings", recs}, {"cache", cache_j}};
}

void Profile::apply_json(const json &j) {
    name = j.value("name", name);
    if (j.contains("created") && j["created"].is_string()) created = j["created"].get<std::string>();
    std::map<std::string, VoiceDef> nv;
    if (j.contains("voices") && j["voices"].is_object())
        for (auto it = j["voices"].begin(); it != j["voices"].end(); ++it) {
            check_name(it.key(), "voice");
            const json &vj = it.value();
            if (!vj.is_object()) fail(VOX_ERR_ARG, "voice '" + it.key() + "' must be an object");
            VoiceDef v;
            v.description = vj.value("description", "");
            v.language = vj.value("language", "auto");
            v.default_style = vj.value("default_style", "default");
            v.options = vj.value("options", json::object());
            if (!v.options.is_object()) fail(VOX_ERR_ARG, "voice '" + it.key() + "': options must be an object");
            if (vj.contains("speakers") && vj["speakers"].is_object())
                for (auto s = vj["speakers"].begin(); s != vj["speakers"].end(); ++s)
                    if (s.value().is_string() && !s.value().get<std::string>().empty()) v.speakers[s.key()] = s.value().get<std::string>();
            if (vj.contains("styles") && vj["styles"].is_object())
                for (auto s = vj["styles"].begin(); s != vj["styles"].end(); ++s) {
                    check_name(s.key(), "style");
                    const json &sj = s.value();
                    if (!sj.is_object()) fail(VOX_ERR_ARG, "style '" + s.key() + "' must be an object");
                    StyleDef st;
                    st.ref = sj.value("ref", "");
                    st.instruct = sj.value("instruct", "");
                    st.description = sj.value("description", "");
                    st.options = sj.value("options", json::object());
                    if (!st.options.is_object()) fail(VOX_ERR_ARG, "style '" + s.key() + "': options must be an object");
                    v.styles[s.key()] = st;
                }
            if (v.default_style.empty()) v.default_style = "default";
            if (!v.styles.count(v.default_style)) v.styles[v.default_style] = StyleDef{};
            nv[it.key()] = std::move(v);
        }
    voices = std::move(nv);
    default_voice = j.value("default_voice", default_voice);
    if (!voices.empty() && !voices.count(default_voice)) default_voice = voices.begin()->first;
    if (voices.empty()) default_voice.clear();
}

Profile profile_from_voice(const VoiceData &v0, const std::string &voice_name) {
    Profile p;
    p.name = v0.name;
    p.created = v0.created.empty() ? today_iso() : v0.created;
    std::string vn = voice_name;
    for (auto &c : vn)
        if (c == '[' || c == ']' || c == ':' || c == ',' || c == '\n' || c == '\r' || c == '=') c = '_';
    if (vn.empty()) vn = "voice";
    if (vn.size() > 64) vn.resize(64);
    VoiceDef vd;
    vd.language = v0.language.empty() ? "auto" : v0.language;
    vd.options = v0.sampling.is_object() ? v0.sampling : json::object();
    StyleDef st;
    if (v0.mode == VOX_VOICE_SPEAKER) {
        if (!v0.speaker.empty()) {
            vd.speakers[v0.model_family + "/" + v0.model_variant] = v0.speaker;
            st.instruct = v0.instruct;
        } else {
            vd.description = v0.instruct;   // a designed voice (VoiceDesign): the instruction is what the voice is
        }
    }
    if (!v0.ref_audio.empty()) st.ref = p.add_recording(v0.ref_audio, v0.ref_text, v0.language);
    vd.styles["default"] = st;
    p.voices[vn] = vd;
    p.default_voice = vn;
    if (v0.mode == VOX_VOICE_SPEAKER || !v0.model_family.empty()) {
        auto d = std::make_shared<VoiceData>(v0);
        d->sampling = json::object();   // moved into the voice options
        p.cache.push_back({vn, "default", d});
    }
    return p;
}

Profile profile_load(const std::string &path) {
    SafetensorsFile f = safetensors_load(path);
    auto fmt = f.metadata.find("format");
    if (fmt == f.metadata.end() || (f.metadata.count("voice") == 0 && f.metadata.count("profile") == 0))
        fail(VOX_ERR_FORMAT, "not a .tvoice file (missing metadata): " + path);
    if (fmt->second.empty() || (fmt->second[0] != '1' && fmt->second[0] != '2'))
        fail(VOX_ERR_FORMAT, ".tvoice format " + fmt->second + " is not supported by this runtime (update vox)");
    auto parse = [&](const char *key) {
        try { return json::parse(f.metadata[key]); } catch (const std::exception &e) {
            fail(VOX_ERR_FORMAT, std::string("bad .tvoice metadata '") + key + "': " + e.what());
        }
    };
    if (!f.metadata.count("profile")) {   // 1.0: one voice built for one model
        const json j = parse("voice");
        VoiceData v = voice_get(f, j, "");
        return profile_from_voice(v, v.name);
    }
    const json pj = parse("profile");
    Profile p;
    p.apply_json(pj);
    if (pj.contains("recordings") && pj["recordings"].is_object())
        for (auto it = pj["recordings"].begin(); it != pj["recordings"].end(); ++it) {
            Recording r;
            r.text = it.value().value("text", "");
            r.language = it.value().value("language", "");
            r.tensor = it.value().value("tensor", "ref." + it.key());
            auto t = f.tensors.find(r.tensor);
            if (t == f.tensors.end()) fail(VOX_ERR_FORMAT, ".tvoice profile: recording '" + it.key() + "' has no tensor " + r.tensor);
            r.wav = t->second.bytes;
            p.recordings[it.key()] = std::move(r);
        }
    if (pj.contains("cache") && pj["cache"].is_array())
        for (const json &cj : pj["cache"]) {
            const std::string prefix = cj.value("prefix", "");
            const json meta = cj.contains("meta") ? cj["meta"] : (prefix.empty() && f.metadata.count("voice") ? parse("voice") : json());
            if (!meta.is_object()) continue;
            auto d = std::make_shared<VoiceData>(voice_get(f, meta, prefix));
            d->sampling = json::object();
            p.cache.push_back({cj.value("voice", ""), cj.value("style", ""), d});
        }
    return p;
}

void profile_save(const Profile &p0, const std::string &path) {
    Profile p = p0;
    SafetensorsFile f;
    // the legacy view: the default voice's default style as built for some model (first such cache entry)
    int legacy = -1;
    if (!p.default_voice.empty() && p.voices.count(p.default_voice)) {
        const std::string ds = p.voices.at(p.default_voice).default_style;
        for (size_t i = 0; i < p.cache.size(); ++i)
            if (p.cache[i].voice == p.default_voice && p.cache[i].style == ds) { legacy = int(i); break; }
    }
    json pj = p.to_json(true);
    json cache = json::array();
    for (size_t i = 0; i < p.cache.size(); ++i) {
        const bool lg = int(i) == legacy;
        const std::string prefix = lg ? "" : "c" + std::to_string(i) + ".";
        VoiceData d = *p.cache[i].data;
        if (lg) {   // 1.x readers see this voice: its name, sampling defaults and recording are those of the profile voice
            const VoiceDef &vd = p.voices.at(p.default_voice);
            d.name = p.name;
            d.sampling = vd.options;
            if (d.ref_audio.empty()) {
                const StyleDef &st = vd.styles.at(vd.default_style);
                if (!st.ref.empty() && p.recordings.count(st.ref)) d.ref_audio = p.recordings.at(st.ref).wav;
            }
            f.metadata["voice"] = voice_to_json(d).dump();
        }
        voice_put(f, d, prefix, lg);
        json cj = pj["cache"][i];
        cj["prefix"] = prefix;
        cj["meta"] = voice_to_json(d);
        cache.push_back(cj);
        if (lg)   // a recording equal to the legacy ref_audio is stored once
            for (auto &[id, r] : p.recordings)
                if (r.wav == d.ref_audio) r.tensor = "ref_audio";
    }
    pj["cache"] = cache;
    for (auto &[id, r] : p.recordings) {
        if (r.tensor != "ref_audio") {
            r.tensor = "ref." + id;
            f.tensors[r.tensor] = Tensor::from(DType::U8, {int64_t(r.wav.size())}, r.wav);
        }
        pj["recordings"][id]["tensor"] = r.tensor;
    }
    f.metadata["format"] = legacy >= 0 ? "1.1" : "2.0";
    f.metadata["profile"] = pj.dump();
    safetensors_save(path, f);
}

void profile_patch(Profile &p, const json &patch) {
    if (!patch.is_object()) fail(VOX_ERR_ARG, "profile patch must be a JSON object");
    json before = p.to_json(false);
    json after = before;
    json pp = patch;
    pp.erase("cache");
    json recpatch = pp.contains("recordings") ? pp["recordings"] : json();
    pp.erase("recordings");
    pp.erase("format");
    after.merge_patch(pp);
    Profile q = p;
    q.apply_json(after);
    // recordings: only text / language can change here, null removes (new recordings come through vox_voice_add_recording)
    if (recpatch.is_object())
        for (auto it = recpatch.begin(); it != recpatch.end(); ++it) {
            if (!q.recordings.count(it.key())) fail(VOX_ERR_ARG, "no recording '" + it.key() + "' in the profile");
            if (it.value().is_null()) { q.recordings.erase(it.key()); continue; }
            if (it.value().contains("text")) q.recordings[it.key()].text = it.value()["text"].get<std::string>();
            if (it.value().contains("language")) q.recordings[it.key()].language = it.value()["language"].get<std::string>();
        }
    for (const auto &[vn, v] : q.voices)
        for (const auto &[sn, s] : v.styles)
            if (!s.ref.empty() && !q.recordings.count(s.ref))
                fail(VOX_ERR_ARG, "voice '" + vn + "' style '" + sn + "' refers to recording '" + s.ref + "', which the profile does not have");
    // cache entries that no longer describe the voice: a removed voice/style; for voices built from a recording, another recording or text;
    // for built-in speakers, another speaker mapping or instruction; for designed voices, another description or instruction
    auto stale = [&](const CacheEntry &c) {
        const std::string &vn = c.voice, &sn = c.style;
        if (!q.voices.count(vn) || !q.voices.at(vn).styles.count(sn)) return true;
        if (!p.voices.count(vn) || !p.voices.at(vn).styles.count(sn)) return true;
        const VoiceDef &a = p.voices.at(vn), &b = q.voices.at(vn);
        const StyleDef &sa = a.styles.at(sn), &sb = b.styles.at(sn);
        if (c.data->mode == VOX_VOICE_SPEAKER) {
            if (sa.instruct != sb.instruct) return true;
            return c.data->speaker.empty() ? a.description != b.description : a.speakers != b.speakers;
        }
        auto ref_of = [](const VoiceDef &v, const StyleDef &st) {
            return !st.ref.empty() ? st.ref : v.styles.count(v.default_style) ? v.styles.at(v.default_style).ref : std::string();
        };
        const std::string ra = ref_of(a, sa), rb = ref_of(b, sb);
        auto text_of = [](const Profile &x, const std::string &id) { return x.recordings.count(id) ? x.recordings.at(id).text : std::string("\x01"); };
        return ra != rb || text_of(p, ra) != text_of(q, rb);
    };
    std::vector<CacheEntry> keep;
    for (const auto &c : q.cache)
        if (!stale(c)) keep.push_back(c);
    q.cache = std::move(keep);
    q.drop_unused_recordings();
    p = std::move(q);
}

VoiceData voice_load(const std::string &path) {
    Profile p = profile_load(path);
    if (p.default_voice.empty()) fail(VOX_ERR_FORMAT, path + ": the profile has no voices");
    const std::string ds = p.voices.at(p.default_voice).default_style;
    for (const auto &c : p.cache)
        if (c.voice == p.default_voice && c.style == ds) {
            VoiceData v = *c.data;
            v.sampling = p.voices.at(p.default_voice).options;
            return v;
        }
    fail(VOX_ERR_FORMAT, path + ": the default voice is not built for any model yet (open it with a model: vox profile prepare)");
}

}  // namespace vox
