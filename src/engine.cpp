// Реализация C ABI. Все исключения ловятся здесь и превращаются в vox_status.
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <cstring>
#include <chrono>
#include <future>

#include "backend.hpp"
#include "family.hpp"
#include "gpu.hpp"
#include "tune.hpp"
#include "kernels.hpp"
#include "markup.hpp"
#include "profile.hpp"
#include "sampling.hpp"
#include "voice.hpp"
#include "wav.hpp"

using namespace vox;

#ifndef VOX_VERSION_STRING
#define VOX_VERSION_STRING "0.19.0"
#endif

extern std::string vox_internal_last_error();

struct vox_engine {
    std::shared_ptr<TmodFile> tmod;
    std::unique_ptr<Backend> backend;
    std::string device;
    ResourceConfig config;
    ResourcePlan plan;
    std::shared_ptr<ThreadPool> pool;        // потоки CPU этого движка (число, привязка к ядрам, приоритет — из плана)
    MemoryBudget budget;                     // память сессий (KV-кэши, состояния); лимит = ram_limit − веса на хосте
    std::atomic<uint32_t> sessions{0};
    std::atomic<int64_t> pull_us{0}, samples{0}, created{0};
    Capabilities caps;
    json tuned;                              // the vox tune result the engine opened with (null = none)
};

// A voice handle is a profile (one or many voices, styles, recordings, built-voice cache); sessions share it and build styles lazily.
struct vox_voice {
    std::shared_ptr<ProfileRef> pr;
};

// What one stretch of speech is spoken with: a profile voice and style, plus what markup asked for on top.
struct SegSpec {
    std::string voice, style;   // "" = the profile's default voice / the voice's default style
    double speed = 1.0;         // [speed x]
    std::string extra_instruct; // set when an emotion has no style in the profile and the model takes instructions
    std::string extra_opts;     // JSON object of generation options standing in for an emotion (manifest "emotion_options")
};
struct QueuedSeg {
    SegSpec spec;
    std::string text;           // text for this segment (markup already removed)
    bool closed = false;        // a later tag started the next segment
    double pause_ms = 0;        // silence before it
};

struct vox_session {
    vox_engine *e = nullptr;
    std::shared_ptr<const VoiceData> voice;   // the voice of the active segment, as built for this model
    std::shared_ptr<ProfileRef> prof;
    // Segments (0.15): markup can switch voice / style / speed or insert a pause; each switch ends the active segment and queues the
    // next one. The active segment owns the backend state below (st, frame, codes, …); queued ones only collect text.
    MarkupParser mk;
    SegSpec active_spec;
    bool active_has_text = false, active_closed = false;
    std::deque<QueuedSeg> queue;
    uint32_t seg_index = 0;
    uint64_t seed = 1;
    vox_gen_params params{};        // the session's explicit parameters (re-applied to every segment)
    json session_extra = json::object();
    std::string language;
    json notes = json::array();     // what markup did that the model could not do as asked (dropped / mapped hints)
    uint64_t hints_applied = 0;
    std::string ws_hold;            // blanks before a tag, sent only if more text follows in the same segment
    bool after_tag = false;         // the last thing parsed was a tag: blanks that follow it are dropped
    std::unique_ptr<GenState> st;
    SampleParams talker, sub;
    DetRng rng{1};
    uint32_t chunk_frames = 4, max_frames = 2048;

    // Двусторонний стриминг: поданный, но ещё не переданный модели текст (pending) и политика его коммита.
    enum class Commit { Auto, Sentence, Manual };
    std::string pending;            // текст, ещё не отданный модели (незаконченное слово/предложение, либо всё при commit=manual)
    uint64_t committed_bytes = 0;   // сколько байт текста уже передано модели (откатить нельзя: звук уже считается)
    Commit commit = Commit::Auto;
    uint32_t idle_ms = 0;           // >0: если текст не приходит столько мс, а pending не пуст — коммитим его (flush)
    bool idle_armed = false;        // таймер простоя запущен (после последней подачи текста)
    std::chrono::steady_clock::time_point last_feed;
    uint64_t flushes = 0, idle_flushes = 0;
    bool text_ended = false, finished = false, flushed = false;
    uint64_t frame = 0;
    std::vector<int32_t> history0;  // для repetition penalty
    std::vector<int16_t> codes;     // накопленные кадры текущего чанка
    uint32_t buffered = 0;
    std::vector<float> pcm;
    size_t pcm_pos = 0;
    std::vector<float> logits;
    std::vector<int32_t> frame_codes;
    std::atomic<bool> cancelled{false};
    uint64_t reserved_ram = 0;      // учтено в vox_engine::budget
    bool counted = false;           // учтена в e->sessions / e->budget
    bool direct = false;            // модель без покадровых кодов: звук делает backend->direct_step

    // Конвейер «GPU генерирует чанк k+1, пока CPU декодирует чанк k» (только если backend->decode_overlappable()).
    bool first_decoded = false;
    std::vector<int16_t> inflight_codes;
    std::vector<float> inflight_pcm;
    std::future<void> inflight;  // объявлен последним: ~vox_session дождётся потока до разрушения остального
    ~vox_session() {
        if (inflight.valid()) { try { inflight.wait(); } catch (...) {} }
        if (e && counted) { e->budget.release(reserved_ram); e->sessions.fetch_sub(1); }
    }
};

namespace {

template <typename F>
vox_status guard(F &&f) {
    try {
        f();
        return VOX_OK;
    } catch (const Error &e) {
        set_last_error(e.what());
        return e.status;
    } catch (const std::bad_alloc &) {
        set_last_error("out of memory");
        return VOX_ERR_OOM;
    } catch (const std::exception &e) {
        set_last_error(e.what());
        return VOX_ERR_FORMAT;
    }
}

template <typename T>
T read_struct(const T *in, void (*init)(T *)) {
    T out;
    init(&out);
    if (in) {
        if (in->struct_size < sizeof(uint32_t)) fail(VOX_ERR_ARG, "struct_size is not set");
        std::memcpy(&out, in, std::min<size_t>(in->struct_size, sizeof(T)));
        out.struct_size = sizeof(T);
    }
    return out;
}

vox_status write_json(const std::string &s, char *buf, size_t cap, size_t *len) {
    if (len) *len = s.size();
    if (buf && cap) {
        size_t n = std::min(cap - 1, s.size());
        std::memcpy(buf, s.data(), n);
        buf[n] = '\0';
    }
    return VOX_OK;
}

// Значение сэмплинга: явный параметр → профиль голоса → .tmod → жёсткий дефолт.
float pick(float explicit_v, const json &voice, const json &model, const char *key, float dflt) {
    if (explicit_v >= 0) return explicit_v;
    if (voice.contains(key)) return voice[key].get<float>();
    if (model.contains(key)) return model[key].get<float>();
    return dflt;
}

void fill_voice(Backend &b, const json &manifest, const float *pcm, size_t n, uint32_t sr, const std::string &ref_text,
                vox_voice_mode mode, VoiceData &v) {
    v.mode = mode;
    v.ref_text = ref_text;
    v.model_family = manifest.at("model").value("family", "");
    v.model_variant = manifest.at("model").value("variant", "");
    v.weights_sha256 = manifest.at("model").value("weights_sha256", "");
    v.features.clear();
    if (b.direct()) {  // модель без кодов: вместо эмбеддинга и кодов — признаки записи, считает сама модель
        v.speaker_emb.clear();
        v.ref_codes.clear(); v.ref_frames = v.ref_codebooks = 0;
        v.ref_text_tokens.clear();
        b.prepare_voice(v, pcm, n, sr);
        v.kv_cache.clear();
        v.ref_audio = wav_encode(pcm, n, sr, true);
        if (v.created.empty()) v.created = today_iso();
        return;
    }
    v.speaker_emb = b.speaker_embedding(pcm, n, sr);
    if (mode == VOX_VOICE_ICL) {
        v.ref_codes = b.encode_codes(pcm, n, sr, v.ref_frames);
        v.ref_codebooks = b.info().num_codebooks;
        v.ref_text_tokens = b.tokenize_ref(ref_text);
    } else {
        v.ref_codes.clear(); v.ref_frames = v.ref_codebooks = 0; v.ref_text_tokens.clear();
    }
    v.kv_cache.clear();  // кэши старой сборки больше не валидны
    v.ref_audio = wav_encode(pcm, n, sr, true);
    if (v.created.empty()) v.created = today_iso();
}


// ---------------- profiles: a (voice, style) of a profile → a voice built for this engine's model ----------------
std::string model_key(const vox_engine *e) {
    const json &mm = e->tmod->manifest().at("model");
    return mm.value("family", "") + "/" + mm.value("variant", "");
}

bool cache_matches(const VoiceData &d, const json &mm) {
    const std::string fam = mm.value("family", ""), var = mm.value("variant", ""), sha = mm.value("weights_sha256", "");
    if (!d.model_family.empty() && d.model_family != fam) return false;
    if (d.mode == VOX_VOICE_SPEAKER) return d.model_variant.empty() || d.model_variant == var;
    return d.weights_sha256.empty() || sha.empty() || d.weights_sha256 == sha;
}

// canonical key of a name in a map, case-insensitive; "" if absent
template <typename M>
std::string find_key(const M &m, const std::string &name) {
    if (m.count(name)) return name;
    const std::string l = profile_key_lower(name);
    for (const auto &kv : m)
        if (profile_key_lower(kv.first) == l) return kv.first;
    return "";
}

void check_speaker_names(const Capabilities &c, const std::string &spk_l) {
    size_t a0 = 0;
    while (a0 <= spk_l.size()) {
        size_t b0 = spk_l.find(',', a0);
        if (b0 == std::string::npos) b0 = spk_l.size();
        const std::string one = spk_l.substr(a0, b0 - a0);
        if (std::find(c.speakers.begin(), c.speakers.end(), one) == c.speakers.end()) {
            std::string all;
            for (const auto &x : c.speakers) all += (all.empty() ? "" : ", ") + x;
            fail(VOX_ERR_ARG, "unknown speaker '" + one + "'; this model has: " + all);
        }
        a0 = b0 + 1;
    }
}

// The built voice for (voice, style) of the profile on e's model: from the cache, else built now (from a recording, a built-in speaker
// named for this model, or the voice description) and added to the cache. Caller holds pr.mu.
std::shared_ptr<const VoiceData> resolve_base(vox_engine *e, Profile &p, const std::string &voice_in, const std::string &style_in,
                                              json *notes) {
    if (p.voices.empty()) fail(VOX_ERR_ARG, "the profile has no voices");
    const std::string vkey = voice_in.empty() ? p.default_voice : find_key(p.voices, voice_in);
    if (vkey.empty()) p.voice(voice_in);   // throws with the list of voices
    const VoiceDef &vd = p.voices.at(vkey);
    std::string skey = style_in.empty() ? vd.default_style : find_key(vd.styles, style_in);
    if (skey.empty()) {
        std::string all;
        for (const auto &kv : vd.styles) all += (all.empty() ? "" : ", ") + kv.first;
        fail(VOX_ERR_ARG, "voice '" + vkey + "' has no style '" + style_in + "'; it has: " + all);
    }
    const json &mm = e->tmod->manifest().at("model");
    for (const auto &c : p.cache)
        if (c.voice == vkey && c.style == skey && cache_matches(*c.data, mm)) return c.data;

    const StyleDef &st = vd.styles.at(skey);
    const std::string ref = !st.ref.empty() ? st.ref : vd.styles.count(vd.default_style) ? vd.styles.at(vd.default_style).ref : "";
    const std::string fam = mm.value("family", ""), var = mm.value("variant", "");
    std::string spk;
    for (const std::string &k : {fam + "/" + var, fam}) {
        const std::string kk = find_key(vd.speakers, k);
        if (!kk.empty()) { spk = vd.speakers.at(kk); break; }
    }
    const Capabilities &c = e->caps;
    auto d = std::make_shared<VoiceData>();
    auto note = [&](const std::string &n) { if (notes) notes->push_back(n); };
    if (!spk.empty() && c.named_speakers) {
        spk = profile_key_lower(spk);
        check_speaker_names(c, spk);
        d->mode = VOX_VOICE_SPEAKER;
        d->speaker = spk;
        if (c.instruct) d->instruct = st.instruct;
        else if (!st.instruct.empty()) note("style '" + skey + "': this model takes no instructions, the instruction is not used");
    } else if (c.requires_instruct) {
        std::string ins = vd.description;
        if (!st.instruct.empty()) ins += (ins.empty() ? "" : " ") + st.instruct;
        if (ins.empty())
            fail(VOX_ERR_ARG, "voice '" + vkey + "' has no description: this model (" + model_key(e) + ") makes voices only from a text description "
                              "(set it: vox profile set FILE --voice " + vkey + " --description \"…\")");
        d->mode = VOX_VOICE_SPEAKER;
        d->instruct = ins;
    } else if (!ref.empty() && p.recordings.count(ref) && (c.clone_icl || c.clone_xvector)) {
        const Recording &r = p.recordings.at(ref);
        vox_voice_mode mode;
        if (c.clone_icl && !r.text.empty()) mode = VOX_VOICE_ICL;
        else if (c.clone_xvector) mode = VOX_VOICE_XVECTOR_ONLY;
        else fail(VOX_ERR_ARG, "voice '" + vkey + "' style '" + skey + "': this model clones from a recording and its text, and the recording has no text");
        const Audio a = wav_decode(r.wav.data(), r.wav.size());
        PoolScope scope(e->pool.get());
        fill_voice(*e->backend, e->tmod->manifest(), a.pcm.data(), a.pcm.size(), a.sample_rate, r.text, mode, *d);
        d->ref_audio = r.wav;   // the exact bytes of the recording (the legacy view of the file then shares them)
        if (c.instruct) d->instruct = st.instruct;   // models that take an instruction with a cloned voice (CosyVoice2)
        else if (!st.instruct.empty()) note("style '" + skey + "': this model takes no instructions, the style is its recording");
    } else {
        std::string why = c.named_speakers ? "it has built-in speakers but the voice names none for it" :
                          (c.clone_icl || c.clone_xvector) ? "the voice has no recording" : "it cannot clone voices";
        fail(VOX_ERR_ARG, "voice '" + vkey + "' cannot be made on " + model_key(e) + ": " + why +
                          (c.named_speakers ? " (map one: vox profile set FILE --voice " + vkey + " --speaker " + fam + "=NAME)" : ""));
    }
    d->name = vkey;
    d->model_family = fam;
    d->model_variant = var;
    d->weights_sha256 = mm.value("weights_sha256", "");
    d->language = vd.language;
    if (d->created.empty()) d->created = today_iso();
    d->sampling = json::object();
    p.cache.push_back({vkey, skey, d});
    return d;
}

// English instructions for emotions on models that take style instructions (Qwen3 CustomVoice 1.7B, VoiceDesign)
const char *emotion_instruct(const std::string &e) {
    static const std::map<std::string, const char *> m = {
        {"happy", "Speak in a happy, cheerful tone."}, {"sad", "Speak in a sad, sorrowful tone."}, {"angry", "Speak in an angry tone."},
        {"excited", "Speak with excitement and energy."}, {"calm", "Speak calmly and slowly."}, {"whisper", "Whisper softly."},
        {"shout", "Speak very loudly, almost shouting."}, {"fear", "Speak in a fearful, trembling voice."}, {"surprised", "Speak with surprise."},
        {"serious", "Speak in a serious tone."}, {"tender", "Speak gently and tenderly."}, {"tired", "Speak in a tired, sleepy voice."},
        {"sarcastic", "Speak sarcastically."}};
    auto it = m.find(e);
    return it == m.end() ? "" : it->second;
}
// the model's own wording when its manifest has one ("instruct_emotions": {"happy": "…"}), the English default otherwise
std::string emotion_instruct_for(const vox_engine *e, const std::string &emo) {
    const json &m = e->tmod->manifest();
    if (m.contains("instruct_emotions") && m["instruct_emotions"].is_object()) {
        const json &t = m["instruct_emotions"];
        if (t.contains(emo) && t[emo].is_string()) return t[emo].get<std::string>();
    }
    return emotion_instruct(emo);
}
// generation options the model's manifest gives for an emotion ("emotion_options": {"excited": {"exaggeration": 0.8}}), as JSON; "" if none
std::string emotion_options_for(const vox_engine *e, const std::string &emo) {
    const json &m = e->tmod->manifest();
    if (m.contains("emotion_options") && m["emotion_options"].is_object() && m["emotion_options"].contains(emo) && m["emotion_options"][emo].is_object())
        return m["emotion_options"][emo].dump();
    return "";
}
// speaking-rate factor standing in for an emotion on models that only have a speed (F5, Kokoro)
double emotion_speed(const std::string &e) {
    static const std::map<std::string, double> m = {{"happy", 1.05}, {"excited", 1.12}, {"angry", 1.08}, {"shout", 1.08}, {"fear", 1.06},
                                                   {"surprised", 1.05}, {"sad", 0.9}, {"calm", 0.92}, {"tired", 0.88}, {"tender", 0.93},
                                                   {"whisper", 0.95}, {"serious", 0.97}, {"sarcastic", 1.0}};
    auto it = m.find(e);
    return it == m.end() ? 1.0 : it->second;
}

bool profile_has_style(const Profile &p, const std::string &voice, const std::string &style) {
    const std::string vk = voice.empty() ? p.default_voice : find_key(p.voices, voice);
    if (vk.empty() || !p.voices.count(vk)) return false;
    return !find_key(p.voices.at(vk).styles, style).empty();
}

// Позиция (в байтах) конца последнего законченного предложения в pending; 0 — нет. Знак конца считается законченным, если за ним
// пробел/перевод строки (так "3.5" и "e.g." не рвутся посередине) или он полноширинный (。！？…); `final` — конец текста: любой знак в конце.
size_t sentence_cut(const std::string &t, bool final) {
    size_t best = 0;
    for (size_t i = 0; i < t.size();) {
        const unsigned char c = static_cast<unsigned char>(t[i]);
        size_t len = 1;
        bool term = false, full = false;
        if (c < 0x80) {
            term = c == '.' || c == '!' || c == '?' || c == '\n';
        } else {
            len = (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
            len = std::min(len, t.size() - i);
            static const char *kFull[] = {"。", "！", "？", "…"};
            for (const char *f : kFull)
                if (t.compare(i, std::strlen(f), f) == 0) { term = full = true; break; }
        }
        if (term) {
            size_t j = i + len;
            const bool ws_after = j < t.size() && (t[j] == ' ' || t[j] == '\t' || t[j] == '\n' || t[j] == '\r');
            if (full || ws_after || final || c == '\n') {
                while (j < t.size() && (t[j] == ' ' || t[j] == '\t' || t[j] == '\n' || t[j] == '\r')) ++j;
                // многоточие "..." и "?!" не рвём посередине: знак конца сливается со следующим знаком
                const bool glued = !full && j == i + len && j < t.size() && (t[j] == '.' || t[j] == '!' || t[j] == '?');
                if (!glued) best = j;
            }
        }
        i += len;
    }
    return best;
}

// ---------------- segments ----------------
bool inner_ended(const vox_session *s) { return s->active_closed || (s->text_ended && s->queue.empty()); }

// (Re)starts the backend state of the active segment for spec: resolves the voice, applies the style / markup options and sampling.
void begin_segment(vox_session *s, const SegSpec &spec) {
    vox_engine *e = s->e;
    const Capabilities &c = e->caps;
    std::shared_ptr<const VoiceData> base;
    json vopts = json::object(), sopts = json::object();
    {
        std::lock_guard<std::mutex> lk(s->prof->mu);
        Profile &p = s->prof->p;
        base = resolve_base(e, p, spec.voice, spec.style, &s->notes);
        const std::string vk = spec.voice.empty() ? p.default_voice : find_key(p.voices, spec.voice);
        const VoiceDef &vd = p.voices.at(vk);
        const std::string sk = spec.style.empty() ? vd.default_style : find_key(vd.styles, spec.style);
        vopts = vd.options;
        sopts = vd.styles.at(sk).options;
    }
    auto v = std::make_shared<VoiceData>(*base);
    json samp = vopts;
    for (auto it = sopts.begin(); it != sopts.end(); ++it) samp[it.key()] = it.value();
    v->sampling = samp;
    if (!spec.extra_instruct.empty() && c.instruct)
        v->instruct += (v->instruct.empty() ? "" : " ") + spec.extra_instruct;
    s->voice = v;

    const vox_gen_params &p = s->params;
    const json &vs = v->sampling;
    const json md = e->tmod->manifest().value("generation", json::object());
    s->talker.temperature = pick(p.temperature, vs, md, "temperature", 0.9f);
    s->talker.top_p = pick(p.top_p, vs, md, "top_p", 1.0f);
    s->talker.top_k = int32_t(pick(float(p.top_k), vs, md, "top_k", 50));
    s->talker.repetition_penalty = pick(p.repetition_penalty, vs, md, "repetition_penalty", 1.05f);
    s->sub.temperature = pick(p.sub_temperature, vs, md, "sub_temperature", 0.9f);
    s->sub.top_p = pick(p.sub_top_p, vs, md, "sub_top_p", 1.0f);
    s->sub.top_k = int32_t(pick(float(p.sub_top_k), vs, md, "sub_top_k", 50));
    const uint64_t seed0 = p.seed ? p.seed : vs.value("seed", uint64_t(0));
    s->seed = seed0 ? seed0 : 1;
    s->rng = DetRng(s->seed + s->seg_index);   // segment 0 = the plain session; later segments get their own stream
    SessionOptions so;
    so.decoder_context_frames = p.decoder_context_frames;
    so.decoder_mode = p.decoder_mode;
    so.max_frames = s->max_frames;
    so.seed = s->seed + s->seg_index;
    so.options = md;                                   // манифест < голос < стиль < extra_json сессии
    for (auto it = vs.begin(); it != vs.end(); ++it) so.options[it.key()] = it.value();
    for (auto it = s->session_extra.begin(); it != s->session_extra.end(); ++it) so.options[it.key()] = it.value();
    if (spec.speed != 1.0 && c.speed) so.options["speed"] = so.options.value("speed", 1.0) * spec.speed;
    if (!spec.extra_opts.empty()) {
        const json eo = json::parse(spec.extra_opts);
        for (auto it = eo.begin(); it != eo.end(); ++it) so.options[it.key()] = it.value();
    }
    const std::string lang = !s->language.empty() ? s->language : v->language;
    s->st = e->backend->begin(*v, lang);
    e->backend->configure(*s->st, so);
    s->active_spec = spec;
    s->active_has_text = false;
    s->finished = s->flushed = false;
    s->frame = 0;
    s->history0.clear();
    s->codes.clear();
    s->buffered = 0;
    s->first_decoded = false;
}

void send_to_active(vox_session *s, const std::string &piece) {
    if (piece.empty()) return;
    if (s->direct) {
        s->e->backend->direct_text(*s->st, piece);
    } else {
        auto toks = s->e->backend->tokenize(piece);
        if (!toks.empty()) s->e->backend->append_text(*s->st, toks);
    }
    s->active_has_text = true;
}

// Around markup, blanks belong to no one: "Hi. [pause 1s] Bye." speaks "Hi." and "Bye." exactly as if they were said alone. Blanks
// before a tag are held back (and sent only if more text follows in the same segment), blanks right after a tag are dropped.
// Text without tags is passed through untouched.
void route_text(vox_session *s, const std::string &t0, bool tag_follows) {
    std::string t = s->ws_hold + t0;
    s->ws_hold.clear();
    if (s->after_tag) {
        size_t a = 0;
        while (a < t.size() && (t[a] == ' ' || t[a] == '\t' || t[a] == '\n' || t[a] == '\r')) ++a;
        t.erase(0, a);
        if (!t.empty()) s->after_tag = false;
    }
    if (tag_follows) {
        size_t b = t.size();
        while (b > 0 && (t[b - 1] == ' ' || t[b - 1] == '\t' || t[b - 1] == '\n' || t[b - 1] == '\r')) --b;
        s->ws_hold = t.substr(b);
        t.resize(b);
    }
    if (t.empty()) return;
    if (s->queue.empty()) send_to_active(s, t);
    else s->queue.back().text += t;
}

// The spec of the segment text currently goes to (the last queued one, or the active one).
SegSpec &input_spec(vox_session *s) { return s->queue.empty() ? s->active_spec : s->queue.back().spec; }
bool input_has_text(const vox_session *s) { return s->queue.empty() ? s->active_has_text : !s->queue.back().text.empty(); }

// Text from now on is spoken with `spec`, after `pause_ms` of silence.
void switch_segment(vox_session *s, const SegSpec &spec, double pause_ms) {
    s->ws_hold.clear();   // blanks around a segment boundary belong to no one
    s->after_tag = true;
    const bool untouched_active = s->queue.empty() && !s->active_has_text && s->frame == 0 && !s->finished;
    if (pause_ms <= 0 && untouched_active) {   // nothing spoken yet: just restart the active segment with the new voice / style
        begin_segment(s, spec);
        return;
    }
    if (pause_ms <= 0 && !s->queue.empty() && s->queue.back().text.empty()) {   // the queued one has no text yet: retarget it
        s->queue.back().spec = spec;
        return;
    }
    if (s->queue.empty()) {
        s->active_closed = true;
        if (s->direct) s->e->backend->direct_seal(*s->st);
    } else {
        s->queue.back().closed = true;
    }
    QueuedSeg q;
    q.spec = spec;
    q.pause_ms = pause_ms;
    s->queue.push_back(std::move(q));
}

void apply_ops(vox_session *s, const std::vector<MarkupOp> &ops, bool tag_next) {
    const Capabilities &c = s->e->caps;
    for (size_t oi = 0; oi < ops.size(); ++oi) {
        const MarkupOp &op = ops[oi];
        if (op.kind == MarkupOp::Text) { route_text(s, op.s, oi + 1 < ops.size() || tag_next); continue; }
        SegSpec spec = input_spec(s);
        switch (op.kind) {
            case MarkupOp::Voice: {
                std::lock_guard<std::mutex> lk(s->prof->mu);
                const std::string vk = find_key(s->prof->p.voices, op.s);
                if (vk.empty()) { s->notes.push_back("[voice:" + op.s + "]: the profile has no such voice, ignored"); continue; }
                spec.voice = vk; spec.style.clear(); spec.extra_instruct.clear(); spec.extra_opts.clear(); spec.speed = 1.0;
                break;
            }
            case MarkupOp::Style: {
                bool has;
                {
                    std::lock_guard<std::mutex> lk(s->prof->mu);
                    has = profile_has_style(s->prof->p, spec.voice, op.s);
                }
                spec.extra_instruct.clear();
                spec.extra_opts.clear();
                if (has) { spec.style = op.s; ++s->hints_applied; break; }
                const std::string emo = markup_emotion(op.s);
                spec.style.clear();
                if (emo.empty()) { s->notes.push_back("[" + op.s + "]: no such style in the voice, ignored"); continue; }
                if (!emotion_options_for(s->e, emo).empty()) { spec.extra_opts = emotion_options_for(s->e, emo); ++s->hints_applied; }
                else if (c.instruct) { spec.extra_instruct = emotion_instruct_for(s->e, emo); ++s->hints_applied; }
                else if (c.speed) { spec.speed = emotion_speed(emo); ++s->hints_applied; s->notes.push_back("[" + op.s + "]: no style in the profile; mapped to speed " + std::to_string(spec.speed).substr(0, 4)); }
                else { s->notes.push_back("[" + op.s + "]: the voice has no such style and this model cannot express it (add a style with its own recording)"); continue; }
                break;
            }
            case MarkupOp::Reset:
                spec.style.clear(); spec.extra_instruct.clear(); spec.extra_opts.clear(); spec.speed = 1.0;
                break;
            case MarkupOp::Speed:
                if (!c.speed) { s->notes.push_back("[speed]: this model has no speaking-rate control, ignored"); continue; }
                spec.speed = op.num;
                ++s->hints_applied;
                break;
            case MarkupOp::Pause:
                ++s->hints_applied;
                switch_segment(s, spec, op.num);
                continue;
            case MarkupOp::Event:
                if (std::find(c.tags.begin(), c.tags.end(), op.s) != c.tags.end()) { s->after_tag = false; route_text(s, " [" + op.s + "] ", false); ++s->hints_applied; continue; }
                s->notes.push_back("[" + op.s + "]: the model has no such sound; a short pause instead");
                switch_segment(s, spec, 250);
                continue;
            case MarkupOp::Unknown:
                s->notes.push_back(op.s + ": not markup this engine knows, dropped");
                continue;
            default: continue;
        }
        const SegSpec &cur = input_spec(s);
        if (cur.voice == spec.voice && cur.style == spec.style && cur.speed == spec.speed && cur.extra_instruct == spec.extra_instruct && cur.extra_opts == spec.extra_opts) continue;
        if (!input_has_text(s)) { input_spec(s) = spec; s->after_tag = true; if (s->queue.empty()) switch_segment(s, spec, 0); continue; }
        switch_segment(s, spec, 0);
    }
    if (s->notes.size() > 64) s->notes.erase(s->notes.begin(), s->notes.begin() + long(s->notes.size() - 64));
}

// Active segment done and drained: start the next queued one (silence first). false → no more segments.
bool next_segment(vox_session *s) {
    if (s->queue.empty()) return false;
    QueuedSeg q = std::move(s->queue.front());
    s->queue.pop_front();
    if (s->pcm_pos >= s->pcm.size()) { s->pcm.clear(); s->pcm_pos = 0; }
    const size_t sil = size_t(std::llround(q.pause_ms * 1e-3 * double(s->e->backend->info().sample_rate)));
    s->pcm.insert(s->pcm.end(), sil, 0.0f);
    ++s->seg_index;
    begin_segment(s, q.spec);
    s->active_closed = q.closed;
    send_to_active(s, q.text);
    if (s->direct && (q.closed || (s->text_ended && s->queue.empty()))) s->e->backend->direct_seal(*s->st);
    return true;
}

// Переносит готовую к коммиту часть pending в модель по политике. force — flush/конец текста: коммитим всё, включая слово без пробела.
// Возвращает число переданных байт.
size_t commit_pending(vox_session *s, bool force) {
    std::string &t = s->pending;
    if (t.empty()) return 0;
    size_t cut = 0;
    if (force) cut = t.size();
    else switch (s->commit) {
        case vox_session::Commit::Manual: cut = 0; break;
        case vox_session::Commit::Sentence: cut = sentence_cut(t, false); break;
        case vox_session::Commit::Auto: cut = s->direct ? t.size() : s->e->backend->text_safe_cut(t); break;
    }
    if (!force) cut = s->mk.safe_cut(t, cut);   // не рвём тег разметки и «имя:» в начале строки
    if (cut == 0) return 0;
    const std::string piece = t.substr(0, cut);
    t.erase(0, cut);
    s->committed_bytes += cut;
    apply_ops(s, s->mk.parse(piece), !t.empty() && t[0] == '[');
    // модели, озвучивающие куски целиком: «законченный кусок» — повод озвучить его сразу, не ожидая продолжения
    if (s->direct && s->queue.empty() && (force || s->commit != vox_session::Commit::Auto)) s->e->backend->direct_seal(*s->st);
    return cut;
}

// Простой: текст не поступал idle_ms, а в pending что-то осталось — коммитим (клиент опрашивает pull, так что отдельный поток не нужен).
void check_idle(vox_session *s) {
    if (!s->idle_ms || !s->idle_armed || s->text_ended || s->pending.empty()) return;
    const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - s->last_feed).count();
    if (waited < int64_t(s->idle_ms)) return;
    s->idle_armed = false;
    if (commit_pending(s, true)) { ++s->flushes; ++s->idle_flushes; }
}

void decode_buffered(vox_session *s) {
    if (s->buffered == 0) return;
    if (s->pcm_pos >= s->pcm.size()) { s->pcm.clear(); s->pcm_pos = 0; }
    s->e->backend->decode(*s->st, s->codes.data(), s->buffered, s->pcm);
    s->codes.clear();
    s->buffered = 0;
    s->first_decoded = true;
}

void start_async_decode(vox_session *s) {
    s->inflight_codes.swap(s->codes);
    s->codes.clear();
    const size_t n = s->buffered;
    s->buffered = 0;
    s->inflight_pcm.clear();
    s->inflight = std::async(std::launch::async, [s, n] {
        PoolScope scope(s->e->pool.get());
        s->e->backend->decode(*s->st, s->inflight_codes.data(), n, s->inflight_pcm);
    });
}

void collect_async_decode(vox_session *s) {
    s->inflight.get();  // пробрасывает исключение потока декодера
    if (s->pcm_pos >= s->pcm.size()) { s->pcm.clear(); s->pcm_pos = 0; }
    s->pcm.insert(s->pcm.end(), s->inflight_pcm.begin(), s->inflight_pcm.end());
    s->inflight_pcm.clear();
}

// Генерирует кадры до заполнения чанка, конца речи или нехватки текста.
void generate(vox_session *s) {
    Backend &b = *s->e->backend;
    const ModelInfo &mi = b.info();
    while (s->buffered < s->chunk_frames) {
        if (s->cancelled.load(std::memory_order_relaxed)) fail(VOX_ERR_CANCELLED, "session cancelled");
        if (b.needs_text(*s->st, inner_ended(s))) return;
        if (s->frame >= s->max_frames) { s->finished = true; return; }

        b.talker_logits(*s->st, inner_ended(s), s->logits);
        for (int32_t i = mi.suppress_from; i < mi.suppress_to && size_t(i) < s->logits.size(); ++i)
            if (i != mi.eos_code) s->logits[size_t(i)] = -INFINITY;
        static const bool no_eos = std::getenv("VOX_DEBUG_NO_EOS") != nullptr;  // тесты: случайные веса сразу выдают eos
        if ((no_eos || s->frame < mi.min_eos_frames) && mi.eos_code >= 0 && size_t(mi.eos_code) < s->logits.size())
            s->logits[size_t(mi.eos_code)] = -INFINITY;
        int32_t c0 = sample_logits(s->logits, s->talker, s->rng.uniform(s->frame, 0), &s->history0);
        if (c0 == mi.eos_code) { s->finished = true; return; }
        s->history0.push_back(c0);
        b.accept_code0(*s->st, c0);
        s->codes.push_back(int16_t(c0));

        s->frame_codes.assign(mi.num_codebooks, 0);
        if (b.predict_frame(*s->st, s->sub, s->rng.key0(), s->rng.key1(), s->frame, s->frame_codes.data())) {
            for (uint32_t cb = 1; cb < mi.num_codebooks; ++cb) s->codes.push_back(int16_t(s->frame_codes[cb]));
        } else for (uint32_t cb = 1; cb < mi.num_codebooks; ++cb) {
            b.predictor_logits(*s->st, cb, s->logits);
            int32_t c = sample_logits(s->logits, s->sub, s->rng.uniform(s->frame, cb));
            b.accept_code(*s->st, cb, c);
            s->codes.push_back(int16_t(c));
        }
        b.end_frame(*s->st);
        ++s->frame;
        ++s->buffered;
    }
}

}  // namespace

extern "C" {

VOX_API const char *vox_version(void) { return VOX_VERSION_STRING; }

VOX_API void vox_engine_opts_default(vox_engine_opts *o) {
    if (!o) return;
    std::memset(o, 0, sizeof *o);
    o->struct_size = sizeof *o;
    o->device = "auto";
    o->max_sessions = 1;
    o->config_json = nullptr;
}

VOX_API void vox_gen_params_default(vox_gen_params *p) {
    if (!p) return;
    std::memset(p, 0, sizeof *p);
    p->struct_size = sizeof *p;
    p->temperature = p->top_p = p->repetition_penalty = -1;
    p->sub_temperature = p->sub_top_p = -1;
    p->top_k = p->sub_top_k = -1;
}

// ---- Engine ----
namespace {
// A configuration measured by `vox tune` for this model on this computer, when nothing was asked for explicitly (device
// "auto", no GPU index, placement, layers or threads; {"tuned": false} or VOX_TUNE=0 turn it off). Returns the entry used.
json apply_tune(ResourceConfig &cfg, const TmodFile &f) {
    const char *tenv = std::getenv("VOX_TUNE");
    const bool defaults = cfg.device == "auto" && cfg.gpu < 0 && cfg.place.empty() && cfg.gpu_layers.empty() && !cfg.threads;
    if (!cfg.tuned || !defaults || (tenv && tenv[0] == '0')) return json();
    json t;
    try { t = tune_entry(f); } catch (const std::exception &) { return json(); }
    if (!t.is_object() || !t.contains("device") || !t["device"].is_string()) return json();
    try {
        ResourceConfig tc = cfg;
        tc.apply_device_string(t["device"].get<std::string>());
        if (t.value("threads", 0) > 0 && !tc.threads) tc.threads = uint32_t(t.value("threads", 0));
        cfg = tc;
        if (t.contains("gemm") && t["gemm"].is_string()) gpu_gemm_preset(t["gemm"].get<std::string>());
        return t;
    } catch (const std::exception &) {
        return json();   // a stale entry: plan as if there were none
    }
}
}  // namespace

VOX_API vox_status vox_engine_open(const char *path, const vox_engine_opts *opts, vox_engine **out) {
    return guard([&] {
        if (!path || !out) fail(VOX_ERR_ARG, "path and out are required");
        *out = nullptr;
        vox_engine_opts o = read_struct(opts, vox_engine_opts_default);
        auto e = std::make_unique<vox_engine>();
        e->device = o.device ? o.device : "auto";
        // строка device (устаревшая форма) → поля; config_json перекрывает; отдельные числовые поля дополняют
        ResourceConfig cfg;
        cfg.apply_device_string(e->device);
        if (o.config_json && o.config_json[0]) {
            json j;
            try { j = json::parse(o.config_json); } catch (const std::exception &ex) { fail(VOX_ERR_ARG, std::string("config_json is not valid JSON: ") + ex.what()); }
            ResourceConfig c2 = ResourceConfig::from_json(j);
            if (j.contains("device")) { cfg = ResourceConfig(); cfg.apply_device_string(e->device); cfg.apply_device_string(j["device"].get<std::string>()); }
            auto take = [&](const char *k) { return j.contains(k); };
            if (take("mode")) cfg.mode = c2.mode;
            if (take("gpu")) cfg.gpu = c2.gpu;
            if (take("threads")) cfg.threads = c2.threads;
            if (take("cpu_affinity")) cfg.cpu_affinity = c2.cpu_affinity;
            if (take("priority")) cfg.nice = c2.nice;
            if (take("ram_limit")) cfg.ram_limit = c2.ram_limit;
            if (take("vram_limit")) cfg.vram_limit = c2.vram_limit;
            if (take("place")) for (auto &kv : c2.place) cfg.place[kv.first] = kv.second;
            if (take("gpu_layers")) for (auto &kv : c2.gpu_layers) cfg.gpu_layers[kv.first] = kv.second;
            if (take("strict")) cfg.strict = c2.strict;
            if (take("enforce")) cfg.enforce = c2.enforce;
            if (take("overlap")) cfg.overlap = c2.overlap;
            if (take("release_host_weights")) cfg.release_host_weights = c2.release_host_weights;
            if (take("tuned")) cfg.tuned = c2.tuned;
            if (take("max_sessions")) cfg.max_sessions = c2.max_sessions;
            if (take("planned_sessions")) cfg.planned_sessions = c2.planned_sessions;
        }
        if (!cfg.threads && o.n_threads) cfg.threads = o.n_threads;
        if (!cfg.ram_limit && o.memory_limit) cfg.ram_limit = o.memory_limit;
        if (o.max_sessions > cfg.planned_sessions) cfg.planned_sessions = o.max_sessions;
        e->tmod = std::make_shared<TmodFile>(path);
        e->tuned = apply_tune(cfg, *e->tmod);
        e->config = cfg;
        e->plan = plan_for_model(*e->tmod, cfg);
        if (e->tuned.is_object())
            e->plan.notes.push_back("tuned: " + e->tuned.value("device", std::string()) + " (vox tune, " + e->tuned.value("when", std::string()) + ")");
        if (!e->plan.feasible) {
            std::string why;
            for (const auto &w : e->plan.warnings) why += (why.empty() ? "" : "; ") + w;
            const bool memory = e->plan.ram_budget && e->plan.ram_estimate > e->plan.ram_budget;
            fail(memory ? VOX_ERR_OOM : VOX_ERR_DEVICE, "cannot open the model within the requested resources: " + why +
                                                          "\n--- plan ---\n" + e->plan.summary());
        }
        if (std::getenv("VOX_VERBOSE")) std::fprintf(stderr, "vox: resource plan\n%s", e->plan.summary().c_str());
        e->pool = make_thread_pool(int(e->plan.threads), e->plan.tuning);
        e->budget.set_limit(cfg.ram_limit ? (cfg.ram_limit > e->plan.ram_static ? cfg.ram_limit - e->plan.ram_static : 1) : 0);
        {
            PoolScope scope(e->pool.get());
            e->backend = create_backend(e->tmod, e->plan);
        }
        e->caps = find_family(e->tmod->manifest()).capabilities(*e->tmod);
        *out = e.release();
    });
}

VOX_API void vox_engine_close(vox_engine *e) { delete e; }

VOX_API uint32_t vox_engine_sample_rate(const vox_engine *e) { return e ? e->backend->info().sample_rate : 0; }

VOX_API vox_status vox_engine_info(const vox_engine *e, char *buf, size_t cap, size_t *len) {
    return guard([&] {
        if (!e) fail(VOX_ERR_ARG, "engine is null");
        json m = e->tmod->manifest();
        size_t nblobs = m.contains("blobs") ? m["blobs"].size() : 0;
        m.erase("blobs");
        const ModelInfo &mi = e->backend->info();
        json j = {{"version", VOX_VERSION_STRING},
                  {"backend", e->backend->name()},
                  {"device", e->device},
                  {"fingerprint", e->backend->fingerprint()},
                  {"specialized", e->tmod->specialized()},
                  {"sample_rate", mi.sample_rate},
                  {"samples_per_frame", mi.samples_per_frame},
                  {"num_codebooks", mi.num_codebooks},
                  {"blob_count", nblobs},
                  {"family", find_family(e->tmod->manifest()).name()},
                  {"capabilities", e->caps.to_json()},
                  {"plan", e->plan.to_json()},
                  {"tuned", e->tuned},
                  {"manifest", m}};
        write_json(j.dump(), buf, cap, len);
    });
}

VOX_API vox_status vox_engine_stats(const vox_engine *e, char *buf, size_t cap, size_t *len) {
    return guard([&] {
        if (!e) fail(VOX_ERR_ARG, "engine is null");
        const SystemInfo sys = system_info();
        json mem = {{"rss", sys.rss}, {"peak_rss", sys.peak_rss}, {"ram_available", sys.ram_available}, {"ram_budget", e->plan.ram_budget},
                    {"ram_static_estimate", e->plan.ram_static}, {"sessions_reserved", e->budget.used()}, {"sessions_reserved_peak", e->budget.peak()},
                    {"sessions_limit", e->budget.limit()}, {"active_sessions", e->sessions.load()}};
        json j = {{"device", e->plan.backend}, {"threads", e->plan.threads}, {"memory", mem}, {"plan", e->plan.to_json()},
                  {"backend", e->backend->stats()}};
        if (e->plan.uses_gpu()) {
            const GpuMemInfo gm = gpu_mem_info();
            const GpuStats gs = gpu_stats();
            j["gpu"] = {{"index", gm.device_index}, {"name", e->plan.gpu_name}, {"vram_used_by_vox", gm.used}, {"vram_limit", gm.limit},
                        {"vram_device_free", gm.device_free}, {"vram_device_total", gm.device_total},
                        {"matrices_on_gpu", gs.mats_ok}, {"matrices_on_cpu", gs.mats_failed}, {"last_error", gs.last_error}};
        }
        const double secs = double(e->samples.load()) / double(e->backend->info().sample_rate), wall = double(e->pull_us.load()) / 1e6;
        j["throughput"] = {{"sessions_created", e->created.load()}, {"audio_seconds", secs}, {"compute_seconds", wall}, {"rtf", secs > 0 ? wall / secs : 0.0}};
        write_json(j.dump(), buf, cap, len);
    });
}

VOX_API vox_status vox_engine_phonemize(vox_engine *e, const char *utf8, const char *lang, char *buf, size_t cap, size_t *len) {
    return guard([&] {
        if (!e || !utf8) fail(VOX_ERR_ARG, "engine and text are required");
        PoolScope scope(e->pool.get());
        write_json(e->backend->phonemize(utf8, lang ? lang : ""), buf, cap, len);
    });
}

VOX_API vox_status vox_devices_json(char *buf, size_t cap, size_t *len) {
    return guard([&] {
        json gpus = json::array();
        for (const GpuDeviceInfo &d : gpu_enumerate())
            gpus.push_back({{"index", d.index}, {"name", d.name}, {"type", d.type}, {"software", d.software}, {"unified_memory", d.unified_memory},
                            {"vram_total", d.vram_total}, {"vram_free", d.vram_free}, {"subgroup_arith", d.subgroup_arith},
                            {"cooperative_matrix", d.coop_matrix}, {"memory_budget", d.memory_budget}, {"api_version", d.api_version},
                            {"vendor_id", d.vendor_id}, {"max_storage_range", d.max_storage_range}});
        json j = {{"version", VOX_VERSION_STRING}, {"cpu", system_info().to_json()}, {"gpus", gpus}, {"families", family_names()}};
        write_json(j.dump(), buf, cap, len);
    });
}

VOX_API vox_status vox_model_inspect(const char *path, char *buf, size_t cap, size_t *len) {
    return guard([&] {
        if (!path) fail(VOX_ERR_ARG, "path is required");
        TmodFile f(path);
        write_json(inspect_model(f).dump(), buf, cap, len);
    });
}

VOX_API vox_status vox_plan_json(const char *path, const char *config_json, char *buf, size_t cap, size_t *len) {
    return guard([&] {
        if (!path) fail(VOX_ERR_ARG, "path is required");
        json j = config_json && config_json[0] ? json::parse(config_json) : json::object();
        ResourceConfig cfg = ResourceConfig::from_json(j);
        TmodFile f(path);
        const json tuned = apply_tune(cfg, f);
        ResourcePlan p = plan_for_model(f, cfg);
        if (tuned.is_object()) p.notes.push_back("tuned: " + tuned.value("device", std::string()) + " (vox tune, " + tuned.value("when", std::string()) + ")");
        json out = p.to_json();
        out["config"] = cfg.to_json();
        out["tuned"] = tuned;
        out["summary"] = p.summary();
        write_json(out.dump(), buf, cap, len);
    });
}

VOX_API vox_status vox_tune_info(const char *path, char *buf, size_t cap, size_t *len) {
    return guard([&] {
        if (!path) fail(VOX_ERR_ARG, "path is required");
        TmodFile f(path);
        const json out = {{"cache", tune_cache_path()}, {"model_key", tune_model_key(f)}, {"hardware_key", tune_hw_key()},
                          {"hardware", tune_hardware()}, {"entry", tune_entry(f)}, {"candidates", tune_candidates(f)}};
        write_json(out.dump(), buf, cap, len);
    });
}

VOX_API vox_status vox_tune_save(const char *path, const char *entry_json) {
    return guard([&] {
        if (!path || !entry_json) fail(VOX_ERR_ARG, "path and entry_json are required");
        json e;
        try { e = json::parse(entry_json); } catch (const std::exception &ex) { fail(VOX_ERR_ARG, std::string("entry_json is not valid JSON: ") + ex.what()); }
        TmodFile f(path);
        tune_store(f, e);
    });
}

VOX_API vox_status vox_tune_forget(const char *path) {
    return guard([&] {
        if (!path) fail(VOX_ERR_ARG, "path is required");
        TmodFile f(path);
        tune_forget(f);
    });
}

VOX_API vox_status vox_write_test_model(const char *path) {
    return guard([&] {
        if (!path) fail(VOX_ERR_ARG, "path is required");
        write_test_model(path);
    });
}

// ---- Voice ----
namespace {
vox_voice *wrap(Profile p) {
    auto *v = new vox_voice{std::make_shared<ProfileRef>()};
    v->pr->p = std::move(p);
    return v;
}
// info: the profile, plus the 1.x fields of its default voice (name, mode, speaker, model …) for older clients
json voice_info_json(const Profile &p) {
    json j = p.to_json(false);
    json legacy;
    if (!p.default_voice.empty()) {
        const std::string ds = p.voices.at(p.default_voice).default_style;
        for (const auto &c : p.cache)
            if (c.voice == p.default_voice && c.style == ds) { legacy = voice_to_json(*c.data); break; }
    }
    if (legacy.is_null()) legacy = {{"mode", "profile"}, {"speaker", ""}, {"instruct", ""}, {"ref_text", ""},
                                    {"model", {{"family", ""}, {"variant", ""}, {"weights_sha256", ""}}}};
    for (auto it = legacy.begin(); it != legacy.end(); ++it)
        if (!j.contains(it.key())) j[it.key()] = it.value();
    j["format"] = "2.0";
    j["name"] = p.name;
    j["sampling"] = p.default_voice.empty() ? json::object() : p.voices.at(p.default_voice).options;
    j["language"] = p.default_voice.empty() ? "auto" : p.voices.at(p.default_voice).language;
    return j;
}
std::string opt_str(const char *s) { return s ? s : ""; }
}  // namespace

VOX_API vox_status vox_voice_create(vox_engine *e, const float *pcm, size_t n, uint32_t sr, const char *ref_text,
                                    vox_voice_mode mode, vox_voice **out) {
    return guard([&] {
        if (!e || !pcm || !n || !sr || !out) fail(VOX_ERR_ARG, "engine, pcm, n_samples, sample_rate and out are required");
        *out = nullptr;
        std::string text = ref_text ? ref_text : "";
        if (mode == VOX_VOICE_SPEAKER) fail(VOX_ERR_ARG, "use vox_voice_from_speaker for speaker/instruction voices");
        if (!e->caps.clone_icl && !e->caps.clone_xvector)
            fail(VOX_ERR_UNSUPPORTED, "this model cannot clone voices from a recording (no speaker encoder: CustomVoice/VoiceDesign type, or rebuild the .tmod from a Base checkpoint)");
        // the mode follows what the model can do: a model that clones only without a transcript (Chatterbox) takes the
        // recording alone, whatever was asked; one that clones only with it needs the text
        if (mode == VOX_VOICE_ICL && !e->caps.clone_icl) mode = VOX_VOICE_XVECTOR_ONLY;
        if (mode == VOX_VOICE_XVECTOR_ONLY && !e->caps.clone_xvector) {
            if (text.empty())
                fail(VOX_ERR_ARG, "this model clones a voice only from a recording and its transcript: give the text of the recording "
                                  "(vox voice create --text …; Python: clone_voice(audio, ref_text))");
            mode = VOX_VOICE_ICL;
        }
        if (mode == VOX_VOICE_ICL && text.empty())
            fail(VOX_ERR_ARG, "the text of the recording is needed to clone in the full (ICL) mode: give it (vox voice create --text …; "
                              "Python: clone_voice(audio, ref_text)) or clone from the voice alone, without the text "
                              "(vox voice create --xvector-only; Python: xvector_only=True; C: the mode VOX_VOICE_XVECTOR_ONLY)");
        PoolScope scope(e->pool.get());
        VoiceData v;
        fill_voice(*e->backend, e->tmod->manifest(), pcm, n, sr, text, mode, v);
        *out = wrap(profile_from_voice(v, v.name));
    });
}

VOX_API vox_status vox_voice_from_speaker(vox_engine *e, const char *speaker, const char *instruct, vox_voice **out) {
    return guard([&] {
        if (!e || !out) fail(VOX_ERR_ARG, "engine and out are required");
        *out = nullptr;
        std::string spk = speaker ? speaker : "", ins = instruct ? instruct : "";
        const Capabilities &c = e->caps;
        if (spk.empty() && ins.empty()) fail(VOX_ERR_ARG, "give a speaker name, an instruction, or both");
        if (!spk.empty()) {
            if (!c.named_speakers) fail(VOX_ERR_ARG, "this model has no built-in speakers (model type: use a CustomVoice model, or clone a voice from a recording)");
            spk = profile_key_lower(spk);   // "a,b,c" mixes speakers (Kokoro averages the voice packs): every name must exist
            check_speaker_names(c, spk);
        }
        if (!ins.empty() && !c.instruct) fail(VOX_ERR_ARG, "this model does not take style instructions (VoiceDesign and CustomVoice 1.7B do)");
        if (c.requires_instruct) {
            if (ins.empty()) fail(VOX_ERR_ARG, "this model (VoiceDesign) needs an instruction describing the voice");
            if (!spk.empty()) fail(VOX_ERR_ARG, "this model (VoiceDesign) has no named speakers: describe the voice with an instruction");
        }
        if (!c.named_speakers && !c.requires_instruct) fail(VOX_ERR_ARG, "this model cannot make voices without a recording");
        VoiceData v;
        v.mode = VOX_VOICE_SPEAKER;
        v.speaker = spk;
        v.instruct = ins;
        v.name = spk.empty() ? "designed" : spk;
        const json &mm = e->tmod->manifest().at("model");
        v.model_family = mm.value("family", "");
        v.model_variant = mm.value("variant", "");
        v.weights_sha256 = mm.value("weights_sha256", "");
        v.created = today_iso();
        *out = wrap(profile_from_voice(v, v.name));
    });
}

VOX_API vox_status vox_voice_load(vox_engine *e, const char *path, vox_voice **out) {
    return guard([&] {
        if (!e || !path || !out) fail(VOX_ERR_ARG, "engine, path and out are required");
        *out = nullptr;
        std::unique_ptr<vox_voice> v(wrap(profile_load(path)));
        // the default voice must work on this model (built now from the stored recording if the profile has no build for it)
        std::lock_guard<std::mutex> lk(v->pr->mu);
        resolve_base(e, v->pr->p, "", "", nullptr);
        *out = v.release();
    });
}

VOX_API vox_status vox_profile_new(const char *name, vox_voice **out) {
    return guard([&] {
        if (!out) fail(VOX_ERR_ARG, "out is required");
        Profile p;
        p.name = name && *name ? name : "profile";
        p.created = today_iso();
        *out = wrap(std::move(p));
    });
}

VOX_API vox_status vox_profile_open(const char *path, vox_voice **out) {
    return guard([&] {
        if (!path || !out) fail(VOX_ERR_ARG, "path and out are required");
        *out = nullptr;
        *out = wrap(profile_load(path));
    });
}

VOX_API vox_status vox_voice_add_recording(vox_voice *v, const char *voice, const char *style, const float *pcm, size_t n, uint32_t sr,
                                           const char *ref_text, const char *language) {
    return guard([&] {
        if (!v || !voice || !*voice || !pcm || !n || !sr) fail(VOX_ERR_ARG, "voice handle, voice name, pcm, n_samples and sample_rate are required");
        std::lock_guard<std::mutex> lk(v->pr->mu);
        Profile &p = v->pr->p;
        std::string vk = find_key(p.voices, voice);
        const bool newv = vk.empty();
        if (newv) vk = voice;
        std::string sk = style && *style ? style : "";
        if (!newv) {
            const VoiceDef &vd = p.voices.at(vk);
            if (sk.empty()) sk = vd.default_style;
            else if (const std::string k = find_key(vd.styles, sk); !k.empty()) sk = k;
        } else if (sk.empty()) {
            sk = "default";
        }
        const std::string id = p.add_recording(wav_encode(pcm, n, sr, true), opt_str(ref_text), opt_str(language));
        json patch = {{"voices", {{vk, {{"styles", {{sk, {{"ref", id}}}}}}}}}};
        if (newv) {
            patch["voices"][vk]["language"] = language && *language ? language : "auto";
            patch["voices"][vk]["default_style"] = sk;
        }
        profile_patch(p, patch);
    });
}

VOX_API vox_status vox_voice_merge(vox_voice *dst, const vox_voice *src, const char *src_voice, const char *as_voice, const char *as_style) {
    return guard([&] {
        if (!dst || !src) fail(VOX_ERR_ARG, "dst and src are required");
        Profile sp;
        {
            std::lock_guard<std::mutex> lk(src->pr->mu);
            sp = src->pr->p;
        }
        if (sp.voices.empty()) fail(VOX_ERR_ARG, "the source profile has no voices");
        const std::string svk = src_voice && *src_voice ? find_key(sp.voices, src_voice) : sp.default_voice;
        if (svk.empty()) sp.voice(src_voice);
        const VoiceDef &sv = sp.voices.at(svk);
        std::lock_guard<std::mutex> lk(dst->pr->mu);   // the source was copied above, under its own lock
        Profile &p = dst->pr->p;
        const std::string target = as_voice && *as_voice ? as_voice : svk;
        std::string tvk = find_key(p.voices, target);
        if (tvk.empty()) tvk = target;
        // recordings of the source voice, renamed if their ids clash
        std::map<std::string, std::string> idmap;
        for (const auto &[sn, st] : sv.styles)
            if (!st.ref.empty() && sp.recordings.count(st.ref) && !idmap.count(st.ref)) {
                const Recording &r = sp.recordings.at(st.ref);
                idmap[st.ref] = p.add_recording(r.wav, r.text, r.language);
            }
        json pj = p.to_json(false);
        json svj;
        {
            Profile tmp;
            tmp.voices[svk] = sv;
            svj = tmp.to_json(false)["voices"][svk];
        }
        for (auto it = svj["styles"].begin(); it != svj["styles"].end(); ++it) {
            const std::string r = it.value().value("ref", "");
            if (!r.empty()) it.value()["ref"] = idmap[r];
        }
        std::vector<std::pair<std::string, std::string>> cache_map;   // source style → target style
        if (as_style && *as_style) {   // the source voice's default style becomes one style of the target voice
            json st = svj["styles"][sv.default_style];
            if (st.value("ref", "").empty() && svj["styles"].contains(sv.default_style)) st["ref"] = svj["styles"][sv.default_style].value("ref", "");
            if (!pj["voices"].contains(tvk)) pj["voices"][tvk] = {{"language", sv.language}, {"description", sv.description}, {"styles", json::object()},
                                                                   {"speakers", sv.speakers}, {"options", sv.options}};
            pj["voices"][tvk]["styles"][as_style] = st;
            for (const auto &[k, sname] : sv.speakers) if (!pj["voices"][tvk]["speakers"].contains(k)) pj["voices"][tvk]["speakers"][k] = sname;
            cache_map.push_back({sv.default_style, as_style});
        } else {
            pj["voices"][tvk] = svj;
            for (const auto &[sn, _] : sv.styles) cache_map.push_back({sn, sn});
        }
        if (pj.value("default_voice", std::string()).empty()) pj["default_voice"] = tvk;
        Profile q = p;
        q.apply_json(pj);
        std::vector<CacheEntry> keep;
        for (const auto &c : q.cache)
            if (!(c.voice == tvk && std::any_of(cache_map.begin(), cache_map.end(), [&](const auto &m) { return m.second == c.style; }))) keep.push_back(c);
        for (const auto &c : sp.cache)
            if (c.voice == svk)
                for (const auto &m : cache_map)
                    if (m.first == c.style) {
                        auto d = std::make_shared<VoiceData>(*c.data);
                        d->name = tvk;
                        keep.push_back({tvk, m.second, d});
                    }
        q.cache = std::move(keep);
        q.drop_unused_recordings();
        p = std::move(q);
    });
}

VOX_API vox_status vox_voice_update(vox_voice *v, const char *patch_json) {
    return guard([&] {
        if (!v || !patch_json) fail(VOX_ERR_ARG, "voice and patch_json are required");
        json patch;
        try { patch = json::parse(patch_json); } catch (const std::exception &err) { fail(VOX_ERR_ARG, std::string("patch_json is not valid JSON: ") + err.what()); }
        std::lock_guard<std::mutex> lk(v->pr->mu);
        profile_patch(v->pr->p, patch);
    });
}

VOX_API vox_status vox_voice_prepare(vox_engine *e, vox_voice *v, const char *voice, const char *style, char *buf, size_t cap, size_t *len) {
    return guard([&] {
        if (!e || !v) fail(VOX_ERR_ARG, "engine and voice are required");
        std::lock_guard<std::mutex> lk(v->pr->mu);
        Profile &p = v->pr->p;
        json built = json::array(), notes = json::array();
        std::vector<std::string> vs;
        if (voice && *voice) { const std::string k = find_key(p.voices, voice); if (k.empty()) p.voice(voice); vs.push_back(k); }
        else for (const auto &kv : p.voices) vs.push_back(kv.first);
        for (const std::string &vk : vs) {
            std::vector<std::string> ss;
            const VoiceDef &vd = p.voices.at(vk);
            if (style && *style) { const std::string k = find_key(vd.styles, style); if (k.empty()) fail(VOX_ERR_ARG, "voice '" + vk + "' has no style '" + style + "'"); ss.push_back(k); }
            else for (const auto &kv : vd.styles) ss.push_back(kv.first);
            for (const std::string &sk : ss) {
                const size_t before = p.cache.size();
                resolve_base(e, p, vk, sk, &notes);
                built.push_back({{"voice", vk}, {"style", sk}, {"built", p.cache.size() > before}});
            }
        }
        write_json(json({{"model", model_key(e)}, {"voices", built}, {"notes", notes}}).dump(), buf, cap, len);
    });
}

VOX_API vox_status vox_voice_save(const vox_voice *v, const char *path) {
    return guard([&] {
        if (!v || !path) fail(VOX_ERR_ARG, "voice and path are required");
        Profile copy;
        {
            std::lock_guard<std::mutex> lk(v->pr->mu);
            copy = v->pr->p;
        }
        profile_save(copy, path);
    });
}

VOX_API vox_status vox_voice_set_name(vox_voice *v, const char *name) {
    return guard([&] {
        if (!v || !name || !*name) fail(VOX_ERR_ARG, "voice and name are required");
        std::lock_guard<std::mutex> lk(v->pr->mu);
        Profile &p = v->pr->p;
        p.name = name;
        // a one-voice profile (what 1.x called a voice): the voice is renamed too, when the name is a valid voice name
        if (p.voices.size() == 1 && p.voices.begin()->first != name) {
            const std::string old = p.voices.begin()->first;
            json j = p.to_json(false);
            json vj = j["voices"][old];
            j["voices"] = json::object();
            j["voices"][name] = vj;
            j["default_voice"] = name;
            Profile q = p;
            try { q.apply_json(j); } catch (const Error &) { return; }   // not a valid voice name: only the profile is renamed
            for (auto &c : q.cache)
                if (c.voice == old) {
                    auto d = std::make_shared<VoiceData>(*c.data);
                    d->name = name;
                    c = {name, c.style, d};
                }
            p = std::move(q);
        }
    });
}

VOX_API vox_status vox_voice_set_defaults(vox_voice *v, const vox_gen_params *pin) {
    return guard([&] {
        if (!v || !pin) fail(VOX_ERR_ARG, "voice and params are required");
        vox_gen_params p = read_struct(pin, vox_gen_params_default);
        std::lock_guard<std::mutex> lk(v->pr->mu);
        Profile &pf = v->pr->p;
        if (pf.default_voice.empty()) fail(VOX_ERR_STATE, "the profile has no voices");
        VoiceDef &vd = pf.voices.at(pf.default_voice);
        json &j = vd.options;
        auto setf = [&](const char *k, float x) { if (x >= 0) j[k] = x; };
        setf("temperature", p.temperature); setf("top_p", p.top_p);
        setf("repetition_penalty", p.repetition_penalty);
        setf("sub_temperature", p.sub_temperature); setf("sub_top_p", p.sub_top_p);
        if (p.top_k >= 0) j["top_k"] = p.top_k;
        if (p.sub_top_k >= 0) j["sub_top_k"] = p.sub_top_k;
        if (p.seed) j["seed"] = p.seed;
        if (p.language) vd.language = p.language;
        if (p.extra_json && *p.extra_json) {
            json ex;
            try { ex = json::parse(p.extra_json); } catch (const std::exception &err) { fail(VOX_ERR_ARG, std::string("extra_json is not valid JSON: ") + err.what()); }
            if (!ex.is_object()) fail(VOX_ERR_ARG, "extra_json must be a JSON object");
            for (auto it = ex.begin(); it != ex.end(); ++it) j[it.key()] = it.value();
        }
    });
}

VOX_API vox_status vox_voice_info(const vox_voice *v, char *buf, size_t cap, size_t *len) {
    return guard([&] {
        if (!v) fail(VOX_ERR_ARG, "voice is null");
        std::lock_guard<std::mutex> lk(v->pr->mu);
        write_json(voice_info_json(v->pr->p).dump(), buf, cap, len);
    });
}

VOX_API void vox_voice_free(vox_voice *v) { delete v; }

// ---- Session ----
VOX_API vox_status vox_session_create(vox_engine *e, const vox_voice *v, const vox_gen_params *pin, vox_session **out) {
    return guard([&] {
        if (!e || !v || !out) fail(VOX_ERR_ARG, "engine, voice and out are required");
        *out = nullptr;
        vox_gen_params p = read_struct(pin, vox_gen_params_default);
        const json md = e->tmod->manifest().value("generation", json::object());

        PoolScope scope(e->pool.get());
        auto s = std::make_unique<vox_session>();
        s->prof = v->pr;
        s->params = p;
        s->params.language = nullptr;   // the pointer does not outlive this call: the language is kept below
        s->chunk_frames = p.chunk_frames ? p.chunk_frames : md.value("chunk_frames", 4u);
        s->max_frames = p.max_frames ? p.max_frames : md.value("max_frames", 2048u);
        if (p.decoder_mode > VOX_DECODER_REFERENCE) fail(VOX_ERR_ARG, "unknown decoder_mode " + std::to_string(p.decoder_mode));
        json ex = json::object();
        if (p.extra_json && *p.extra_json) {
            try { ex = json::parse(p.extra_json); } catch (const std::exception &err) { fail(VOX_ERR_ARG, std::string("extra_json is not valid JSON: ") + err.what()); }
            if (!ex.is_object()) fail(VOX_ERR_ARG, "extra_json must be a JSON object");
        }
        s->params.extra_json = nullptr;
        // profile selection and markup (0.15): {"voice": "anna", "style": "happy", "markup": "auto" | "strict" | "off"}
        SegSpec spec;
        if (ex.contains("voice")) { if (!ex["voice"].is_string()) fail(VOX_ERR_ARG, "extra_json.voice must be a string"); spec.voice = ex["voice"].get<std::string>(); }
        if (ex.contains("style")) { if (!ex["style"].is_string()) fail(VOX_ERR_ARG, "extra_json.style must be a string"); spec.style = ex["style"].get<std::string>(); }
        const std::string markup = ex.value("markup", std::string("auto"));
        if (markup == "off") s->mk.mode = MarkupParser::Off;
        else if (markup == "strict") s->mk.mode = MarkupParser::Strict;
        else if (markup == "auto") s->mk.mode = MarkupParser::Auto;
        else fail(VOX_ERR_ARG, "extra_json.markup must be \"auto\", \"strict\" or \"off\", got \"" + markup + "\"");
        ex.erase("voice"); ex.erase("style"); ex.erase("markup");
        s->session_extra = ex;
        {
            std::lock_guard<std::mutex> lk(v->pr->mu);
            const Profile &pf = v->pr->p;
            if (!spec.voice.empty()) {
                const std::string vk = find_key(pf.voices, spec.voice);
                if (vk.empty()) pf.voice(spec.voice);   // throws with the list
                spec.voice = vk;
            }
            if (!spec.style.empty() && !profile_has_style(pf, spec.voice, spec.style)) {
                const std::string emo = markup_emotion(spec.style);
                if (emo.empty()) {
                    const VoiceDef &vd = pf.voice(spec.voice.empty() ? pf.default_voice : spec.voice);
                    std::string all;
                    for (const auto &kv : vd.styles) all += (all.empty() ? "" : ", ") + kv.first;
                    fail(VOX_ERR_ARG, "the voice has no style '" + spec.style + "'; it has: " + all);
                }
                spec.style.clear();   // an emotion without a style of its own: mapped like the [emotion] tag
                if (!emotion_options_for(e, emo).empty()) spec.extra_opts = emotion_options_for(e, emo);
                else if (e->caps.instruct) spec.extra_instruct = emotion_instruct_for(e, emo);
                else if (e->caps.speed) spec.speed = emotion_speed(emo);
            }
            std::vector<std::string> names;
            for (const auto &kv : pf.voices) names.push_back(profile_key_lower(kv.first));
            s->mk.dialogue = names.size() >= 2;
            auto pr = v->pr;
            s->mk.is_voice = [names](const std::string &n) {
                return std::find(names.begin(), names.end(), profile_key_lower(n)) != names.end();
            };
            s->mk.is_style = [pr, sp = s.get()](const std::string &n) {   // called with pr->mu not held (parse runs outside the lock)
                std::lock_guard<std::mutex> lk2(pr->mu);
                return profile_has_style(pr->p, sp->queue.empty() ? sp->active_spec.voice : sp->queue.back().spec.voice, n);
            };
        }
        if (p.language) s->language = p.language;
        s->direct = e->backend->direct();
        if (ex.contains("stream") && ex["stream"].is_object()) {
            const json &sj = ex["stream"];
            const std::string c = sj.value("commit", std::string("auto"));
            if (c == "auto") s->commit = vox_session::Commit::Auto;
            else if (c == "sentence") s->commit = vox_session::Commit::Sentence;
            else if (c == "manual") s->commit = vox_session::Commit::Manual;
            else fail(VOX_ERR_ARG, "stream.commit must be \"auto\", \"sentence\" or \"manual\", got \"" + c + "\"");
            if (sj.contains("idle_ms")) {
                if (!sj["idle_ms"].is_number() || sj["idle_ms"].get<double>() < 0) fail(VOX_ERR_ARG, "stream.idle_ms must be a non-negative number");
                s->idle_ms = uint32_t(sj["idle_ms"].get<double>());
            }
        }
        // допуск сессии: предел числа сессий и бюджет памяти (KV-кэши, состояние декодера)
        if (e->plan.max_sessions && e->sessions.load() >= e->plan.max_sessions)
            fail(VOX_ERR_OOM, "session limit reached (" + std::to_string(e->plan.max_sessions) + " concurrent sessions allowed by the resource config)");
        s->e = e;
        begin_segment(s.get(), spec);   // builds the voice for this model if the profile does not have it yet
        SessionOptions so;
        so.max_frames = s->max_frames;
        so.options = ex;
        const SessionCost cost = e->backend->session_cost(*s->voice, so);
        if (!e->budget.try_reserve(cost.ram)) {
            fail(VOX_ERR_OOM, "memory limit: this session needs ~" + format_size(cost.ram) + " for KV caches and decoder state, " +
                                  format_size(e->budget.limit() - std::min(e->budget.limit(), e->budget.used())) + " of " + format_size(e->budget.limit()) +
                                  " is left (lower max_frames, close other sessions, or raise ram_limit)");
        }
        s->reserved_ram = cost.ram;
        s->counted = true;
        e->sessions.fetch_add(1);
        e->created.fetch_add(1);
        *out = s.release();
    });
}

VOX_API vox_status vox_session_feed_text(vox_session *s, const char *utf8) {
    return guard([&] {
        if (!s || !utf8) fail(VOX_ERR_ARG, "session and text are required");
        if (s->text_ended) fail(VOX_ERR_STATE, "text already ended");
        if (!*utf8) return;
        s->pending += utf8;
        s->last_feed = std::chrono::steady_clock::now();
        s->idle_armed = true;
        commit_pending(s, false);
    });
}

VOX_API vox_status vox_session_flush(vox_session *s) {
    return guard([&] {
        if (!s) fail(VOX_ERR_ARG, "session is null");
        if (s->text_ended) return;
        if (commit_pending(s, true)) ++s->flushes;
        else if (s->direct) s->e->backend->direct_seal(*s->st);  // нечего коммитить, но озвученный отрезок надо закрыть (отдать хвост)
        s->idle_armed = false;
    });
}

VOX_API vox_status vox_session_set_pending(vox_session *s, const char *utf8) {
    return guard([&] {
        if (!s || !utf8) fail(VOX_ERR_ARG, "session and text are required");
        if (s->text_ended) fail(VOX_ERR_STATE, "text already ended");
        s->pending = utf8;
        s->last_feed = std::chrono::steady_clock::now();
        s->idle_armed = !s->pending.empty();
        commit_pending(s, false);
    });
}

VOX_API vox_status vox_session_end_text(vox_session *s) {
    return guard([&] {
        if (!s) fail(VOX_ERR_ARG, "session is null");
        if (s->text_ended) return;
        commit_pending(s, true);
        s->text_ended = true;
    });
}

VOX_API vox_status vox_session_info(const vox_session *s, char *buf, size_t cap, size_t *len) {
    return guard([&] {
        if (!s) fail(VOX_ERR_ARG, "session is null");
        const char *cm = s->commit == vox_session::Commit::Auto ? "auto" : s->commit == vox_session::Commit::Sentence ? "sentence" : "manual";
        json j = {{"committed_bytes", s->committed_bytes}, {"pending", s->pending}, {"pending_bytes", s->pending.size()},
                  {"text_ended", s->text_ended}, {"finished", s->finished}, {"frames", s->frame}, {"commit", cm}, {"idle_ms", s->idle_ms},
                  {"flushes", s->flushes}, {"idle_flushes", s->idle_flushes}, {"direct", s->direct},
                  {"segment", s->seg_index}, {"queued_segments", s->queue.size()},
                  {"voice", s->voice ? s->voice->name : std::string()}, {"style", s->active_spec.style},
                  {"markup", s->mk.mode == MarkupParser::Off ? "off" : s->mk.mode == MarkupParser::Strict ? "strict" : "auto"},
                  {"hints_applied", s->hints_applied}, {"notes", s->notes}};
        if (s->st) {
            json mi = s->e->backend->state_info(*s->st);
            if (!mi.is_null()) j["model"] = mi;
        }
        write_json(j.dump(), buf, cap, len);
    });
}

VOX_API vox_status vox_session_pull(vox_session *s, float *out, size_t cap, size_t *written) {
    if (written) *written = 0;
    vox_status result = VOX_AGAIN;
    const auto t_start = std::chrono::steady_clock::now();
    vox_status st = guard([&] {
        if (!s || !out || !cap) fail(VOX_ERR_ARG, "session, buffer and capacity are required");
        PoolScope scope(s->e->pool.get());
        if (s->cancelled.load()) fail(VOX_ERR_CANCELLED, "session cancelled");
        check_idle(s);
        for (;;) {
            // a segment that got no text (a pause or a tag right before the next one): nothing to synthesize
            if (!s->finished && inner_ended(s) && !s->active_has_text && !s->inflight.valid() && s->buffered == 0) {
                s->finished = s->flushed = true;
                s->first_decoded = true;
            }
            if (s->direct && s->pcm_pos >= s->pcm.size() && !s->finished) {
                // модель без покадровых кодов: шаг модели дописывает звук (по предложению), пока не озвучен весь текст
                if (s->pcm_pos >= s->pcm.size()) { s->pcm.clear(); s->pcm_pos = 0; }
                const size_t before = s->pcm.size();
                if (s->cancelled.load()) fail(VOX_ERR_CANCELLED, "session cancelled");
                const bool done = s->e->backend->direct_step(*s->st, inner_ended(s), s->pcm);
                if (done) s->finished = s->flushed = true;
                if (s->pcm.size() > before) continue;
                if (done) { if (next_segment(s)) continue; result = VOX_DONE; return; }
                if (inner_ended(s)) continue;   // текст весь: кусок без звука, модель идёт дальше сама
                result = VOX_AGAIN;  // ждёт текст
                return;
            }
            if (s->pcm_pos < s->pcm.size()) {
                size_t n = std::min(cap, s->pcm.size() - s->pcm_pos);
                std::memcpy(out, s->pcm.data() + s->pcm_pos, n * sizeof(float));
                s->pcm_pos += n;
                if (written) *written = n;
                result = VOX_OK;
                return;
            }
            if (s->inflight.valid()) {
                // Декодер крутится на CPU — тем временем GPU считает следующий чанк (если есть текст).
                if (!s->finished && s->buffered < s->chunk_frames) {
                    try { generate(s); } catch (...) { try { collect_async_decode(s); } catch (...) {} throw; }
                }
                collect_async_decode(s);
                continue;
            }
            if (s->direct || (s->finished && s->flushed)) {   // сегмент озвучен и отдан: следующий сегмент разметки или конец
                if (s->finished && next_segment(s)) continue;
                if (s->direct && !s->finished) { result = VOX_AGAIN; return; }
                result = VOX_DONE;
                return;
            }
            if (s->finished) {
                decode_buffered(s);
                s->e->backend->flush(*s->st, s->pcm);
                s->flushed = true;
                continue;
            }
            generate(s);
            if (s->buffered >= s->chunk_frames && !s->finished && s->first_decoded && s->e->backend->decode_overlappable()) {
                start_async_decode(s);  // первый чанк декодируем синхронно (низкая задержка до первого звука)
                continue;
            }
            if (s->buffered >= s->chunk_frames || s->finished) { decode_buffered(s); continue; }
            // Не хватает текста: отдаём то, что уже есть, ради низкой задержки.
            if (s->buffered) { decode_buffered(s); continue; }
            result = VOX_AGAIN;
            return;
        }
    });
    if (s && st == VOX_OK) {
        s->e->pull_us.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_start).count());
        if (result == VOX_OK && written) s->e->samples.fetch_add(int64_t(*written));
    }
    return st == VOX_OK ? result : st;
}

VOX_API void vox_session_cancel(vox_session *s) {
    if (s) s->cancelled.store(true);
}

VOX_API void vox_session_free(vox_session *s) { delete s; }

VOX_API const char *vox_last_error(void) {
    static thread_local std::string copy;
    copy = vox_internal_last_error();
    return copy.c_str();
}

}  // extern "C"
