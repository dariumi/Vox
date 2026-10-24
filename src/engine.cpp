// Реализация C ABI. Все исключения ловятся здесь и превращаются в vox_status.
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <future>

#include "backend.hpp"
#include "family.hpp"
#include "kernels.hpp"
#include "sampling.hpp"
#include "voice.hpp"
#include "wav.hpp"

using namespace vox;

#ifndef VOX_VERSION_STRING
#define VOX_VERSION_STRING "0.14.0"
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
};

struct vox_voice {
    std::shared_ptr<VoiceData> data;
};

struct vox_session {
    vox_engine *e = nullptr;
    std::shared_ptr<const VoiceData> voice;
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
    bool direct = false;            // модель без покадровых кодов: звук делает backend->direct_step

    // Конвейер «GPU генерирует чанк k+1, пока CPU декодирует чанк k» (только если backend->decode_overlappable()).
    bool first_decoded = false;
    std::vector<int16_t> inflight_codes;
    std::vector<float> inflight_pcm;
    std::future<void> inflight;  // объявлен последним: ~vox_session дождётся потока до разрушения остального
    ~vox_session() {
        if (inflight.valid()) { try { inflight.wait(); } catch (...) {} }
        if (e) { e->budget.release(reserved_ram); e->sessions.fetch_sub(1); }
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
    if (cut == 0) return 0;
    const std::string piece = t.substr(0, cut);
    if (s->direct) {
        s->e->backend->direct_text(*s->st, piece);
    } else {
        auto toks = s->e->backend->tokenize(piece);
        if (!toks.empty()) s->e->backend->append_text(*s->st, toks);
    }
    t.erase(0, cut);
    s->committed_bytes += cut;
    // модели, озвучивающие куски целиком: «законченный кусок» — повод озвучить его сразу, не ожидая продолжения
    if (s->direct && (force || s->commit != vox_session::Commit::Auto)) s->e->backend->direct_seal(*s->st);
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
        if (b.needs_text(*s->st, s->text_ended)) return;
        if (s->frame >= s->max_frames) { s->finished = true; return; }

        b.talker_logits(*s->st, s->text_ended, s->logits);
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
            if (take("max_sessions")) cfg.max_sessions = c2.max_sessions;
            if (take("planned_sessions")) cfg.planned_sessions = c2.planned_sessions;
        }
        if (!cfg.threads && o.n_threads) cfg.threads = o.n_threads;
        if (!cfg.ram_limit && o.memory_limit) cfg.ram_limit = o.memory_limit;
        if (o.max_sessions > cfg.planned_sessions) cfg.planned_sessions = o.max_sessions;
        e->config = cfg;
        e->tmod = std::make_shared<TmodFile>(path);
        e->plan = plan_for_model(*e->tmod, cfg);
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
        const ResourceConfig cfg = ResourceConfig::from_json(j);
        TmodFile f(path);
        const ResourcePlan p = plan_for_model(f, cfg);
        json out = p.to_json();
        out["config"] = cfg.to_json();
        out["summary"] = p.summary();
        write_json(out.dump(), buf, cap, len);
    });
}

VOX_API vox_status vox_write_test_model(const char *path) {
    return guard([&] {
        if (!path) fail(VOX_ERR_ARG, "path is required");
        write_test_model(path);
    });
}

// ---- Voice ----
VOX_API vox_status vox_voice_create(vox_engine *e, const float *pcm, size_t n, uint32_t sr, const char *ref_text,
                                    vox_voice_mode mode, vox_voice **out) {
    return guard([&] {
        if (!e || !pcm || !n || !sr || !out) fail(VOX_ERR_ARG, "engine, pcm, n_samples, sample_rate and out are required");
        *out = nullptr;
        std::string text = ref_text ? ref_text : "";
        if (mode == VOX_VOICE_SPEAKER) fail(VOX_ERR_ARG, "use vox_voice_from_speaker for speaker/instruction voices");
        if (!e->caps.clone_icl && !e->caps.clone_xvector)
            fail(VOX_ERR_UNSUPPORTED, "this model cannot clone voices from a recording (no speaker encoder: CustomVoice/VoiceDesign type, or rebuild the .tmod from a Base checkpoint)");
        if (mode == VOX_VOICE_ICL && text.empty())
            fail(VOX_ERR_ARG, "ref_text is required in ICL mode (or use VOX_VOICE_XVECTOR_ONLY)");
        PoolScope scope(e->pool.get());
        auto v = std::make_shared<VoiceData>();
        fill_voice(*e->backend, e->tmod->manifest(), pcm, n, sr, text, mode, *v);
        *out = new vox_voice{v};
    });
}

VOX_API vox_status vox_voice_from_speaker(vox_engine *e, const char *speaker, const char *instruct, vox_voice **out) {
    return guard([&] {
        if (!e || !out) fail(VOX_ERR_ARG, "engine and out are required");
        *out = nullptr;
        std::string spk = speaker ? speaker : "", ins = instruct ? instruct : "";
        const Capabilities &c = e->caps;
        if (spk.empty() && ins.empty()) fail(VOX_ERR_ARG, "give a speaker name, an instruction, or both");
        std::string spk_l = spk;
        for (auto &ch : spk_l) ch = char(std::tolower(static_cast<unsigned char>(ch)));
        if (!spk.empty()) {
            if (!c.named_speakers) fail(VOX_ERR_ARG, "this model has no built-in speakers (model type: use a CustomVoice model, or clone a voice from a recording)");
            // "a,b,c" mixes speakers (Kokoro averages the voice packs): every name must exist
            size_t a0 = 0;
            while (a0 <= spk_l.size()) {
                size_t b0 = spk_l.find(',', a0);
                if (b0 == std::string::npos) b0 = spk_l.size();
                const std::string one = spk_l.substr(a0, b0 - a0);
                if (std::find(c.speakers.begin(), c.speakers.end(), one) == c.speakers.end()) {
                    std::string all;
                    for (const auto &s : c.speakers) all += (all.empty() ? "" : ", ") + s;
                    fail(VOX_ERR_ARG, "unknown speaker '" + one + "'; this model has: " + all);
                }
                a0 = b0 + 1;
            }
            spk = spk_l;
        }
        if (!ins.empty() && !c.instruct) fail(VOX_ERR_ARG, "this model does not take style instructions (VoiceDesign and CustomVoice 1.7B do)");
        if (c.requires_instruct) {
            if (ins.empty()) fail(VOX_ERR_ARG, "this model (VoiceDesign) needs an instruction describing the voice");
            if (!spk.empty()) fail(VOX_ERR_ARG, "this model (VoiceDesign) has no named speakers: describe the voice with an instruction");
        }
        if (!c.named_speakers && !c.requires_instruct) fail(VOX_ERR_ARG, "this model cannot make voices without a recording");
        auto v = std::make_shared<VoiceData>();
        v->mode = VOX_VOICE_SPEAKER;
        v->speaker = spk;
        v->instruct = ins;
        v->name = spk.empty() ? "designed" : spk;
        const json &mm = e->tmod->manifest().at("model");
        v->model_family = mm.value("family", "");
        v->model_variant = mm.value("variant", "");
        v->weights_sha256 = mm.value("weights_sha256", "");
        *out = new vox_voice{v};
    });
}

VOX_API vox_status vox_voice_load(vox_engine *e, const char *path, vox_voice **out) {
    return guard([&] {
        if (!e || !path || !out) fail(VOX_ERR_ARG, "engine, path and out are required");
        *out = nullptr;
        PoolScope scope(e->pool.get());
        auto v = std::make_shared<VoiceData>(voice_load(path));
        const json &mm = e->tmod->manifest().at("model");
        std::string fam = mm.value("family", ""), sha = mm.value("weights_sha256", "");
        bool fam_ok = v->model_family.empty() || v->model_family == fam;
        bool sha_ok = v->weights_sha256.empty() || sha.empty() || v->weights_sha256 == sha;
        if (v->mode == VOX_VOICE_SPEAKER) {
            if (!fam_ok || (v->model_variant != mm.value("variant", "") && !v->model_variant.empty()))
                fail(VOX_ERR_FORMAT, "speaker voice '" + v->name + "' belongs to " + v->model_family + "/" + v->model_variant +
                                         ": speaker ids differ between models, create it again for this model");
        } else if (!fam_ok || !sha_ok) {
            // Профиль от другой модели или других весов: пересоздаём из сохранённой записи.
            if (v->ref_audio.empty())
                fail(VOX_ERR_FORMAT, "voice was created for " + v->model_family + " with other weights and has no "
                                     "ref_audio to rebuild from; create it again from the reference recording");
            Audio a = wav_decode(v->ref_audio.data(), v->ref_audio.size());
            fill_voice(*e->backend, e->tmod->manifest(), a.pcm.data(), a.pcm.size(), a.sample_rate, v->ref_text,
                       v->mode, *v);
        }
        *out = new vox_voice{v};
    });
}

VOX_API vox_status vox_voice_save(const vox_voice *v, const char *path) {
    return guard([&] {
        if (!v || !path) fail(VOX_ERR_ARG, "voice and path are required");
        voice_save(*v->data, path);
    });
}

VOX_API vox_status vox_voice_set_name(vox_voice *v, const char *name) {
    return guard([&] {
        if (!v || !name) fail(VOX_ERR_ARG, "voice and name are required");
        v->data->name = name;
    });
}

VOX_API vox_status vox_voice_set_defaults(vox_voice *v, const vox_gen_params *pin) {
    return guard([&] {
        if (!v || !pin) fail(VOX_ERR_ARG, "voice and params are required");
        vox_gen_params p = read_struct(pin, vox_gen_params_default);
        json &j = v->data->sampling;
        auto setf = [&](const char *k, float x) { if (x >= 0) j[k] = x; };
        setf("temperature", p.temperature); setf("top_p", p.top_p);
        setf("repetition_penalty", p.repetition_penalty);
        setf("sub_temperature", p.sub_temperature); setf("sub_top_p", p.sub_top_p);
        if (p.top_k >= 0) j["top_k"] = p.top_k;
        if (p.sub_top_k >= 0) j["sub_top_k"] = p.sub_top_k;
        if (p.seed) j["seed"] = p.seed;
        if (p.language) v->data->language = p.language;
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
        write_json(voice_to_json(*v->data).dump(), buf, cap, len);
    });
}

VOX_API void vox_voice_free(vox_voice *v) { delete v; }

// ---- Session ----
VOX_API vox_status vox_session_create(vox_engine *e, const vox_voice *v, const vox_gen_params *pin, vox_session **out) {
    return guard([&] {
        if (!e || !v || !out) fail(VOX_ERR_ARG, "engine, voice and out are required");
        *out = nullptr;
        vox_gen_params p = read_struct(pin, vox_gen_params_default);
        const json &vs = v->data->sampling;
        const json md = e->tmod->manifest().value("generation", json::object());

        PoolScope scope(e->pool.get());
        auto s = std::make_unique<vox_session>();
        s->voice = v->data;
        s->talker.temperature = pick(p.temperature, vs, md, "temperature", 0.9f);
        s->talker.top_p = pick(p.top_p, vs, md, "top_p", 1.0f);
        s->talker.top_k = int32_t(pick(float(p.top_k), vs, md, "top_k", 50));
        s->talker.repetition_penalty = pick(p.repetition_penalty, vs, md, "repetition_penalty", 1.05f);
        s->sub.temperature = pick(p.sub_temperature, vs, md, "sub_temperature", 0.9f);
        s->sub.top_p = pick(p.sub_top_p, vs, md, "sub_top_p", 1.0f);
        s->sub.top_k = int32_t(pick(float(p.sub_top_k), vs, md, "sub_top_k", 50));
        uint64_t seed = p.seed ? p.seed : vs.value("seed", uint64_t(0));
        s->rng = DetRng(seed ? seed : 1);  // без явного seed результат всё равно воспроизводим
        s->chunk_frames = p.chunk_frames ? p.chunk_frames : md.value("chunk_frames", 4u);
        s->max_frames = p.max_frames ? p.max_frames : md.value("max_frames", 2048u);
        std::string lang = p.language ? p.language : v->data->language;
        SessionOptions so;
        so.decoder_context_frames = p.decoder_context_frames;
        if (p.decoder_mode > VOX_DECODER_REFERENCE) fail(VOX_ERR_ARG, "unknown decoder_mode " + std::to_string(p.decoder_mode));
        so.decoder_mode = p.decoder_mode;
        so.max_frames = s->max_frames;
        so.seed = seed ? seed : 1;
        so.options = md;                                   // манифест < профиль голоса < extra_json сессии
        for (auto it = vs.begin(); it != vs.end(); ++it) so.options[it.key()] = it.value();
        if (p.extra_json && *p.extra_json) {
            json ex;
            try { ex = json::parse(p.extra_json); } catch (const std::exception &err) { fail(VOX_ERR_ARG, std::string("extra_json is not valid JSON: ") + err.what()); }
            if (!ex.is_object()) fail(VOX_ERR_ARG, "extra_json must be a JSON object");
            for (auto it = ex.begin(); it != ex.end(); ++it) so.options[it.key()] = it.value();
        }
        // допуск сессии: предел числа сессий и бюджет памяти (KV-кэши, состояние декодера)
        if (e->plan.max_sessions && e->sessions.load() >= e->plan.max_sessions)
            fail(VOX_ERR_OOM, "session limit reached (" + std::to_string(e->plan.max_sessions) + " concurrent sessions allowed by the resource config)");
        const SessionCost cost = e->backend->session_cost(*v->data, so);
        if (!e->budget.try_reserve(cost.ram))
            fail(VOX_ERR_OOM, "memory limit: this session needs ~" + format_size(cost.ram) + " for KV caches and decoder state, " +
                                  format_size(e->budget.limit() - std::min(e->budget.limit(), e->budget.used())) + " of " + format_size(e->budget.limit()) +
                                  " is left (lower max_frames, close other sessions, or raise ram_limit)");
        s->e = e;
        s->reserved_ram = cost.ram;
        e->sessions.fetch_add(1);
        e->created.fetch_add(1);
        s->st = e->backend->begin(*v->data, lang);
        e->backend->configure(*s->st, so);
        s->direct = e->backend->direct();
        if (so.options.contains("stream") && so.options["stream"].is_object()) {
            const json &sj = so.options["stream"];
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
                  {"flushes", s->flushes}, {"idle_flushes", s->idle_flushes}, {"direct", s->direct}};
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
            if (s->direct && s->pcm_pos >= s->pcm.size() && !s->finished) {
                // модель без покадровых кодов: шаг модели дописывает звук (по предложению), пока не озвучен весь текст
                if (s->pcm_pos >= s->pcm.size()) { s->pcm.clear(); s->pcm_pos = 0; }
                const size_t before = s->pcm.size();
                if (s->cancelled.load()) fail(VOX_ERR_CANCELLED, "session cancelled");
                const bool done = s->e->backend->direct_step(*s->st, s->text_ended, s->pcm);
                if (done) s->finished = s->flushed = true;
                if (s->pcm.size() > before) continue;
                if (done) { result = VOX_DONE; return; }
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
            if (s->direct) { result = VOX_DONE; return; }  // всё озвучено и отдано
            if (s->finished) {
                if (s->flushed) { result = VOX_DONE; return; }
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
