// Низкоуровневый модуль vox._vox: тонкая обёртка над C ABI.
// Удобный API — в python/vox/__init__.py.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <cstring>
#include <optional>
#include <string>

#include "py_family.hpp"
#include "vox.h"

namespace nb = nanobind;
using namespace nb::literals;

using F32Array = nb::ndarray<nb::numpy, float, nb::ndim<1>>;
using F32In = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

static nb::object g_error;  // vox.VoxError
static std::vector<std::string> g_py_families;

[[noreturn]] static void raise(vox_status st) {
    // g_error — фабрика (status, message) → исключение нужного подкласса VoxError
    nb::object exc = g_error(int(st), std::string(vox_last_error()));
    PyErr_SetObject(reinterpret_cast<PyObject *>(Py_TYPE(exc.ptr())), exc.ptr());
    throw nb::python_error();
}
static void check(vox_status st) {
    if (st < 0) raise(st);
}

static std::string json_free(vox_status (*fn)(char *, size_t, size_t *)) {
    size_t n = 0;
    check(fn(nullptr, 0, &n));
    std::string s(n + 1, '\0');
    check(fn(s.data(), s.size(), &n));
    s.resize(n);
    return s;
}

static std::string json_of(vox_status (*fn)(const void *, char *, size_t, size_t *), const void *obj) {
    size_t n = 0;
    check(fn(obj, nullptr, 0, &n));
    std::string s(n + 1, '\0');
    check(fn(obj, s.data(), s.size(), &n));
    s.resize(n);
    return s;
}

struct Engine {
    vox_engine *h = nullptr;
    Engine(const std::string &path, const std::string &device, uint32_t threads, const std::string &config_json) {
        vox_engine_opts o;
        vox_engine_opts_default(&o);
        o.device = device.c_str();
        o.n_threads = threads;
        if (!config_json.empty()) o.config_json = config_json.c_str();
        vox_status st;
        {
            nb::gil_scoped_release r;
            st = vox_engine_open(path.c_str(), &o, &h);
        }
        check(st);
    }
    ~Engine() { vox_engine_close(h); }
    uint32_t sample_rate() const { return vox_engine_sample_rate(h); }
    std::string info() const {
        return json_of([](const void *p, char *b, size_t c, size_t *l) {
            return vox_engine_info(static_cast<const vox_engine *>(p), b, c, l); }, h);
    }
    std::string stats() const {
        return json_of([](const void *p, char *b, size_t c, size_t *l) {
            return vox_engine_stats(static_cast<const vox_engine *>(p), b, c, l); }, h);
    }
    std::string phonemize(const std::string &text, const std::string &lang) const {
        struct Arg { vox_engine *e; const std::string *t, *l; } arg{h, &text, &lang};
        return json_of([](const void *p, char *b, size_t c, size_t *l) {
            const Arg *x = static_cast<const Arg *>(p);
            return vox_engine_phonemize(x->e, x->t->c_str(), x->l->c_str(), b, c, l); }, &arg);
    }
};

struct Voice {
    vox_voice *h = nullptr;
    nb::object engine;  // держит Engine живым, пока жив голос
    ~Voice() { vox_voice_free(h); }
    void save(const std::string &path) const { check(vox_voice_save(h, path.c_str())); }
    void set_name(const std::string &n) { check(vox_voice_set_name(h, n.c_str())); }
    std::string info() const {
        return json_of([](const void *p, char *b, size_t c, size_t *l) {
            return vox_voice_info(static_cast<const vox_voice *>(p), b, c, l); }, h);
    }
};

// Параметры генерации из kwargs Python: None = не задано.
struct Params {
    vox_gen_params p;
    std::string lang, extra;
    Params(std::optional<float> temperature, std::optional<int> top_k, std::optional<float> top_p,
           std::optional<float> repetition_penalty, std::optional<float> sub_temperature,
           std::optional<int> sub_top_k, std::optional<float> sub_top_p, std::optional<uint64_t> seed,
           std::optional<uint32_t> chunk_frames, std::optional<uint32_t> max_frames,
           std::optional<std::string> language, std::optional<uint32_t> decoder_context_frames,
           std::optional<std::string> decoder_mode, std::optional<std::string> extra_json) {
        vox_gen_params_default(&p);
        if (extra_json && !extra_json->empty()) { extra = *extra_json; p.extra_json = extra.c_str(); }
        if (temperature) p.temperature = *temperature;
        if (top_k) p.top_k = *top_k;
        if (top_p) p.top_p = *top_p;
        if (repetition_penalty) p.repetition_penalty = *repetition_penalty;
        if (sub_temperature) p.sub_temperature = *sub_temperature;
        if (sub_top_k) p.sub_top_k = *sub_top_k;
        if (sub_top_p) p.sub_top_p = *sub_top_p;
        if (seed) p.seed = *seed;
        if (chunk_frames) p.chunk_frames = *chunk_frames;
        if (max_frames) p.max_frames = *max_frames;
        if (language) { lang = *language; p.language = lang.c_str(); }
        if (decoder_context_frames) p.decoder_context_frames = *decoder_context_frames;
        if (decoder_mode) {
            if (*decoder_mode == "stream") p.decoder_mode = VOX_DECODER_STREAM;
            else if (*decoder_mode == "reference") p.decoder_mode = VOX_DECODER_REFERENCE;
            else throw std::invalid_argument("decoder_mode must be 'stream' or 'reference'");
        }
    }
};

struct Session {
    vox_session *h = nullptr;
    nb::object engine, voice;
    ~Session() { vox_session_free(h); }
    void feed(const std::string &t) { check(vox_session_feed_text(h, t.c_str())); }
    void end() { check(vox_session_end_text(h)); }
    void flush() { check(vox_session_flush(h)); }
    void set_pending(const std::string &t) { check(vox_session_set_pending(h, t.c_str())); }
    std::string info() {
        size_t n = 0;
        check(vox_session_info(h, nullptr, 0, &n));
        std::string s(n + 1, '\0');
        check(vox_session_info(h, s.data(), s.size(), &n));
        s.resize(n);
        return s;
    }
    void cancel() { vox_session_cancel(h); }
    // Возвращает (status, chunk). chunk — numpy float32 без лишних копий.
    std::pair<int, F32Array> pull(size_t max_samples) {
        float *buf = new float[max_samples];
        size_t n = 0;
        vox_status st;
        {
            nb::gil_scoped_release r;
            st = vox_session_pull(h, buf, max_samples, &n);
        }
        if (st < 0) { delete[] buf; raise(st); }
        nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<float *>(p); });
        return {int(st), F32Array(buf, {n}, owner)};
    }
};

NB_MODULE(_vox, m) {
    m.doc() = "vox low-level bindings (C ABI)";
    m.attr("VERSION") = vox_version();
    m.attr("OK") = int(VOX_OK);
    m.attr("AGAIN") = int(VOX_AGAIN);
    m.attr("DONE") = int(VOX_DONE);

    nb::class_<Engine>(m, "Engine")
        .def(nb::init<const std::string &, const std::string &, uint32_t, const std::string &>(), "path"_a, "device"_a = "auto",
             "threads"_a = 0, "config_json"_a = "")
        .def_prop_ro("sample_rate", &Engine::sample_rate)
        .def("info", &Engine::info)
        .def("stats", &Engine::stats)
        .def("phonemize", &Engine::phonemize);

    nb::class_<Voice>(m, "Voice")
        .def("save", &Voice::save)
        .def("set_name", &Voice::set_name)
        .def("info", &Voice::info);

    nb::class_<Params>(m, "Params")
        .def(nb::init<std::optional<float>, std::optional<int>, std::optional<float>, std::optional<float>,
                      std::optional<float>, std::optional<int>, std::optional<float>, std::optional<uint64_t>,
                      std::optional<uint32_t>, std::optional<uint32_t>, std::optional<std::string>, std::optional<uint32_t>, std::optional<std::string>, std::optional<std::string>>(),
             "temperature"_a = nb::none(), "top_k"_a = nb::none(), "top_p"_a = nb::none(),
             "repetition_penalty"_a = nb::none(), "sub_temperature"_a = nb::none(), "sub_top_k"_a = nb::none(),
             "sub_top_p"_a = nb::none(), "seed"_a = nb::none(), "chunk_frames"_a = nb::none(),
             "max_frames"_a = nb::none(), "language"_a = nb::none(), "decoder_context_frames"_a = nb::none(),
             "decoder_mode"_a = nb::none(), "extra_json"_a = nb::none());

    nb::class_<Session>(m, "Session")
        .def("feed", &Session::feed)
        .def("end", &Session::end)
        .def("flush", &Session::flush)
        .def("set_pending", &Session::set_pending)
        .def("info", &Session::info)
        .def("cancel", &Session::cancel)
        .def("pull", &Session::pull, "max_samples"_a = 8192);

    m.def("devices_json", [] { return json_free(vox_devices_json); });
    m.def("write_test_model", [](const std::string &path) { check(vox_write_test_model(path.c_str())); });
    m.def("inspect_json", [](const std::string &path) {
        size_t n = 0;
        check(vox_model_inspect(path.c_str(), nullptr, 0, &n));
        std::string s(n + 1, '\0');
        check(vox_model_inspect(path.c_str(), s.data(), s.size(), &n));
        s.resize(n);
        return s;
    }, "path"_a);
    m.def("plan_json", [](const std::string &path, const std::string &config_json) {
        size_t n = 0;
        check(vox_plan_json(path.c_str(), config_json.empty() ? nullptr : config_json.c_str(), nullptr, 0, &n));
        std::string s(n + 1, '\0');
        check(vox_plan_json(path.c_str(), config_json.empty() ? nullptr : config_json.c_str(), s.data(), s.size(), &n));
        s.resize(n);
        return s;
    }, "path"_a, "config_json"_a = "");

    nb::class_<vox_py::TmodView>(m, "TmodView")
        .def_prop_ro("manifest", &vox_py::TmodView::manifest)
        .def("blob_names", &vox_py::TmodView::blob_names)
        .def("blob", &vox_py::TmodView::blob, "name"_a);
    m.def("register_family", [](nb::object impl) {
        auto fam = std::make_shared<vox_py::PyFamily>(std::move(impl));
        g_py_families.push_back(fam->name());
        vox::register_family(std::move(fam));
    }, "family"_a);
    // При выходе из интерпретатора: статический реестр C++ разрушается уже после Py_Finalize, и деструктор
    // Python-семейства трогал бы мёртвый интерпретатор. Поэтому семейства и объект ошибки выгружаются заранее.
    m.def("_shutdown", [] {
        for (auto &n : g_py_families) vox::unregister_family(n);
        g_py_families.clear();
        g_error = nb::object();
    });
    nb::module_::import_("atexit").attr("register")(m.attr("_shutdown"));
    m.def("family_names", [] { return vox::family_names(); });

    m.def("set_error_type", [](nb::object t) { g_error = t; });

    m.def("voice_create", [](nb::object eng, F32In pcm, uint32_t sr, std::optional<std::string> ref_text,
                             bool xvector_only) {
        Engine &e = nb::cast<Engine &>(eng);
        auto *v = new Voice();
        v->engine = eng;
        vox_status st;
        const char *t = ref_text ? ref_text->c_str() : nullptr;
        {
            nb::gil_scoped_release r;
            st = vox_voice_create(e.h, pcm.data(), pcm.shape(0), sr, t,
                                  xvector_only ? VOX_VOICE_XVECTOR_ONLY : VOX_VOICE_ICL, &v->h);
        }
        if (st < 0) { delete v; raise(st); }
        return v;
    }, nb::rv_policy::take_ownership);

    m.def("voice_from_speaker", [](nb::object eng, std::optional<std::string> speaker, std::optional<std::string> instruct) {
        Engine &e = nb::cast<Engine &>(eng);
        auto *v = new Voice();
        v->engine = eng;
        vox_status st = vox_voice_from_speaker(e.h, speaker ? speaker->c_str() : nullptr, instruct ? instruct->c_str() : nullptr, &v->h);
        if (st < 0) { delete v; raise(st); }
        return v;
    }, nb::rv_policy::take_ownership);

    m.def("voice_load", [](nb::object eng, const std::string &path) {
        Engine &e = nb::cast<Engine &>(eng);
        auto *v = new Voice();
        v->engine = eng;
        vox_status st;
        {
            nb::gil_scoped_release r;
            st = vox_voice_load(e.h, path.c_str(), &v->h);
        }
        if (st < 0) { delete v; raise(st); }
        return v;
    }, nb::rv_policy::take_ownership);

    m.def("voice_set_defaults", [](Voice &v, const Params &p) { check(vox_voice_set_defaults(v.h, &p.p)); });

    m.def("session_create", [](nb::object eng, nb::object voice, const Params &p) {
        auto *s = new Session();
        s->engine = eng;
        s->voice = voice;
        vox_status st;
        {
            nb::gil_scoped_release r;   // may build a profile voice for this model (speaker encoder, codec encoder)
            st = vox_session_create(nb::cast<Engine &>(eng).h, nb::cast<Voice &>(voice).h, &p.p, &s->h);
        }
        if (st < 0) { delete s; raise(st); }
        return s;
    }, nb::rv_policy::take_ownership);

    // ---- profiles (ABI 0.5) ----
    m.def("profile_new", [](const std::string &name) {
        auto *v = new Voice();
        vox_status st = vox_profile_new(name.c_str(), &v->h);
        if (st < 0) { delete v; raise(st); }
        return v;
    }, nb::rv_policy::take_ownership);
    m.def("profile_open", [](const std::string &path) {
        auto *v = new Voice();
        vox_status st = vox_profile_open(path.c_str(), &v->h);
        if (st < 0) { delete v; raise(st); }
        return v;
    }, nb::rv_policy::take_ownership);
    m.def("voice_add_recording", [](Voice &v, const std::string &voice, std::optional<std::string> style, F32In pcm, uint32_t sr,
                                    std::optional<std::string> text, std::optional<std::string> language) {
        check(vox_voice_add_recording(v.h, voice.c_str(), style ? style->c_str() : nullptr, pcm.data(), pcm.shape(0), sr,
                                      text ? text->c_str() : nullptr, language ? language->c_str() : nullptr));
    });
    m.def("voice_merge", [](Voice &dst, Voice &src, std::optional<std::string> src_voice, std::optional<std::string> as_voice,
                            std::optional<std::string> as_style) {
        check(vox_voice_merge(dst.h, src.h, src_voice ? src_voice->c_str() : nullptr, as_voice ? as_voice->c_str() : nullptr,
                              as_style ? as_style->c_str() : nullptr));
    });
    m.def("voice_update", [](Voice &v, const std::string &patch) { check(vox_voice_update(v.h, patch.c_str())); });
    m.def("voice_prepare", [](nb::object eng, Voice &v, std::optional<std::string> voice, std::optional<std::string> style) {
        Engine &e = nb::cast<Engine &>(eng);
        std::string out(1 << 20, '\0');
        size_t n = 0;
        vox_status st;
        {
            nb::gil_scoped_release r;
            st = vox_voice_prepare(e.h, v.h, voice ? voice->c_str() : nullptr, style ? style->c_str() : nullptr, out.data(), out.size(), &n);
        }
        if (st < 0) raise(st);
        out.resize(std::min(n, out.size() - 1));
        return out;
    });
}
