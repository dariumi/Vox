// Семейства моделей, написанные на Python: vox.register_family(obj).
// Движок вызывает методы Python-бэкенда под GIL; остальной рантайм (сессии, потоковый текст, детерминированный сэмплинг,
// ресурсы, форматы, сервер) работает как с родным бэкендом. Вычисления в Python-бэкенде идут на CPU: GPU-стадии доступны
// только родным семействам.
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "../src/backend.hpp"
#include "../src/family.hpp"

namespace vox_py {
namespace nb = nanobind;
using namespace vox;
using namespace nb::literals;

inline nb::object to_py(const json &j) { return nb::module_::import_("json").attr("loads")(j.dump()); }
inline json from_py(nb::handle o) {
    return json::parse(nb::cast<std::string>(nb::module_::import_("json").attr("dumps")(o, "default"_a = nb::cpp_function([](nb::handle h) -> nb::object {
        // numpy-скаляры и массивы в JSON-словарях
        if (nb::hasattr(h, "tolist")) return h.attr("tolist")();
        throw nb::type_error("object is not JSON serializable");
    }))));
}

// Python-вид .tmod: манифест и блобы без копирования (numpy-массивы смотрят прямо в mmap).
struct TmodView {
    std::shared_ptr<TmodFile> f;
    nb::object manifest() const { return to_py(f->manifest()); }
    std::vector<std::string> blob_names() const {
        std::vector<std::string> v;
        for (const auto &kv : f->blobs()) v.push_back(kv.first);
        return v;
    }
    // (uint8-массив с байтами, имя dtype в стиле safetensors, shape)
    nb::tuple blob(const std::string &name) const {
        if (!f->has(name)) throw nb::key_error(("no blob " + name).c_str());
        const TensorView &t = f->get(name);
        auto *keep = new std::shared_ptr<TmodFile>(f);
        nb::capsule owner(keep, [](void *p) noexcept { delete static_cast<std::shared_ptr<TmodFile> *>(p); });
        nb::ndarray<nb::numpy, const uint8_t, nb::ndim<1>> a(t.data, {t.nbytes}, owner);
        return nb::make_tuple(a, std::string(dtype_name(t.dtype)), t.shape);
    }
};

[[noreturn]] inline void rethrow(const char *what, nb::python_error &e) {
    fail(VOX_ERR_UNSUPPORTED, std::string("python backend: ") + what + ": " + e.what());
}

struct PyState : GenState {
    nb::object obj;
    ~PyState() override {
        nb::gil_scoped_acquire g;
        obj = nb::object();
    }
};

template <class T> const char *np_name();
template <> inline const char *np_name<float>() { return "float32"; }
template <> inline const char *np_name<int16_t>() { return "int16"; }
template <> inline const char *np_name<int32_t>() { return "int32"; }

template <class T>
inline std::vector<T> vec_from(nb::handle o) {
    nb::gil_scoped_acquire g;
    nb::object arr = nb::module_::import_("numpy").attr("ascontiguousarray")(o, np_name<T>());
    auto a = nb::cast<nb::ndarray<const T, nb::c_contig, nb::device::cpu>>(arr);
    return std::vector<T>(a.data(), a.data() + a.size());
}

class PyBackend final : public Backend {
public:
    explicit PyBackend(nb::object impl) : impl_(std::move(impl)) {
        nb::gil_scoped_acquire g;
        try {
            json m = from_py(impl_.attr("info")());
            info_.sample_rate = m.value("sample_rate", 24000u);
            info_.samples_per_frame = m.value("samples_per_frame", 1920u);
            info_.num_codebooks = m.value("num_codebooks", 16u);
            info_.codebook_size = m.value("codebook_size", 2048u);
            info_.eos_code = m.value("eos_code", int32_t(info_.codebook_size));
            info_.suppress_from = m.value("suppress_from", 0);
            info_.suppress_to = m.value("suppress_to", 0);
            info_.min_eos_frames = m.value("min_eos_frames", 0u);
        } catch (nb::python_error &e) { rethrow("info()", e); }
    }
    ~PyBackend() override {
        nb::gil_scoped_acquire g;
        impl_ = nb::object();
    }

    std::string name() const override { return call_str("name", "python"); }
    const ModelInfo &info() const override { return info_; }
    std::string fingerprint() const override { return call_str("fingerprint", "py-backend"); }

    std::vector<int32_t> tokenize(const std::string &t) override {
        nb::gil_scoped_acquire g;
        try { return vec_from<int32_t>(impl_.attr("tokenize")(t)); } catch (nb::python_error &e) { rethrow("tokenize", e); }
    }
    std::vector<int32_t> tokenize_ref(const std::string &t) override {
        nb::gil_scoped_acquire g;
        try {
            if (!nb::hasattr(impl_, "tokenize_ref")) return tokenize(t);
            return vec_from<int32_t>(impl_.attr("tokenize_ref")(t));
        } catch (nb::python_error &e) { rethrow("tokenize_ref", e); }
    }
    size_t text_safe_cut(const std::string &pending) const override {
        nb::gil_scoped_acquire g;
        try {
            if (!nb::hasattr(impl_, "text_safe_cut")) return Backend::text_safe_cut(pending);
            return nb::cast<size_t>(impl_.attr("text_safe_cut")(nb::bytes(pending.data(), pending.size())));
        } catch (nb::python_error &e) { rethrow("text_safe_cut", e); }
    }

    std::vector<float> speaker_embedding(const float *pcm, size_t n, uint32_t sr) override {
        nb::gil_scoped_acquire g;
        try { return vec_from<float>(impl_.attr("speaker_embedding")(pcm_array(pcm, n), sr)); } catch (nb::python_error &e) { rethrow("speaker_embedding", e); }
    }
    std::vector<int16_t> encode_codes(const float *pcm, size_t n, uint32_t sr, int64_t &frames) override {
        nb::gil_scoped_acquire g;
        try {
            nb::object r = impl_.attr("encode_codes")(pcm_array(pcm, n), sr);  // массив [frames, codebooks]
            auto v = vec_from<int16_t>(r);
            frames = int64_t(v.size() / info_.num_codebooks);
            return v;
        } catch (nb::python_error &e) { rethrow("encode_codes", e); }
    }

    std::unique_ptr<GenState> begin(const VoiceData &v, const std::string &language) override {
        nb::gil_scoped_acquire g;
        try {
            nb::dict d;
            d["name"] = v.name;
            d["ref_text"] = v.ref_text;
            d["language"] = v.language;
            d["mode"] = int(v.mode);
            d["speaker_emb"] = floats(v.speaker_emb);
            d["ref_frames"] = v.ref_frames;
            d["ref_codebooks"] = v.ref_codebooks;
            d["ref_codes"] = int16s(v.ref_codes, v.ref_frames, v.ref_codebooks);
            d["ref_text_tokens"] = nb::cast(v.ref_text_tokens);
            d["sampling"] = to_py(v.sampling);
            auto st = std::make_unique<PyState>();
            st->obj = impl_.attr("begin")(d, language);
            return st;
        } catch (nb::python_error &e) { rethrow("begin", e); }
    }
    void append_text(GenState &s, const std::vector<int32_t> &t) override {
        nb::gil_scoped_acquire g;
        try { impl_.attr("append_text")(obj(s), nb::cast(t)); } catch (nb::python_error &e) { rethrow("append_text", e); }
    }
    bool needs_text(const GenState &s, bool ended) const override {
        nb::gil_scoped_acquire g;
        try { return nb::cast<bool>(impl_.attr("needs_text")(obj(s), ended)); } catch (nb::python_error &e) { rethrow("needs_text", e); }
    }
    void talker_logits(GenState &s, bool ended, std::vector<float> &out) override {
        nb::gil_scoped_acquire g;
        try { out = vec_from<float>(impl_.attr("talker_logits")(obj(s), ended)); } catch (nb::python_error &e) { rethrow("talker_logits", e); }
    }
    void accept_code0(GenState &s, int32_t code) override {
        nb::gil_scoped_acquire g;
        try { impl_.attr("accept_code0")(obj(s), code); } catch (nb::python_error &e) { rethrow("accept_code0", e); }
    }
    void predictor_logits(GenState &s, uint32_t cb, std::vector<float> &out) override {
        nb::gil_scoped_acquire g;
        try { out = vec_from<float>(impl_.attr("predictor_logits")(obj(s), cb)); } catch (nb::python_error &e) { rethrow("predictor_logits", e); }
    }
    void accept_code(GenState &s, uint32_t cb, int32_t code) override {
        nb::gil_scoped_acquire g;
        try {
            if (nb::hasattr(impl_, "accept_code")) impl_.attr("accept_code")(obj(s), cb, code);
        } catch (nb::python_error &e) { rethrow("accept_code", e); }
    }
    void end_frame(GenState &s) override {
        nb::gil_scoped_acquire g;
        try { impl_.attr("end_frame")(obj(s)); } catch (nb::python_error &e) { rethrow("end_frame", e); }
    }
    void decode(GenState &s, const int16_t *codes, size_t n_frames, std::vector<float> &pcm) override {
        nb::gil_scoped_acquire g;
        try {
            std::vector<int16_t> c(codes, codes + n_frames * info_.num_codebooks);
            nb::object r = impl_.attr("decode")(obj(s), int16s(c, int64_t(n_frames), info_.num_codebooks));
            auto v = vec_from<float>(r);
            pcm.insert(pcm.end(), v.begin(), v.end());
        } catch (nb::python_error &e) { rethrow("decode", e); }
    }
    void flush(GenState &s, std::vector<float> &pcm) override {
        nb::gil_scoped_acquire g;
        try {
            if (!nb::hasattr(impl_, "flush")) return;
            auto v = vec_from<float>(impl_.attr("flush")(obj(s)));
            pcm.insert(pcm.end(), v.begin(), v.end());
        } catch (nb::python_error &e) { rethrow("flush", e); }
    }
    json stats() const override {
        nb::gil_scoped_acquire g;
        try {
            if (!nb::hasattr(impl_, "stats")) return json::object();
            return from_py(impl_.attr("stats")());
        } catch (nb::python_error &e) { rethrow("stats", e); }
    }

private:
    static nb::object obj(const GenState &s) { return static_cast<const PyState &>(s).obj; }
    std::string call_str(const char *m, const char *dflt) const {
        nb::gil_scoped_acquire g;
        try {
            if (!nb::hasattr(impl_, m)) return dflt;
            return nb::cast<std::string>(impl_.attr(m)());
        } catch (nb::python_error &e) { rethrow(m, e); }
    }
    static nb::object pcm_array(const float *p, size_t n) {
        auto *copy = new std::vector<float>(p, p + n);
        nb::capsule owner(copy, [](void *q) noexcept { delete static_cast<std::vector<float> *>(q); });
        return nb::cast(nb::ndarray<nb::numpy, float, nb::ndim<1>>(copy->data(), {n}, owner));
    }
    static nb::object floats(const std::vector<float> &v) { return pcm_array(v.data(), v.size()); }
    static nb::object int16s(const std::vector<int16_t> &v, int64_t rows, int64_t cols) {
        auto *copy = new std::vector<int16_t>(v);
        nb::capsule owner(copy, [](void *q) noexcept { delete static_cast<std::vector<int16_t> *>(q); });
        return nb::cast(nb::ndarray<nb::numpy, int16_t, nb::ndim<2>>(copy->data(), {size_t(rows), size_t(cols)}, owner));
    }

    nb::object impl_;
    ModelInfo info_;
};

class PyFamily final : public Family {
public:
    explicit PyFamily(nb::object impl) : impl_(std::move(impl)) {
        nb::gil_scoped_acquire g;
        name_ = nb::cast<std::string>(impl_.attr("name"));
    }
    ~PyFamily() override {
        nb::gil_scoped_acquire g;
        impl_ = nb::object();
    }
    std::string name() const override { return name_; }
    std::string description() const override {
        nb::gil_scoped_acquire g;
        return nb::hasattr(impl_, "description") ? nb::cast<std::string>(impl_.attr("description")) : "python-defined family";
    }
    bool matches(const json &manifest) const override {
        nb::gil_scoped_acquire g;
        try {
            if (!nb::hasattr(impl_, "matches")) return Family::matches(manifest);
            return nb::cast<bool>(impl_.attr("matches")(to_py(manifest)));
        } catch (nb::python_error &e) { rethrow("matches", e); }
    }
    ModelFootprint footprint(const TmodFile &f) const override {
        nb::gil_scoped_acquire g;
        ModelFootprint fp;
        fp.family = name_;
        fp.variant = f.manifest().contains("model") ? f.manifest()["model"].value("variant", "") : "";
        try {
            if (nb::hasattr(impl_, "footprint")) {
                json j = from_py(impl_.attr("footprint")(to_py(f.manifest())));
                fp.variant = j.value("variant", fp.variant);
                fp.always_ram_bytes = j.value("always_ram_bytes", uint64_t(0));
                for (const auto &s : j.value("stages", json::array())) {
                    StageDesc d;
                    d.name = s.value("name", "model");
                    d.gpu_capable = false;  // Python-бэкенд считает на CPU
                    d.weight_bytes = s.value("weight_bytes", uint64_t(0));
                    if (!d.weight_bytes && s.contains("prefixes")) d.weight_bytes = blob_bytes(f, s["prefixes"].get<std::vector<std::string>>());
                    d.lookup_only = s.value("lookup_only", false);
                    d.bytes_per_frame = s.value("bytes_per_frame", 0.0);
                    d.flops_per_frame = s.value("flops_per_frame", 0.0);
                    d.session_ram_bytes = s.value("session_ram_bytes", uint64_t(0));
                    fp.stages.push_back(d);
                }
            }
        } catch (nb::python_error &e) { rethrow("footprint", e); }
        if (fp.stages.empty()) {
            StageDesc d;
            d.name = "model";
            d.gpu_capable = false;
            d.weight_bytes = blob_bytes(f, {""});
            fp.stages.push_back(d);
        }
        return fp;
    }
    Capabilities capabilities(const TmodFile &f) const override {
        nb::gil_scoped_acquire g;
        Capabilities c;
        c.clone_icl = c.clone_xvector = true;
        try {
            if (nb::hasattr(impl_, "capabilities")) {
                json j = from_py(impl_.attr("capabilities")(to_py(f.manifest())));
                c.clone_icl = j.value("clone_icl", c.clone_icl);
                c.clone_xvector = j.value("clone_xvector", c.clone_xvector);
                c.named_speakers = j.value("named_speakers", false);
                c.instruct = j.value("instruct", false);
                c.streaming_text = j.value("streaming_text", true);
                c.requires_instruct = j.value("requires_instruct", false);
                c.speakers = j.value("speakers", std::vector<std::string>{});
                c.languages = j.value("languages", std::vector<std::string>{"auto"});
            }
        } catch (nb::python_error &e) { rethrow("capabilities", e); }
        return c;
    }
    std::unique_ptr<Backend> create(std::shared_ptr<TmodFile> f, const ResourcePlan &plan) const override {
        nb::gil_scoped_acquire g;
        try {
            TmodView view{std::move(f)};
            nb::object b = impl_.attr("create")(nb::cast(view), to_py(plan.to_json()));
            return std::make_unique<PyBackend>(std::move(b));
        } catch (nb::python_error &e) { rethrow("create", e); }
    }

private:
    nb::object impl_;
    std::string name_;
};

}  // namespace vox_py
