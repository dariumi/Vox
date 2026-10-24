#include "family.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace vox {
namespace {
struct Registry {
    std::mutex mu;
    std::vector<std::shared_ptr<Family>> families;
    Registry() {
        families.push_back(make_qwen3_family());
        families.push_back(make_test_family());
        families.push_back(make_f5_family());
        families.push_back(make_kokoro_family());
        families.push_back(make_cosyvoice_family());
    }
};
Registry &registry() {
    static Registry r;  // встроенные семейства регистрируются при первом обращении (без зависимости от порядка статической инициализации)
    return r;
}
}  // namespace

void register_family(std::shared_ptr<Family> f) {
    if (!f) fail(VOX_ERR_ARG, "family is null");
    Registry &r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    for (auto &x : r.families)
        if (x->name() == f->name()) { x = std::move(f); return; }
    r.families.push_back(std::move(f));
}

void unregister_family(const std::string &name) {
    std::shared_ptr<Family> dead;  // уничтожается вне блокировки
    Registry &r = registry();
    {
        std::lock_guard<std::mutex> lk(r.mu);
        for (auto it = r.families.begin(); it != r.families.end(); ++it)
            if ((*it)->name() == name) { dead = std::move(*it); r.families.erase(it); break; }
    }
}

std::vector<std::string> family_names() {
    Registry &r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    std::vector<std::string> out;
    for (auto &x : r.families) out.push_back(x->name());
    return out;
}

const Family &find_family(const json &manifest) {
    Registry &r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    for (auto &x : r.families)
        if (x->matches(manifest)) return *x;
    std::string known;
    for (auto &x : r.families) known += (known.empty() ? "" : ", ") + x->name();
    const std::string fam = manifest.contains("model") ? manifest["model"].value("family", "") : "";
    fail(VOX_ERR_UNSUPPORTED, "unknown model family '" + fam + "' (this build knows: " + known + ")");
}

uint64_t blob_bytes(const TmodFile &f, const std::vector<std::string> &prefixes, const std::vector<std::string> &exclude) {
    uint64_t total = 0;
    for (const auto &[name, v] : f.blobs()) {
        bool in = false;
        for (const auto &p : prefixes) in = in || name.compare(0, p.size(), p) == 0;
        for (const auto &p : exclude) in = in && name.compare(0, p.size(), p) != 0;
        if (in) total += v.nbytes;
    }
    return total;
}

ResourcePlan plan_for_model(const TmodFile &f, const ResourceConfig &cfg_in) {
    ResourceConfig cfg = cfg_in;
    if (const char *e = std::getenv("VOX_GPU_STRICT")) if (e[0] && e[0] != '0') cfg.strict = true;
    if (const char *e = std::getenv("VOX_VK_DEVICE")) if (cfg.gpu < 0 && e[0]) cfg.gpu = std::atoi(e);
    const Family &fam = find_family(f.manifest());
    ModelFootprint fp = fam.footprint(f);
    for (StageDesc &s : fp.stages)
        if (!s.weight_bytes && !s.prefixes.empty()) s.weight_bytes = blob_bytes(f, s.prefixes, s.exclude);
    std::vector<GpuDeviceInfo> gpus;
    if (cfg.device != "cpu") gpus = gpu_enumerate();
    return make_plan(cfg, fp, system_info(), gpus);
}

std::unique_ptr<Backend> create_backend(std::shared_ptr<TmodFile> f, const ResourcePlan &plan) {
    const Family &fam = find_family(f->manifest());
    if (plan.uses_gpu()) {
        std::string err;
        if (!gpu_init("vulkan:" + std::to_string(plan.gpu_index), err)) fail(VOX_ERR_DEVICE, err);
        const GpuMemInfo mi = gpu_mem_info();
        if (mi.device_index != plan.gpu_index)
            fail(VOX_ERR_DEVICE, "this process already uses vulkan:" + std::to_string(mi.device_index) + "; one Vulkan device per process is supported, "
                                 "cannot open vulkan:" + std::to_string(plan.gpu_index));
        gpu_set_vram_limit(plan.vram_limit);
    }
    return fam.create(std::move(f), plan);
}

json inspect_model(const TmodFile &f) {
    const Family &fam = find_family(f.manifest());
    ModelFootprint fp = fam.footprint(f);
    json stages = json::array();
    for (StageDesc &s : fp.stages) {
        if (!s.weight_bytes && !s.prefixes.empty()) s.weight_bytes = blob_bytes(f, s.prefixes, s.exclude);
        json e = {{"name", s.name}, {"weight_bytes", s.weight_bytes}, {"gpu_capable", s.gpu_capable}, {"splittable", s.splittable},
                  {"session_vram_bytes", s.session_vram_bytes}, {"session_ram_bytes", s.session_ram_bytes}};
        if (s.layers) e["layers"] = s.layers;
        stages.push_back(e);
    }
    uint64_t total = 0;
    for (const auto &[n, v] : f.blobs()) total += v.nbytes;
    json m = f.manifest();
    m.erase("blobs");
    return {{"family", fam.name()}, {"description", fam.description()}, {"variant", f.manifest().at("model").value("variant", "")},
            {"capabilities", fam.capabilities(f).to_json()}, {"stages", stages}, {"file_bytes", total}, {"blob_count", f.blobs().size()},
            {"always_ram_bytes", fp.always_ram_bytes}, {"manifest", m}};
}

std::unique_ptr<Backend> make_backend(std::shared_ptr<TmodFile> tmod, const std::string &device, uint32_t n_threads) {
    ResourceConfig cfg;
    cfg.apply_device_string(device);
    if (n_threads) cfg.threads = n_threads;
    ResourcePlan plan = plan_for_model(*tmod, cfg);
    if (!plan.feasible) {
        std::string why;
        for (const auto &w : plan.warnings) why += (why.empty() ? "" : "; ") + w;
        fail(VOX_ERR_DEVICE, "cannot place the model as requested (" + device + "): " + why);
    }
    return create_backend(std::move(tmod), plan);
}

}  // namespace vox
