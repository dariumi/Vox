#include "resources.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#  include <sys/mman.h>
#  include <sys/resource.h>
#  include <unistd.h>
#endif
#if defined(__linux__)
#  include <sched.h>
#endif
#if defined(__APPLE__)
#  include <sys/sysctl.h>
#endif
#if defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>
#endif

namespace vox {

// ====================================================== строки и размеры
const char *place_name(Place p) { return p == Place::Cpu ? "cpu" : p == Place::Gpu ? "gpu" : "auto"; }
const char *mode_name(Mode m) {
    switch (m) {
        case Mode::Speed: return "speed";
        case Mode::Balanced: return "balanced";
        case Mode::Memory: return "memory";
        case Mode::Background: return "background";
        default: return "auto";
    }
}
Place place_from_string(const std::string &s) {
    if (s == "auto") return Place::Auto;
    if (s == "cpu") return Place::Cpu;
    if (s == "gpu" || s == "vulkan") return Place::Gpu;
    fail(VOX_ERR_ARG, "placement must be auto, cpu or gpu, got '" + s + "'");
}
Mode mode_from_string(const std::string &s) {
    if (s == "auto" || s.empty()) return Mode::Auto;
    if (s == "speed") return Mode::Speed;
    if (s == "balanced") return Mode::Balanced;
    if (s == "memory") return Mode::Memory;
    if (s == "background") return Mode::Background;
    fail(VOX_ERR_ARG, "mode must be auto, speed, balanced, memory or background, got '" + s + "'");
}

uint64_t parse_size(const json &v, uint64_t whole) {
    if (v.is_number()) {
        if (v.get<double>() < 0) fail(VOX_ERR_ARG, "size must not be negative");
        return static_cast<uint64_t>(v.get<double>());
    }
    if (!v.is_string()) fail(VOX_ERR_ARG, "size must be a number of bytes or a string like \"6GiB\" / \"50%\"");
    std::string s = v.get<std::string>();
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    size_t j = i;
    while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '.')) ++j;
    if (j == i) fail(VOX_ERR_ARG, "cannot parse size '" + s + "'");
    const double num = std::stod(s.substr(i, j - i));
    std::string unit = s.substr(j);
    unit.erase(std::remove_if(unit.begin(), unit.end(), [](unsigned char c) { return std::isspace(c); }), unit.end());
    std::transform(unit.begin(), unit.end(), unit.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (unit == "%") {
        if (!whole) fail(VOX_ERR_ARG, "a percentage size needs a known total (size '" + s + "')");
        return static_cast<uint64_t>(num / 100.0 * double(whole));
    }
    double mult = 1;
    if (unit.empty() || unit == "b") mult = 1;
    else if (unit == "k" || unit == "kb" || unit == "kib") mult = 1024.0;
    else if (unit == "m" || unit == "mb" || unit == "mib") mult = 1024.0 * 1024;
    else if (unit == "g" || unit == "gb" || unit == "gib") mult = 1024.0 * 1024 * 1024;
    else if (unit == "t" || unit == "tb" || unit == "tib") mult = 1024.0 * 1024 * 1024 * 1024;
    else fail(VOX_ERR_ARG, "unknown size unit in '" + s + "' (use B, K, M, G, T, %)");
    return static_cast<uint64_t>(num * mult);
}

std::string format_size(uint64_t b) {
    char buf[64];
    if (b >= (1ull << 30)) std::snprintf(buf, sizeof buf, "%.2f GiB", double(b) / double(1ull << 30));
    else if (b >= (1ull << 20)) std::snprintf(buf, sizeof buf, "%.1f MiB", double(b) / double(1ull << 20));
    else std::snprintf(buf, sizeof buf, "%.1f KiB", double(b) / 1024.0);
    return buf;
}

namespace {
std::vector<std::string> split(const std::string &s, char sep) {
    std::vector<std::string> out;
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(sep, p);
        out.push_back(s.substr(p, q == std::string::npos ? std::string::npos : q - p));
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return out;
}
// «0-3,6,8-9» → список ядер
std::vector<int> parse_cpu_list(const std::string &s) {
    std::vector<int> out;
    for (const std::string &part : split(s, ',')) {
        if (part.empty()) continue;
        auto dash = part.find('-');
        try {
            if (dash == std::string::npos) out.push_back(std::stoi(part));
            else for (int c = std::stoi(part.substr(0, dash)); c <= std::stoi(part.substr(dash + 1)); ++c) out.push_back(c);
        } catch (const std::exception &) {
            fail(VOX_ERR_ARG, "cannot parse cpu_affinity '" + s + "' (expected like \"0-3,6\")");
        }
    }
    return out;
}
}  // namespace

// ====================================================== ResourceConfig
void ResourceConfig::apply_device_string(const std::string &spec) {
    const std::vector<std::string> parts = split(spec, ',');
    std::string base = parts.empty() ? "auto" : parts[0];
    if (base.empty()) base = "auto";
    int index = -1;
    auto colon = base.find(':');
    if (colon != std::string::npos) {
        try { index = std::stoi(base.substr(colon + 1)); } catch (const std::exception &) { fail(VOX_ERR_DEVICE, "bad device index in '" + spec + "'"); }
        base = base.substr(0, colon);
    }
    if (base == "auto" || base == "cpu") device = base;
    else if (base == "gpu" || base == "vulkan") device = "gpu";
    else fail(VOX_ERR_DEVICE, "unknown device '" + spec + "' (use auto, cpu, gpu[:N], vulkan[:N]; CUDA, Metal and WebGPU backends do not exist in this build)");
    if (index >= 0) gpu = index;
    for (size_t i = 1; i < parts.size(); ++i) {
        const std::string &opt = parts[i];
        if (opt.empty()) continue;
        const auto eq = opt.find('=');
        const std::string k = opt.substr(0, eq), v = eq == std::string::npos ? "" : opt.substr(eq + 1);
        try {
            if (k == "strict") strict = true;
            else if (k == "codec" || k == "talker" || k == "predictor") place[k] = place_from_string(v);
            else if (k == "layers") gpu_layers["talker"] = std::stoi(v);
            else if (k == "mode") mode = mode_from_string(v);
            else if (k == "threads") threads = uint32_t(std::stoul(v));
            else if (k == "ram") ram_limit = parse_size(json(v));
            else if (k == "vram") vram_limit = parse_size(json(v));
            // any other stage of the model by name (llm=cpu, flow=gpu, t3=cpu, …: the stages `vox plan` lists)
            else if (eq != std::string::npos && (v == "cpu" || v == "gpu" || v == "auto") && !k.empty()) place[k] = place_from_string(v);
            else fail(VOX_ERR_DEVICE, "unknown device option '" + opt + "' (strict, <stage>=cpu|gpu such as codec=cpu, layers=N, mode=, threads=N, ram=, vram=)");
        } catch (const Error &) {
            throw;
        } catch (const std::exception &) {
            fail(VOX_ERR_DEVICE, "bad value in device option '" + opt + "'");
        }
    }
}

ResourceConfig ResourceConfig::from_json(const json &j) {
    ResourceConfig c;
    if (j.is_null()) return c;
    if (!j.is_object()) fail(VOX_ERR_ARG, "resource config must be a JSON object");
    static const char *known[] = {"device", "mode", "gpu", "threads", "cpu_affinity", "priority", "ram_limit", "vram_limit", "place",
                                  "gpu_layers", "strict", "enforce", "overlap", "release_host_weights", "max_sessions", "planned_sessions", "tuned"};
    for (auto it = j.begin(); it != j.end(); ++it) {
        bool ok = false;
        for (const char *k : known) ok = ok || it.key() == k;
        if (!ok) {
            std::string list;
            for (const char *k : known) list += std::string(list.empty() ? "" : ", ") + k;
            fail(VOX_ERR_ARG, "unknown resource option '" + it.key() + "' (known: " + list + ")");
        }
    }
    const SystemInfo sys = system_info();
    if (j.contains("device")) c.apply_device_string(j["device"].get<std::string>());
    if (j.contains("mode")) c.mode = mode_from_string(j["mode"].get<std::string>());
    if (j.contains("gpu") && !j["gpu"].is_null()) c.gpu = j["gpu"].get<int>();
    if (j.contains("threads")) c.threads = j["threads"].get<uint32_t>();
    if (j.contains("cpu_affinity")) {
        const json &a = j["cpu_affinity"];
        c.cpu_affinity = a.is_string() ? parse_cpu_list(a.get<std::string>()) : a.get<std::vector<int>>();
    }
    if (j.contains("priority")) {
        const json &p = j["priority"];
        if (p.is_number()) c.nice = p.get<int>();
        else {
            const std::string s = p.get<std::string>();
            c.nice = s == "low" ? 10 : s == "normal" ? 0 : s == "idle" ? 19 : 1000;
            if (s != "low" && s != "normal" && s != "idle" && s != "auto") fail(VOX_ERR_ARG, "priority must be low, normal, idle or a nice value");
        }
    }
    if (j.contains("ram_limit")) c.ram_limit = parse_size(j["ram_limit"], sys.ram_total);
    if (j.contains("vram_limit")) c.vram_limit = parse_size(j["vram_limit"]);
    if (j.contains("place"))
        for (auto it = j["place"].begin(); it != j["place"].end(); ++it) c.place[it.key()] = place_from_string(it.value().get<std::string>());
    if (j.contains("gpu_layers"))
        for (auto it = j["gpu_layers"].begin(); it != j["gpu_layers"].end(); ++it) c.gpu_layers[it.key()] = it.value().get<int>();
    if (j.contains("strict")) c.strict = j["strict"].get<bool>();
    if (j.contains("enforce")) c.enforce = j["enforce"].get<bool>() ? 1 : 0;
    if (j.contains("overlap")) c.overlap = j["overlap"].get<bool>() ? 1 : 0;
    if (j.contains("release_host_weights")) c.release_host_weights = j["release_host_weights"].get<bool>();
    if (j.contains("tuned")) c.tuned = j["tuned"].get<bool>();
    if (j.contains("max_sessions")) c.max_sessions = j["max_sessions"].get<uint32_t>();
    if (j.contains("planned_sessions")) c.planned_sessions = std::max<uint32_t>(1, j["planned_sessions"].get<uint32_t>());
    return c;
}

json ResourceConfig::to_json() const {
    json j = {{"device", device}, {"mode", mode_name(mode)}, {"threads", threads}, {"ram_limit", ram_limit}, {"vram_limit", vram_limit},
              {"strict", strict}, {"release_host_weights", release_host_weights}, {"max_sessions", max_sessions},
              {"planned_sessions", planned_sessions}};
    if (gpu >= 0) j["gpu"] = gpu;
    if (!cpu_affinity.empty()) j["cpu_affinity"] = cpu_affinity;
    if (nice != 1000) j["priority"] = nice;
    if (enforce >= 0) j["enforce"] = enforce != 0;
    if (overlap >= 0) j["overlap"] = overlap != 0;
    json pl = json::object(), gl = json::object();
    for (auto &[k, v] : place) pl[k] = place_name(v);
    for (auto &[k, v] : gpu_layers) gl[k] = v;
    if (!pl.empty()) j["place"] = pl;
    if (!gl.empty()) j["gpu_layers"] = gl;
    return j;
}

// ====================================================== SystemInfo
namespace {
uint64_t read_u64_file(const char *path, bool *ok = nullptr) {
    std::ifstream f(path);
    std::string s;
    if (ok) *ok = false;
    if (!(f >> s)) return 0;
    if (s == "max") return 0;
    try {
        if (ok) *ok = true;
        return std::stoull(s);
    } catch (const std::exception &) {
        if (ok) *ok = false;
        return 0;
    }
}
}  // namespace

uint64_t SystemInfo::usable_ram() const {
    uint64_t v = ram_available ? ram_available : ram_total;
    if (cgroup_limit) {
        // остаток лимита контейнера = лимит − то, что процессы контейнера уже заняли (rss вместе с кэшем страниц)
        v = v ? std::min(v, cgroup_limit) : cgroup_limit;
    }
    return v;
}

json SystemInfo::to_json() const {
    return {{"os", os}, {"cpu_logical", cpu_logical}, {"cpu_allowed", cpu_allowed}, {"ram_total", ram_total}, {"ram_available", ram_available},
            {"cgroup_limit", cgroup_limit}, {"rss", rss}, {"peak_rss", peak_rss}};
}

SystemInfo system_info() {
    SystemInfo s;
    const unsigned hc = std::thread::hardware_concurrency();
    s.cpu_logical = s.cpu_allowed = hc ? int(hc) : 1;
#if defined(__linux__)
    s.os = "linux";
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) s.cpu_allowed = std::max(1, int(CPU_COUNT(&set)));
    std::ifstream mi("/proc/meminfo");
    std::string key;
    uint64_t val;
    std::string unit;
    while (mi >> key >> val >> unit) {
        if (key == "MemTotal:") s.ram_total = val * 1024;
        else if (key == "MemAvailable:") s.ram_available = val * 1024;
    }
    // ограничение контейнера: cgroup v2, затем v1; «огромное» значение v1 = без ограничения
    bool ok = false;
    uint64_t lim = read_u64_file("/sys/fs/cgroup/memory.max", &ok);
    uint64_t cur = ok ? read_u64_file("/sys/fs/cgroup/memory.current") : 0;
    if (!ok) {
        lim = read_u64_file("/sys/fs/cgroup/memory/memory.limit_in_bytes", &ok);
        cur = ok ? read_u64_file("/sys/fs/cgroup/memory/memory.usage_in_bytes") : 0;
        if (ok && lim > (1ull << 60)) ok = false;
    }
    if (ok && lim && (!s.ram_total || lim < s.ram_total)) {
        s.cgroup_limit = lim > cur ? lim - cur : 0;
        if (!s.cgroup_limit) s.cgroup_limit = 1;
    }
    std::ifstream st("/proc/self/status");
    std::string line;
    while (std::getline(st, line)) {
        if (line.rfind("VmRSS:", 0) == 0) s.rss = std::strtoull(line.c_str() + 6, nullptr, 10) * 1024;
        else if (line.rfind("VmHWM:", 0) == 0) s.peak_rss = std::strtoull(line.c_str() + 6, nullptr, 10) * 1024;
    }
#elif defined(__APPLE__)
    s.os = "macos";
    uint64_t mem = 0;
    size_t len = sizeof mem;
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) { s.ram_total = mem; s.ram_available = mem / 2; }  // оценка: точного MemAvailable нет
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
    s.peak_rss = uint64_t(ru.ru_maxrss);
    s.rss = s.peak_rss;
#elif defined(_WIN32)
    s.os = "windows";
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) { s.ram_total = ms.ullTotalPhys; s.ram_available = ms.ullAvailPhys; }
#else
    s.os = "other";
#endif
    return s;
}

// ====================================================== планировщик
Place ResourcePlan::place_of(const std::string &stage) const {
    for (const auto &s : stages) if (s.stage == stage) return s.place;
    return Place::Cpu;
}
int ResourcePlan::gpu_layers_of(const std::string &stage) const {
    for (const auto &s : stages) if (s.stage == stage) return s.gpu_layers;
    return -1;
}
bool ResourcePlan::uses_gpu() const {
    for (const auto &s : stages) if (s.place == Place::Gpu) return true;
    return false;
}

namespace {
constexpr uint64_t kGpuOverhead = 160ull << 20;  // рабочие буферы трансформера, staging-буфер, контекст драйвера

struct Perf { double cpu_bw, cpu_flops, gpu_bw, gpu_flops; };

// Грубые оценки, нужны только чтобы упорядочить стадии по выгоде от переноса на GPU (в оценку скорости они не идут).
double benefit(const StageDesc &s, const Perf &p) {
    auto t = [&](double bw, double fl) { return std::max(s.bytes_per_frame / bw, s.flops_per_frame / fl); };
    return t(p.cpu_bw, p.cpu_flops) - t(p.gpu_bw, p.gpu_flops);
}

struct Candidate {
    ResourcePlan plan;
    double score = 0;
    uint64_t free = 0;
    int rank = 0;
};

Candidate plan_for(const ResourceConfig &cfg, const ModelFootprint &fp, const SystemInfo &, const GpuDeviceInfo *dev, Mode mode,
                   const ResourcePlan &base) {
    Candidate C;
    ResourcePlan &P = C.plan;
    P = base;
    const bool unified = dev && dev->unified_memory;
    const double margin = mode == Mode::Speed ? 0.02 : mode == Mode::Balanced ? 0.08 : mode == Mode::Memory ? 0.12 : 0.10;
    Perf perf{20e9, double(std::max<uint32_t>(1, P.threads)) * 25e9, 150e9, 4e12};
    if (dev && dev->type == "integrated") { perf.gpu_bw = 40e9; perf.gpu_flops = 5e11; }
    if (dev && dev->software) { perf.gpu_bw = 5e9; perf.gpu_flops = 5e10; }

    uint64_t remaining = 0;
    if (dev) {
        uint64_t budget = static_cast<uint64_t>(double(dev->vram_free) * (1.0 - margin));
        if (cfg.vram_limit) budget = std::min(budget, cfg.vram_limit);
        P.vram_budget = budget;
        P.gpu_index = dev->index;
        P.gpu_name = dev->name;
        remaining = budget > kGpuOverhead ? budget - kGpuOverhead : 0;
    }

    std::vector<size_t> order;
    for (size_t i = 0; i < fp.stages.size(); ++i) order.push_back(i);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const StageDesc &x = fp.stages[a], &y = fp.stages[b];
        const double dx = benefit(x, perf) / double(x.weight_bytes + (1u << 20)), dy = benefit(y, perf) / double(y.weight_bytes + (1u << 20));
        return dx > dy;
    });

    std::map<std::string, StagePlacement> placed;
    uint64_t vram_used = 0;
    auto left = [&] { return remaining - std::min(remaining, vram_used); };
    for (size_t idx : order) {
        const StageDesc &s = fp.stages[idx];
        StagePlacement sp;
        sp.stage = s.name;
        sp.weight_bytes = s.weight_bytes;
        const auto pit = cfg.place.find(s.name);
        const Place want = pit == cfg.place.end() ? Place::Auto : pit->second;
        const auto lit = cfg.gpu_layers.find(s.name);
        const uint64_t sess = s.session_vram_bytes * P.planned_sessions;
        const uint64_t need = s.weight_bytes + sess;
        sp.place = Place::Cpu;

        if (!dev || !s.gpu_capable || want == Place::Cpu) {
            sp.reason = want == Place::Cpu ? "requested: cpu" : !s.gpu_capable ? "stage has no GPU implementation" : "no GPU in use";
        } else if (lit != cfg.gpu_layers.end()) {
            if (!s.splittable) fail(VOX_ERR_ARG, "stage '" + s.name + "' cannot be split between GPU and CPU (gpu_layers)");
            const int k = std::min(std::max(0, lit->second), s.layers);
            const uint64_t per = s.layer_bytes + (s.layers ? s.session_vram_bytes * P.planned_sessions / uint64_t(s.layers) : 0);
            sp.place = k > 0 ? Place::Gpu : Place::Cpu;
            sp.gpu_layers = k == s.layers ? -1 : k;
            sp.vram_bytes = k == s.layers ? need : per * uint64_t(k);
            sp.reason = "requested: " + std::to_string(k) + " of " + std::to_string(s.layers) + " layers on the GPU";
            if (sp.vram_bytes > left()) {
                P.warnings.push_back("stage '" + s.name + "': requested " + std::to_string(k) + " GPU layers need " + format_size(sp.vram_bytes) +
                                     ", only " + format_size(left()) + " of the VRAM budget is left");
                if (cfg.strict || P.enforce) P.feasible = false;
            }
        } else if (need <= left()) {
            sp.place = Place::Gpu;
            sp.vram_bytes = need;
            sp.reason = want == Place::Gpu ? "requested: gpu" : "fits in the VRAM budget";
        } else if (s.splittable && s.layers > 0) {
            const uint64_t avail = left();
            const uint64_t per = s.layer_bytes + s.session_vram_bytes * P.planned_sessions / uint64_t(s.layers);
            const int k = per ? int(std::min<uint64_t>(uint64_t(s.layers), avail / per)) : 0;
            if (k >= 1) {
                sp.place = Place::Gpu;
                sp.gpu_layers = k;
                sp.vram_bytes = per * uint64_t(k);
                sp.reason = std::to_string(k) + " of " + std::to_string(s.layers) + " layers fit in the VRAM budget; the rest runs on the CPU";
            } else {
                sp.reason = "not even one layer fits in the VRAM budget";
            }
        } else {
            sp.reason = "does not fit in the remaining VRAM budget (" + format_size(need) + " needed, " +
                        format_size(left()) + " left)";
            if (want == Place::Gpu) {
                P.warnings.push_back("stage '" + s.name + "' was requested on the GPU but " + sp.reason);
                if (cfg.strict || P.enforce) P.feasible = false;
            }
        }
        vram_used += sp.vram_bytes;
        if (sp.place == Place::Gpu) {
            const double frac = sp.gpu_layers < 0 || s.layers == 0 ? 1.0 : double(sp.gpu_layers) / double(s.layers);
            C.score += benefit(s, perf) * frac;
        }
        placed[s.name] = sp;
    }

    // RAM: веса стадий на CPU + то, что остаётся при любом размещении + KV-кэши сессий на CPU
    uint64_t ram = fp.always_ram_bytes, sess_ram = 0;
    for (const StageDesc &s : fp.stages) {
        StagePlacement &sp = placed[s.name];
        const uint64_t on_gpu = sp.place == Place::Gpu ? (sp.gpu_layers < 0 ? s.weight_bytes : s.layer_bytes * uint64_t(sp.gpu_layers)) : 0;
        const uint64_t host = s.weight_bytes - std::min(s.weight_bytes, on_gpu);
        const uint64_t kept = on_gpu && !P.release_host_weights ? on_gpu : 0;  // страницы mmap остаются резидентными
        sp.ram_bytes = host + kept;
        if (sp.place != Place::Gpu) sess_ram += s.session_ram_bytes * P.planned_sessions;
        else if (sp.gpu_layers >= 0 && s.layers > 0)  // гибрид: KV остальных слоёв лежит в RAM
            sess_ram += s.session_ram_bytes * P.planned_sessions * uint64_t(s.layers - sp.gpu_layers) / uint64_t(s.layers);
        ram += sp.ram_bytes;
        P.stages.push_back(sp);
    }
    P.vram_estimate = dev ? vram_used + kGpuOverhead : 0;
    P.ram_static = ram + (unified ? vram_used : 0);
    P.ram_estimate = P.ram_static + sess_ram;
    if (P.stages.size() > 1) {
        // порядок вывода — как в описании модели
        std::vector<StagePlacement> sorted;
        for (const StageDesc &s : fp.stages) sorted.push_back(placed[s.name]);
        P.stages = sorted;
    }
    C.free = dev ? dev->vram_free : 0;
    C.rank = !dev ? 0 : dev->software ? 1 : dev->type == "discrete" ? 4 : dev->type == "integrated" ? 3 : 2;
    return C;
}
}  // namespace

ResourcePlan make_plan(const ResourceConfig &cfg_in, const ModelFootprint &fp, const SystemInfo &sys, const std::vector<GpuDeviceInfo> &gpus) {
    ResourceConfig cfg = cfg_in;
    const Mode mode = cfg.mode == Mode::Auto ? Mode::Balanced : cfg.mode;
    ResourcePlan base;
    base.mode = mode;
    base.strict = cfg.strict;
    base.release_host_weights = cfg.release_host_weights;
    base.max_sessions = cfg.max_sessions;
    base.planned_sessions = std::max<uint32_t>(1, cfg.planned_sessions);
    base.enforce = cfg.enforce >= 0 ? cfg.enforce != 0 : (cfg.ram_limit != 0 || cfg.vram_limit != 0);
    base.vram_limit = cfg.vram_limit;
    base.ram_limit = cfg.ram_limit;

    // ---- CPU: потоки, привязка, приоритет ----
    const int hc = std::max(1, sys.cpu_allowed);
    uint32_t threads = cfg.threads;
    if (!threads) {
        threads = mode == Mode::Speed ? uint32_t(hc) : mode == Mode::Balanced ? uint32_t(hc > 2 ? hc - 1 : hc)  // один свободный поток для системы
                : mode == Mode::Memory ? uint32_t(std::max(1, hc / 2)) : uint32_t(std::max(1, hc / 4));
        if (!cfg.cpu_affinity.empty()) threads = std::min<uint32_t>(threads, uint32_t(cfg.cpu_affinity.size()));
    }
    base.threads = std::max<uint32_t>(1, threads);
    base.tuning.affinity = cfg.cpu_affinity;
    base.tuning.nice = cfg.nice != 1000 ? cfg.nice : (mode == Mode::Background ? 10 : 0);
    base.tuning.spin = mode == Mode::Speed || mode == Mode::Balanced;
    base.overlap = cfg.overlap >= 0 ? cfg.overlap != 0 : (mode == Mode::Speed || mode == Mode::Balanced);
    if (cfg.threads > uint32_t(hc))
        base.warnings.push_back("threads=" + std::to_string(cfg.threads) + " exceeds the " + std::to_string(hc) + " cores available to this process");

    // ---- память ----
    const uint64_t usable = sys.usable_ram();
    base.ram_budget = cfg.ram_limit ? cfg.ram_limit : static_cast<uint64_t>(double(usable) * (mode == Mode::Speed ? 0.9 : 0.8));

    // ---- кандидаты-устройства ----
    const bool explicit_gpu = cfg.device == "gpu" || cfg.gpu >= 0;
    std::vector<const GpuDeviceInfo *> cands;
    if (cfg.device == "cpu") {
        // только CPU
    } else if (explicit_gpu) {
        const int want = cfg.gpu >= 0 ? cfg.gpu : -1;
        for (const auto &d : gpus) if (want < 0 ? !d.software || gpus.size() == 1 : d.index == want) cands.push_back(&d);
        if (cands.empty() && want >= 0) fail(VOX_ERR_DEVICE, "GPU " + std::to_string(want) + " not found (Vulkan reports " + std::to_string(gpus.size()) + " device(s); see `vox devices`)");
        if (cands.empty()) fail(VOX_ERR_DEVICE, "device 'gpu' requested but no Vulkan device is available");
        if (want < 0) cands.resize(std::min<size_t>(cands.size(), gpus.size()));
    } else if (mode != Mode::Background) {
        for (const auto &d : gpus) if (!d.software) cands.push_back(&d);  // auto: программные рендереры не берём
    }

    std::vector<Candidate> results;
    results.push_back(plan_for(cfg, fp, sys, nullptr, mode, base));
    for (const GpuDeviceInfo *d : cands) results.push_back(plan_for(cfg, fp, sys, d, mode, base));

    size_t best = 0;
    if (cfg.device == "cpu" || cands.empty()) best = 0;
    else {
        best = 1;
        for (size_t i = 1; i < results.size(); ++i) {
            const Candidate &a = results[i], &b = results[best];
            const bool tie = std::fabs(a.score - b.score) <= 1e-12;
            if (a.score > b.score + 1e-12 || (tie && (a.rank > b.rank || (a.rank == b.rank && a.free > b.free)))) best = i;
        }
        // auto: GPU, на котором ничего не поместилось, хуже CPU
        if (!explicit_gpu && !results[best].plan.uses_gpu()) best = 0;
    }
    ResourcePlan P = results[best].plan;
    P.backend = P.uses_gpu() ? "vulkan" : "cpu";
    if (!P.uses_gpu()) { P.gpu_index = -1; P.gpu_name.clear(); P.vram_budget = P.vram_estimate = 0; }
    if (!explicit_gpu && cfg.device == "auto" && cands.size() > 1 && P.uses_gpu()) {
        std::string others;
        for (const GpuDeviceInfo *d : cands) if (d->index != P.gpu_index) others += (others.empty() ? "" : ", ") + std::string("vulkan:") + std::to_string(d->index) + " (" + d->name + ", free " + format_size(d->vram_free) + ")";
        P.notes.push_back("auto: chose vulkan:" + std::to_string(P.gpu_index) + " (" + P.gpu_name + "); other candidates: " + others);
    }
    if (cfg.device == "auto" && cands.empty() && !gpus.empty() && mode != Mode::Background)
        P.notes.push_back("auto: only software Vulkan renderers found, running on the CPU");
    if (mode == Mode::Background && cfg.device == "auto") P.notes.push_back("background mode: the GPU is not used unless a device is named explicitly");

    // ---- проверки лимитов ----
    if (P.ram_budget && P.ram_estimate > P.ram_budget) {
        P.warnings.push_back("estimated host memory " + format_size(P.ram_estimate) + " exceeds the budget " + format_size(P.ram_budget) +
                             (P.uses_gpu() ? "" : "; moving stages to the GPU (device=gpu) or using a q8 model would reduce it"));
        if (P.enforce && cfg.ram_limit) P.feasible = false;
    }
    if (cfg.strict && cfg.device != "cpu" && !P.uses_gpu() && !fp.stages.empty()) {
        P.warnings.push_back("strict: nothing could be placed on the GPU");
        P.feasible = false;
    }
    return P;
}

json ResourcePlan::to_json() const {
    json st = json::array();
    for (const auto &s : stages) {
        json e = {{"stage", s.stage}, {"place", place_name(s.place)}, {"weight_bytes", s.weight_bytes}, {"ram_bytes", s.ram_bytes},
                  {"vram_bytes", s.vram_bytes}, {"reason", s.reason}};
        if (s.gpu_layers >= 0) e["gpu_layers"] = s.gpu_layers;
        st.push_back(e);
    }
    json j = {{"mode", mode_name(mode)}, {"backend", backend}, {"gpu", gpu_index >= 0 ? json(gpu_index) : json(nullptr)}, {"gpu_name", gpu_name},
              {"threads", threads}, {"cpu_affinity", tuning.affinity}, {"nice", tuning.nice}, {"spin", tuning.spin}, {"overlap", overlap},
              {"strict", strict}, {"enforce", enforce}, {"release_host_weights", release_host_weights}, {"max_sessions", max_sessions},
              {"planned_sessions", planned_sessions}, {"ram_budget", ram_budget}, {"ram_estimate", ram_estimate}, {"ram_static", ram_static}, {"vram_budget", vram_budget},
              {"vram_estimate", vram_estimate}, {"feasible", feasible}, {"stages", st}, {"notes", notes}, {"warnings", warnings}};
    return j;
}

std::string ResourcePlan::summary() const {
    std::ostringstream o;
    o << "mode " << mode_name(mode) << ", " << threads << " CPU thread(s)";
    if (tuning.nice) o << ", nice " << tuning.nice;
    if (!tuning.affinity.empty()) { o << ", cores "; for (size_t i = 0; i < tuning.affinity.size(); ++i) o << (i ? "," : "") << tuning.affinity[i]; }
    o << "\n";
    o << "device: " << (uses_gpu() ? "vulkan:" + std::to_string(gpu_index) + " (" + gpu_name + ")" : std::string("cpu")) << "\n";
    for (const auto &s : stages) {
        o << "  " << s.stage << ": " << place_name(s.place);
        if (s.gpu_layers >= 0) o << " (" << s.gpu_layers << " layers)";
        o << "  weights " << format_size(s.weight_bytes) << "  — " << s.reason << "\n";
    }
    o << "memory: host ~" << format_size(ram_estimate);
    if (ram_budget) o << " / budget " << format_size(ram_budget);
    if (uses_gpu()) { o << ", VRAM ~" << format_size(vram_estimate); if (vram_budget) o << " / budget " << format_size(vram_budget); }
    o << "\n";
    for (const auto &n : notes) o << "note: " << n << "\n";
    for (const auto &w : warnings) o << "warning: " << w << "\n";
    o << (feasible ? "plan: feasible\n" : "plan: NOT feasible with these limits\n");
    return o.str();
}

// ====================================================== MemoryBudget, release_pages
bool MemoryBudget::try_reserve(uint64_t bytes) {
    uint64_t cur = used_.load();
    for (;;) {
        if (limit_ && cur + bytes > limit_) return false;
        if (used_.compare_exchange_weak(cur, cur + bytes)) break;
    }
    uint64_t pk = peak_.load();
    while (cur + bytes > pk && !peak_.compare_exchange_weak(pk, cur + bytes)) {}
    return true;
}
void MemoryBudget::release(uint64_t bytes) {
    uint64_t cur = used_.load();
    while (!used_.compare_exchange_weak(cur, cur > bytes ? cur - bytes : 0)) {}
}

void release_pages(const void *p, size_t bytes) {
#if defined(__unix__) || defined(__APPLE__)
    const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t a = (reinterpret_cast<uintptr_t>(p) + page - 1) & ~(page - 1);
    const uintptr_t b = (reinterpret_cast<uintptr_t>(p) + bytes) & ~(page - 1);
    if (b > a) madvise(reinterpret_cast<void *>(a), b - a, MADV_DONTNEED);
#else
    (void)p; (void)bytes;
#endif
}

}  // namespace vox
