#include "tune.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>

#include "family.hpp"
#include "gpu.hpp"
#include "resources.hpp"

#if defined(__x86_64__) || defined(__i386__)
#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
#endif
#if defined(__APPLE__)
#  include <sys/sysctl.h>
#endif
#if defined(_WIN32)
#  include <direct.h>
#else
#  include <sys/stat.h>
#endif

namespace vox {

namespace {

std::string cpu_brand() {
#if defined(__x86_64__) || defined(__i386__)
    unsigned r[12] = {0};
#  if defined(_MSC_VER)
    int ext[4];
    __cpuid(ext, 0x80000000);
    if (unsigned(ext[0]) >= 0x80000004u)
        for (unsigned i = 0; i < 3; ++i) __cpuid(reinterpret_cast<int *>(r + 4 * i), int(0x80000002u + i));
#  else
    if (__get_cpuid_max(0x80000000u, nullptr) >= 0x80000004u)
        for (unsigned i = 0; i < 3; ++i) __get_cpuid(0x80000002u + i, &r[4 * i], &r[4 * i + 1], &r[4 * i + 2], &r[4 * i + 3]);
#  endif
    std::string s(reinterpret_cast<const char *>(r), 48);
    s = s.c_str();
    const auto a = s.find_first_not_of(' ');
    if (a != std::string::npos && !s.empty()) return s.substr(a);
#endif
#if defined(__APPLE__)
    char buf[256];
    size_t n = sizeof buf;
    if (sysctlbyname("machdep.cpu.brand_string", buf, &n, nullptr, 0) == 0) return std::string(buf);
#endif
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line))
        for (const char *k : {"model name", "Hardware", "CPU part"})
            if (line.compare(0, std::strlen(k), k) == 0) {
                const auto c = line.find(':');
                if (c != std::string::npos) return line.substr(c + 2);
            }
    return "unknown CPU";
}

std::string home_cache_dir() {
    if (const char *x = std::getenv("XDG_CACHE_HOME")) if (x[0]) return std::string(x) + "/vox";
#if defined(_WIN32)
    if (const char *x = std::getenv("LOCALAPPDATA")) if (x[0]) return std::string(x) + "\\vox";
    if (const char *x = std::getenv("USERPROFILE")) if (x[0]) return std::string(x) + "\\.cache\\vox";
#endif
    if (const char *x = std::getenv("HOME")) if (x[0]) return std::string(x) + "/.cache/vox";
    return ".vox-cache";
}

void make_dirs(const std::string &path) {   // the parent folders of a file
    for (size_t i = 1; i < path.size(); ++i)
        if (path[i] == '/' || path[i] == '\\') {
            const std::string d = path.substr(0, i);
#if defined(_WIN32)
            _mkdir(d.c_str());
#else
            mkdir(d.c_str(), 0755);
#endif
        }
}

json read_cache() {
    std::ifstream in(tune_cache_path());
    if (!in) return json::object();
    try {
        json j = json::parse(in);
        return j.is_object() ? j : json::object();
    } catch (const std::exception &) {
        return json::object();   // a damaged cache is a missing cache
    }
}

void write_cache(const json &j) {
    const std::string p = tune_cache_path(), tmp = p + ".tmp" + std::to_string(std::rand());
    make_dirs(p);
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) fail(VOX_ERR_IO, "cannot write the tuning cache " + p);
        out << j.dump(1) << "\n";
    }
    std::remove(p.c_str());   // Windows: rename does not replace
    if (std::rename(tmp.c_str(), p.c_str()) != 0) { std::remove(tmp.c_str()); fail(VOX_ERR_IO, "cannot write the tuning cache " + p); }
}

std::string entry_key(const TmodFile &f) { return tune_model_key(f) + "@" + tune_hw_key(); }

bool software_ok() {
    const char *e = std::getenv("VOX_TUNE_SOFTWARE");   // tests: tune on lavapipe too
    return e && e[0] && e[0] != '0';
}

}  // namespace

std::string tune_cache_path() {
    if (const char *x = std::getenv("VOX_TUNE_CACHE")) if (x[0]) return x;
    return home_cache_dir() + "/tune.json";
}

json tune_hardware() {
    const SystemInfo si = system_info();
    json gpus = json::array();
    for (const GpuDeviceInfo &d : gpu_enumerate()) {
        if (d.software && !software_ok()) continue;
        gpus.push_back({{"index", d.index}, {"name", d.name}, {"driver", d.driver_version}, {"vram_gib", int((d.vram_total + (1ull << 29)) >> 30)}});
    }
    return {{"cpu", cpu_brand()}, {"cores", si.cpu_allowed}, {"ram_gib", int((si.ram_total + (1ull << 29)) >> 30)}, {"gpus", gpus}};
}

std::string tune_hw_key() {
    static const std::string key = hex64(fnv1a64(tune_hardware().dump()));   // the hardware does not change while we run
    return key;
}

std::string tune_model_key(const TmodFile &f) {
    const json &m = f.manifest();
    uint64_t bytes = 0;
    for (const auto &kv : f.blobs()) bytes += kv.second.nbytes;
    const json mm = m.value("model", json::object());
    std::string s = mm.value("family", "") + "|" + mm.value("variant", "") + "|" + mm.value("weights_sha256", "") + "|" +
                    m.value("precision", json::object()).dump() + "|" + std::to_string(bytes) + "|" + std::to_string(f.blobs().size());
    return hex64(fnv1a64(s));
}

json tune_entry(const TmodFile &f) {
    const json c = read_cache();
    const json &e = c.value("entries", json::object());
    const std::string k = entry_key(f);
    return e.contains(k) ? e.at(k) : json();
}

void tune_store(const TmodFile &f, json entry) {
    if (!entry.is_object() || !entry.contains("device") || !entry["device"].is_string())
        fail(VOX_ERR_ARG, "a tuning result needs at least {\"device\": \"…\"}");
    // the device string must parse (it is applied when an engine opens)
    ResourceConfig probe;
    probe.apply_device_string(entry["device"].get<std::string>());
    json c = read_cache();
    if (!c.contains("entries") || !c["entries"].is_object()) c["entries"] = json::object();
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char when[32];
    std::strftime(when, sizeof when, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    entry["model_key"] = tune_model_key(f);
    entry["hardware_key"] = tune_hw_key();
    entry["hardware"] = tune_hardware();
    entry["when"] = when;
    entry["vox"] = VOX_VERSION_STRING;
    const json &mm = f.manifest().value("model", json::object());
    entry["model"] = {{"family", mm.value("family", "")}, {"variant", mm.value("variant", "")}};
    c["format"] = 1;
    c["entries"][entry_key(f)] = entry;
    write_cache(c);
}

bool tune_forget(const TmodFile &f) {
    json c = read_cache();
    if (!c.contains("entries") || !c["entries"].contains(entry_key(f))) return false;
    c["entries"].erase(entry_key(f));
    write_cache(c);
    return true;
}

json tune_candidates(const TmodFile &f) {
    json out = json::array();
    const SystemInfo si = system_info();
    out.push_back({{"label", "cpu"}, {"device", "cpu"}});
    if (si.cpu_allowed >= 4) {   // SMT siblings share the FMA units: the physical cores alone are often as fast or faster
        const int half = si.cpu_allowed / 2;
        out.push_back({{"label", "cpu, " + std::to_string(half) + " threads"}, {"device", "cpu,threads=" + std::to_string(half)}});
    }
    const Family &fam = find_family(f.manifest());
    const ModelFootprint fp = fam.footprint(f);
    std::vector<std::string> gpu_stages;
    for (const StageDesc &s : fp.stages)
        if (s.gpu_capable) gpu_stages.push_back(s.name);
    if (gpu_stages.empty()) return out;
    for (const GpuDeviceInfo &d : gpu_enumerate()) {
        if (d.software && !software_ok()) continue;
        const std::string dev = "vulkan:" + std::to_string(d.index);
        out.push_back({{"label", dev + " (" + d.name + ")"}, {"device", dev}});
        if (gpu_stages.size() > 1)
            for (const std::string &s : gpu_stages)
                out.push_back({{"label", dev + ", " + s + " on the CPU"}, {"device", dev + "," + s + "=cpu"}});
    }
    return out;
}

}  // namespace vox
