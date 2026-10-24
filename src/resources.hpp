#pragma once
// Менеджер ресурсов: что и где считать, сколько памяти и ядер можно занять.
//
//   ResourceConfig  — пожелания пользователя (устройство, режим, лимиты памяти, потоки, размещение стадий);
//   ModelFootprint  — что нужно модели (веса и трафик по стадиям); его даёт семейство модели (family.hpp);
//   SystemInfo      — что есть на машине (ядра, RAM с учётом cgroup, видеопамять каждого GPU);
//   ResourcePlan    — результат планирования: устройство, размещение каждой стадии, оценка RAM/VRAM, потоки.
//
// make_plan() — чистая функция: ничего не загружает и не обращается к железу, поэтому план можно получить до
// открытия модели (`vox plan`, vox.plan() в Python) и проверить тестами на выдуманных устройствах.
#include <atomic>
#include <map>
#include <string>
#include <vector>

#include "formats.hpp"
#include "gpu.hpp"
#include "kernels.hpp"

namespace vox {

enum class Place { Auto, Cpu, Gpu };
enum class Mode { Auto, Speed, Balanced, Memory, Background };

const char *place_name(Place p);
const char *mode_name(Mode m);
Place place_from_string(const std::string &s);  // "auto" | "cpu" | "gpu"
Mode mode_from_string(const std::string &s);    // "auto" | "speed" | "balanced" | "memory" | "background"
// Размер: число байт или строка «6GiB», «512M», «1.5G», «50%» (доля от whole; без whole процент недопустим).
uint64_t parse_size(const json &v, uint64_t whole = 0);
std::string format_size(uint64_t bytes);

struct ResourceConfig {
    // "auto" | "cpu" | "gpu" | "gpu:N" | "vulkan" | "vulkan:N", далее через запятую: strict, codec=cpu|gpu,
    // talker=cpu|gpu, predictor=cpu|gpu, layers=N, mode=..., threads=N, ram=..., vram=...
    std::string device = "auto";
    Mode mode = Mode::Auto;
    int gpu = -1;                                // индекс устройства Vulkan (перекрывает номер в device)
    uint32_t threads = 0;                        // потоки CPU, 0 = по режиму
    std::vector<int> cpu_affinity;               // ядра, на которых живут рабочие потоки (пусто = как решит ОС)
    int nice = 1000;                             // приоритет рабочих потоков (Linux); 1000 = по режиму
    uint64_t ram_limit = 0, vram_limit = 0;      // байт; 0 = автоматически по доступной памяти
    std::map<std::string, Place> place;          // стадия → устройство (talker, predictor, codec, ...)
    std::map<std::string, int> gpu_layers;       // стадия → сколько слоёв держать на GPU (остальные на CPU)
    bool strict = false;                         // не откатываться молча на CPU, если что-то не поместилось
    int enforce = -1;                            // отказ открыть модель, если план превышает лимиты (-1 = да, если лимит задан явно)
    int overlap = -1;                            // кодек на CPU параллельно GPU (-1 = по режиму)
    bool release_host_weights = true;            // после загрузки на GPU вернуть системе страницы mmap с весами
    bool tuned = true;                           // при "auto" без явных настроек взять замеренную конфигурацию (tune.hpp)
    uint32_t max_sessions = 0;                   // жёсткий предел одновременных сессий, 0 = без предела
    uint32_t planned_sessions = 1;               // на сколько сессий резервировать KV-кэш при планировании

    static ResourceConfig from_json(const json &j);
    json to_json() const;
    // Разбор строки устройства (см. device выше); ошибки — VOX_ERR_DEVICE.
    void apply_device_string(const std::string &spec);
};

// ---------- что нужно модели ----------
struct StageDesc {
    std::string name;                       // "talker", "predictor", "codec"
    std::vector<std::string> prefixes;      // префиксы блобов .tmod, из которых складывается вес стадии
    std::vector<std::string> exclude;       // …за вычетом блобов с этими префиксами
    uint64_t weight_bytes = 0;              // заполняется из манифеста, если 0
    bool gpu_capable = true;
    bool lookup_only = false;               // таблицы, из которых берутся отдельные строки: RSS почти не растёт
    double bytes_per_frame = 0;             // сколько байт весов прочитывается на кадр (predictor: 15 проходов)
    double flops_per_frame = 0;
    int layers = 0;                         // >0 и splittable: часть слоёв можно держать на GPU, остальные на CPU
    bool splittable = false;
    uint64_t layer_bytes = 0;               // вес одного слоя
    uint64_t fixed_gpu_bytes = 0;           // что едет на GPU вместе со стадией (голова, проекции)
    uint64_t session_vram_bytes = 0;        // KV-кэш одной сессии на GPU
    uint64_t session_ram_bytes = 0;         // и на CPU
};
struct ModelFootprint {
    std::string family, variant;
    std::vector<StageDesc> stages;
    uint64_t always_ram_bytes = 0;          // что остаётся в RAM при любом размещении
};

// ---------- что есть на машине ----------
struct SystemInfo {
    int cpu_logical = 1, cpu_allowed = 1;   // всего и доступно процессу (affinity, cpuset)
    int cpu_physical = 0;                   // 0 = неизвестно; физических ядер среди доступных (SMT-соседи считаются одним ядром)
    uint64_t ram_total = 0, ram_available = 0;
    uint64_t cgroup_limit = 0;              // 0 = нет ограничения контейнера
    uint64_t rss = 0, peak_rss = 0;
    std::string os;
    uint64_t usable_ram() const;            // что можно занять: min(доступно, остаток лимита контейнера)
    json to_json() const;
};
SystemInfo system_info();

// ---------- результат ----------
struct StagePlacement {
    std::string stage;
    Place place = Place::Cpu;
    int gpu_layers = -1;                    // для разделяемых стадий: слоёв на GPU (-1 = все)
    uint64_t weight_bytes = 0, ram_bytes = 0, vram_bytes = 0;
    std::string reason;
};
struct ResourcePlan {
    Mode mode = Mode::Balanced;
    std::string backend = "cpu";            // "cpu" | "vulkan"
    int gpu_index = -1;
    std::string gpu_name;
    uint32_t threads = 1;
    CpuTuning tuning;
    bool strict = false, overlap = true, release_host_weights = true;
    uint32_t max_sessions = 0, planned_sessions = 1;
    std::vector<StagePlacement> stages;
    uint64_t ram_budget = 0, vram_budget = 0, ram_estimate = 0, vram_estimate = 0, vram_limit = 0;
    uint64_t ram_static = 0;                // веса и таблицы на хосте без сессий (то, что не зависит от числа сессий)
    uint64_t ram_limit = 0;                 // жёсткий лимит пользователя (0 = нет); сессии делят ram_limit − ram_static
    bool feasible = true;
    bool enforce = false;
    std::vector<std::string> notes, warnings;

    Place place_of(const std::string &stage) const;
    int gpu_layers_of(const std::string &stage) const;  // -1 = все слои (если стадия на GPU)
    bool uses_gpu() const;
    json to_json() const;
    std::string summary() const;            // читаемое описание для `vox plan`
};

ResourcePlan make_plan(const ResourceConfig &cfg, const ModelFootprint &fp, const SystemInfo &sys, const std::vector<GpuDeviceInfo> &gpus);

// Учёт памяти, которую движок занимает сам (KV-кэши и состояния сессий). Сессия, не влезающая в лимит, не создаётся.
class MemoryBudget {
public:
    explicit MemoryBudget(uint64_t limit = 0) : limit_(limit) {}
    bool try_reserve(uint64_t bytes);
    void release(uint64_t bytes);
    uint64_t used() const { return used_.load(); }
    uint64_t peak() const { return peak_.load(); }
    uint64_t limit() const { return limit_; }
    void set_limit(uint64_t limit) { limit_ = limit; }  // до создания первой сессии

private:
    uint64_t limit_;
    std::atomic<uint64_t> used_{0}, peak_{0};
};

// Возвращает системе страницы mmap с весами (Linux/macOS: madvise DONTNEED). Данные остаются доступными (файл), только
// при следующем обращении страницы подгрузятся заново, поэтому вызывать можно для чего угодно, но имеет смысл после
// загрузки в видеопамять.
void release_pages(const void *p, size_t bytes);

}  // namespace vox
