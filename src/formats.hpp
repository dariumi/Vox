#pragma once
// Форматы файлов vox:
//   .tvoice — safetensors (тензоры + __metadata__ со строками JSON)
//   .tmod   — собственный контейнер: заголовок 64 байта, JSON-манифест,
//             область данных с блобами, выровненными по 4096 байт (для mmap).
#include <map>
#include <memory>
#include <string>

#include "../third_party/json.hpp"
#include "common.hpp"

namespace vox {

using json = nlohmann::json;

// ---------- mmap ----------
class MappedFile {
public:
    explicit MappedFile(const std::string &path);
    ~MappedFile();
    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;
    const uint8_t *data() const { return data_; }
    size_t size() const { return size_; }

private:
    const uint8_t *data_ = nullptr;
    size_t size_ = 0;
#if defined(_WIN32)
    void *file_ = nullptr, *mapping_ = nullptr;
#endif
};

// ---------- safetensors ----------
struct SafetensorsFile {
    std::map<std::string, Tensor> tensors;
    std::map<std::string, std::string> metadata;
};

SafetensorsFile safetensors_load(const std::string &path);
void safetensors_save(const std::string &path, const SafetensorsFile &f);

// ---------- .tmod ----------
constexpr char TMOD_MAGIC[4] = {'V', 'O', 'X', 'M'};
constexpr uint16_t TMOD_VERSION_MAJOR = 1;
constexpr uint16_t TMOD_VERSION_MINOR = 0;
constexpr size_t TMOD_ALIGN = 4096;

#pragma pack(push, 1)
struct TmodHeader {
    char magic[4];
    uint16_t version_major;
    uint16_t version_minor;
    uint32_t flags;            // бит 0: specialized (есть ядра под цель)
    uint32_t reserved0;
    uint64_t manifest_offset;
    uint64_t manifest_size;
    uint64_t data_offset;
    uint64_t data_size;
    uint8_t reserved[16];
};
#pragma pack(pop)
static_assert(sizeof(TmodHeader) == 64, "TmodHeader must be 64 bytes");

// Открытый .tmod: манифест + невладеющие взгляды на блобы внутри mmap.
class TmodFile {
public:
    explicit TmodFile(const std::string &path);
    const json &manifest() const { return manifest_; }
    bool has(const std::string &name) const { return blobs_.count(name) != 0; }
    const TensorView &get(const std::string &name) const;
    const std::map<std::string, TensorView> &blobs() const { return blobs_; }
    bool specialized() const { return flags_ & 1u; }

private:
    std::unique_ptr<MappedFile> map_;
    json manifest_;
    uint32_t flags_ = 0;
    std::map<std::string, TensorView> blobs_;
};

// Запись .tmod: манифест дополняется таблицей "blobs" автоматически.
void tmod_write(const std::string &path, json manifest, const std::map<std::string, Tensor> &blobs,
                uint32_t flags = 0);

}  // namespace vox
