#include "formats.hpp"

#include <cstdio>
#include <fstream>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace vox {

// ---------- common ----------
thread_local std::string g_last_error;
void set_last_error(const std::string &msg) { g_last_error = msg; }
}  // namespace vox
std::string vox_internal_last_error() { return vox::g_last_error; }
namespace vox {

size_t dtype_size(DType t) {
    switch (t) {
        case DType::F32: case DType::I32: return 4;
        case DType::F16: case DType::BF16: case DType::I16: return 2;
        case DType::I8: case DType::U8: return 1;
    }
    return 1;
}
const char *dtype_name(DType t) {
    switch (t) {
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::BF16: return "BF16";
        case DType::I32: return "I32";
        case DType::I16: return "I16";
        case DType::I8: return "I8";
        case DType::U8: return "U8";
    }
    return "U8";
}
DType dtype_from_name(const std::string &s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "I32") return DType::I32;
    if (s == "I16") return DType::I16;
    if (s == "I8") return DType::I8;
    if (s == "U8") return DType::U8;
    fail(VOX_ERR_FORMAT, "unsupported dtype: " + s);
}
std::string hex64(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

// ---------- mmap ----------
MappedFile::MappedFile(const std::string &path) {
#if defined(_WIN32)
    file_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) fail(VOX_ERR_IO, "cannot open " + path);
    LARGE_INTEGER sz;
    GetFileSizeEx(file_, &sz);
    size_ = static_cast<size_t>(sz.QuadPart);
    if (size_ == 0) fail(VOX_ERR_FORMAT, "empty file: " + path);
    mapping_ = CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping_) fail(VOX_ERR_IO, "cannot map " + path);
    data_ = static_cast<const uint8_t *>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
    if (!data_) fail(VOX_ERR_IO, "cannot map view " + path);
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) fail(VOX_ERR_IO, "cannot open " + path);
    struct stat st {};
    if (fstat(fd, &st) != 0) { ::close(fd); fail(VOX_ERR_IO, "cannot stat " + path); }
    size_ = static_cast<size_t>(st.st_size);
    if (size_ == 0) { ::close(fd); fail(VOX_ERR_FORMAT, "empty file: " + path); }
    void *p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) fail(VOX_ERR_IO, "mmap failed: " + path);
    data_ = static_cast<const uint8_t *>(p);
#endif
}

MappedFile::~MappedFile() {
#if defined(_WIN32)
    if (data_) UnmapViewOfFile(data_);
    if (mapping_) CloseHandle(mapping_);
    if (file_ && file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
#else
    if (data_) ::munmap(const_cast<uint8_t *>(data_), size_);
#endif
}

// ---------- safetensors ----------
static std::vector<int64_t> parse_shape(const json &j) {
    std::vector<int64_t> s;
    for (auto &d : j) s.push_back(d.get<int64_t>());
    return s;
}

SafetensorsFile safetensors_load(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail(VOX_ERR_IO, "cannot open " + path);
    uint64_t hlen = 0;
    in.read(reinterpret_cast<char *>(&hlen), 8);
    if (!in || hlen == 0 || hlen > (64ull << 20)) fail(VOX_ERR_FORMAT, "bad safetensors header: " + path);
    std::string hdr(hlen, '\0');
    in.read(hdr.data(), static_cast<std::streamsize>(hlen));
    if (!in) fail(VOX_ERR_FORMAT, "truncated safetensors header: " + path);
    std::vector<uint8_t> body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    json j;
    try { j = json::parse(hdr); } catch (const std::exception &e) {
        fail(VOX_ERR_FORMAT, std::string("bad safetensors JSON: ") + e.what());
    }
    SafetensorsFile f;
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.key() == "__metadata__") {
            for (auto m = it.value().begin(); m != it.value().end(); ++m)
                f.metadata[m.key()] = m.value().get<std::string>();
            continue;
        }
        const json &t = it.value();
        Tensor ten;
        ten.dtype = dtype_from_name(t.at("dtype").get<std::string>());
        ten.shape = parse_shape(t.at("shape"));
        uint64_t b = t.at("data_offsets")[0].get<uint64_t>(), e = t.at("data_offsets")[1].get<uint64_t>();
        if (e < b || e > body.size()) fail(VOX_ERR_FORMAT, "tensor out of range: " + it.key());
        int64_t expect = 1;
        for (auto d : ten.shape) expect *= d;
        if (static_cast<uint64_t>(expect) * dtype_size(ten.dtype) != e - b)
            fail(VOX_ERR_FORMAT, "tensor size mismatch: " + it.key());
        ten.bytes.assign(body.begin() + static_cast<ptrdiff_t>(b), body.begin() + static_cast<ptrdiff_t>(e));
        f.tensors.emplace(it.key(), std::move(ten));
    }
    return f;
}

void safetensors_save(const std::string &path, const SafetensorsFile &f) {
    json j = json::object();
    uint64_t off = 0;
    for (auto &[name, t] : f.tensors) {
        j[name] = {{"dtype", dtype_name(t.dtype)}, {"shape", t.shape}, {"data_offsets", {off, off + t.bytes.size()}}};
        off += t.bytes.size();
    }
    if (!f.metadata.empty()) {
        json m = json::object();
        for (auto &[k, v] : f.metadata) m[k] = v;
        j["__metadata__"] = m;
    }
    std::string hdr = j.dump();
    while (hdr.size() % 8) hdr.push_back(' ');  // выравнивание данных по 8 байт
    std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        if (!out) fail(VOX_ERR_IO, "cannot write " + path);
        uint64_t hlen = hdr.size();
        out.write(reinterpret_cast<const char *>(&hlen), 8);
        out.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
        for (auto &[name, t] : f.tensors)
            out.write(reinterpret_cast<const char *>(t.bytes.data()), static_cast<std::streamsize>(t.bytes.size()));
        if (!out) fail(VOX_ERR_IO, "write failed: " + path);
    }
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) fail(VOX_ERR_IO, "cannot rename to " + path);
}

// ---------- .tmod ----------
TmodFile::TmodFile(const std::string &path) : map_(std::make_unique<MappedFile>(path)) {
    if (map_->size() < sizeof(TmodHeader)) fail(VOX_ERR_FORMAT, "file too small for .tmod: " + path);
    TmodHeader h;
    std::memcpy(&h, map_->data(), sizeof h);
    if (std::memcmp(h.magic, TMOD_MAGIC, 4) != 0) fail(VOX_ERR_FORMAT, "not a .tmod file: " + path);
    if (h.version_major != TMOD_VERSION_MAJOR)
        fail(VOX_ERR_FORMAT, ".tmod format " + std::to_string(h.version_major) + ".x is not supported by this runtime");
    if (h.manifest_offset + h.manifest_size > map_->size() || h.data_offset + h.data_size > map_->size())
        fail(VOX_ERR_FORMAT, "truncated .tmod: " + path);
    flags_ = h.flags;
    try {
        manifest_ = json::parse(std::string(reinterpret_cast<const char *>(map_->data() + h.manifest_offset),
                                            h.manifest_size));
    } catch (const std::exception &e) {
        fail(VOX_ERR_FORMAT, std::string("bad .tmod manifest: ") + e.what());
    }
    const uint8_t *base = map_->data() + h.data_offset;
    if (manifest_.contains("blobs")) {
        for (auto it = manifest_["blobs"].begin(); it != manifest_["blobs"].end(); ++it) {
            const json &b = it.value();
            TensorView v;
            v.dtype = dtype_from_name(b.at("dtype").get<std::string>());
            v.shape = parse_shape(b.at("shape"));
            uint64_t off = b.at("offset").get<uint64_t>(), n = b.at("nbytes").get<uint64_t>();
            if (off + n > h.data_size) fail(VOX_ERR_FORMAT, "blob out of range: " + it.key());
            v.data = base + off;
            v.nbytes = n;
            blobs_.emplace(it.key(), v);
        }
    }
}

const TensorView &TmodFile::get(const std::string &name) const {
    auto it = blobs_.find(name);
    if (it == blobs_.end()) fail(VOX_ERR_FORMAT, "missing blob in .tmod: " + name);
    return it->second;
}

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

void tmod_write(const std::string &path, json manifest, const std::map<std::string, Tensor> &blobs,
                uint32_t flags) {
    json table = json::object();
    uint64_t off = 0;
    for (auto &[name, t] : blobs) {
        off = align_up(off, TMOD_ALIGN);
        table[name] = {{"dtype", dtype_name(t.dtype)}, {"shape", t.shape}, {"offset", off}, {"nbytes", t.bytes.size()}};
        off += t.bytes.size();
    }
    manifest["blobs"] = table;
    std::string m = manifest.dump(2);

    TmodHeader h{};
    std::memcpy(h.magic, TMOD_MAGIC, 4);
    h.version_major = TMOD_VERSION_MAJOR;
    h.version_minor = TMOD_VERSION_MINOR;
    h.flags = flags;
    h.manifest_offset = sizeof(TmodHeader);
    h.manifest_size = m.size();
    h.data_offset = align_up(h.manifest_offset + h.manifest_size, TMOD_ALIGN);
    h.data_size = off;

    std::ofstream out(path, std::ios::binary);
    if (!out) fail(VOX_ERR_IO, "cannot write " + path);
    out.write(reinterpret_cast<const char *>(&h), sizeof h);
    out.write(m.data(), static_cast<std::streamsize>(m.size()));
    std::vector<char> zeros(TMOD_ALIGN, 0);
    uint64_t pos = h.manifest_offset + h.manifest_size;
    out.write(zeros.data(), static_cast<std::streamsize>(h.data_offset - pos));
    uint64_t dpos = 0;
    for (auto &[name, t] : blobs) {
        uint64_t a = align_up(dpos, TMOD_ALIGN);
        out.write(zeros.data(), static_cast<std::streamsize>(a - dpos));
        out.write(reinterpret_cast<const char *>(t.bytes.data()), static_cast<std::streamsize>(t.bytes.size()));
        dpos = a + t.bytes.size();
    }
    if (!out) fail(VOX_ERR_IO, "write failed: " + path);
}

}  // namespace vox
