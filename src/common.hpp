#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vox.h"

namespace vox {

// Внутренние ошибки — исключения. На границе C ABI они превращаются в vox_status.
struct Error : std::runtime_error {
    vox_status status;
    Error(vox_status s, const std::string &msg) : std::runtime_error(msg), status(s) {}
};

[[noreturn]] inline void fail(vox_status s, const std::string &msg) { throw Error(s, msg); }

void set_last_error(const std::string &msg);

enum class DType { F32, F16, BF16, I32, I16, I8, U8 };

size_t dtype_size(DType t);
const char *dtype_name(DType t);   // имена как в safetensors: "F32", "I16", ...
DType dtype_from_name(const std::string &s);

// Невладеющий взгляд на тензор (данные в mmap или в чужом буфере).
struct TensorView {
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    const uint8_t *data = nullptr;
    size_t nbytes = 0;

    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

// Владеющий тензор.
struct Tensor {
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;

    TensorView view() const { return {dtype, shape, bytes.data(), bytes.size()}; }

    template <typename T>
    static Tensor from(DType dt, std::vector<int64_t> shape, const std::vector<T> &v) {
        Tensor t;
        t.dtype = dt;
        t.shape = std::move(shape);
        t.bytes.resize(v.size() * sizeof(T));
        if (!v.empty()) std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
        return t;
    }
    template <typename T>
    std::vector<T> as() const {
        std::vector<T> v(bytes.size() / sizeof(T));
        if (!v.empty()) std::memcpy(v.data(), bytes.data(), bytes.size());
        return v;
    }
};

// FNV-1a 64: быстрый некриптографический хэш для отпечатков сборки.
inline uint64_t fnv1a64(const void *data, size_t n, uint64_t h = 1469598103934665603ULL) {
    auto p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
inline uint64_t fnv1a64(const std::string &s, uint64_t h = 1469598103934665603ULL) {
    return fnv1a64(s.data(), s.size(), h);
}
std::string hex64(uint64_t v);

}  // namespace vox
