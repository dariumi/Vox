#pragma once
// Минимальный WAV: чтение PCM 16/24/32 и float32 (с переводом в mono), запись float32 или PCM16.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "common.hpp"

namespace vox {

struct Audio {
    std::vector<float> pcm;  // mono
    uint32_t sample_rate = 0;
};

inline Audio wav_decode(const uint8_t *d, size_t n) {
    auto u16 = [&](size_t o) { uint16_t v; std::memcpy(&v, d + o, 2); return v; };
    auto u32 = [&](size_t o) { uint32_t v; std::memcpy(&v, d + o, 4); return v; };
    if (n < 12 || std::memcmp(d, "RIFF", 4) || std::memcmp(d + 8, "WAVE", 4)) fail(VOX_ERR_FORMAT, "not a WAV file");
    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t sr = 0;
    const uint8_t *data = nullptr;
    size_t data_n = 0;
    for (size_t o = 12; o + 8 <= n;) {
        uint32_t sz = u32(o + 4);
        if (!std::memcmp(d + o, "fmt ", 4) && o + 24 <= n) {
            fmt = u16(o + 8); ch = u16(o + 10); sr = u32(o + 12); bits = u16(o + 22);
            if (fmt == 0xFFFE && sz >= 40 && o + 34 <= n) fmt = u16(o + 32);  // WAVE_FORMAT_EXTENSIBLE
        } else if (!std::memcmp(d + o, "data", 4)) {
            data = d + o + 8;
            data_n = std::min<size_t>(sz, n - o - 8);
            break;
        }
        o += 8 + sz + (sz & 1);
    }
    if (!data || !ch || !sr) fail(VOX_ERR_FORMAT, "WAV without fmt/data chunk");
    const size_t bps = bits / 8, frames = data_n / (bps * ch);
    Audio a;
    a.sample_rate = sr;
    a.pcm.resize(frames);
    for (size_t f = 0; f < frames; ++f) {
        double acc = 0;
        for (uint16_t c = 0; c < ch; ++c) {
            const uint8_t *p = data + (f * ch + c) * bps;
            if (fmt == 3 && bits == 32) { float v; std::memcpy(&v, p, 4); acc += v; }
            else if (fmt == 1 && bits == 16) { int16_t v; std::memcpy(&v, p, 2); acc += v / 32768.0; }
            else if (fmt == 1 && bits == 24) { int32_t v = (p[0] << 8) | (p[1] << 16) | (p[2] << 24); acc += (v >> 8) / 8388608.0; }
            else if (fmt == 1 && bits == 32) { int32_t v; std::memcpy(&v, p, 4); acc += v / 2147483648.0; }
            else fail(VOX_ERR_FORMAT, "unsupported WAV encoding (format " + std::to_string(fmt) + ", " + std::to_string(bits) + " bit)");
        }
        a.pcm[f] = float(acc / ch);
    }
    return a;
}

inline Audio wav_read(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail(VOX_ERR_IO, "cannot open " + path);
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return wav_decode(b.data(), b.size());
}

inline std::vector<uint8_t> wav_encode(const float *pcm, size_t n, uint32_t sr, bool pcm16) {
    const uint16_t bits = pcm16 ? 16 : 32, fmt = pcm16 ? 1 : 3;
    const uint32_t data_n = uint32_t(n * bits / 8);
    std::vector<uint8_t> o(44 + data_n);
    auto w16 = [&](size_t at, uint16_t v) { std::memcpy(&o[at], &v, 2); };
    auto w32 = [&](size_t at, uint32_t v) { std::memcpy(&o[at], &v, 4); };
    std::memcpy(&o[0], "RIFF", 4); w32(4, 36 + data_n); std::memcpy(&o[8], "WAVE", 4);
    std::memcpy(&o[12], "fmt ", 4); w32(16, 16); w16(20, fmt); w16(22, 1); w32(24, sr);
    w32(28, sr * bits / 8); w16(32, bits / 8); w16(34, bits);
    std::memcpy(&o[36], "data", 4); w32(40, data_n);
    for (size_t i = 0; i < n; ++i) {
        if (pcm16) {
            // round(x · 32768) clamped: the exact inverse of wav_decode (v / 32768), so decode → encode keeps the bytes
            const float x = pcm[i] * 32768.0f;
            const int16_t v = int16_t(x <= -32768.0f ? -32768 : x >= 32767.0f ? 32767 : int(std::lrint(x)));
            std::memcpy(&o[44 + i * 2], &v, 2);
        } else {
            std::memcpy(&o[44 + i * 4], &pcm[i], 4);
        }
    }
    return o;
}

inline void wav_write(const std::string &path, const float *pcm, size_t n, uint32_t sr, bool pcm16 = true) {
    auto b = wav_encode(pcm, n, sr, pcm16);
    std::ofstream out(path, std::ios::binary);
    if (!out) fail(VOX_ERR_IO, "cannot write " + path);
    out.write(reinterpret_cast<const char *>(b.data()), std::streamsize(b.size()));
}

}  // namespace vox
