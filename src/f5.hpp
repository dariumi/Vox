#pragma once
// F5-TTS: диффузионный трансформер (DiT) с условным сопоставлением потоков (CFM) + вокодер Vocos.
// Модель не авторегрессионная: по mel-спектрограмме образца, тексту образца и тексту фразы она за NFE шагов Эйлера
// превращает шум в mel-спектрограмму всей фразы, а Vocos переводит её в звук. Здесь — вся математика на CPU
// (линейные слои могут лежать на GPU через Mat::gpu); Backend (backend_f5.cpp) только управляет сессиями.
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "f5_ops.hpp"
#include "formats.hpp"

namespace vox {

struct F5Cfg {
    int dim = 1024, depth = 22, heads = 16, dim_head = 64, ff_mult = 2;
    int text_dim = 512, text_rows = 2546, mel_dim = 100, conv_layers = 4, conv_mult = 2;
    bool mask_padding = true;
    MelCfg mel;
    int vocos_dim = 512, vocos_inter = 1536, vocos_layers = 8, vocos_n_fft = 1024, vocos_hop = 256;
    uint32_t sample_rate = 24000;
};

// Текст → токены. "pinyin": китайские иероглифы заменяются слогами с тоном (таблица из манифеста), латиница — посимвольно;
// "char": каждый символ UTF-8 — токен. Неизвестный символ → 0 (как в F5: vocab_char_map.get(c, 0)).
class F5Tokenizer {
public:
    void load(const json &tok);
    std::vector<int32_t> encode(const std::string &utf8) const;
    // Последовательность «символов» после преобразования (для тестов): каждый элемент — одна строка UTF-8.
    std::vector<std::string> symbols(const std::string &utf8) const;
    bool pinyin_mode() const { return pinyin_; }

private:
    bool pinyin_ = true;
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<uint32_t, std::string> pinyin_map_;
};

struct F5Sampler {
    int nfe = 32;
    float cfg = 2.0f, sway = -1.0f;
    bool epss = true;
};

class F5Model {
public:
    // gpu: поднять линейные слои на GPU (если он уже инициализирован).
    F5Model(std::shared_ptr<TmodFile> f, bool gpu);
    ~F5Model();
    F5Model(const F5Model &) = delete;
    F5Model &operator=(const F5Model &) = delete;

    const F5Cfg &cfg() const { return cfg_; }
    const F5Tokenizer &tokenizer() const { return tok_; }

    // mel как у F5 (после нормировки громкости и ресемплинга до 24 кГц): [frames][mel_dim], frames = 1 + n / hop.
    void ref_mel(const float *pcm24k, size_t n, std::vector<float> &mel, int &frames) const;

    // Шум N(0,1) для куска chunk: счётчиковый Philox, не зависит от потоков/порядка.
    static void noise(uint64_t seed, uint32_t chunk, size_t n, float *out);

    // Сетка времени: linspace или EPSS, затем sway.
    static std::vector<float> timesteps(const F5Sampler &s);

    // Эмбеддинг текста (для cond: drop=false, для безусловной ветки: drop=true): [N][text_dim].
    void text_embed(const std::vector<int32_t> &ids, int N, bool drop, std::vector<float> &out) const;

    // Один проход DiT над B последовательностями длины N (B = 1 или 2: условная и безусловная ветки CFG).
    // x, cond: [B][N][mel_dim], temb: [B][N][text_dim], t — момент времени. out: [B][N][mel_dim].
    void dit_forward(int B, int N, const float *x, const float *cond, const float *temb, float t, float *out) const;

    // Полный цикл CFM: cond_mel [cond_frames][mel_dim], text-токены, длительность в кадрах, шум [duration][mel_dim].
    // out: [duration][mel_dim]; первые cond_frames кадров заменяются на cond_mel, как в F5.
    // progress(step, nfe) вызывается после каждого шага.
    void sample(const std::vector<float> &cond_mel, int cond_frames, const std::vector<int32_t> &ids, int duration, const F5Sampler &s,
                const float *noise, std::vector<float> &out, const std::function<void(int, int)> &progress = nullptr) const;

    // Vocos: mel [frames][mel_dim] → звук (frames * hop отсчётов).
    void vocode(const float *mel, int frames, std::vector<float> &pcm) const;

    // Размер весов DiT в байтах (для планировщика и статистики).
    uint64_t weight_bytes() const { return weight_bytes_; }
    int gpu_mats() const { return gpu_mats_; }
    int linear_mats() const { return linear_mats_; }
    double ms_dit() const { return ms_dit_; }
    double ms_vocos() const { return ms_vocos_; }

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    std::shared_ptr<TmodFile> file_;
    F5Cfg cfg_;
    F5Tokenizer tok_;
    std::vector<float> mel_fb_;
    uint64_t weight_bytes_ = 0;
    int gpu_mats_ = 0, linear_mats_ = 0;
    mutable double ms_dit_ = 0, ms_vocos_ = 0;
};

}  // namespace vox
