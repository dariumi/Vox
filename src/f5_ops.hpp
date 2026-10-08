#pragma once
// Операции, нужные моделям на потоках вещественных признаков (F5-TTS): БПФ, mel-спектрограмма как у torchaudio,
// свёртки по времени, обратное STFT как у Vocos. Всё считается на CPU в float32 (БПФ — в double).
#include <cstdint>
#include <vector>

#include "kernels.hpp"

namespace vox {

struct MelCfg {
    int sample_rate = 24000, n_fft = 1024, hop = 256, win = 1024, n_mels = 100;
};

// mel-спектрограмма как torchaudio.transforms.MelSpectrogram(power=1, center=True, pad_mode="reflect", norm=None) с
// последующим log(clamp(·, 1e-5)). fb — банк фильтров [n_mels][n_fft/2+1] (строится конвертером, чтобы совпадать с эталоном).
// out: [frames][n_mels], frames = 1 + n / hop.
void mel_log_spectrogram(const MelCfg &cfg, const float *fb, const float *pcm, size_t n, std::vector<float> &out, int &frames);

// Свёртка по времени с "same"-дополнением нулями (нечётное K), группами. x: [T][cin], y: [T][cout].
// w: PyTorch Conv1d.weight [cout][cin/groups][K] (любой тип весов, приводится к float при загрузке).
struct Conv1d {
    int cin = 0, cout = 0, K = 1, groups = 1;
    std::vector<float> w;   // [groups][cout/groups][K * cin/groups], внутри K-мажорно: [k][i]
    std::vector<float> b;   // [cout] или пусто
    // src — данные PyTorch-раскладки [cout][cin/groups][K] в float
    void load(const float *src, int cin_, int cout_, int K_, int groups_, const float *bias);
    void forward(const float *x, int T, float *y) const;
    bool depthwise() const { return groups == cin && cin == cout; }
};

// Обратное STFT в стиле Vocos (padding="same"): mag, phase — [T][n_fft/2+1]; выход (T * hop) отсчётов.
void istft_same(const float *mag, const float *phase, int T, int n_fft, int hop, std::vector<float> &out);

// Вспомогательное: приведение любой матрицы к float (rows * cols).
void mat_to_float(const Mat &W, std::vector<float> &out);

}  // namespace vox
