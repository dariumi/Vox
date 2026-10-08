#pragma once
// Native voice creation for Qwen3-TTS-12Hz: speaker embedding (mel + ECAPA-TDNN) and reference codes
// (Mimi encoder + split RVQ), matching qwen_tts `create_voice_clone_prompt` without torch.
#include <memory>
#include <string>
#include <vector>

#include "formats.hpp"
#include "kernels.hpp"

namespace vox {

// Windowed-sinc (Kaiser) resampler. For sr_from == sr_to returns a copy.
std::vector<float> resample_sinc(const float *x, size_t n, uint32_t sr_from, uint32_t sr_to);

class Qwen3VoiceEncoder {
public:
    explicit Qwen3VoiceEncoder(std::shared_ptr<TmodFile> f);
    ~Qwen3VoiceEncoder();
    bool has_speaker_encoder() const;
    bool has_codec_encoder() const;
    // pcm at 24 kHz mono. Embedding size = enc_dim.
    std::vector<float> speaker_embedding(const float *pcm, size_t n) const;
    // Log-mel spectrogram used by the speaker encoder, [frames][128] (time-major).
    std::vector<float> mel(const float *pcm, size_t n, int64_t &frames) const;
    // Reference codes [frames][valid_quantizers] (int16), frames = ceil(n / 1920).
    std::vector<int16_t> encode(const float *pcm, size_t n, int64_t &frames, int &codebooks) const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace vox
