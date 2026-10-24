#pragma once
// Chatterbox (Resemble AI, English): T3 — a 0.5B Llama over S3 speech tokens conditioned on a speaker embedding, a perceiver
// summary of prompt speech tokens and an "exaggeration" scalar, sampled with classifier-free guidance — then S3Gen, the same
// flow matching + HiFT as CosyVoice2 (cosyvoice.hpp, run without its LLM). The voice encoder for T3 is a 3-layer LSTM over
// 40-band mels (VoiceEncoder). Mirrors chatterbox.tts.ChatterboxTTS.generate (tests/test_chatterbox_tiny.py).
#include <memory>
#include <string>
#include <vector>

#include "formats.hpp"
#include "hf_tokenizer.hpp"
#include "kernels.hpp"

namespace vox {

struct T3Cfg {
    int hidden = 1024, layers = 30, heads = 16, kv_heads = 16, head_dim = 64, inter = 4096;
    float eps = 1e-5f;
    double theta = 500000.0;
    double rope_factor = 8.0, low_freq = 1.0, high_freq = 4.0, orig_ctx = 8192.0;   // rope_type llama3
    int text_vocab = 704, speech_vocab = 8194, start_text = 255, stop_text = 0, start_speech = 6561, stop_speech = 6562;
    int max_text = 2050, max_speech = 4100, prompt_len = 150, speaker_dim = 256, perceiver_heads = 4;
};

struct T3Sampling {
    float temperature = 0.8f, cfg_weight = 0.5f, repetition_penalty = 1.2f, min_p = 0.05f, top_p = 1.0f;
    int max_new_tokens = 1000;   // ChatterboxTTS.generate: 1000 (40 s of speech per piece)
};

class ChatterboxT3 {
public:
    explicit ChatterboxT3(std::shared_ptr<TmodFile> f);
    ~ChatterboxT3();
    const T3Cfg &cfg() const { return cfg_; }

    // text → token ids with [START] … [STOP] (punc_norm, spaces → [SPACE], tokenizer.json)
    std::vector<int32_t> text_tokens(const std::string &text) const;
    static std::string punc_norm(const std::string &text);

    // Conditioning rows: speaker embedding [256], prompt speech tokens (≤ 150), exaggeration → [rows][hidden]
    std::vector<float> cond_embeds(const std::vector<float> &spk, const std::vector<int32_t> &prompt_tokens, float exaggeration) const;

    struct Trace { std::vector<float> cond, logits; };   // logits: CFG-combined, before sampling, [steps][speech_vocab]
    // Speech tokens until [STOP_SPEECH] (excluded) or max_new_tokens; forced: the tokens to feed instead of sampling (tests)
    std::vector<int32_t> generate(const std::vector<float> &cond, const std::vector<int32_t> &text, const T3Sampling &sp, uint64_t seed,
                                  uint32_t piece, const std::vector<int32_t> *forced = nullptr, Trace *tr = nullptr) const;

    // VoiceEncoder: 16 kHz audio → 256-d speaker embedding (trim at 20 dB, 40-band mels, LSTM over 160-frame partials)
    std::vector<float> voice_embedding(const float *pcm16, size_t n) const;
    bool has_voice_encoder() const;

    double ms_t3 = 0;
    mutable double ms_gen = 0;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    std::shared_ptr<TmodFile> file_;
    T3Cfg cfg_;
    std::unique_ptr<HfTokenizer> tok_;
};

}  // namespace vox
