#pragma once
// CosyVoice2 (FunAudioLLM/CosyVoice2-0.5B), all on the CPU:
//   text (Qwen2 BPE) → Qwen2 LM over speech tokens (25 Hz, 6561 FSQ codes; RAS sampling) → flow: token embedding,
//   upsampling Conformer encoder (relative positions, ×2 to 50 Hz), conditional flow matching (U-Net estimator with
//   transformer blocks, 10 Euler steps, classifier-free guidance 0.7, fixed noise) → 80-band mel → HiFT vocoder
//   (F0 predictor, harmonic NSF source, transposed convolutions with Snake residual blocks, iSTFT n_fft 16) → 24 kHz.
// Voice cloning frontend: 24 kHz mel of the recording (flow prompt), CAM++ speaker x-vector from a Kaldi fbank (16 kHz),
// S3 speech tokens (S3TokenizerV2: Whisper log-mel, conv + FSMN attention encoder, FSQ) of the recording.
// Mirrors cosyvoice.cli.model.CosyVoice2Model operation by operation (tests/test_cosyvoice_tiny.py).
#include <memory>
#include <string>
#include <vector>

#include "formats.hpp"
#include "kernels.hpp"

namespace vox {

struct CosyCfg {
    // LLM
    int llm_hidden = 896, llm_layers = 24, llm_heads = 14, llm_kv_heads = 2, llm_head_dim = 64, llm_inter = 4864;
    float llm_eps = 1e-6f;
    double llm_theta = 1e6;
    int speech_tokens = 6561;               // FSQ codebook; +0 eos, +1, +2 are the stop tokens
    float top_p = 0.8f;
    int top_k = 25, win_size = 10;
    float tau_r = 0.1f;
    float min_ratio = 2.0f, max_ratio = 20.0f;   // generated tokens per text token
    // flow
    int enc_dim = 512, enc_heads = 8, enc_units = 2048, enc_blocks = 6, enc_up_blocks = 4, pre_lookahead = 3, token_mel_ratio = 2;
    int mel_dim = 80, spk_dim = 192;
    int est_ch = 256, est_heads = 8, est_head_dim = 64, est_blocks = 4, est_mid = 12;
    int n_timesteps = 10;
    float cfg_rate = 0.7f;
    bool cosine = true;
    // HiFT
    int hift_ch = 512, nb_harmonics = 8, n_fft = 16, hop = 4;
    std::vector<int> up_rates{8, 5, 3}, up_kernels{16, 11, 7}, rb_kernels{3, 7, 11}, src_rb_kernels{7, 7, 11};
    std::vector<std::vector<int>> rb_dils{{1, 3, 5}, {1, 3, 5}, {1, 3, 5}}, src_rb_dils{{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    float lrelu = 0.1f, audio_limit = 0.99f;
    uint32_t sample_rate = 24000;
    // frontend
    int mel_n_fft = 1920, mel_hop = 480;
    bool has_campplus = false, has_s3 = false;
    bool campplus_pool_window = false;  // last incomplete context segment: sum / 100 (campplus.onnx) instead of the mean (PyTorch)
    // variants used by Chatterbox's S3Gen (same networks): no LLM in the file, Gaussian noise drawn per utterance instead of the
    // fixed table, Euler steps dt = t[i+1] − t[i], the older SineGen
    bool has_llm = true, fixed_noise = true, dt_diff = false;
    int sinegen = 2;
    int s3_dim = 1280, s3_heads = 20, s3_layers = 6, s3_mels = 128;
};

// Source noise of HiFT (SineGen2): initial phases of the 8 overtones and Gaussian noise [samples][9];
// from the session seed (Philox) or injected by tests.
struct CosyNoise {
    std::vector<float> rand_ini, sine_noise;
};

class CosyVoiceModel {
public:
    explicit CosyVoiceModel(std::shared_ptr<TmodFile> f);
    ~CosyVoiceModel();
    CosyVoiceModel(const CosyVoiceModel &) = delete;
    CosyVoiceModel &operator=(const CosyVoiceModel &) = delete;
    const CosyCfg &cfg() const { return cfg_; }

    // ---- LLM ----
    struct LmState {
        KVCache kv;             // kv.gpu_seq >= 0: the KV cache is in video memory (enable_gpu_llm)
        std::vector<float> h;   // normed hidden state of the last position
        int64_t text_len = 0;
        ~LmState();
    };
    // lm_input = [sos, embed(prompt_text + text), task_id, speech_embedding(prompt_tokens)]; returns the state after the prompt
    std::unique_ptr<LmState> lm_begin(const std::vector<int32_t> &prompt_text, const std::vector<int32_t> &text,
                                      const std::vector<int32_t> &prompt_tokens) const;
    // log-softmax over the speech vocabulary (6561 + 3) for the next token
    void lm_logp(LmState &s, std::vector<float> &logp) const;
    void lm_accept(LmState &s, int32_t token) const;
    // RAS sampling of CosyVoice (nucleus top-p / top-k, then a resample from the full distribution when the token repeats
    // inside the window); u0, u1 uniform in [0, 1). decoded: tokens so far.
    int32_t ras_sample(std::vector<float> &logp, const std::vector<int32_t> &decoded, float u0, float u1) const;
    // Whole generation: min / max lengths from the text length, eos ignored before min_len (resampled, as sampling_ids does).
    std::vector<int32_t> lm_generate(const std::vector<int32_t> &prompt_text, const std::vector<int32_t> &text,
                                     const std::vector<int32_t> &prompt_tokens, uint64_t seed, uint32_t piece) const;
    // the Qwen2 token-embedding table width (for checks)
    int text_vocab() const;

    // ---- flow ----
    struct FlowTrace { std::vector<float> spk, mu, mel; };
    // tokens = prompt tokens + new tokens; prompt_feat [Tp][80]; xvec [192] → mel of the new tokens [T][80]
    // seed / piece: the starting noise when the model has no fixed table; z0 [80][T] (prompt + new frames) overrides it (tests)
    void flow(const std::vector<int32_t> &prompt_tokens, const std::vector<int32_t> &tokens, const std::vector<float> &prompt_feat,
              const std::vector<float> &xvec, std::vector<float> &mel, FlowTrace *tr = nullptr, uint64_t seed = 1, uint32_t piece = 0,
              const std::vector<float> *z0 = nullptr) const;

    // ---- HiFT ----
    struct HiftTrace { std::vector<float> f0, source; };
    void hift(const std::vector<float> &mel, int T, uint64_t seed, uint32_t piece, const CosyNoise *inject, std::vector<float> &pcm,
              HiftTrace *tr = nullptr) const;
    static void make_noise(uint64_t seed, uint32_t piece, size_t samples, CosyNoise &n);

    // ---- frontend (recordings) ----
    std::vector<float> mel24(const float *pcm24, size_t n, int &frames) const;            // [frames][80]
    std::vector<float> fbank16(const float *pcm16, size_t n, int &frames) const;          // Kaldi fbank [frames][80], mean removed
    std::vector<float> xvector(const std::vector<float> &fbank, int frames) const;        // CAM++ [192]
    std::vector<float> whisper_mel(const float *pcm16, size_t n, int &frames) const;      // [128][frames]
    std::vector<int32_t> speech_tokens(const std::vector<float> &wmel, int frames) const; // S3TokenizerV2
    bool can_clone() const { return cfg_.has_campplus && cfg_.has_s3; }

    // The flow estimator on the GPU (gpu_active() first): its weights go to video memory and every Euler step is one
    // submission. false (reason on stderr) → it stays on the CPU. The encoder and HiFT stay on the CPU.
    bool enable_gpu();
    bool gpu_flow() const;
    // The Qwen2 LM on the GPU (the transformer of Qwen3-TTS's GPU path, KV cache in video memory, ≤ 3072 positions;
    // a longer piece runs on the CPU). false → the CPU.
    bool enable_gpu_llm();
    bool gpu_llm() const;

    uint64_t weight_bytes() const { return weight_bytes_; }
    mutable double ms_llm = 0, ms_flow = 0, ms_hift = 0;

private:
    struct Impl;
    struct Front;
    struct FrontFree { void operator()(Front *) const; };   // defined with Front (cosyvoice_front.cpp)
    void load_front();
    std::unique_ptr<Impl> p_;
    std::unique_ptr<Front, FrontFree> fp_;
    std::shared_ptr<TmodFile> file_;
    CosyCfg cfg_;
    uint64_t weight_bytes_ = 0;
};

}  // namespace vox
