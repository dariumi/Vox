#pragma once
// Kokoro (StyleTTS2 + iSTFTNet, hexgrad/Kokoro-82M): phonemes + style vector → 24 kHz audio, all on the CPU.
//   PL-BERT (ALBERT, shared layers) → prosody predictor (BiLSTM duration encoder with AdaLayerNorm, durations,
//   F0 / energy with AdaIN residual blocks) → text encoder (CNN + BiLSTM) → alignment by durations → decoder
//   (AdaIN residual blocks) → iSTFTNet generator (harmonic NSF source, transposed convolutions, Snake residual blocks,
//   inverse STFT with n_fft 20). Mirrors kokoro.model.KModel.forward operation by operation (tests/test_kokoro_tiny.py).
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "formats.hpp"
#include "kernels.hpp"

namespace vox {

struct KokoroCfg {
    int n_token = 178, hidden = 512, style_dim = 128, n_layer = 3, max_dur = 50, text_kernel = 5, context = 512;
    int bert_hidden = 768, bert_heads = 12, bert_inter = 2048, bert_layers = 12, bert_emb = 128, bert_max_pos = 512;
    std::vector<int> up_rates{10, 6}, up_kernels{20, 12}, rb_kernels{3, 7, 11};
    std::vector<std::vector<int>> rb_dils{{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    int up_initial = 512, n_fft = 20, hop = 5;
    uint32_t sample_rate = 24000;
};

// Source noise of the harmonic-plus-noise excitation: initial phases of the 8 overtones (the fundamental has none)
// and Gaussian noise [samples][9]. Generated from the session seed (Philox) or injected by tests.
struct KokoroNoise {
    std::vector<float> rand_ini;    // [9]
    std::vector<float> sine_noise;  // [L][9]
    std::vector<float> har;         // tests: the source spectrogram itself [frames][n_fft + 2] (bypasses the source module)
};

class KokoroModel {
public:
    explicit KokoroModel(std::shared_ptr<TmodFile> f);
    ~KokoroModel();
    KokoroModel(const KokoroModel &) = delete;
    KokoroModel &operator=(const KokoroModel &) = delete;

    const KokoroCfg &cfg() const { return cfg_; }
    // phoneme string (UTF-8) → token ids (unknown symbols are dropped, as in KModel.forward)
    std::vector<int32_t> phonemes_to_ids(const std::string &ps) const;
    size_t phoneme_count(const std::string &ps) const;   // number of code points (what the voice pack is indexed with)
    const std::vector<std::string> &voices() const { return voices_; }
    // style vector [256] = pack[n_phonemes - 1] of a named voice; "a,b,c" averages the packs (KPipeline.load_voice)
    void style(const std::string &voice, size_t n_phonemes, std::vector<float> &ref_s) const;
    bool has_voice(const std::string &name) const;

    struct Trace {   // intermediate tensors for `vox verify`
        std::vector<float> bert, d_en, d, duration, F0, N, t_en, asr, dec, har_src, har, post;
        std::vector<int32_t> pred_dur;
    };
    // ids (without the 0 padding tokens; they are added here), style [256], speed; noise for the source (sized here if empty:
    // filled from seed/chunk). Returns PCM. trace may be null.
    void synthesize(const std::vector<int32_t> &ids, const float *ref_s, float speed, uint64_t seed, uint32_t chunk,
                    const KokoroNoise *inject, std::vector<float> &pcm, Trace *trace = nullptr) const;
    // decoder + iSTFTNet generator alone: asr [F][hidden], F0 / N [2F] (what the prosody predictor gives) → PCM
    void decode(const std::vector<float> &asr, int F, const std::vector<float> &F0, const std::vector<float> &N, const float *ref_s,
                uint64_t seed, uint32_t chunk, const KokoroNoise *inject, std::vector<float> &pcm, Trace *trace = nullptr) const;
    // just the source module and generator inputs, for tests
    static void make_noise(uint64_t seed, uint32_t chunk, size_t samples, KokoroNoise &n);

    // the embedded English lexicons (blob g2p.<name>), JSON text; empty if absent
    std::string lexicon(const std::string &name) const;

    double ms_prosody() const { return ms_prosody_; }
    double ms_decoder() const { return ms_decoder_; }
    uint64_t weight_bytes() const { return weight_bytes_; }

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    std::shared_ptr<TmodFile> file_;
    KokoroCfg cfg_;
    std::map<std::string, int32_t> vocab_;
    std::vector<std::string> voices_;
    uint64_t weight_bytes_ = 0;
    mutable double ms_prosody_ = 0, ms_decoder_ = 0;
};

}  // namespace vox
