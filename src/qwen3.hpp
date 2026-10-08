#pragma once
// Qwen3-TTS: конфигурация, веса (взгляды на mmap), стриминговый декодер кодека.
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "formats.hpp"
#include "kernels.hpp"
#include "resources.hpp"

namespace vox {

struct TalkerCfg {
    int hidden = 0, inter = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, vocab = 0;
    int text_hidden = 0, text_vocab = 0, groups = 0;
    float eps = 1e-6f;
    double theta = 1e6;
    int eos = 0, think = 0, nothink = 0, think_bos = 0, think_eos = 0, pad = 0, bos = 0;
    std::map<std::string, int> lang;
    std::map<std::string, int> spk_id;                  // готовые дикторы (CustomVoice): имя в нижнем регистре → id в codec_embedding
    std::map<std::string, std::string> spk_dialect;     // диктор → язык-диалект, подменяющий chinese/auto
};
struct PredCfg {
    int hidden = 0, inter = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, vocab = 0;
    float eps = 1e-6f;
    double theta = 1e6;
};
struct CodecCfg {
    int codebook_size = 0, hidden = 0, latent = 0, heads = 0, kv_heads = 0, head_dim = 0, window = 0, inter = 0;
    int layers = 0, quantizers = 0, decoder_dim = 0, codebook_dim = 0;
    float layer_scale = 0.01f, eps = 1e-5f;
    double theta = 10000.0;
    std::vector<int> up_rates, up_ratios;
    int samples_per_frame() const {
        int p = 1;
        for (int r : up_rates) p *= r;
        for (int r : up_ratios) p *= r;
        return p;
    }
};
struct SpecialIds { int im_start = 0, im_end = 0, tts_pad = 0, tts_bos = 0, tts_eos = 0; };

// Стриминговые свёртки/состояния декодера кодека
struct CConv { Mat w; int K = 1, cin = 0, cout = 0, dil = 1; };               // w: [cout][K*cin]
struct CDw { const float *w = nullptr, *b = nullptr; int K = 7, C = 0; };     // depthwise: w [C][K]
struct CTrans { Mat w; int K = 0, stride = 1, cin = 0, cout = 0; };           // w: [K*cout][cin], bias в w.bias? нет — отдельно
struct Snake { std::vector<float> a, inv_b; };                                  // exp(alpha), 1/(exp(beta)+1e-9)

// Состояние декодера кодека в видеопамяти (истории свёрток, хвосты, кольцевой KV); освобождается деструктором.
struct GpuCodecStateHandle {
    int id = -1;
    explicit GpuCodecStateHandle(int i) : id(i) {}
    ~GpuCodecStateHandle();
    GpuCodecStateHandle(const GpuCodecStateHandle &) = delete;
    GpuCodecStateHandle &operator=(const GpuCodecStateHandle &) = delete;
};

struct CodecState {
    std::shared_ptr<GpuCodecStateHandle> gpu;  // != null → декодер считается на GPU (копия CodecState делит это состояние!)
    std::vector<float> pre_conv_h, conv0_h, final_h;
    KVCache kv;
    std::vector<std::vector<float>> up_dw_h;
    struct Blk { std::vector<float> tail; std::vector<float> ru_h[3]; };
    std::vector<Blk> blk;
    int64_t frames = 0;
};

class Qwen3Weights {
public:
    explicit Qwen3Weights(std::shared_ptr<TmodFile> f);

    std::shared_ptr<TmodFile> file;
    TalkerCfg tc;
    PredCfg pc;
    CodecCfg cc;
    SpecialIds ids;

    // talker
    TransformerW talker;
    Mat codec_emb, text_emb, codec_head, tp1, tp2;
    // code predictor
    TransformerW pred;
    std::vector<Mat> pred_emb, pred_head;
    Mat small_proj;
    bool has_small = false;

    // Переносит в видеопамять стадии, которые план размещает на GPU (после загрузки весов). Страницы mmap с уже
    // загруженными весами возвращаются системе (plan.release_host_weights).
    void upload_gpu(const ResourcePlan &plan);

    // декодер кодека
    CodecState new_codec_state() const;
    // Глубокая копия (для кэша прогрева): состояние на GPU тоже копируется.
    CodecState clone_codec_state(const CodecState &s) const;
    int codec_gpu = -1;  // декодер кодека на GPU (gpu_codec_create), -1 = на CPU
    // codes[F][Q] (int16) → PCM (добавляется в pcm_out), состояние продвигается
    void codec_forward(CodecState &st, const int16_t *codes, int F, std::vector<float> &pcm_out) const;

    void load_codec();

    // ---- веса декодера кодека (читаются и GPU-бэкендом, src/gpu_codec.cpp) ----

    // квантизатор: по таблице на кодбук (уже разделённой на usage), затем output_proj
    std::vector<Mat> q_first, q_rest;  // [dim] эмбеддинги [codebook_size][dim]
    Mat q_first_out, q_rest_out;       // [codebook_dim][dim]
    CConv pre_conv;
    Mat pt_in, pt_out;
    TransformerW pt;
    struct UpStage { CTrans tr; const float *tr_b = nullptr; CDw dw; const float *ln_w = nullptr, *ln_b = nullptr, *gamma = nullptr; Mat pw1, pw2; };
    std::vector<UpStage> up;
    CConv conv0;
    struct DecBlock { Snake act; CTrans tr; const float *tr_b = nullptr; Snake a1[3], a2[3]; CConv c1[3], c2[3]; };
    std::vector<DecBlock> blocks;
    Snake final_act;
    CConv final_conv;
};

}  // namespace vox
