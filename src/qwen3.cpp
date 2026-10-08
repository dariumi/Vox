#include "qwen3.hpp"

#include "gpu.hpp"

#include <algorithm>
#include <cstdlib>

namespace vox {
namespace {

// ---------- загрузка тензоров ----------
const float *vec_f32(const TmodFile &f, const std::string &name) {
    const TensorView &t = f.get(name);
    if (t.dtype != DType::F32) fail(VOX_ERR_FORMAT, "tensor must be F32: " + name);
    return reinterpret_cast<const float *>(t.data);
}

Mat get_mat(const TmodFile &f, const std::string &name, bool with_bias = false) {
    const TensorView &t = f.get(name);
    Mat m;
    m.data = t.data;
    m.rows = t.shape.at(0);
    m.cols = t.shape.size() > 1 ? t.shape[1] : 1;
    switch (t.dtype) {
        case DType::F32: m.type = WType::F32; break;
        case DType::F16: m.type = WType::F16; break;
        case DType::BF16: m.type = WType::BF16; break;
        case DType::I8:
            m.type = WType::Q8;
            m.scale = vec_f32(f, name + ".scale");
            break;
        case DType::U8:  // Q4: [rows][cols/2] байт (см. kernels.hpp) + масштабы
            m.type = WType::Q4;
            m.cols *= 2;
            m.scale = vec_f32(f, name + ".scale");
            break;
        default: fail(VOX_ERR_FORMAT, "unsupported weight dtype for " + name);
    }
    if (with_bias) {
        std::string b = name.substr(0, name.size() - 6) + "bias";  // ".weight" → "bias"
        if (f.has(b)) m.bias = vec_f32(f, b);
    }
    return m;
}

void load_transformer(const TmodFile &f, const std::string &prefix, TransformerW &w, int layers, int hidden, int heads,
                      int kv_heads, int head_dim, int inter, float eps, double theta, bool qk_norm, bool layer_scale) {
    w.hidden = hidden; w.heads = heads; w.kv_heads = kv_heads; w.head_dim = head_dim; w.inter = inter;
    w.eps = eps; w.theta = theta;
    w.inv_freq = rope_inv_freq(head_dim, theta);
    w.layers.resize(size_t(layers));
    for (int i = 0; i < layers; ++i) {
        const std::string p = prefix + "layers." + std::to_string(i) + ".";
        LayerW &L = w.layers[size_t(i)];
        L.q = get_mat(f, p + "self_attn.q_proj.weight", true);
        L.k = get_mat(f, p + "self_attn.k_proj.weight", true);
        L.v = get_mat(f, p + "self_attn.v_proj.weight", true);
        L.o = get_mat(f, p + "self_attn.o_proj.weight", true);
        L.gate = get_mat(f, p + "mlp.gate_proj.weight");
        L.up = get_mat(f, p + "mlp.up_proj.weight");
        L.down = get_mat(f, p + "mlp.down_proj.weight");
        L.ln1 = vec_f32(f, p + "input_layernorm.weight");
        L.ln2 = vec_f32(f, p + "post_attention_layernorm.weight");
        if (qk_norm) {
            L.qn = vec_f32(f, p + "self_attn.q_norm.weight");
            L.kn = vec_f32(f, p + "self_attn.k_norm.weight");
        }
        if (layer_scale) {
            L.attn_scale = vec_f32(f, p + "self_attn_layer_scale.scale");
            L.mlp_scale = vec_f32(f, p + "mlp_layer_scale.scale");
        }
    }
    w.final_norm = vec_f32(f, prefix + "norm.weight");
}

Snake load_snake(const TmodFile &f, const std::string &p) {
    const TensorView &a = f.get(p + ".alpha"), &b = f.get(p + ".beta");
    const float *pa = reinterpret_cast<const float *>(a.data), *pb = reinterpret_cast<const float *>(b.data);
    Snake s;
    for (int64_t i = 0; i < a.numel(); ++i) {
        s.a.push_back(std::exp(pa[i]));
        s.inv_b.push_back(1.0f / (std::exp(pb[i]) + 1e-9f));
    }
    return s;
}
void snake_apply(const Snake &s, float *x, int T, int C) {
    parallel_for(T, 16, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t)
            for (int c = 0; c < C; ++c) {
                float v = x[size_t(t) * size_t(C) + size_t(c)], sn = std::sin(v * s.a[size_t(c)]);
                x[size_t(t) * size_t(C) + size_t(c)] = v + s.inv_b[size_t(c)] * sn * sn;
            }
    });
}

CConv load_conv(const TmodFile &f, const std::string &p, int K, int cin, int cout, int dil = 1) {
    CConv c;
    c.w = get_mat(f, p + ".weight", true);
    c.K = K; c.cin = cin; c.cout = cout; c.dil = dil;
    if (c.w.rows != cout || c.w.cols != int64_t(K) * cin) fail(VOX_ERR_FORMAT, "conv shape mismatch: " + p);
    return c;
}

// ---------- стриминговые операции ----------
void conv_stream(const CConv &c, std::vector<float> &hist, const float *x, int T, float *y) {
    const int ctx = (c.K - 1) * c.dil;
    if (c.K == 1) { matmul(c.w, x, T, y); return; }
    if (int(hist.size()) != ctx * c.cin) hist.assign(size_t(ctx) * size_t(c.cin), 0.f);
    std::vector<float> ext(size_t(ctx + T) * size_t(c.cin));
    std::memcpy(ext.data(), hist.data(), size_t(ctx) * size_t(c.cin) * 4);
    std::memcpy(ext.data() + size_t(ctx) * size_t(c.cin), x, size_t(T) * size_t(c.cin) * 4);
    std::vector<float> col(size_t(T) * size_t(c.K) * size_t(c.cin));
    parallel_for(T, 64, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t)
            for (int k = 0; k < c.K; ++k)
                std::memcpy(&col[(size_t(t) * size_t(c.K) + size_t(k)) * size_t(c.cin)],
                            &ext[size_t(t + k * c.dil) * size_t(c.cin)], size_t(c.cin) * 4);
    });
    matmul(c.w, col.data(), T, y);
    std::memcpy(hist.data(), ext.data() + size_t(T) * size_t(c.cin), size_t(ctx) * size_t(c.cin) * 4);
}

void dwconv_stream(const CDw &c, std::vector<float> &hist, const float *x, int T, float *y) {
    const int ctx = c.K - 1, C = c.C;
    if (int(hist.size()) != ctx * C) hist.assign(size_t(ctx) * size_t(C), 0.f);
    std::vector<float> ext(size_t(ctx + T) * size_t(C));
    std::memcpy(ext.data(), hist.data(), size_t(ctx) * size_t(C) * 4);
    std::memcpy(ext.data() + size_t(ctx) * size_t(C), x, size_t(T) * size_t(C) * 4);
    parallel_for(T, 16, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t)
            for (int ch = 0; ch < C; ++ch) {
                float acc = c.b[ch];
                for (int k = 0; k < c.K; ++k) acc += c.w[size_t(ch) * size_t(c.K) + size_t(k)] * ext[size_t(t + k) * size_t(C) + size_t(ch)];
                y[size_t(t) * size_t(C) + size_t(ch)] = acc;
            }
    });
    std::memcpy(hist.data(), ext.data() + size_t(T) * size_t(C), size_t(ctx) * size_t(C) * 4);
}

// Транспонированная свёртка с обрезкой правого края (K - stride) и переносом хвоста между чанками.
// Выход p собирается «на вытягивание» из кадров t = p/stride и t-1 (K <= 2*stride), без гонок записи.
void trans_stream(const CTrans &c, const float *bias, std::vector<float> &tail, const float *x, int T, float *y) {
    const int K = c.K, s = c.stride, co = c.cout, tr = K - s;
    std::vector<float> tmp(size_t(T) * size_t(K) * size_t(co));
    matmul(c.w, x, T, tmp.data());  // [T][K*co], строка (k*co + o)
    if (int(tail.size()) != tr * co) tail.assign(size_t(tr) * size_t(co), 0.f);
    std::vector<float> new_tail(size_t(tr) * size_t(co));
    parallel_for(int64_t(T) * s + tr, 32, [&](int64_t p0, int64_t p1) {
        for (int64_t p = p0; p < p1; ++p) {
            const int64_t t = p / s, k = p - t * s;
            for (int o = 0; o < co; ++o) {
                float v = p < tr ? tail[size_t(p) * size_t(co) + size_t(o)] : 0.f;
                if (t < T && k < K) v += tmp[(size_t(t) * size_t(K) + size_t(k)) * size_t(co) + size_t(o)];
                if (t >= 1 && k + s < K) v += tmp[(size_t(t - 1) * size_t(K) + size_t(k + s)) * size_t(co) + size_t(o)];
                if (p < int64_t(T) * s) y[size_t(p) * size_t(co) + size_t(o)] = v + bias[o];
                else new_tail[size_t(p - int64_t(T) * s) * size_t(co) + size_t(o)] = v;
            }
        }
    });
    tail.swap(new_tail);
}

}  // namespace

// ====================== Qwen3Weights ======================
Qwen3Weights::Qwen3Weights(std::shared_ptr<TmodFile> f) : file(std::move(f)) {
    const json &q = file->manifest().at("qwen3");
    const json &t = q.at("talker"), &p = q.at("predictor"), &c = q.at("codec"), &i = q.at("ids");

    tc.hidden = t.at("hidden_size"); tc.inter = t.at("intermediate_size"); tc.layers = t.at("num_hidden_layers");
    tc.heads = t.at("num_attention_heads"); tc.kv_heads = t.at("num_key_value_heads");
    tc.head_dim = t.value("head_dim", tc.hidden / tc.heads);
    tc.vocab = t.at("vocab_size"); tc.text_hidden = t.at("text_hidden_size"); tc.text_vocab = t.at("text_vocab_size");
    tc.groups = t.at("num_code_groups"); tc.eps = t.value("rms_norm_eps", 1e-6f); tc.theta = t.value("rope_theta", 1e6);
    tc.eos = t.at("codec_eos_token_id"); tc.think = t.at("codec_think_id"); tc.nothink = t.at("codec_nothink_id");
    tc.think_bos = t.at("codec_think_bos_id"); tc.think_eos = t.at("codec_think_eos_id");
    tc.pad = t.at("codec_pad_id"); tc.bos = t.at("codec_bos_id");
    if (t.contains("codec_language_id") && t["codec_language_id"].is_object())
        for (auto it = t["codec_language_id"].begin(); it != t["codec_language_id"].end(); ++it) tc.lang[it.key()] = it.value();
    if (t.contains("spk_id") && t["spk_id"].is_object())
        for (auto it = t["spk_id"].begin(); it != t["spk_id"].end(); ++it) {
            std::string k = it.key();
            for (auto &ch : k) ch = char(std::tolower(static_cast<unsigned char>(ch)));
            // в конфигурации id бывает числом или списком чисел (первый — id в codec_embedding)
            tc.spk_id[k] = it.value().is_array() ? int(it.value().at(0)) : int(it.value());
        }
    if (t.contains("spk_is_dialect") && t["spk_is_dialect"].is_object())
        for (auto it = t["spk_is_dialect"].begin(); it != t["spk_is_dialect"].end(); ++it) {
            std::string k = it.key();
            for (auto &ch : k) ch = char(std::tolower(static_cast<unsigned char>(ch)));
            if (it.value().is_string()) tc.spk_dialect[k] = it.value();
        }

    pc.hidden = p.at("hidden_size"); pc.inter = p.at("intermediate_size"); pc.layers = p.at("num_hidden_layers");
    pc.heads = p.at("num_attention_heads"); pc.kv_heads = p.at("num_key_value_heads");
    pc.head_dim = p.value("head_dim", pc.hidden / pc.heads);
    pc.vocab = p.at("vocab_size"); pc.eps = p.value("rms_norm_eps", 1e-6f); pc.theta = p.value("rope_theta", 1e6);

    ids.im_start = i.at("im_start"); ids.im_end = i.at("im_end"); ids.tts_pad = i.at("tts_pad");
    ids.tts_bos = i.at("tts_bos"); ids.tts_eos = i.at("tts_eos");

    cc.codebook_size = c.at("codebook_size"); cc.hidden = c.at("hidden_size"); cc.latent = c.at("latent_dim");
    cc.heads = c.at("num_attention_heads"); cc.kv_heads = c.at("num_key_value_heads");
    cc.head_dim = c.value("head_dim", cc.hidden / cc.heads);
    cc.window = c.at("sliding_window"); cc.inter = c.at("intermediate_size"); cc.layers = c.at("num_hidden_layers");
    cc.quantizers = c.at("num_quantizers"); cc.decoder_dim = c.at("decoder_dim"); cc.codebook_dim = c.at("codebook_dim");
    cc.layer_scale = c.value("layer_scale_initial_scale", 0.01f); cc.eps = c.value("rms_norm_eps", 1e-5f);
    cc.theta = c.value("rope_theta", 10000.0);
    cc.up_rates = c.at("upsample_rates").get<std::vector<int>>();
    cc.up_ratios = c.at("upsampling_ratios").get<std::vector<int>>();

    const TmodFile &F = *file;
    codec_emb = get_mat(F, "talker.model.codec_embedding.weight");
    text_emb = get_mat(F, "talker.model.text_embedding.weight");
    codec_head = get_mat(F, "talker.codec_head.weight");
    tp1 = get_mat(F, "talker.text_projection.linear_fc1.weight", true);
    tp2 = get_mat(F, "talker.text_projection.linear_fc2.weight", true);
    load_transformer(F, "talker.model.", talker, tc.layers, tc.hidden, tc.heads, tc.kv_heads, tc.head_dim, tc.inter,
                     tc.eps, tc.theta, true, false);

    load_transformer(F, "talker.code_predictor.model.", pred, pc.layers, pc.hidden, pc.heads, pc.kv_heads, pc.head_dim,
                     pc.inter, pc.eps, pc.theta, true, false);
    for (int g = 0; g < tc.groups - 1; ++g) {
        pred_emb.push_back(get_mat(F, "talker.code_predictor.model.codec_embedding." + std::to_string(g) + ".weight"));
        pred_head.push_back(get_mat(F, "talker.code_predictor.lm_head." + std::to_string(g) + ".weight"));
    }
    has_small = F.has("talker.code_predictor.small_to_mtp_projection.weight");
    if (has_small) small_proj = get_mat(F, "talker.code_predictor.small_to_mtp_projection.weight", true);
    else if (pc.hidden != tc.hidden) fail(VOX_ERR_FORMAT, "predictor hidden differs but small_to_mtp_projection is missing");

    load_codec();
}

void Qwen3Weights::load_codec() {
    const TmodFile &F = *file;
    const CodecCfg &c = cc;
    const int dim = c.codebook_dim / 2;
    auto table = [&](const std::string &p) { return get_mat(F, p); };
    q_first.push_back(table("codec.quantizer.rvq_first.vq.layers.0._codebook.embedding"));
    for (int i = 0; i < c.quantizers - 1; ++i)
        q_rest.push_back(table("codec.quantizer.rvq_rest.vq.layers." + std::to_string(i) + "._codebook.embedding"));
    q_first_out = get_mat(F, "codec.quantizer.rvq_first.output_proj.weight");
    q_rest_out = get_mat(F, "codec.quantizer.rvq_rest.output_proj.weight");
    if (q_first.front().cols != dim || q_first_out.cols != dim || q_first_out.rows != c.codebook_dim)
        fail(VOX_ERR_FORMAT, "codec quantizer shape mismatch");

    pre_conv = load_conv(F, "codec.pre_conv.conv", 3, c.codebook_dim, c.latent);
    pt_in = get_mat(F, "codec.pre_transformer.input_proj.weight", true);
    pt_out = get_mat(F, "codec.pre_transformer.output_proj.weight", true);
    load_transformer(F, "codec.pre_transformer.", pt, c.layers, c.hidden, c.heads, c.kv_heads, c.head_dim, c.inter,
                     c.eps, c.theta, false, true);
    pt.window = c.window;

    for (size_t s = 0; s < c.up_ratios.size(); ++s) {
        UpStage u;
        const std::string p = "codec.upsample." + std::to_string(s) + ".";
        u.tr.w = get_mat(F, p + "0.conv.weight");
        u.tr.K = u.tr.stride = c.up_ratios[s]; u.tr.cin = u.tr.cout = c.latent;
        u.tr_b = vec_f32(F, p + "0.conv.bias");
        u.dw.w = vec_f32(F, p + "1.dwconv.conv.weight"); u.dw.b = vec_f32(F, p + "1.dwconv.conv.bias");
        u.dw.K = 7; u.dw.C = c.latent;
        u.ln_w = vec_f32(F, p + "1.norm.weight"); u.ln_b = vec_f32(F, p + "1.norm.bias");
        u.gamma = vec_f32(F, p + "1.gamma");
        u.pw1 = get_mat(F, p + "1.pwconv1.weight", true);
        u.pw2 = get_mat(F, p + "1.pwconv2.weight", true);
        up.push_back(std::move(u));
    }
    conv0 = load_conv(F, "codec.decoder.0.conv", 7, c.latent, c.decoder_dim);
    for (size_t b = 0; b < c.up_rates.size(); ++b) {
        DecBlock d;
        const int in_dim = c.decoder_dim >> b, out_dim = c.decoder_dim >> (b + 1), r = c.up_rates[b];
        const std::string p = "codec.decoder." + std::to_string(b + 1) + ".block.";
        d.act = load_snake(F, p + "0");
        d.tr.w = get_mat(F, p + "1.conv.weight");
        d.tr.K = 2 * r; d.tr.stride = r; d.tr.cin = in_dim; d.tr.cout = out_dim;
        d.tr_b = vec_f32(F, p + "1.conv.bias");
        const int dil[3] = {1, 3, 9};
        for (int u = 0; u < 3; ++u) {
            const std::string q = p + std::to_string(u + 2) + ".";
            d.a1[u] = load_snake(F, q + "act1");
            d.c1[u] = load_conv(F, q + "conv1.conv", 7, out_dim, out_dim, dil[u]);
            d.a2[u] = load_snake(F, q + "act2");
            d.c2[u] = load_conv(F, q + "conv2.conv", 1, out_dim, out_dim);
        }
        blocks.push_back(std::move(d));
    }
    const int nb = int(c.up_rates.size());
    final_act = load_snake(F, "codec.decoder." + std::to_string(nb + 1));
    final_conv = load_conv(F, "codec.decoder." + std::to_string(nb + 2) + ".conv", 7, c.decoder_dim >> nb, 1);
}

namespace {
// Возвращает системе страницы mmap, из которых матрица уже скопирована в видеопамять.
void release_mat(const Mat &m) {
    if (!m.valid()) return;
    release_pages(m.data, size_t(m.rows) * mat_row_bytes(m.type, m.cols));
    if (m.scale) release_pages(m.scale, size_t(m.rows) * size_t(m.cols / 32) * 4);
    if (m.bias) release_pages(m.bias, size_t(m.rows) * 4);
}
}  // namespace

void Qwen3Weights::upload_gpu(const ResourcePlan &plan) {
    const bool rel = plan.release_host_weights;
    auto up = [&](Mat &m) {
        if (!m.valid()) return;
        m.gpu = gpu_upload(m);
        if (m.gpu >= 0 && rel) release_mat(m);
    };
    auto relm = [&](const Mat &m) { if (rel) release_mat(m); };
    auto tf = [&](TransformerW &t) {
        for (auto &L : t.layers) for (Mat *m : {&L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down}) up(*m);
    };
    // talker / predictor: gpu_xfmr_create uploads the layer weights itself, stacked (q|k|v and gate|up) — no separate copies
    if (plan.place_of("talker") == Place::Gpu) {
        const int k = plan.gpu_layers_of("talker");
        talker.gpu_id = gpu_xfmr_create(talker, "talker", k, relm);
        talker.gpu_layers = talker.gpu_id >= 0 && k >= 0 && k < int(talker.layers.size()) ? k : -1;
        if (talker.gpu_id >= 0 && talker.gpu_layers < 0) { up(codec_head); up(tp1); up(tp2); }  // голова и проекция — только при полном размещении
        else if (talker.gpu_id >= 0) up(tp1), up(tp2);
    }
    if (plan.place_of("predictor") == Place::Gpu) {
        pred.gpu_id = gpu_xfmr_create(pred, "predictor", -1, relm);
        for (auto &m : pred_head) up(m);
        for (auto &m : pred_emb) up(m);  // embedding tables for the on-GPU predictor loop
        if (has_small) up(small_proj);
    }
    if (plan.place_of("codec") != Place::Gpu) return;  // codec=cpu: веса декодера не трогаем, он целиком считается на CPU
    tf(pt);
    up(pt_in); up(pt_out);
    up(pre_conv.w); up(conv0.w); up(final_conv.w);
    for (auto &u : this->up) { up(u.tr.w); up(u.pw1); up(u.pw2); }
    for (auto &b : blocks) {
        up(b.tr.w);
        for (int i = 0; i < 3; ++i) { up(b.c1[i].w); up(b.c2[i].w); }
    }
    codec_gpu = gpu_codec_create(*this);  // весь декодер одним командным буфером на чанк; -1 → CPU
    if (codec_gpu < 0) {  // декодер остаётся на CPU целиком: без поштучных GPU-матмулов с синхронизацией
        auto off = [](Mat &m) { m.gpu = -1; };
        for (auto &L : pt.layers) for (Mat *m : {&L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down}) off(*m);
        off(pt_in); off(pt_out); off(pre_conv.w); off(conv0.w); off(final_conv.w);
        for (auto &u : this->up) { off(u.tr.w); off(u.pw1); off(u.pw2); }
        for (auto &b : blocks) { off(b.tr.w); for (int i = 0; i < 3; ++i) { off(b.c1[i].w); off(b.c2[i].w); } }
    }
}

GpuCodecStateHandle::~GpuCodecStateHandle() { if (id >= 0) gpu_codec_state_free(id); }

CodecState Qwen3Weights::new_codec_state() const {
    CodecState s;
    s.kv.init(cc.layers);
    s.up_dw_h.resize(up.size());
    s.blk.resize(blocks.size());
    if (codec_gpu >= 0) {
        const int id = gpu_codec_state_new(codec_gpu);
        if (id < 0) fail(VOX_ERR_DEVICE, "GPU codec state allocation failed");
        s.gpu = std::make_shared<GpuCodecStateHandle>(id);
    }
    return s;
}

CodecState Qwen3Weights::clone_codec_state(const CodecState &src) const {
    CodecState s = src;
    if (src.gpu) {
        const int id = gpu_codec_state_clone(src.gpu->id);
        if (id < 0) fail(VOX_ERR_DEVICE, "GPU codec state copy failed");
        s.gpu = std::make_shared<GpuCodecStateHandle>(id);
    }
    return s;
}

void Qwen3Weights::codec_forward(CodecState &st, const int16_t *codes, int F, std::vector<float> &pcm) const {
    const CodecCfg &c = cc;
    const int Q = c.quantizers, dim = c.codebook_dim / 2, T = F;
    // 1) квантизатор: сумма эмбеддингов по группам → output_proj → сложение
    std::vector<float> s1(size_t(T) * size_t(dim)), s2(size_t(T) * size_t(dim), 0.f), tmp(static_cast<size_t>(dim));
    for (int t = 0; t < T; ++t) {
        const int16_t *cd = codes + size_t(t) * size_t(Q);
        for (int q = 0; q < Q; ++q)
            if (cd[q] < 0 || cd[q] >= c.codebook_size) fail(VOX_ERR_ARG, "codec code out of range");
        mat_gather(q_first[0], cd[0], &s1[size_t(t) * size_t(dim)]);
        for (int q = 1; q < Q; ++q) {
            mat_gather(q_rest[size_t(q - 1)], cd[q], tmp.data());
            for (int i = 0; i < dim; ++i) s2[size_t(t) * size_t(dim) + size_t(i)] += tmp[size_t(i)];
        }
    }
    std::vector<float> z(size_t(T) * size_t(c.codebook_dim)), z2(size_t(T) * size_t(c.codebook_dim));
    matmul(q_first_out, s1.data(), T, z.data());
    matmul(q_rest_out, s2.data(), T, z2.data());
    for (size_t i = 0; i < z.size(); ++i) z[i] += z2[i];

    // Отладка: VOX_CODEC_DEBUG_STAGE=k возвращает в pcm активацию после стадии k (1 pre_conv, 2 трансформер,
    // 3.. апсэмплинг, затем conv0, блоки декодера, финал) — для сравнения CPU и GPU по шагам.
    static const int dbg_stage = std::getenv("VOX_CODEC_DEBUG_STAGE") ? std::atoi(std::getenv("VOX_CODEC_DEBUG_STAGE")) : 0;
    if (st.gpu) {  // всё остальное — на GPU, кусками по kGpuCodecMaxFrames кадров
        for (int f0 = 0; f0 < T; f0 += kGpuCodecMaxFrames) {
            const int n = std::min(kGpuCodecMaxFrames, T - f0);
            if (!gpu_codec_forward(codec_gpu, st.gpu->id, z.data() + size_t(f0) * size_t(c.codebook_dim), n, pcm))
                fail(VOX_ERR_DEVICE, "GPU codec decoder failed");
        }
        st.frames += F;
        return;
    }

    // 2) pre_conv → transformer
    int stage = 0;
    auto dbg = [&](const std::vector<float> &v) {
        if (++stage != dbg_stage) return false;
        pcm.insert(pcm.end(), v.begin(), v.end());
        st.frames += F;
        return true;
    };
    std::vector<float> lat(size_t(T) * size_t(c.latent));
    conv_stream(pre_conv, st.pre_conv_h, z.data(), T, lat.data());
    if (dbg(lat)) return;
    std::vector<float> hin(size_t(T) * size_t(c.hidden)), hout(size_t(T) * size_t(c.hidden));
    matmul(pt_in, lat.data(), T, hin.data());
    transformer_forward(pt, st.kv, hin.data(), T, nullptr, hout.data());
    matmul(pt_out, hout.data(), T, lat.data());
    if (dbg(lat)) return;

    // 3) апсэмплинг (трансп. свёртка + ConvNeXt)
    std::vector<float> cur = std::move(lat);
    int len = T;
    for (size_t s = 0; s < up.size(); ++s) {
        const UpStage &u = up[s];
        const int L = c.latent;
        std::vector<float> y(size_t(len * u.tr.stride) * size_t(L));
        std::vector<float> tail;  // K == stride → хвоста нет
        trans_stream(u.tr, u.tr_b, tail, cur.data(), len, y.data());
        len *= u.tr.stride;
        // ConvNeXt
        std::vector<float> dw(size_t(len) * size_t(L)), nrm(size_t(len) * size_t(L)), h1(size_t(len) * size_t(4 * L)),
            h2(size_t(len) * size_t(L));
        dwconv_stream(u.dw, st.up_dw_h[s], y.data(), len, dw.data());
        parallel_for(len, 16, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) layernorm(&dw[size_t(t) * size_t(L)], u.ln_w, u.ln_b, L, 1e-6f, &nrm[size_t(t) * size_t(L)]);
        });
        matmul(u.pw1, nrm.data(), len, h1.data());
        parallel_for(int64_t(h1.size()), 4096, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; ++i) h1[size_t(i)] = gelu_erf(h1[size_t(i)]); });
        matmul(u.pw2, h1.data(), len, h2.data());
        parallel_for(len, 32, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t)
                for (int i = 0; i < L; ++i)
                    y[size_t(t) * size_t(L) + size_t(i)] += u.gamma[i] * h2[size_t(t) * size_t(L) + size_t(i)];
        });
        cur = std::move(y);
        if (dbg(cur)) return;
    }

    // 4) декодер: conv0 → блоки → snake → conv
    std::vector<float> x(size_t(len) * size_t(c.decoder_dim));
    conv_stream(conv0, st.conv0_h, cur.data(), len, x.data());
    if (dbg(x)) return;
    int ch = c.decoder_dim;
    for (size_t b = 0; b < blocks.size(); ++b) {
        const DecBlock &d = blocks[b];
        const int out_ch = ch / 2;
        snake_apply(d.act, x.data(), len, ch);
        std::vector<float> y(size_t(len * d.tr.stride) * size_t(out_ch));
        trans_stream(d.tr, d.tr_b, st.blk[b].tail, x.data(), len, y.data());
        len *= d.tr.stride;
        ch = out_ch;
        std::vector<float> t1(y.size()), t2(y.size());
        for (int u = 0; u < 3; ++u) {
            t1 = y;
            snake_apply(d.a1[u], t1.data(), len, ch);
            conv_stream(d.c1[u], st.blk[b].ru_h[u], t1.data(), len, t2.data());
            snake_apply(d.a2[u], t2.data(), len, ch);
            std::vector<float> none;  // K == 1: истории нет
            conv_stream(d.c2[u], none, t2.data(), len, t1.data());
            for (size_t i = 0; i < y.size(); ++i) y[i] += t1[i];
        }
        x = std::move(y);
        if (dbg(x)) return;
    }
    snake_apply(final_act, x.data(), len, ch);
    std::vector<float> wav(static_cast<size_t>(len));
    conv_stream(final_conv, st.final_h, x.data(), len, wav.data());
    for (float v : wav) pcm.push_back(std::min(1.0f, std::max(-1.0f, v)));
    st.frames += F;
}

}  // namespace vox
