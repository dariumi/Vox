// Chatterbox T3 (Llama 520M over S3 speech tokens) and its VoiceEncoder. S3Gen is CosyVoiceModel (cosyvoice.cpp) without the LLM.
// Weights from python/vox/build_chatterbox.py: t3.* (t3_cfg.safetensors), ve.* (ve.safetensors), the tokenizer.json blob.
#include "chatterbox.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <set>

#include "gpu.hpp"
#include "nn.hpp"
#include "sampling.hpp"

namespace vox {
using namespace nn;

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// rope_type "llama3": low frequencies divided by the factor, a smooth ramp between (transformers _compute_llama3_parameters)
std::vector<float> llama3_inv_freq(int hd, double theta, double factor, double lowf, double highf, double old_ctx) {
    std::vector<float> f(size_t(hd / 2));
    const double low_wl = old_ctx / lowf, high_wl = old_ctx / highf;
    for (int i = 0; i < hd / 2; ++i) {
        const double inv = 1.0 / std::pow(theta, double(2 * i) / double(hd));
        const double wl = 2.0 * kPi / inv;
        double v = wl > low_wl ? inv / factor : inv;
        if (!(wl < high_wl) && !(wl > low_wl)) {
            const double sm = (old_ctx / wl - lowf) / (highf - lowf);
            v = (1.0 - sm) * v / factor + sm * v;
        }
        f[size_t(i)] = float(v);
    }
    return f;
}

std::vector<uint32_t> cps(const std::string &s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if (i + size_t(len) > s.size()) len = 1;
        uint32_t v = len == 1 ? c : len == 2 ? (c & 0x1Fu) : len == 3 ? (c & 0x0Fu) : (c & 0x07u);
        for (int k = 1; k < len; ++k) v = (v << 6) | (static_cast<unsigned char>(s[i + size_t(k)]) & 0x3Fu);
        out.push_back(v);
        i += size_t(len);
    }
    return out;
}
void put_cp(std::string &o, uint32_t c) {
    if (c < 0x80) o += char(c);
    else if (c < 0x800) { o += char(0xC0 | (c >> 6)); o += char(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { o += char(0xE0 | (c >> 12)); o += char(0x80 | ((c >> 6) & 0x3F)); o += char(0x80 | (c & 0x3F)); }
    else { o += char(0xF0 | (c >> 18)); o += char(0x80 | ((c >> 12) & 0x3F)); o += char(0x80 | ((c >> 6) & 0x3F)); o += char(0x80 | (c & 0x3F)); }
}
bool is_ws(uint32_t c) {
    return (c >= 9 && c <= 13) || c == ' ' || (c >= 0x1C && c <= 0x1F) || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
// str.islower() / str.upper() of one character for Latin, Latin-1, Greek and Cyrillic
bool lower_cp(uint32_t c) { return (c >= 'a' && c <= 'z') || (c >= 0xDF && c <= 0xFF && c != 0xF7) || (c >= 0x3B1 && c <= 0x3C9) || (c >= 0x430 && c <= 0x45F); }
uint32_t upper_cp(uint32_t c) {
    if (c >= 'a' && c <= 'z') return c - 32;
    if (c >= 0xE0 && c <= 0xFE && c != 0xF7) return c - 32;
    if (c >= 0x3B1 && c <= 0x3C9 && c != 0x3C2) return c - 32;
    if (c >= 0x430 && c <= 0x44F) return c - 32;
    if (c >= 0x450 && c <= 0x45F) return c - 80;
    return c;
}
void replace_all(std::string &s, const std::string &a, const std::string &b) {
    for (size_t p = s.find(a); p != std::string::npos; p = s.find(a, p + b.size())) s.replace(p, a.size(), b);
}

struct LstmLayer { Mat ih, hh; std::vector<float> b; int in = 0, hid = 0; };   // b = b_ih + b_hh
}  // namespace

struct ChatterboxT3::Impl {
    TransformerW tf;
    Mat text_emb, speech_emb, text_pos, speech_pos, speech_head;
    Linear spkr;
    std::vector<float> emo;   // [hidden] (emotion_adv_fc, 1 → hidden, no bias)
    std::vector<float> query; // perceiver queries [32][hidden]
    int n_query = 0;
    std::vector<float> pln_w, pln_b;
    Linear pq, pk, pv, po;
    // voice encoder
    std::vector<LstmLayer> lstm;
    Linear ve_proj;
    std::vector<float> ve_mel;   // [40][201]
    bool ve = false;
};

ChatterboxT3::ChatterboxT3(std::shared_ptr<TmodFile> f) : p_(new Impl), file_(std::move(f)) {
    const TmodFile &F = *file_;
    const json &m = F.manifest();
    if (!m.contains("chatterbox")) fail(VOX_ERR_FORMAT, "not a Chatterbox .tmod (no \"chatterbox\" section)");
    const json &c = m.at("chatterbox").at("t3");
    T3Cfg &g = cfg_;
    g.hidden = c.value("hidden_size", g.hidden); g.layers = c.value("num_hidden_layers", g.layers);
    g.heads = c.value("num_attention_heads", g.heads); g.kv_heads = c.value("num_key_value_heads", g.kv_heads);
    g.head_dim = c.value("head_dim", g.hidden / g.heads); g.inter = c.value("intermediate_size", g.inter);
    g.eps = c.value("rms_norm_eps", g.eps); g.theta = c.value("rope_theta", g.theta);
    if (c.contains("rope_scaling") && c["rope_scaling"].is_object()) {
        const json &r = c["rope_scaling"];
        g.rope_factor = r.value("factor", g.rope_factor); g.low_freq = r.value("low_freq_factor", g.low_freq);
        g.high_freq = r.value("high_freq_factor", g.high_freq); g.orig_ctx = r.value("original_max_position_embeddings", g.orig_ctx);
    } else {
        g.rope_factor = 0;
    }
    g.text_vocab = c.value("text_tokens_dict_size", g.text_vocab); g.speech_vocab = c.value("speech_tokens_dict_size", g.speech_vocab);
    g.start_text = c.value("start_text_token", g.start_text); g.stop_text = c.value("stop_text_token", g.stop_text);
    g.start_speech = c.value("start_speech_token", g.start_speech); g.stop_speech = c.value("stop_speech_token", g.stop_speech);
    g.prompt_len = c.value("speech_cond_prompt_len", g.prompt_len); g.speaker_dim = c.value("speaker_embed_size", g.speaker_dim);
    g.perceiver_heads = c.value("perceiver_heads", g.perceiver_heads);
    Impl &P = *p_;
    TransformerW &w = P.tf;
    w.hidden = g.hidden; w.heads = g.heads; w.kv_heads = g.kv_heads; w.head_dim = g.head_dim; w.inter = g.inter; w.eps = g.eps; w.theta = g.theta;
    w.inv_freq = g.rope_factor > 0 ? llama3_inv_freq(g.head_dim, g.theta, g.rope_factor, g.low_freq, g.high_freq, g.orig_ctx)
                                   : rope_inv_freq(g.head_dim, g.theta);
    w.layers.resize(size_t(g.layers));
    auto vec_ptr = [&](const std::string &n) {
        const TensorView &t = F.get(n);
        if (t.dtype != DType::F32) fail(VOX_ERR_FORMAT, "Chatterbox: norms must be F32: " + n);
        return reinterpret_cast<const float *>(t.data);
    };
    for (int i = 0; i < g.layers; ++i) {
        const std::string p = "t3.tfmr.layers." + std::to_string(i) + ".";
        LayerW &L = w.layers[size_t(i)];
        L.q = mat_named(F, p + "self_attn.q_proj.weight"); L.k = mat_named(F, p + "self_attn.k_proj.weight");
        L.v = mat_named(F, p + "self_attn.v_proj.weight"); L.o = mat_named(F, p + "self_attn.o_proj.weight");
        L.gate = mat_named(F, p + "mlp.gate_proj.weight"); L.up = mat_named(F, p + "mlp.up_proj.weight"); L.down = mat_named(F, p + "mlp.down_proj.weight");
        L.ln1 = vec_ptr(p + "input_layernorm.weight"); L.ln2 = vec_ptr(p + "post_attention_layernorm.weight");
    }
    w.final_norm = vec_ptr("t3.tfmr.norm.weight");
    P.text_emb = mat_of(F.get("t3.text_emb.weight"));
    P.speech_emb = mat_of(F.get("t3.speech_emb.weight"));
    P.text_pos = mat_of(F.get("t3.text_pos_emb.emb.weight"));
    P.speech_pos = mat_of(F.get("t3.speech_pos_emb.emb.weight"));
    P.speech_head = mat_named(F, "t3.speech_head.weight");
    g.max_text = int(P.text_pos.rows); g.max_speech = int(P.speech_pos.rows);
    P.spkr.load(F, "t3.cond_enc.spkr_enc");
    P.emo = to_float(F.get("t3.cond_enc.emotion_adv_fc.weight"));
    const TensorView &q = F.get("t3.cond_enc.perceiver.pre_attention_query");
    P.query = to_float(q);
    P.n_query = int(P.query.size() / size_t(g.hidden));
    const std::string a = "t3.cond_enc.perceiver.attn.";
    P.pln_w = to_float(F.get(a + "norm.weight")); P.pln_b = to_float(F.get(a + "norm.bias"));
    P.pq.load(F, a + "to_q"); P.pk.load(F, a + "to_k"); P.pv.load(F, a + "to_v"); P.po.load(F, a + "proj_out");
    // voice encoder
    if (F.has("ve.lstm.weight_ih_l0")) {
        P.ve = true;
        for (int l = 0;; ++l) {
            const std::string s = std::to_string(l);
            if (!F.has("ve.lstm.weight_ih_l" + s)) break;
            LstmLayer L;
            L.ih = mat_of(F.get("ve.lstm.weight_ih_l" + s));
            L.hh = mat_of(F.get("ve.lstm.weight_hh_l" + s));
            L.in = int(L.ih.cols); L.hid = int(L.hh.cols);
            L.b = to_float(F.get("ve.lstm.bias_ih_l" + s));
            const std::vector<float> b2 = to_float(F.get("ve.lstm.bias_hh_l" + s));
            for (size_t i = 0; i < L.b.size(); ++i) L.b[i] += b2[i];
            P.lstm.push_back(L);
        }
        P.ve_proj.load(F, "ve.proj");
        P.ve_mel = to_float(F.get("front.ve_mel_basis"));
    }
    tok_ = std::make_unique<HfTokenizer>([&] {
        const TensorView &t = F.get("tokenizer.json");
        return std::string(reinterpret_cast<const char *>(t.data), t.nbytes);
    }());
}

ChatterboxT3::~ChatterboxT3() = default;

bool ChatterboxT3::has_voice_encoder() const { return p_->ve; }
bool ChatterboxT3::gpu() const { return p_->tf.gpu_id >= 0; }
bool ChatterboxT3::enable_gpu(bool release_host) {
    Impl &P = *p_;
    if (P.tf.gpu_id >= 0) return true;
    if (!gpu_active()) return false;
    const TmodFile &F = *file_;
    auto rel = [&](const Mat &m) {   // the CPU copy is not needed any more (re-read from the file for a piece too long for the GPU)
        if (!release_host || !m.valid()) return;
        F.release(m.data, size_t(m.rows) * mat_row_bytes(m.type, m.cols));
        if (m.scale) F.release(m.scale, size_t(m.rows) * size_t(m.cols / 32) * 4);
    };
    P.tf.gpu_id = gpu_xfmr_create(P.tf, "t3", -1, rel);
    if (P.tf.gpu_id < 0) return false;
    P.speech_head.gpu = gpu_upload(P.speech_head);
    if (P.speech_head.gpu >= 0) rel(P.speech_head);
    return true;
}

std::string ChatterboxT3::punc_norm(const std::string &in) {
    if (in.empty()) return "You need to add some text for me to talk.";
    std::vector<uint32_t> v = cps(in);
    if (lower_cp(v[0])) v[0] = upper_cp(v[0]);
    // " ".join(text.split())
    std::string t;
    bool word = false;
    for (uint32_t c : v) {
        if (is_ws(c)) { word = false; continue; }
        if (!word && !t.empty()) t += ' ';
        word = true;
        put_cp(t, c);
    }
    static const std::pair<const char *, const char *> rep[] = {
        {"...", ", "}, {"…", ", "}, {":", ","}, {" - ", ", "}, {";", ", "}, {"—", "-"}, {"–", "-"}, {" ,", ","},
        {"“", "\""}, {"”", "\""}, {"‘", "'"}, {"’", "'"}};
    for (const auto &r : rep) replace_all(t, r.first, r.second);
    while (!t.empty() && t.back() == ' ') t.pop_back();
    const char e = t.empty() ? 0 : t.back();
    if (!(e == '.' || e == '!' || e == '?' || e == '-' || e == ',')) t += ".";
    return t;
}

std::vector<int32_t> ChatterboxT3::text_tokens(const std::string &text) const {
    std::string t = punc_norm(text);
    replace_all(t, " ", "[SPACE]");
    std::vector<int32_t> ids{cfg_.start_text};
    const std::vector<int32_t> body = tok_->encode(t);
    ids.insert(ids.end(), body.begin(), body.end());
    ids.push_back(cfg_.stop_text);
    return ids;
}

std::vector<float> ChatterboxT3::cond_embeds(const std::vector<float> &spk, const std::vector<int32_t> &prompt, float exaggeration) const {
    const Impl &P = *p_;
    const int H = cfg_.hidden;
    if (int(spk.size()) != cfg_.speaker_dim) fail(VOX_ERR_ARG, "Chatterbox: the speaker embedding must have " + std::to_string(cfg_.speaker_dim) + " values");
    std::vector<float> rows(static_cast<size_t>(H));
    P.spkr.forward(spk.data(), 1, rows.data());
    if (!prompt.empty()) {
        // prompt speech tokens: speech_emb + speech_pos_emb (positions 0…), then the perceiver (cross-, then self-attention, shared block)
        const int n = int(prompt.size());
        if (n > P.speech_pos.rows) fail(VOX_ERR_ARG, "Chatterbox: the prompt has more speech tokens than the model has positions");
        std::vector<float> h(size_t(n) * H), tmp(static_cast<size_t>(H));
        for (int i = 0; i < n; ++i) {
            if (prompt[size_t(i)] < 0 || prompt[size_t(i)] >= P.speech_emb.rows) fail(VOX_ERR_ARG, "Chatterbox: prompt token out of range");
            mat_gather(P.speech_emb, prompt[size_t(i)], &h[size_t(i) * H]);
            mat_gather(P.speech_pos, i, tmp.data());
            for (int d = 0; d < H; ++d) h[size_t(i) * H + d] += tmp[size_t(d)];
        }
        auto block = [&](const std::vector<float> &x1, int n1, const std::vector<float> &x2, int n2) {
            std::vector<float> a = x1, b = x2;
            layer_norm(a.data(), n1, H, P.pln_w.data(), P.pln_b.data(), 1e-5f);
            layer_norm(b.data(), n2, H, P.pln_w.data(), P.pln_b.data(), 1e-5f);
            std::vector<float> Q(size_t(n1) * H), K(size_t(n2) * H), V(size_t(n2) * H), A(size_t(n1) * H), O(size_t(n1) * H);
            P.pq.forward(a.data(), n1, Q.data());
            P.pk.forward(b.data(), n2, K.data());
            P.pv.forward(b.data(), n2, V.data());
            const int nh = cfg_.perceiver_heads, dh = H / nh;
            attention(Q.data(), H, K.data(), H, V.data(), H, n1, n2, nh, dh, 1.0f / std::sqrt(float(dh)), A.data());
            P.po.forward(A.data(), n1, O.data());
            for (size_t i = 0; i < O.size(); ++i) O[i] += x1[i];
            return O;
        };
        const std::vector<float> pre = block(P.query, P.n_query, h, n);
        const std::vector<float> post = block(pre, P.n_query, pre, P.n_query);
        rows.insert(rows.end(), post.begin(), post.end());
    }
    for (int d = 0; d < H; ++d) rows.push_back(P.emo[size_t(d)] * exaggeration);
    return rows;
}

std::vector<int32_t> ChatterboxT3::generate(const std::vector<float> &cond, const std::vector<int32_t> &text, const T3Sampling &sp, uint64_t seed,
                                            uint32_t piece, const std::vector<int32_t> *forced, Trace *tr) const {
    const auto t0 = Clock::now();
    const Impl &P = *p_;
    const int H = cfg_.hidden, V = int(P.speech_head.rows);
    const int nc = int(cond.size() / size_t(H)), nt = int(text.size());
    if (nt > cfg_.max_text) fail(VOX_ERR_ARG, "Chatterbox: the text is too long for one piece");
    const bool use_cfg = sp.cfg_weight > 0.f;
    // [cond | text + text_pos | start speech + pos 0 | the same start again (as T3.inference appends bos_embed)]
    const int n = nc + nt + 2;
    std::vector<float> x(size_t(n) * H), xu, tmp(static_cast<size_t>(H));
    std::memcpy(x.data(), cond.data(), cond.size() * 4);
    for (int i = 0; i < nt; ++i) {
        if (text[size_t(i)] < 0 || text[size_t(i)] >= P.text_emb.rows) fail(VOX_ERR_ARG, "Chatterbox: text token out of range");
        mat_gather(P.text_emb, text[size_t(i)], &x[size_t(nc + i) * H]);
        mat_gather(P.text_pos, i, tmp.data());
        for (int d = 0; d < H; ++d) x[size_t(nc + i) * H + d] += tmp[size_t(d)];
    }
    std::vector<float> bos(static_cast<size_t>(H));
    mat_gather(P.speech_emb, cfg_.start_speech, bos.data());
    mat_gather(P.speech_pos, 0, tmp.data());
    for (int d = 0; d < H; ++d) bos[size_t(d)] += tmp[size_t(d)];
    std::memcpy(&x[size_t(nc + nt) * H], bos.data(), size_t(H) * 4);
    std::memcpy(&x[size_t(nc + nt + 1) * H], bos.data(), size_t(H) * 4);
    if (use_cfg) {   // unconditional: the text embeddings zeroed (their positions stay)
        xu = x;
        for (int i = 0; i < nt; ++i) {
            mat_gather(P.text_pos, i, tmp.data());
            std::memcpy(&xu[size_t(nc + i) * H], tmp.data(), size_t(H) * 4);
        }
    }
    KVCache kc, ku;
    kc.init(cfg_.layers);
    ku.init(cfg_.layers);
    const int maxn = forced ? int(forced->size()) + 1 : std::max(1, sp.max_new_tokens);
    struct SeqGuard {   // GPU KV caches of this piece (both passes or neither)
        KVCache &a, &b;
        ~SeqGuard() { for (KVCache *k : {&a, &b}) if (k->gpu_seq >= 0) gpu_seq_free(k->gpu_seq); }
    } guard{kc, ku};
    if (P.tf.gpu_id >= 0 && int64_t(n) + maxn + 1 <= 3072) {
        kc.gpu_seq = gpu_seq_create(P.tf.gpu_id, int64_t(n) + maxn + 1);
        if (use_cfg && kc.gpu_seq >= 0) {
            ku.gpu_seq = gpu_seq_create(P.tf.gpu_id, int64_t(n) + maxn + 1);
            if (ku.gpu_seq < 0) { gpu_seq_free(kc.gpu_seq); kc.gpu_seq = -1; }
        }
    }
    const bool on_gpu = kc.gpu_seq >= 0;
    std::vector<float> hc(static_cast<size_t>(H)), hu(static_cast<size_t>(H)), lc(static_cast<size_t>(V)), lu(static_cast<size_t>(V));
    transformer_forward(P.tf, kc, x.data(), n, hc.data());
    if (use_cfg) transformer_forward(P.tf, ku, xu.data(), n, hu.data());
    const DetRng rng(seed);
    const uint32_t stream = 0x54330000u + piece;   // "T3"
    std::vector<int32_t> gen;                      // generated ids for the repetition penalty: [start, …]
    gen.push_back(cfg_.start_speech);
    std::vector<int32_t> out;
    std::vector<float> logits(static_cast<size_t>(V));
    std::vector<float> l2next;   // GPU: the logits of both passes came back with the step that made hc / hu
    bool l2_ready = false;
    for (int i = 0; i < maxn; ++i) {
        if (use_cfg) {
            std::vector<float> h2(size_t(2) * H), l2(size_t(2) * V);
            if (l2_ready) {
                l2.swap(l2next);
            } else {
                std::memcpy(h2.data(), hc.data(), size_t(H) * 4);
                std::memcpy(h2.data() + H, hu.data(), size_t(H) * 4);
                matmul(P.speech_head, h2.data(), 2, l2.data());
            }
            std::memcpy(lc.data(), l2.data(), size_t(V) * 4);
            std::memcpy(lu.data(), l2.data() + V, size_t(V) * 4);
            for (int k = 0; k < V; ++k) logits[size_t(k)] = lc[size_t(k)] + sp.cfg_weight * (lc[size_t(k)] - lu[size_t(k)]);
        } else {
            matmul(P.speech_head, hc.data(), 1, lc.data());
            logits = lc;
        }
        if (tr) tr->logits.insert(tr->logits.end(), logits.begin(), logits.end());
        int32_t next;
        if (forced) {
            if (i == int(forced->size())) break;
            next = (*forced)[size_t(i)];
        } else {
            // RepetitionPenaltyLogitsProcessor → temperature → MinP → TopP → softmax → multinomial
            std::vector<float> s = logits;
            std::set<int32_t> seen(gen.begin(), gen.end());
            for (int32_t id : seen) {
                float &z = s[size_t(id)];
                z = z < 0.f ? z * sp.repetition_penalty : z / sp.repetition_penalty;
            }
            if (sp.temperature != 1.0f)
                for (auto &z : s) z /= sp.temperature;
            float mx = -INFINITY;
            for (float z : s) mx = std::max(mx, z);
            std::vector<double> pr(static_cast<size_t>(V));
            double sum = 0;
            for (int k = 0; k < V; ++k) { pr[size_t(k)] = std::exp(double(s[size_t(k)] - mx)); sum += pr[size_t(k)]; }
            const double top = 1.0 / sum;   // the largest probability
            if (sp.min_p > 0.f) {
                const int32_t best = int32_t(std::max_element(s.begin(), s.end()) - s.begin());
                for (int k = 0; k < V; ++k)
                    if (pr[size_t(k)] / sum < double(sp.min_p) * top && k != best) s[size_t(k)] = -INFINITY;
            }
            if (sp.top_p < 1.0f) {
                std::vector<int> idx(static_cast<size_t>(V));
                std::iota(idx.begin(), idx.end(), 0);
                std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return s[size_t(a)] < s[size_t(b)]; });   // ascending
                double m2 = -INFINITY;
                for (float z : s) m2 = std::max(m2, double(z));
                std::vector<double> q(static_cast<size_t>(V));
                double s2 = 0;
                for (int k = 0; k < V; ++k) { q[size_t(k)] = std::isinf(s[size_t(k)]) ? 0.0 : std::exp(double(s[size_t(k)]) - m2); s2 += q[size_t(k)]; }
                double cum = 0;
                for (int r = 0; r < V - 1; ++r) {   // keep at least one
                    cum += q[size_t(idx[size_t(r)])] / s2;
                    if (cum <= 1.0 - double(sp.top_p)) s[size_t(idx[size_t(r)])] = -INFINITY;
                }
            }
            mx = -INFINITY;
            for (float z : s) mx = std::max(mx, z);
            sum = 0;
            for (int k = 0; k < V; ++k) { pr[size_t(k)] = std::isinf(s[size_t(k)]) ? 0.0 : std::exp(double(s[size_t(k)] - mx)); sum += pr[size_t(k)]; }
            const double u = double(rng.uniform(uint64_t(i), stream, 0)) * sum;
            double acc = 0;
            next = V - 1;
            for (int k = 0; k < V; ++k) {
                if (pr[size_t(k)] == 0.0) continue;
                acc += pr[size_t(k)];
                next = k;
                if (u < acc) break;
            }
        }
        if (next == cfg_.stop_speech) break;
        out.push_back(next);
        gen.push_back(next);
        if (i + 1 >= cfg_.max_speech) break;
        mat_gather(P.speech_emb, next, x.data());
        mat_gather(P.speech_pos, i + 1, tmp.data());
        for (int d = 0; d < H; ++d) x[size_t(d)] += tmp[size_t(d)];
        l2_ready = false;
        if (use_cfg && on_gpu) {
            // both passes and the speech head in one submission: the weights are read once for the two rows
            std::vector<float> x2(size_t(2) * H), h2(size_t(2) * H);
            std::memcpy(x2.data(), x.data(), size_t(H) * 4);
            std::memcpy(x2.data() + H, x.data(), size_t(H) * 4);
            l2next.resize(size_t(2) * V);
            const Mat *head = P.speech_head.gpu >= 0 ? &P.speech_head : nullptr;
            if (kc.len == ku.len && gpu_xfmr_forward_pair(P.tf, kc.gpu_seq, ku.gpu_seq, kc.len, x2.data(), head, h2.data(), l2next.data())) {
                kc.len += 1;
                ku.len += 1;
                std::memcpy(hc.data(), h2.data(), size_t(H) * 4);
                std::memcpy(hu.data(), h2.data() + H, size_t(H) * 4);
                l2_ready = head != nullptr;
            } else {
                transformer_forward(P.tf, kc, x.data(), 1, hc.data());
                transformer_forward(P.tf, ku, x.data(), 1, hu.data());
            }
        } else if (use_cfg) {   // both passes in one: the weights are read once
            std::vector<float> x2(size_t(2) * H), h2(size_t(2) * H);
            std::memcpy(x2.data(), x.data(), size_t(H) * 4);
            std::memcpy(x2.data() + H, x.data(), size_t(H) * 4);
            KVCache *both[2] = {&kc, &ku};
            transformer_forward_steps(P.tf, both, 2, x2.data(), h2.data());
            std::memcpy(hc.data(), h2.data(), size_t(H) * 4);
            std::memcpy(hu.data(), h2.data() + H, size_t(H) * 4);
        } else {
            transformer_forward(P.tf, kc, x.data(), 1, hc.data());
        }
    }
    ms_gen += ms_since(t0);
    return out;
}

std::vector<float> ChatterboxT3::voice_embedding(const float *pcm, size_t n) const {
    const Impl &P = *p_;
    if (!P.ve) fail(VOX_ERR_UNSUPPORTED, "Chatterbox: this .tmod has no voice encoder (ve.safetensors)");
    // librosa.effects.trim(top_db=20): RMS of 2048-sample frames every 512 (centred, zero padding), relative to the loudest
    size_t a = 0, b = n;
    {
        const int64_t FL = 2048, HOP = 512, half = FL / 2;
        const int64_t N = int64_t(n), frames = N > 0 ? 1 + N / HOP : 0;
        std::vector<double> mse(size_t(std::max<int64_t>(frames, 0)));
        double mx = 0;
        for (int64_t f = 0; f < frames; ++f) {
            double s = 0;
            for (int64_t k = 0; k < FL; ++k) {
                const int64_t j = f * HOP + k - half;
                if (j >= 0 && j < N) s += double(pcm[j]) * double(pcm[j]);
            }
            mse[size_t(f)] = s / double(FL);
            mx = std::max(mx, mse[size_t(f)]);
        }
        const double ref = 10.0 * std::log10(std::max(1e-10, mx));
        int64_t first = -1, last = -1;
        for (int64_t f = 0; f < frames; ++f)
            if (10.0 * std::log10(std::max(1e-10, mse[size_t(f)])) - ref > -20.0) { if (first < 0) first = f; last = f; }
        if (first < 0) { a = b = 0; }
        else { a = size_t(first * HOP); b = std::min<size_t>(n, size_t((last + 1) * HOP)); }
    }
    if (b <= a) fail(VOX_ERR_ARG, "Chatterbox: the recording is silent");
    // mel: STFT n_fft 400 hop 160 (centred, reflect), |·|², 40 bands up to 8 kHz
    const int NF = 400, HOP = 160, NB = NF / 2 + 1, M = 40;
    const int64_t L = int64_t(b - a);
    const float *y = pcm + a;
    const int T = int(L / HOP + 1);
    RealDFT dft(NF);
    const auto win = hann(NF);
    std::vector<float> fr(size_t(T) * NF), spec(size_t(T) * size_t(2 * NB)), pw(size_t(T) * NB), mel(size_t(T) * M);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < NF; ++i) {
            int64_t j = int64_t(t) * HOP + i - NF / 2;
            if (j < 0) j = -j;
            if (j >= L) j = 2 * (L - 1) - j;
            j = std::min<int64_t>(std::max<int64_t>(j, 0), L - 1);
            fr[size_t(t) * NF + i] = float(double(y[j]) * win[size_t(i)]);
        }
    dft.forward(fr.data(), T, spec.data());
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < NB; ++k) {
            const float re = spec[size_t(t) * (2 * NB) + k], im = spec[size_t(t) * (2 * NB) + NB + k];
            pw[size_t(t) * NB + k] = re * re + im * im;
        }
    matmul(f32_mat(P.ve_mel.data(), M, NB), pw.data(), T, mel.data());
    // partials of 160 frames every 77 (rate 1.3), min coverage 0.8
    const int win_f = 160, step = int(std::lround((16000.0 / 1.3) / 160.0));
    int nw = std::max(T - win_f + step, 0) / step;
    const int rem = std::max(T - win_f + step, 0) % step;
    if (nw == 0 || double(rem + (win_f - step)) / double(win_f) >= 0.8) ++nw;
    const int target = win_f + step * (nw - 1);
    mel.resize(size_t(target) * M, 0.f);
    // LSTM over all partials at once
    const int hid = P.lstm.empty() ? 0 : P.lstm[0].hid;
    std::vector<float> seq(size_t(nw) * size_t(win_f) * M);
    for (int p = 0; p < nw; ++p)
        std::memcpy(&seq[size_t(p) * win_f * M], &mel[size_t(p) * step * M], size_t(win_f) * M * 4);
    int in = M;
    std::vector<float> hlast;
    for (const LstmLayer &Lr : P.lstm) {
        std::vector<float> gx(size_t(nw) * win_f * 4 * hid);
        matmul(Lr.ih, seq.data(), int64_t(nw) * win_f, gx.data());   // [p][t][4H]
        std::vector<float> h(size_t(nw) * hid, 0.f), cst(size_t(nw) * hid, 0.f), gh(size_t(nw) * 4 * hid), outs(size_t(nw) * win_f * hid);
        for (int t = 0; t < win_f; ++t) {
            matmul(Lr.hh, h.data(), nw, gh.data());
            for (int p = 0; p < nw; ++p) {
                const float *gxi = &gx[(size_t(p) * win_f + t) * 4 * hid], *ghi = &gh[size_t(p) * 4 * hid];
                for (int k = 0; k < hid; ++k) {
                    const float ig = 1.0f / (1.0f + std::exp(-(gxi[k] + ghi[k] + Lr.b[size_t(k)])));
                    const float fg = 1.0f / (1.0f + std::exp(-(gxi[hid + k] + ghi[hid + k] + Lr.b[size_t(hid + k)])));
                    const float gg = std::tanh(gxi[2 * hid + k] + ghi[2 * hid + k] + Lr.b[size_t(2 * hid + k)]);
                    const float og = 1.0f / (1.0f + std::exp(-(gxi[3 * hid + k] + ghi[3 * hid + k] + Lr.b[size_t(3 * hid + k)])));
                    float &cc = cst[size_t(p) * hid + k];
                    cc = fg * cc + ig * gg;
                    h[size_t(p) * hid + k] = og * std::tanh(cc);
                }
                std::memcpy(&outs[(size_t(p) * win_f + t) * hid], &h[size_t(p) * hid], size_t(hid) * 4);
            }
        }
        seq.swap(outs);
        hlast = h;
        in = hid;
    }
    (void)in;
    const int E = int(P.ve_proj.out);
    std::vector<float> e(size_t(nw) * E);
    P.ve_proj.forward(hlast.data(), nw, e.data());
    std::vector<double> mean(static_cast<size_t>(E), 0.0);
    for (int p = 0; p < nw; ++p) {
        double nn2 = 0;
        for (int k = 0; k < E; ++k) { float &z = e[size_t(p) * E + k]; z = std::max(z, 0.f); nn2 += double(z) * z; }   // ve_final_relu
        const float inv = float(1.0 / std::sqrt(nn2));
        for (int k = 0; k < E; ++k) mean[size_t(k)] += double(e[size_t(p) * E + k] * inv);
    }
    double nn2 = 0;
    for (auto &v : mean) { v /= nw; nn2 += v * v; }
    std::vector<float> out(static_cast<size_t>(E));
    for (int k = 0; k < E; ++k) out[size_t(k)] = float(mean[size_t(k)] / std::sqrt(nn2));
    return out;
}

}  // namespace vox
