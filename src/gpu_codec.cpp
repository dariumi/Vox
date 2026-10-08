// Qwen3-TTS codec decoder entirely on the GPU (Vulkan).
//
// One call = up to kGpuCodecMaxFrames codec frames recorded into ONE command buffer:
//   z (quantizer output, computed on CPU, [F][codebook_dim]) → pre_conv → pre-transformer (sliding-window
//   attention over a ring KV cache, layer scale) → upsample stages (transposed conv + ConvNeXt) → conv0 →
//   decoder blocks (Snake, transposed conv with overlap-add tail, 3 residual units) → Snake → final conv → clamp.
// Streaming state (conv histories, transposed-conv tails, ring KV) lives in device memory, so nothing but z
// (F*codebook_dim floats) goes up and the PCM (F*samples_per_frame floats) comes back. The math mirrors
// Qwen3Weights::codec_forward (CPU) operation by operation; tests compare the two.
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdio>

#include "gpu_vk_internal.hpp"
#include "qwen3.hpp"

namespace vox {
using namespace vk;

namespace {

struct Buf {
    VkBuffer b = VK_NULL_HANDLE;
    VkDeviceMemory m = VK_NULL_HANDLE;
    VkDeviceSize bytes = 0;
};

const VkBufferUsageFlags kUse = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

Buf dev_buf(VkDeviceSize bytes) {
    Buf b;
    b.bytes = std::max<VkDeviceSize>(16, (bytes + 15) / 16 * 16);
    make_buffer(b.bytes, kUse, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b.b, b.m);
    return b;
}
Buf host_buf(VkDeviceSize bytes, float **map) {
    Buf b;
    b.bytes = std::max<VkDeviceSize>(16, (bytes + 15) / 16 * 16);
    make_buffer(b.bytes, kUse, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b.b, b.m);
    VKC(vkMapMemory(g.dev, b.m, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(map)));
    return b;
}
Buf vec_buf(const float *p, size_t n) {
    Buf b;
    b.bytes = std::max<VkDeviceSize>(16, (n * 4 + 15) / 16 * 16);
    std::vector<float> tmp(b.bytes / 4, 0.f);
    std::copy(p, p + n, tmp.begin());
    upload_buffer(tmp.data(), b.bytes, b.b, b.m);
    return b;
}
Buf vec_buf(const std::vector<float> &v) { return vec_buf(v.data(), v.size()); }
void free_buf(Buf &b) {
    if (b.b) vkDestroyBuffer(g.dev, b.b, nullptr);
    if (b.m) dev_mem_free(b.m);
    b = Buf{};
}

// One streaming history = two buffers (ping-pong) of ctx rows × C.
struct Hist {
    int ctx = 0, C = 0;
    Buf b[2];
    int cur = 0;
};

struct GSnake { Buf ab; int C = 0; };  // [a (C); inv_b (C)]
struct GConv { int mat = -1; int K = 1, cin = 0, cout = 0, dil = 1; };

struct GCodec {
    int cdim = 0, latent = 0, H = 0, heads = 0, kvh = 0, hd = 0, I = 0, W = 0, R = 0, dec = 0, spf = 0;
    float eps = 1e-5f;
    std::vector<float> inv_freq;
    GConv pre_conv;
    int pt_in = -1, pt_out = -1;
    struct L { int q, k, v, o, gate, up, down; Buf ln1, ln2, s_attn, s_mlp; bool has_sa = false, has_sm = false; };
    std::vector<L> layers;
    Buf final_norm;
    struct Up { int tr = -1, K = 0, pw1 = -1, pw2 = -1; Buf tr_b, dwp, lnp, gamma; int dwK = 7; };  // dwp=[w(C*K); b(C)], lnp=[w; b]
    std::vector<Up> ups;
    GConv conv0;
    struct Blk { GSnake act; int tr = -1, K = 0, s = 0, cin = 0, cout = 0; Buf tr_b; GSnake a1[3], a2[3]; GConv c1[3], c2[3]; };
    std::vector<Blk> blocks;
    GSnake fin_act;
    GConv fin;
    // scratch
    Buf zb, out, csb, dbg;
    float *zp = nullptr, *outp = nullptr, *csp = nullptr, *dbgp = nullptr;
    Buf S[5];
    Buf q, k, v, ao, gt, ut, xn, hh;
};

struct GState {
    int codec = -1;
    bool used = false;
    int64_t len = 0;  // positions already in the transformer (absolute)
    Hist pre;
    std::vector<Buf> kr, vr;  // ring KV per layer [R][kd]
    std::vector<Hist> dw;     // per up stage
    Hist c0;
    struct Blk { Hist tail; Hist ru[3]; };
    std::vector<Blk> blk;
    Hist fin;
};

std::vector<GCodec> g_codecs;
std::vector<GState> g_states;

GConv up_conv(const CConv &c) {
    GConv r;
    r.mat = c.w.gpu; r.K = c.K; r.cin = c.cin; r.cout = c.cout; r.dil = c.dil;
    if (r.mat < 0) throw std::runtime_error("codec conv weight is not on the GPU");
    return r;
}
GSnake up_snake(const Snake &s) {
    GSnake r;
    r.C = int(s.a.size());
    std::vector<float> v(s.a);
    v.insert(v.end(), s.inv_b.begin(), s.inv_b.end());
    r.ab = vec_buf(v);
    return r;
}
// Буферы истории; обнуляются командой vkCmdFillBuffer в уже открытом командном буфере (одна отправка на состояние).
Hist make_hist(int ctx, int C) {
    Hist h;
    h.ctx = ctx; h.C = C;
    if (ctx > 0)
        for (auto &b : h.b) {
            b = dev_buf(VkDeviceSize(ctx) * VkDeviceSize(C) * 4);
            vkCmdFillBuffer(g.cmd, b.b, 0, VK_WHOLE_SIZE, 0);
        }
    return h;
}
Hist clone_hist(const Hist &s) {
    Hist h;
    h.ctx = s.ctx; h.C = s.C; h.cur = s.cur;
    for (int i = 0; i < 2; ++i)
        if (s.b[i].b) {
            h.b[i] = dev_buf(s.b[i].bytes);
            VkBufferCopy bc{0, 0, s.b[i].bytes};
            vkCmdCopyBuffer(g.cmd, s.b[i].b, h.b[i].b, 1, &bc);
        }
    return h;
}
void free_hist(Hist &h) { for (auto &b : h.b) free_buf(b); }

// --------------------------- recording helpers ---------------------------
const VkBuffer kNone = VK_NULL_HANDLE;

int gemm_pipe(int mat) {
    const GpuMat &gm = g.mats[size_t(mat)];
    switch (gm.type) {
        case WType::F32: return P_GEMM + 0;
        case WType::F16: return P_GEMM + 1;
        case WType::BF16: return P_GEMM + 2;
        default: throw std::runtime_error("codec weights must be f32/f16/bf16 for the GPU decoder");
    }
}
// Y[T][O] (op) conv(W, [hist; X]); bias from the matrix (if any) unless bias_buf is given.
void gemm(int mat, VkBuffer X, VkBuffer Y, VkBuffer hist, uint32_t T, uint32_t K, uint32_t dil, uint32_t ctx, uint32_t flags,
          VkBuffer bias_buf = VK_NULL_HANDLE, uint32_t bias_n = 0, VkBuffer gamma = VK_NULL_HANDLE) {
    const GpuMat &gm = g.mats[size_t(mat)];
    const uint32_t O = uint32_t(gm.rows), C = uint32_t(gm.cols / K);
    VkBuffer bias = bias_buf ? bias_buf : (gm.b ? gm.b : kNone);
    if (!bias) flags &= ~1u;
    if (bias_buf == VK_NULL_HANDLE) bias_n = O;
    const VkBuffer b[6] = {gm.w, gamma, bias, X, Y, hist};
    const uint32_t pc[8] = {T, O, C, K, dil, ctx, flags, bias_n};
    disp(gemm_pipe(mat), b, pc, 8, (T + 63) / 64, (O + 63) / 64);
}
void hist_update(Hist &h, VkBuffer X, uint32_t T) {
    if (h.ctx == 0) return;
    const VkBuffer b[6] = {kNone, kNone, kNone, X, h.b[1 - h.cur].b, h.b[h.cur].b};
    const uint32_t pc[3] = {T, uint32_t(h.C), uint32_t(h.ctx)};
    disp(P_HIST, b, pc, 3, uint32_t((h.ctx * h.C + 255) / 256), 1);
    h.cur = 1 - h.cur;
}
// conv with streaming history: Y = conv([hist; X]), then hist ← tail of [hist; X]
void conv_stream(const GConv &c, Hist &h, VkBuffer X, VkBuffer Y, uint32_t T, uint32_t flags) {
    const uint32_t ctx = uint32_t((c.K - 1) * c.dil);
    gemm(c.mat, X, Y, ctx ? h.b[h.cur].b : kNone, T, uint32_t(c.K), uint32_t(c.dil), ctx, flags);
    hist_update(h, X, T);
}
void elementwise_groups(uint64_t n, uint32_t &gx, uint32_t &gy) {
    const uint64_t groups = (n + 255) / 256;
    gx = uint32_t(std::min<uint64_t>(groups, 65535));
    gy = uint32_t((groups + 65534) / 65535);
}
void snake(const GSnake &s, VkBuffer X, VkBuffer Y, uint64_t rows) {
    const uint64_t n = rows * uint64_t(s.C);
    const VkBuffer b[6] = {s.ab.b, kNone, kNone, X, Y, kNone};
    const uint32_t pc[4] = {uint32_t(n), uint32_t(s.C), 0u, uint32_t(s.C)};
    uint32_t gx, gy;
    elementwise_groups(n, gx, gy);
    disp(P_SNAKE, b, pc, 4, gx, gy);
}
void rms(const Buf &w, VkBuffer X, VkBuffer Y, uint32_t n, uint32_t rows, float eps) {
    const VkBuffer b[6] = {w.b, kNone, kNone, X, Y, kNone};
    const uint32_t pc[6] = {n, rows, fbits(eps), 0u, 0u, 0u};
    disp(P_RMS, b, pc, 6, rows, 1);
}

}  // namespace

// =========================================================================================
int gpu_codec_create(const Qwen3Weights &w) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok) return -1;
    try {
        const CodecCfg &c = w.cc;
        GCodec G;
        G.cdim = c.codebook_dim; G.latent = c.latent; G.H = c.hidden; G.heads = c.heads; G.kvh = c.kv_heads;
        G.hd = c.head_dim; G.I = c.inter; G.W = c.window; G.dec = c.decoder_dim; G.spf = c.samples_per_frame();
        G.eps = c.eps;
        G.R = G.W + kGpuCodecMaxFrames;
        G.inv_freq = w.pt.inv_freq;
        if (G.W <= 0 || G.W > 1024 || G.hd > 128 || G.hd % 2) throw std::runtime_error("codec transformer shape not supported on GPU");
        for (int u : c.up_ratios) (void)u;
        G.pre_conv = up_conv(w.pre_conv);
        G.pt_in = w.pt_in.gpu; G.pt_out = w.pt_out.gpu;
        if (G.pt_in < 0 || G.pt_out < 0) throw std::runtime_error("codec transformer projections are not on the GPU");
        for (const LayerW &L : w.pt.layers) {
            GCodec::L l;
            const Mat *ms[7] = {&L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down};
            int *ids[7] = {&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down};
            for (int i = 0; i < 7; ++i) {
                if (ms[i]->gpu < 0) throw std::runtime_error("codec transformer weight is not on the GPU");
                if (ms[i]->bias) throw std::runtime_error("codec transformer with biased projections is not supported on GPU");
                *ids[i] = ms[i]->gpu;
            }
            l.ln1 = vec_buf(L.ln1, size_t(G.H));
            l.ln2 = vec_buf(L.ln2, size_t(G.H));
            l.has_sa = L.attn_scale != nullptr; l.has_sm = L.mlp_scale != nullptr;
            if (l.has_sa) l.s_attn = vec_buf(L.attn_scale, size_t(G.H));
            if (l.has_sm) l.s_mlp = vec_buf(L.mlp_scale, size_t(G.H));
            if (L.qn || L.kn) throw std::runtime_error("codec transformer with qk-norm is not supported on GPU");
            G.layers.push_back(l);
        }
        G.final_norm = vec_buf(w.pt.final_norm, size_t(G.H));
        for (const auto &u : w.up) {
            GCodec::Up U;
            U.tr = u.tr.w.gpu; U.K = u.tr.K;
            if (u.tr.K != u.tr.stride) throw std::runtime_error("upsample transposed conv with K != stride is not supported on GPU");
            U.pw1 = u.pw1.gpu; U.pw2 = u.pw2.gpu;
            if (U.tr < 0 || U.pw1 < 0 || U.pw2 < 0) throw std::runtime_error("codec upsample weights are not on the GPU");
            U.tr_b = vec_buf(u.tr_b, size_t(G.latent));
            U.dwK = u.dw.K;
            std::vector<float> dwp(u.dw.w, u.dw.w + size_t(u.dw.C) * size_t(u.dw.K));
            dwp.insert(dwp.end(), u.dw.b, u.dw.b + u.dw.C);
            U.dwp = vec_buf(dwp);
            std::vector<float> lnp(u.ln_w, u.ln_w + G.latent);
            lnp.insert(lnp.end(), u.ln_b, u.ln_b + G.latent);
            U.lnp = vec_buf(lnp);
            U.gamma = vec_buf(u.gamma, size_t(G.latent));
            G.ups.push_back(U);
        }
        G.conv0 = up_conv(w.conv0);
        for (const auto &d : w.blocks) {
            GCodec::Blk B;
            B.act = up_snake(d.act);
            B.tr = d.tr.w.gpu; B.K = d.tr.K; B.s = d.tr.stride; B.cin = d.tr.cin; B.cout = d.tr.cout;
            if (B.tr < 0) throw std::runtime_error("decoder transposed conv is not on the GPU");
            if (B.K > 2 * B.s) throw std::runtime_error("transposed conv with K > 2*stride is not supported");
            B.tr_b = vec_buf(d.tr_b, size_t(B.cout));
            for (int u = 0; u < 3; ++u) {
                B.a1[u] = up_snake(d.a1[u]); B.a2[u] = up_snake(d.a2[u]);
                B.c1[u] = up_conv(d.c1[u]); B.c2[u] = up_conv(d.c2[u]);
            }
            G.blocks.push_back(std::move(B));
        }
        G.fin_act = up_snake(w.final_act);
        G.fin = up_conv(w.final_conv);

        // scratch: the largest activation of every role for kGpuCodecMaxFrames frames
        const int64_t F = kGpuCodecMaxFrames;
        int64_t len = F, mx = F * std::max<int64_t>({int64_t(G.cdim), int64_t(G.latent), int64_t(G.H), int64_t(G.I)});
        for (const auto &u : G.ups) { len *= u.K; mx = std::max(mx, len * 4 * int64_t(G.latent)); }
        mx = std::max(mx, len * int64_t(G.dec));
        for (const auto &b : G.blocks) {
            mx = std::max(mx, len * int64_t(b.K) * b.cout);  // transposed-conv GEMM output
            len *= b.s;
            mx = std::max(mx, len * int64_t(b.cout));
        }
        for (auto &s : G.S) s = dev_buf(VkDeviceSize(mx) * 4);
        if (std::getenv("VOX_CODEC_DEBUG_STAGE")) G.dbg = host_buf(VkDeviceSize(mx) * 4, &G.dbgp);
        const int64_t qd = int64_t(G.heads) * G.hd, kd = int64_t(G.kvh) * G.hd;
        G.q = dev_buf(VkDeviceSize(F * qd) * 4); G.ao = dev_buf(VkDeviceSize(F * qd) * 4);
        G.k = dev_buf(VkDeviceSize(F * kd) * 4); G.v = dev_buf(VkDeviceSize(F * kd) * 4);
        G.gt = dev_buf(VkDeviceSize(F * G.I) * 4); G.ut = dev_buf(VkDeviceSize(F * G.I) * 4);
        G.xn = dev_buf(VkDeviceSize(F * G.H) * 4); G.hh = dev_buf(VkDeviceSize(F * G.H) * 4);
        G.zb = host_buf(VkDeviceSize(F * G.cdim) * 4, &G.zp);
        G.out = host_buf(VkDeviceSize(F * G.spf) * 4, &G.outp);
        G.csb = host_buf(VkDeviceSize(F * G.hd) * 4, &G.csp);
        g_codecs.push_back(std::move(G));
        return int(g_codecs.size()) - 1;
    } catch (const std::exception &e) {
        g.stats.last_error = std::string("GPU codec decoder: ") + e.what();
        std::fprintf(stderr, "vox: GPU codec decoder unavailable (%s); decoding on CPU\n", e.what());
        return -1;
    }
}

namespace {
GState new_state(int codec) {
    const GCodec &G = g_codecs[size_t(codec)];
    GState S;
    S.codec = codec;
    S.used = true;
    S.pre = make_hist((G.pre_conv.K - 1) * G.pre_conv.dil, G.pre_conv.cin);
    const int64_t kd = int64_t(G.kvh) * G.hd;
    for (size_t l = 0; l < G.layers.size(); ++l) {
        S.kr.push_back(dev_buf(VkDeviceSize(G.R * kd) * 4));
        S.vr.push_back(dev_buf(VkDeviceSize(G.R * kd) * 4));
    }
    for (const auto &u : G.ups) S.dw.push_back(make_hist(u.dwK - 1, G.latent));
    S.c0 = make_hist((G.conv0.K - 1) * G.conv0.dil, G.conv0.cin);
    for (const auto &b : G.blocks) {
        GState::Blk B;
        B.tail = make_hist(b.K - b.s, b.cout);
        for (int u = 0; u < 3; ++u) B.ru[u] = make_hist((b.c1[u].K - 1) * b.c1[u].dil, b.c1[u].cin);
        S.blk.push_back(std::move(B));
    }
    S.fin = make_hist((G.fin.K - 1) * G.fin.dil, G.fin.cin);
    return S;
}
void free_state(GState &S) {
    free_hist(S.pre);
    for (auto &b : S.kr) free_buf(b);
    for (auto &b : S.vr) free_buf(b);
    for (auto &h : S.dw) free_hist(h);
    free_hist(S.c0);
    for (auto &B : S.blk) { free_hist(B.tail); for (auto &h : B.ru) free_hist(h); }
    free_hist(S.fin);
    S = GState{};
}
int put_state(GState &&S) {
    for (size_t i = 0; i < g_states.size(); ++i)
        if (!g_states[i].used) { g_states[i] = std::move(S); return int(i); }
    g_states.push_back(std::move(S));
    return int(g_states.size()) - 1;
}
}  // namespace

int gpu_codec_state_new(int codec) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok || codec < 0 || size_t(codec) >= g_codecs.size()) return -1;
    try {
        begin_cmd();
        GState S = new_state(codec);
        VKC(vkEndCommandBuffer(g.cmd));
        submit_wait(g.cmd);
        return put_state(std::move(S));
    } catch (const std::exception &e) {
        std::fprintf(stderr, "vox: GPU codec state allocation failed: %s\n", e.what());
        return -1;
    }
}

int gpu_codec_state_clone(int id) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok || id < 0 || size_t(id) >= g_states.size() || !g_states[size_t(id)].used) return -1;
    try {
        const GState &src = g_states[size_t(id)];
        GState S;
        S.codec = src.codec; S.used = true; S.len = src.len;
        begin_cmd();
        S.pre = clone_hist(src.pre);
        auto cl = [&](const Buf &b) {
            Buf n = dev_buf(b.bytes);
            VkBufferCopy bc{0, 0, b.bytes};
            vkCmdCopyBuffer(g.cmd, b.b, n.b, 1, &bc);
            return n;
        };
        for (const auto &b : src.kr) S.kr.push_back(cl(b));
        for (const auto &b : src.vr) S.vr.push_back(cl(b));
        for (const auto &h : src.dw) S.dw.push_back(clone_hist(h));
        S.c0 = clone_hist(src.c0);
        for (const auto &B : src.blk) {
            GState::Blk n;
            n.tail = clone_hist(B.tail);
            for (int u = 0; u < 3; ++u) n.ru[u] = clone_hist(B.ru[u]);
            S.blk.push_back(std::move(n));
        }
        S.fin = clone_hist(src.fin);
        VKC(vkEndCommandBuffer(g.cmd));
        submit_wait(g.cmd);
        return put_state(std::move(S));
    } catch (const std::exception &e) {
        std::fprintf(stderr, "vox: GPU codec state clone failed: %s\n", e.what());
        return -1;
    }
}

void gpu_codec_state_free(int id) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok || id < 0 || size_t(id) >= g_states.size() || !g_states[size_t(id)].used) return;
    vkDeviceWaitIdle(g.dev);
    free_state(g_states[size_t(id)]);
}

bool gpu_codec_forward(int codec, int sid, const float *z, int F, std::vector<float> &pcm) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok || codec < 0 || size_t(codec) >= g_codecs.size() || sid < 0 || size_t(sid) >= g_states.size()) return false;
    if (F <= 0) return true;
    if (F > kGpuCodecMaxFrames) return false;
    try {
        GCodec &G = g_codecs[size_t(codec)];
        GState &S = g_states[size_t(sid)];
        const uint32_t T = uint32_t(F);
        const int64_t half = G.hd / 2, qd = int64_t(G.heads) * G.hd, kd = int64_t(G.kvh) * G.hd;
        std::memcpy(G.zp, z, size_t(F) * size_t(G.cdim) * 4);
        for (int64_t i = 0; i < F; ++i)  // RoPE table on the CPU (sin/cos of large positions are inaccurate on GPUs)
            for (int64_t p = 0; p < half; ++p) {
                const float f = G.inv_freq[size_t(p)] * float(S.len + i);
                G.csp[(i * half + p) * 2] = std::cos(f);
                G.csp[(i * half + p) * 2 + 1] = std::sin(f);
            }
        VKC(vkResetDescriptorPool(g.dev, g.dyn, 0));
        begin_cmd();
        VkBuffer A = G.S[0].b, B = G.S[1].b, Cb = G.S[2].b, D = G.S[3].b, E = G.S[4].b;
        static const int dbg_stage = std::getenv("VOX_CODEC_DEBUG_STAGE") ? std::atoi(std::getenv("VOX_CODEC_DEBUG_STAGE")) : 0;
        int stage = 0;
        auto dbg = [&](VkBuffer src, uint64_t n) {  // see Qwen3Weights::codec_forward
            if (++stage != dbg_stage) return false;
            VkBufferCopy bc{0, 0, n * 4};
            vkCmdCopyBuffer(g.cmd, src, G.dbg.b, 1, &bc);
            VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
            vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
            VKC(vkEndCommandBuffer(g.cmd));
            submit_wait(g.cmd);
            pcm.insert(pcm.end(), G.dbgp, G.dbgp + n);
            return true;
        };

        // 1) pre_conv (k=3, causal) → A [F][latent]
        prof_stage("codec.pre_conv");
        conv_stream(G.pre_conv, S.pre, G.zb.b, A, T, 1u);
        if (dbg(A, uint64_t(T) * G.latent)) return true;
        // 2) transformer: input_proj → layers → final norm → output_proj
        prof_stage("codec.transformer");
        gemm(G.pt_in, A, G.hh.b, kNone, T, 1, 1, 0, 1u);
        const float scale = 1.0f / std::sqrt(float(G.hd));
        for (size_t li = 0; li < G.layers.size(); ++li) {
            const GCodec::L &L = G.layers[li];
            rms(L.ln1, G.hh.b, G.xn.b, uint32_t(G.H), T, G.eps);
            gemm(L.q, G.xn.b, G.q.b, kNone, T, 1, 1, 0, 0u);
            gemm(L.k, G.xn.b, G.k.b, kNone, T, 1, 1, 0, 0u);
            gemm(L.v, G.xn.b, G.v.b, kNone, T, 1, 1, 0, 0u);
            {
                const VkBuffer bq[6] = {G.final_norm.b, G.csb.b, kNone, G.q.b, kNone, kNone};
                const uint32_t pq[8] = {uint32_t(G.heads), uint32_t(G.hd), T, 0u, fbits(G.eps), 0u, 0u, 0u};
                disp(P_ROPE, bq, pq, 8, uint32_t(G.heads), T);
                const VkBuffer bk[6] = {G.final_norm.b, G.csb.b, kNone, G.k.b, kNone, kNone};
                const uint32_t pk[8] = {uint32_t(G.kvh), uint32_t(G.hd), T, 0u, fbits(G.eps), 0u, 0u, 0u};
                disp(P_ROPE, bk, pk, 8, uint32_t(G.kvh), T);
            }
            for (int kv = 0; kv < 2; ++kv) {
                const VkBuffer br[6] = {kNone, kNone, kNone, kv ? G.v.b : G.k.b, kv ? S.vr[li].b : S.kr[li].b, kNone};
                const uint32_t pr[4] = {T, uint32_t(kd), uint32_t(S.len), uint32_t(G.R)};
                disp(P_RING, br, pr, 4, uint32_t((T * kd + 255) / 256), 1);
            }
            {
                const VkBuffer ba[6] = {S.kr[li].b, S.vr[li].b, kNone, G.q.b, G.ao.b, kNone};
                const uint32_t pa[8] = {uint32_t(G.heads), uint32_t(G.kvh), uint32_t(G.hd), T, uint32_t(S.len), fbits(scale),
                                        uint32_t(G.W), uint32_t(G.R)};
                disp(P_ATTNW, ba, pa, 8, uint32_t(G.heads), T);
            }
            gemm(L.o, G.ao.b, G.hh.b, kNone, T, 1, 1, 0, L.has_sa ? 4u : 8u, kNone, 0, L.has_sa ? L.s_attn.b : kNone);
            rms(L.ln2, G.hh.b, G.xn.b, uint32_t(G.H), T, G.eps);
            gemm(L.gate, G.xn.b, G.gt.b, kNone, T, 1, 1, 0, 0u);
            gemm(L.up, G.xn.b, G.ut.b, kNone, T, 1, 1, 0, 0u);
            {
                const VkBuffer bs[6] = {kNone, kNone, kNone, G.gt.b, G.ut.b, kNone};
                const uint32_t ps[1] = {uint32_t(T * G.I)};
                disp(P_SILU, bs, ps, 1, uint32_t((T * G.I + 255) / 256), 1);
            }
            gemm(L.down, G.gt.b, G.hh.b, kNone, T, 1, 1, 0, L.has_sm ? 4u : 8u, kNone, 0, L.has_sm ? L.s_mlp.b : kNone);
        }
        rms(G.final_norm, G.hh.b, G.xn.b, uint32_t(G.H), T, G.eps);
        gemm(G.pt_out, G.xn.b, A, kNone, T, 1, 1, 0, 1u);
        S.len += F;
        if (dbg(A, uint64_t(T) * G.latent)) return true;

        // 3) upsample: transposed conv (K == stride → GEMM output is already [len*K][latent]) + ConvNeXt
        prof_stage("codec.upsample");
        uint32_t len = T;
        VkBuffer cur = A;
        for (size_t s = 0; s < G.ups.size(); ++s) {
            const GCodec::Up &U = G.ups[s];
            VkBuffer y = (cur == A) ? B : A;
            gemm(U.tr, cur, y, kNone, len, 1, 1, 0, 1u, U.tr_b.b, uint32_t(G.latent));
            len *= uint32_t(U.K);
            {  // depthwise causal conv k=7 with history
                Hist &h = S.dw[s];
                const VkBuffer b[6] = {U.dwp.b, kNone, kNone, y, Cb, h.b[h.cur].b};
                const uint32_t pc[5] = {len, uint32_t(G.latent), uint32_t(U.dwK), 0u, uint32_t(G.latent * U.dwK)};
                disp(P_DW, b, pc, 5, (len * uint32_t(G.latent) + 255) / 256, 1);
                hist_update(h, y, len);
            }
            {
                const VkBuffer b[6] = {U.lnp.b, kNone, kNone, Cb, D, kNone};
                const uint32_t pc[5] = {uint32_t(G.latent), len, fbits(1e-6f), 0u, uint32_t(G.latent)};
                disp(P_LN, b, pc, 5, len, 1);
            }
            gemm(U.pw1, D, E, kNone, len, 1, 1, 0, 1u | 2u);                   // + bias, GELU
            gemm(U.pw2, E, y, kNone, len, 1, 1, 0, 1u | 4u, kNone, 0, U.gamma.b);  // y += gamma * (pw2 + bias)
            cur = y;
            if (dbg(cur, uint64_t(len) * G.latent)) return true;
        }
        // 4) decoder: conv0 → blocks → Snake → final conv (+ clamp)
        VkBuffer x = (cur == A) ? B : A;
        prof_stage("codec.conv0");
        conv_stream(G.conv0, S.c0, cur, x, len, 1u);
        if (dbg(x, uint64_t(len) * G.dec)) return true;
        for (size_t bi = 0; bi < G.blocks.size(); ++bi) {
            const GCodec::Blk &Bk = G.blocks[bi];
            GState::Blk &Sb = S.blk[bi];
            static const char *kBlk[8] = {"codec.block1", "codec.block2", "codec.block3", "codec.block4",
                                          "codec.block5", "codec.block6", "codec.block7", "codec.block8"};
            prof_stage(kBlk[std::min<size_t>(bi, 7)]);
            VkBuffer sa = (x == A) ? B : A;  // snake(x)
            snake(Bk.act, x, sa, len);
            gemm(Bk.tr, sa, Cb, kNone, len, 1, 1, 0, 0u);  // [len][K*cout]
            VkBuffer y = x;                                // x is no longer needed
            {
                const uint32_t tr = uint32_t(Bk.K - Bk.s);
                const VkBuffer b[6] = {Bk.tr_b.b, Sb.tail.b[1 - Sb.tail.cur].b, kNone, Cb, y, Sb.tail.b[Sb.tail.cur].b};
                const uint32_t pc[5] = {len, uint32_t(Bk.s), uint32_t(Bk.K), uint32_t(Bk.cout), 0u};
                uint32_t gx, gy;
                elementwise_groups(uint64_t(len * Bk.s + tr) * uint64_t(Bk.cout), gx, gy);
                disp(P_OLA, b, pc, 5, gx, gy);
                Sb.tail.cur = 1 - Sb.tail.cur;
            }
            len *= uint32_t(Bk.s);
            for (int u = 0; u < 3; ++u) {
                snake(Bk.a1[u], y, sa, len);
                conv_stream(Bk.c1[u], Sb.ru[u], sa, D, len, 1u);
                snake(Bk.a2[u], D, D, len);
                gemm(Bk.c2[u].mat, D, y, kNone, len, 1, 1, 0, 1u | 8u);  // y += conv1x1(...) + bias
            }
            x = y;
            if (dbg(x, uint64_t(len) * Bk.cout)) return true;
        }
        VkBuffer sa = (x == A) ? B : A;
        prof_stage("codec.final");
        snake(G.fin_act, x, sa, len);
        conv_stream(G.fin, S.fin, sa, G.out.b, len, 1u | 16u);
        VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
        VKC(vkEndCommandBuffer(g.cmd));
        submit_wait(g.cmd);
        pcm.insert(pcm.end(), G.outp, G.outp + len);
        (void)qd;
        return true;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "vox: GPU codec decoder failed: %s\n", e.what());
        return false;
    }
}

}  // namespace vox
