// F5-TTS on the GPU (Vulkan): one DiT step and the Vocos backbone each recorded into ONE command buffer.
//
// The CPU path (F5Model::dit_forward / vocode) moves every linear layer to the GPU and back separately, so on a real
// card the PCIe traffic ate the gain (the 0.12 reports: Vulkan 2.8x slower than the CPU). Here the activations stay in
// device memory for the whole step: up goes only the noisy mel x [N][mel] and the time embedding [dim], down comes the
// prediction [B][N][mel]. The constant part of the DiT input (reference mel + text embedding of both CFG branches) is
// uploaded once per sampling run (gpu_f5_begin). Math mirrors f5.cpp operation by operation; tests compare the two.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "gpu_vk_internal.hpp"

namespace vox {
using namespace vk;

namespace {

struct Buf {
    VkBuffer b = VK_NULL_HANDLE;
    VkDeviceMemory m = VK_NULL_HANDLE;
    VkDeviceSize bytes = 0;
    float *map = nullptr;
};

const VkBufferUsageFlags kUse = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
const VkBuffer kNone = VK_NULL_HANDLE;

void free_buf(Buf &b) {
    if (b.map) vkUnmapMemory(g.dev, b.m);
    if (b.b) vkDestroyBuffer(g.dev, b.b, nullptr);
    if (b.m) dev_mem_free(b.m);
    b = Buf{};
}
// device-local buffer of at least `bytes` (grow-only: an existing larger buffer is kept)
void ensure_dev(Buf &b, VkDeviceSize bytes) {
    bytes = std::max<VkDeviceSize>(16, (bytes + 15) / 16 * 16);
    if (b.b && b.bytes >= bytes) return;
    free_buf(b);
    b.bytes = bytes;
    make_buffer(bytes, kUse, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b.b, b.m);
}
void ensure_host(Buf &b, VkDeviceSize bytes) {
    bytes = std::max<VkDeviceSize>(16, (bytes + 15) / 16 * 16);
    if (b.b && b.bytes >= bytes) return;
    free_buf(b);
    b.bytes = bytes;
    make_buffer(bytes, kUse, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b.b, b.m);
    VKC(vkMapMemory(g.dev, b.m, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&b.map)));
}
Buf const_buf(const float *p, size_t n) {
    Buf b;
    b.bytes = std::max<VkDeviceSize>(16, (n * 4 + 15) / 16 * 16);
    std::vector<float> tmp(b.bytes / 4, 0.f);
    if (p) std::copy(p, p + n, tmp.begin());
    upload_buffer(tmp.data(), b.bytes, b.b, b.m);
    return b;
}
Buf pair_buf(const float *a, const float *b, size_t n) {   // [a; b], b may be null (zeros)
    std::vector<float> v(2 * n, 0.f);
    std::copy(a, a + n, v.begin());
    if (b) std::copy(b, b + n, v.begin() + long(n));
    return const_buf(v.data(), v.size());
}

struct GConv { Buf w, b; int cin = 0, cout = 0, K = 1, groups = 1; bool bias = false; };
GConv up_conv(const GpuF5Conv &c) {
    if (!c.w) throw std::runtime_error("convolution weights missing");
    if (c.cin % c.groups || c.cout % c.groups) throw std::runtime_error("bad convolution groups");
    GConv r;
    r.cin = c.cin; r.cout = c.cout; r.K = c.K; r.groups = c.groups; r.bias = c.b != nullptr;
    r.w = const_buf(c.w, size_t(c.cout) * size_t(c.cin / c.groups) * size_t(c.K));
    r.b = c.b ? const_buf(c.b, size_t(c.cout)) : const_buf(nullptr, 4);
    return r;
}

struct GF5 {
    GpuF5Desc d;
    GConv pos1, pos2, v_embed;
    std::vector<GConv> v_dw;
    std::vector<Buf> v_ln, v_gamma;
    Buf v_ln0, v_fln;
    // run state
    int B = 0, N = 0;
    Buf cat, h, nrm, q, k, v, at, o, ff, ada, fin, cs, xb, tsb, outb;
    // vocos scratch
    Buf vm, vx, vy, vh, vz, vhead;
};
std::vector<GF5> g_f5;

void groups_for(uint64_t n, uint32_t &gx, uint32_t &gy) {
    const uint64_t groups = (n + 255) / 256;
    gx = uint32_t(std::min<uint64_t>(groups, 65535));
    gy = uint32_t((groups + 65534) / 65535);
}
// Y[T][rows] (+)= W X (+ bias); flags: 2 = GELU (erf) after the bias, 8 = accumulate into Y
void gemm(int mat, VkBuffer X, VkBuffer Y, uint32_t T, uint32_t flags) {
    const GpuMat &gm = g.mats[size_t(mat)];
    if (gm.b) flags |= 1u;
    const VkBuffer b[6] = {gm.w, gm.s ? gm.s : kNone, gm.b ? gm.b : kNone, X, Y, kNone};
    // The register-blocked 128×128 GEMM (gemmb.comp) only with VOX_F5_GEMM=new: on real NVIDIA cards (RTX 5050 / 3050) it was
    // 5–14× slower than the 64×64 convgemm tile (0.14.1–0.17.0), so convgemm is the default again until it is measured faster.
    static const bool big_ok = [] { const char *e = std::getenv("VOX_F5_GEMM"); return e && std::strcmp(e, "new") == 0; }();
    if (big_ok && T >= 64 && gm.cols % 8 == 0 && int(gm.type) <= 3 && !((flags & 4u) && gm.s)) {
        const uint32_t pc[5] = {T, uint32_t(gm.rows), uint32_t(gm.cols), flags, uint32_t(gm.rows)};
        disp(P_GEMMB + int(gm.type), b, pc, 5, (T + 127) / 128, uint32_t((gm.rows + 127) / 128));
        return;
    }
    const uint32_t pc[8] = {T, uint32_t(gm.rows), uint32_t(gm.cols), 1u, 1u, 0u, flags, uint32_t(gm.rows)};
    disp(P_GEMM + int(gm.type), b, pc, 8, (T + 63) / 64, uint32_t((gm.rows + 63) / 64));
}
void conv(const GConv &c, VkBuffer X, VkBuffer Y, uint32_t N, uint32_t S, uint32_t flags) {
    if (c.bias) flags |= 1u;
    const VkBuffer b[6] = {c.w.b, kNone, c.b.b, X, Y, kNone};
    const uint32_t cog = uint32_t(c.cout / c.groups);
    const uint32_t pc[6] = {N, uint32_t(c.cin), uint32_t(c.cout), uint32_t(c.K), uint32_t(c.groups), flags};
    disp3(P_F5CONV, b, pc, 6, (N + 63) / 64, uint32_t(c.groups) * ((cog + 63) / 64), S);
}
void layernorm(VkBuffer P, uint32_t off_w, uint32_t off_b, uint32_t mode, VkBuffer X, VkBuffer Y, uint32_t n, uint32_t rows, float eps) {
    const VkBuffer b[6] = {P, kNone, kNone, X, Y, kNone};
    const uint32_t pc[6] = {n, rows, fbits(eps), off_w, off_b, mode};
    disp(P_F5LN, b, pc, 6, rows, 1);
}
void ew(uint32_t mode, VkBuffer G, uint32_t off, VkBuffer X, VkBuffer Y, uint64_t n, uint32_t D, uint32_t N = 0) {
    const VkBuffer b[6] = {G, kNone, kNone, X, Y, kNone};
    const uint32_t pc[5] = {uint32_t(n), D, mode, off, N};
    uint32_t gx, gy;
    groups_for(n, gx, gy);
    disp(P_F5EW, b, pc, 5, gx, gy);
}
// copy host data into a device buffer (own submit)
void upload_into(Buf &dst, const float *src, size_t n, Buf &stage) {
    ensure_host(stage, VkDeviceSize(n) * 4);
    std::memcpy(stage.map, src, n * 4);
    begin_cmd();
    VkBufferCopy bc{0, 0, VkDeviceSize(n) * 4};
    vkCmdCopyBuffer(g.cmd, stage.b, dst.b, 1, &bc);
    VKC(vkEndCommandBuffer(g.cmd));
    submit_wait(g.cmd);
}
Buf g_stage;

}  // namespace

int gpu_f5_create(const GpuF5Desc &d) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok) return -1;
    try {
        if (d.dim_head > 64 || d.dim_head % 2 || d.dim_head <= 0) throw std::runtime_error("head_dim > 64 is not supported by the GPU attention");
        if (d.dim % 1 || d.heads <= 0) throw std::runtime_error("bad DiT shape");
        auto need = [](int id, const char *what) { if (id < 0) throw std::runtime_error(std::string(what) + " is not on the GPU"); };
        need(d.in_proj, "input projection"); need(d.norm_out, "norm_out"); need(d.proj_out, "proj_out");
        for (const auto &b : d.blocks) for (int id : {b.ada, b.q, b.k, b.v, b.o, b.ff1, b.ff2}) need(id, "DiT block weight");
        GF5 G;
        G.d = d;
        G.pos1 = up_conv(d.pos1);
        G.pos2 = up_conv(d.pos2);
        if (d.vocos_dim > 0) {
            need(d.v_head, "Vocos head");
            G.v_embed = up_conv(d.v_embed);
            G.v_ln0 = pair_buf(d.v_ln_w, d.v_ln_b, size_t(d.vocos_dim));
            G.v_fln = pair_buf(d.v_fln_w, d.v_fln_b, size_t(d.vocos_dim));
            for (const auto &b : d.v_blocks) {
                need(b.pw1, "Vocos pwconv1"); need(b.pw2, "Vocos pwconv2");
                G.v_dw.push_back(up_conv(b.dw));
                G.v_ln.push_back(pair_buf(b.ln_w, b.ln_b, size_t(d.vocos_dim)));
                G.v_gamma.push_back(const_buf(b.gamma, size_t(d.vocos_dim)));
            }
        }
        g_f5.push_back(std::move(G));
        return int(g_f5.size()) - 1;
    } catch (const std::exception &e) {
        g.stats.last_error = std::string("F5 on the GPU: ") + e.what();
        std::fprintf(stderr, "vox: F5-TTS graph on the GPU unavailable (%s); only the linear layers use the GPU\n", e.what());
        return -1;
    }
}

bool gpu_f5_begin(int id, int B, int N, const float *cat0, const float *rope) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok || id < 0 || size_t(id) >= g_f5.size() || B <= 0 || N <= 0) return false;
    try {
        GF5 &G = g_f5[size_t(id)];
        const GpuF5Desc &d = G.d;
        const VkDeviceSize R = VkDeviceSize(B) * VkDeviceSize(N), D = VkDeviceSize(d.dim), inner = VkDeviceSize(d.heads) * VkDeviceSize(d.dim_head);
        const VkDeviceSize catw = VkDeviceSize(2 * d.mel_dim + d.text_dim);
        ensure_dev(G.cat, R * catw * 4);
        for (Buf *b : {&G.h, &G.nrm, &G.o}) ensure_dev(*b, R * D * 4);
        for (Buf *b : {&G.q, &G.k, &G.v, &G.at}) ensure_dev(*b, R * inner * 4);
        ensure_dev(G.ff, R * D * VkDeviceSize(d.ff_mult) * 4);
        ensure_dev(G.ada, 6 * D * 4);
        ensure_dev(G.fin, 2 * D * 4);
        ensure_dev(G.cs, VkDeviceSize(N) * VkDeviceSize(d.dim_head) * 4);
        ensure_host(G.xb, VkDeviceSize(N) * VkDeviceSize(d.mel_dim) * 4);
        ensure_host(G.tsb, D * 4);
        ensure_host(G.outb, R * VkDeviceSize(d.mel_dim) * 4);
        upload_into(G.cat, cat0, size_t(R * catw), g_stage);
        upload_into(G.cs, rope, size_t(N) * size_t(d.dim_head), g_stage);
        G.B = B; G.N = N;
        return true;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "vox: F5 GPU buffers: %s\n", e.what());
        return false;
    }
}

bool gpu_f5_step(int id, const float *x, const float *ts, float *out) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok || id < 0 || size_t(id) >= g_f5.size() || g_f5[size_t(id)].B <= 0) return false;
    try {
        GF5 &G = g_f5[size_t(id)];
        const GpuF5Desc &d = G.d;
        const uint32_t B = uint32_t(G.B), N = uint32_t(G.N), R = B * N, D = uint32_t(d.dim), M = uint32_t(d.mel_dim);
        const uint32_t H = uint32_t(d.heads), HD = uint32_t(d.dim_head), INNER = H * HD, FF = D * uint32_t(d.ff_mult);
        std::memcpy(G.xb.map, x, size_t(N) * M * 4);
        std::memcpy(G.tsb.map, ts, size_t(D) * 4);
        VKC(vkResetDescriptorPool(g.dev, g.dyn, 0));
        begin_cmd();
        prof_stage("f5.input");
        ew(2, kNone, uint32_t(2 * d.mel_dim + d.text_dim), G.xb.b, G.cat.b, uint64_t(B) * N * M, M, N);
        gemm(d.in_proj, G.cat.b, G.h.b, R, 0u);
        conv(G.pos1, G.h.b, G.nrm.b, N, B, 2u);        // a = mish(conv1(h))
        conv(G.pos2, G.nrm.b, G.h.b, N, B, 2u | 4u);   // h += mish(conv2(a))
        const float scale = 1.0f / std::sqrt(float(HD));
        for (const auto &L : d.blocks) {
            prof_stage("f5.block");
            rec_mm(L.ada, G.tsb.b, G.ada.b, 1, false);
            // ada = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
            layernorm(G.ada.b, D, 0, 1, G.h.b, G.nrm.b, D, R, 1e-6f);
            gemm(L.q, G.nrm.b, G.q.b, R, 0u);
            gemm(L.k, G.nrm.b, G.k.b, R, 0u);
            gemm(L.v, G.nrm.b, G.v.b, R, 0u);
            {
                const VkBuffer b[6] = {G.cs.b, kNone, kNone, G.q.b, G.k.b, kNone};
                const uint32_t pc[6] = {R, N, H, HD, INNER, uint32_t(std::max(0, d.pe_attn_head))};
                uint32_t gx, gy;
                groups_for(uint64_t(R) * H * (HD / 2), gx, gy);
                disp(P_F5ROPE, b, pc, 6, gx, gy);
            }
            {
                const VkBuffer b[6] = {G.q.b, G.k.b, G.v.b, kNone, G.at.b, kNone};
                const uint32_t pc[4] = {N, HD, INNER, fbits(scale)};
                disp3(P_F5ATTN, b, pc, 4, (N + 31) / 32, H, B);
            }
            gemm(L.o, G.at.b, G.o.b, R, 0u);
            ew(0, G.ada.b, 2 * D, G.o.b, G.h.b, uint64_t(R) * D, D);
            layernorm(G.ada.b, 4 * D, 3 * D, 1, G.h.b, G.nrm.b, D, R, 1e-6f);
            gemm(L.ff1, G.nrm.b, G.ff.b, R, 0u);
            ew(1, kNone, 0, kNone, G.ff.b, uint64_t(R) * FF, FF);
            gemm(L.ff2, G.ff.b, G.o.b, R, 0u);
            ew(0, G.ada.b, 5 * D, G.o.b, G.h.b, uint64_t(R) * D, D);
        }
        prof_stage("f5.output");
        rec_mm(d.norm_out, G.tsb.b, G.fin.b, 1, false);   // fin = [scale, shift]
        layernorm(G.fin.b, 0, D, 1, G.h.b, G.nrm.b, D, R, 1e-6f);
        gemm(d.proj_out, G.nrm.b, G.outb.b, R, 0u);
        host_barrier();
        VKC(vkEndCommandBuffer(g.cmd));
        submit_wait(g.cmd);
        std::memcpy(out, G.outb.map, size_t(R) * M * 4);
        return true;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "vox: F5 DiT step on the GPU failed: %s\n", e.what());
        return false;
    }
}

bool gpu_f5_vocos(int id, const float *mel, int T, std::vector<float> &head) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok || id < 0 || size_t(id) >= g_f5.size() || g_f5[size_t(id)].d.vocos_dim <= 0 || T <= 0) return false;
    try {
        GF5 &G = g_f5[size_t(id)];
        const GpuF5Desc &d = G.d;
        const uint32_t Tu = uint32_t(T), Dv = uint32_t(d.vocos_dim), M = uint32_t(d.mel_dim);
        const uint32_t I = uint32_t(d.vocos_inter), O = uint32_t(d.vocos_out);
        ensure_host(G.vm, VkDeviceSize(T) * M * 4);
        ensure_dev(G.vx, VkDeviceSize(T) * Dv * 4);
        ensure_dev(G.vy, VkDeviceSize(T) * Dv * 4);
        ensure_dev(G.vz, VkDeviceSize(T) * Dv * 4);
        ensure_dev(G.vh, VkDeviceSize(T) * I * 4);
        ensure_host(G.vhead, VkDeviceSize(T) * O * 4);
        std::memcpy(G.vm.map, mel, size_t(T) * M * 4);
        VKC(vkResetDescriptorPool(g.dev, g.dyn, 0));
        begin_cmd();
        prof_stage("f5.vocos");
        conv(G.v_embed, G.vm.b, G.vx.b, Tu, 1, 0u);
        layernorm(G.v_ln0.b, 0, Dv, 0, G.vx.b, G.vx.b, Dv, Tu, 1e-6f);
        for (size_t i = 0; i < d.v_blocks.size(); ++i) {
            const auto &b = d.v_blocks[i];
            conv(G.v_dw[i], G.vx.b, G.vy.b, Tu, 1, 0u);
            layernorm(G.v_ln[i].b, 0, Dv, 0, G.vy.b, G.vy.b, Dv, Tu, 1e-6f);
            gemm(b.pw1, G.vy.b, G.vh.b, Tu, 2u);        // + bias, GELU (erf)
            gemm(b.pw2, G.vh.b, G.vz.b, Tu, 0u);
            ew(0, G.v_gamma[i].b, 0, G.vz.b, G.vx.b, uint64_t(Tu) * Dv, Dv);
        }
        layernorm(G.v_fln.b, 0, Dv, 0, G.vx.b, G.vx.b, Dv, Tu, 1e-6f);
        gemm(d.v_head, G.vx.b, G.vhead.b, Tu, 0u);
        host_barrier();
        VKC(vkEndCommandBuffer(g.cmd));
        submit_wait(g.cmd);
        head.assign(G.vhead.map, G.vhead.map + size_t(T) * O);
        return true;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "vox: Vocos on the GPU failed: %s\n", e.what());
        return false;
    }
}

void gpu_f5_end(int id) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok || id < 0 || size_t(id) >= g_f5.size()) return;
    GF5 &G = g_f5[size_t(id)];
    G.B = G.N = 0;   // buffers are kept (grow-only) for the next chunk
}

}  // namespace vox
