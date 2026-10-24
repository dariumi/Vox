// Vulkan-бэкенд: matmul на GPU. Веса — в device-local буферах (загружаются один раз), активации — через
// host-visible буферы. Один рабочий блок = строка весов × до 4 строк входа (src/shaders/matmul.comp).
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

#include "gpu_vk_internal.hpp"

namespace vox {
namespace vk {
#include "vk_shaders.inc"

Ctx g;

uint32_t find_mem(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g.mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (g.mp.memoryTypes[i].propertyFlags & want) == want) return i;
    throw std::runtime_error("no suitable Vulkan memory type");
}

void make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer &buf, VkDeviceMemory &mem) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VKC(vkCreateBuffer(g.dev, &bi, nullptr, &buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g.dev, buf, &mr);
    const bool dl = (props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    if (dl && g.vram_limit && g.vram_used + mr.size > g.vram_limit) {
        vkDestroyBuffer(g.dev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        throw std::runtime_error("VRAM limit reached: " + std::to_string(g.vram_limit >> 20) + " MiB allowed, " +
                                 std::to_string(g.vram_used >> 20) + " MiB in use, " + std::to_string((mr.size + (1u << 20) - 1) >> 20) + " MiB requested");
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = find_mem(mr.memoryTypeBits, props);
    VkResult r = vkAllocateMemory(g.dev, &ai, nullptr, &mem);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(g.dev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        mem = VK_NULL_HANDLE;
        throw std::runtime_error(std::string("vkAllocateMemory failed (") + (r == VK_ERROR_OUT_OF_DEVICE_MEMORY ? "out of device memory" : "VkResult " + std::to_string(int(r))) +
                                 "), " + std::to_string(mr.size >> 20) + " MiB requested, " + std::to_string(g.vram_used >> 20) + " MiB already used by vox");
    }
    VKC(vkBindBufferMemory(g.dev, buf, mem, 0));
    if (dl) { g.dev_mem[mem] = mr.size; g.vram_used += mr.size; }
}

void dev_mem_free(VkDeviceMemory m) {
    if (!m) return;
    auto it = g.dev_mem.find(m);
    if (it != g.dev_mem.end()) { g.vram_used -= std::min(g.vram_used, it->second); g.dev_mem.erase(it); }
    vkFreeMemory(g.dev, m, nullptr);
}

void prof_collect() {
    if (!g.prof || g.qn < 2) return;
    std::vector<uint64_t> ts(g.qn);
    if (vkGetQueryPoolResults(g.dev, g.qpool, 0, g.qn, ts.size() * 8, ts.data(), 8,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) return;
    for (uint32_t i = 1; i < g.qn; ++i) {
        const std::string &lab = g.qlabels[i];
        auto it = std::find_if(g.prof_acc.begin(), g.prof_acc.end(), [&](const auto &p) { return p.first == lab; });
        if (it == g.prof_acc.end()) { g.prof_acc.push_back({lab, {}}); it = g.prof_acc.end() - 1; }
        it->second.ns += double(ts[i] - ts[i - 1]) * g.ts_ns;
        it->second.calls++;
    }
}

void prof_print() {
    if (!g.prof || g.prof_acc.empty()) return;
    auto v = g.prof_acc;
    std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) { return a.second.ns > b.second.ns; });
    double total = 0;
    for (auto &p : v) total += p.second.ns;
    std::fprintf(stderr, "[vox gpu profile] %s: GPU busy %.1f ms in %.0f submits\n  %-28s %8s %10s %9s %6s\n", g.name.c_str(),
                 total / 1e6, g.submits, "stage/shader", "calls", "total ms", "avg us", "share");
    for (auto &p : v)
        std::fprintf(stderr, "  %-28s %8llu %10.2f %9.1f %5.1f%%\n", p.first.c_str(), (unsigned long long)p.second.calls, p.second.ns / 1e6,
                     p.second.ns / 1e3 / double(std::max<uint64_t>(1, p.second.calls)), 100.0 * p.second.ns / std::max(1.0, total));
}

void submit_wait(VkCommandBuffer cmd) {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VKC(vkQueueSubmit(g.q, 1, &si, g.fence));
    VKC(vkWaitForFences(g.dev, 1, &g.fence, VK_TRUE, UINT64_MAX));
    VKC(vkResetFences(g.dev, 1, &g.fence));
    g.submits += 1;
    prof_collect();
    g.qn = 0;
}

void begin_cmd() {
    VKC(vkResetCommandBuffer(g.cmd, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKC(vkBeginCommandBuffer(g.cmd, &bi));
    if (g.prof) {
        vkCmdResetQueryPool(g.cmd, g.qpool, 0, g.qcap);
        g.qlabels.assign(1, "start");
        vkCmdWriteTimestamp(g.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g.qpool, 0);
        g.qn = 1;
    }
}

void Ctx::shutdown() {
    if (!dev) { if (inst) vkDestroyInstance(inst, nullptr); inst = VK_NULL_HANDLE; return; }
    vkDeviceWaitIdle(dev);
    prof_print();
    if (qpool) vkDestroyQueryPool(dev, qpool, nullptr);
    for (auto &m : mats) {
        for (VkBuffer b : {m.w, m.s, m.b}) if (b) vkDestroyBuffer(dev, b, nullptr);
        for (VkDeviceMemory x : {m.mw, m.ms, m.mb}) if (x) vkFreeMemory(dev, x, nullptr);
    }
    mats.clear();
    for (VkBuffer b : {xb, yb, dummy, stage}) if (b) vkDestroyBuffer(dev, b, nullptr);
    for (VkDeviceMemory x : {xm, ym, dm, stm}) if (x) vkFreeMemory(dev, x, nullptr);
    if (dpool) vkDestroyDescriptorPool(dev, dpool, nullptr);
    for (VkBuffer b : {csb, sh, sxn, sq, sao, sg, su}) if (b) vkDestroyBuffer(dev, b, nullptr);
    for (VkDeviceMemory x : {csm, mh, mxn, mq, mao, mg, mup}) if (x) vkFreeMemory(dev, x, nullptr);
    for (auto &x : xf) { if (x.norms) vkDestroyBuffer(dev, x.norms, nullptr); if (x.nm) vkFreeMemory(dev, x.nm, nullptr); }
    for (auto &sq_ : seqs) {
        for (VkBuffer b : sq_.k) if (b) vkDestroyBuffer(dev, b, nullptr);
        for (VkBuffer b : sq_.v) if (b) vkDestroyBuffer(dev, b, nullptr);
        for (VkDeviceMemory m : sq_.km) if (m) vkFreeMemory(dev, m, nullptr);
        for (VkDeviceMemory m : sq_.vm) if (m) vkFreeMemory(dev, m, nullptr);
    }
    if (dyn) vkDestroyDescriptorPool(dev, dyn, nullptr);
    for (auto p : pipe) if (p) vkDestroyPipeline(dev, p, nullptr);
    if (sm) vkDestroyShaderModule(dev, sm, nullptr);
    for (auto m : extra_modules) if (m) vkDestroyShaderModule(dev, m, nullptr);
    if (pl) vkDestroyPipelineLayout(dev, pl, nullptr);
    if (dsl) vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    if (fence) vkDestroyFence(dev, fence, nullptr);
    if (pool) vkDestroyCommandPool(dev, pool, nullptr);
    vkDestroyDevice(dev, nullptr);
    if (inst) vkDestroyInstance(inst, nullptr);
    dev = VK_NULL_HANDLE; inst = VK_NULL_HANDLE; ok = false;
}

void init_impl(const std::string &spec) {
    int want_index = -1;
    auto colon = spec.find(':');
    if (colon != std::string::npos) want_index = std::stoi(spec.substr(colon + 1));
    if (const char *e = std::getenv("VOX_VK_DEVICE")) want_index = std::atoi(e);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vox";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VKC(vkCreateInstance(&ici, nullptr, &g.inst));

    uint32_t n = 0;
    VKC(vkEnumeratePhysicalDevices(g.inst, &n, nullptr));
    if (!n) throw std::runtime_error("no Vulkan devices found");
    std::vector<VkPhysicalDevice> devs(n);
    VKC(vkEnumeratePhysicalDevices(g.inst, &n, devs.data()));
    int best = -1, best_score = -1;
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
        if (want_index == int(i)) { best = int(i); break; }
        if (want_index < 0 && score > best_score) { best = int(i); best_score = score; }
    }
    if (best < 0 || best >= int(n)) throw std::runtime_error("Vulkan device index out of range (found " + std::to_string(n) + ")");
    g.phys = devs[size_t(best)];
    g.dev_index = best;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g.phys, &props);
    g.name = props.deviceName;
    g.max_range = props.limits.maxStorageBufferRange;
    vkGetPhysicalDeviceMemoryProperties(g.phys, &g.mp);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nq, qfs.data());
    g.qf = UINT32_MAX;
    for (uint32_t i = 0; i < nq; ++i)
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g.qf = i; break; }
    if (g.qf == UINT32_MAX) throw std::runtime_error("no compute queue");
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = g.qf;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    VKC(vkCreateDevice(g.phys, &di, nullptr, &g.dev));
    vkGetDeviceQueue(g.dev, g.qf, 0, &g.q);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = g.qf;
    VKC(vkCreateCommandPool(g.dev, &pci, nullptr, &g.pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = g.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VKC(vkAllocateCommandBuffers(g.dev, &cai, &g.cmd));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKC(vkCreateFence(g.dev, &fi, nullptr, &g.fence));

    VkDescriptorSetLayoutBinding bind[6];
    for (uint32_t i = 0; i < 6; ++i) bind[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 6;
    dli.pBindings = bind;
    VKC(vkCreateDescriptorSetLayout(g.dev, &dli, nullptr, &g.dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g.dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VKC(vkCreatePipelineLayout(g.dev, &pli, nullptr, &g.pl));
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    {  // subgroup arithmetic in compute shaders → fast reduction, otherwise the variant without subgroups
        VkPhysicalDeviceSubgroupProperties sp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sp;
        vkGetPhysicalDeviceProperties2(g.phys, &p2);
        g.subgroups = (sp.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) && (sp.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) &&
                      (sp.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) && !std::getenv("VOX_VK_NO_SUBGROUPS");
        g.subgroup_size = sp.subgroupSize;
    }
    smi.codeSize = g.subgroups ? sizeof(k_matmul_spv) : sizeof(k_matmul_nosg_spv);
    smi.pCode = g.subgroups ? k_matmul_spv : k_matmul_nosg_spv;
    VKC(vkCreateShaderModule(g.dev, &smi, nullptr, &g.sm));
    auto make_module = [&](const uint32_t *code, size_t bytes) {
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = bytes;
        mi.pCode = code;
        VkShaderModule m;
        VKC(vkCreateShaderModule(g.dev, &mi, nullptr, &m));
        return m;
    };
    auto make_pipe = [&](VkShaderModule m, const uint32_t *spec) {
        VkSpecializationMapEntry me{0, 0, 4};
        VkSpecializationInfo si{1, &me, 4, spec};
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", spec ? &si : nullptr};
        cpi.layout = g.pl;
        VkPipeline p;
        VKC(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &p));
        return p;
    };
    for (uint32_t t = 0; t < 5; ++t) { uint32_t wtype = t; g.pipe[t] = make_pipe(g.sm, &wtype); }
    g.extra_modules[0] = make_module(k_rmsnorm_spv, sizeof(k_rmsnorm_spv));
    g.extra_modules[1] = make_module(k_qkrope_spv, sizeof(k_qkrope_spv));
    g.extra_modules[2] = make_module(k_attention_spv, sizeof(k_attention_spv));
    g.extra_modules[3] = make_module(k_silumul_spv, sizeof(k_silumul_spv));
    for (int i = 0; i < 4; ++i) g.pipe[P_RMS + i] = make_pipe(g.extra_modules[i], nullptr);
    g.extra_modules[4] = make_module(k_convgemm_spv, sizeof(k_convgemm_spv));
    for (uint32_t t = 0; t < 5; ++t) { uint32_t wtype = t; g.pipe[P_GEMM + int(t)] = make_pipe(g.extra_modules[4], &wtype); }
    {
        const uint32_t *codes[10] = {k_snake_spv, k_dwconv_spv, k_layernorm_spv, k_ola_spv, k_hist_spv, k_attnw_spv, k_ringput_spv,
                                     k_qkvsplit_spv, k_swiglu_spv, k_sample_spv};
        const size_t sizes[10] = {sizeof(k_snake_spv), sizeof(k_dwconv_spv), sizeof(k_layernorm_spv), sizeof(k_ola_spv),
                                  sizeof(k_hist_spv), sizeof(k_attnw_spv), sizeof(k_ringput_spv), sizeof(k_qkvsplit_spv),
                                  sizeof(k_swiglu_spv), sizeof(k_sample_spv)};
        for (int i = 0; i < 10; ++i) {
            g.extra_modules[5 + i] = make_module(codes[i], sizes[i]);
            g.pipe[P_SNAKE + i] = make_pipe(g.extra_modules[5 + i], nullptr);
        }
        g.extra_modules[15] = make_module(k_gather_spv, sizeof(k_gather_spv));
        for (uint32_t t = 0; t < 3; ++t) { uint32_t wtype = t; g.pipe[P_GATHER + int(t)] = make_pipe(g.extra_modules[15], &wtype); }
        const uint32_t *f5c[5] = {k_f5attn_spv, k_f5conv_spv, k_f5ln_spv, k_f5ew_spv, k_f5rope_spv};
        const size_t f5s[5] = {sizeof(k_f5attn_spv), sizeof(k_f5conv_spv), sizeof(k_f5ln_spv), sizeof(k_f5ew_spv), sizeof(k_f5rope_spv)};
        for (int i = 0; i < 5; ++i) {
            g.extra_modules[16 + i] = make_module(f5c[i], f5s[i]);
            g.pipe[P_F5ATTN + i] = make_pipe(g.extra_modules[16 + i], nullptr);
        }
        g.extra_modules[21] = make_module(k_gemmb_spv, sizeof(k_gemmb_spv));
        for (uint32_t t = 0; t < 4; ++t) { uint32_t wtype = t; g.pipe[P_GEMMB + int(t)] = make_pipe(g.extra_modules[21], &wtype); }
    }
    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 * 4096};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 4096;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &dps;
    VKC(vkCreateDescriptorPool(g.dev, &dpi, nullptr, &g.dpool));

    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    make_buffer(kScratchFloats * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, g.xb, g.xm);
    make_buffer(kScratchFloats * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, g.yb, g.ym);
    make_buffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, g.dummy, g.dm);
    VKC(vkMapMemory(g.dev, g.xm, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&g.xp)));
    VKC(vkMapMemory(g.dev, g.ym, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&g.yp)));
    VkDescriptorPoolSize dps2{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 * 4096};
    VkDescriptorPoolCreateInfo dpi2{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi2.maxSets = 4096;
    dpi2.poolSizeCount = 1;
    dpi2.pPoolSizes = &dps2;
    VKC(vkCreateDescriptorPool(g.dev, &dpi2, nullptr, &g.dyn));
    make_buffer(size_t(kMaxM) * 64 * 2 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, g.csb, g.csm);
    VKC(vkMapMemory(g.dev, g.csm, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&g.csp)));
    g.mats.reserve(4096);
    {  // профайлер GPU (VOX_PROFILE=gpu)
        const char *pe = std::getenv("VOX_PROFILE");
        if (pe && std::strstr(pe, "gpu")) {
            if (props.limits.timestampComputeAndGraphics || qfs[g.qf].timestampValidBits > 0) {
                VkQueryPoolCreateInfo qi2{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                qi2.queryType = VK_QUERY_TYPE_TIMESTAMP;
                qi2.queryCount = g.qcap = 16384;
                VKC(vkCreateQueryPool(g.dev, &qi2, nullptr, &g.qpool));
                g.ts_ns = props.limits.timestampPeriod;
                g.prof = true;
            } else {
                std::fprintf(stderr, "vox: VOX_PROFILE=gpu: this device has no compute timestamps\n");
            }
        }
    }
    g.ok = true;
}

// Бит 2 флагов matmul-шейдера: cols кратно числу элементов в 16 байтах → веса читаются по uvec4.
uint32_t vec_flag(const GpuMat &m) {
    const int64_t epc = m.type == WType::F32 ? 4 : (m.type == WType::Q8 ? 16 : m.type == WType::Q4 ? 32 : 8);
    return m.cols % epc == 0 ? 4u : 0u;
}

// Буфер в видеопамяти с данными из CPU (через staging).
void upload_buffer(const void *data, VkDeviceSize bytes, VkBuffer &buf, VkDeviceMemory &mem) {
    const VkDeviceSize padded = (bytes + 15) / 16 * 16;  // шейдер читает веса по uvec4
    make_buffer(padded, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buf, mem);
    if (g.stage_size < padded) {
        if (g.stage) {
            vkUnmapMemory(g.dev, g.stm);
            vkDestroyBuffer(g.dev, g.stage, nullptr);
            vkFreeMemory(g.dev, g.stm, nullptr);
        }
        make_buffer(padded, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, g.stage, g.stm);
        VKC(vkMapMemory(g.dev, g.stm, 0, VK_WHOLE_SIZE, 0, &g.stp));
        g.stage_size = padded;
    }
    std::memset(g.stp, 0, size_t(padded));
    std::memcpy(g.stp, data, size_t(bytes));
    begin_cmd();
    VkBufferCopy bc{0, 0, padded};
    vkCmdCopyBuffer(g.cmd, g.stage, buf, 1, &bc);
    VKC(vkEndCommandBuffer(g.cmd));
    submit_wait(g.cmd);
}
}  // namespace vk
using namespace vk;

bool gpu_init(const std::string &device, std::string &err) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (g.ok) return true;
    if (!g.init_error.empty()) { err = g.init_error; return false; }
    try {
        init_impl(device);
        return true;
    } catch (const std::exception &e) {
        g.init_error = std::string("Vulkan init failed: ") + e.what();
        err = g.init_error;
        g.shutdown();
        return false;
    }
}
bool gpu_active() { return g.ok; }
GpuStats gpu_stats() { std::lock_guard<std::mutex> lk(g.mu); return g.stats; }
namespace {
// Размер и свободный объём device-local кучи; free — по VK_EXT_memory_budget, если он есть (бюджет процесса минус его использование).
void heap_info(VkPhysicalDevice d, bool budget_ext, uint64_t &total, uint64_t &free_) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    if (budget_ext) mp2.pNext = &bp;
    vkGetPhysicalDeviceMemoryProperties2(d, &mp2);
    total = 0; free_ = 0;
    uint32_t best = 0;
    for (uint32_t h = 0; h < mp2.memoryProperties.memoryHeapCount; ++h)
        if ((mp2.memoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && mp2.memoryProperties.memoryHeaps[h].size >= total) {
            total = mp2.memoryProperties.memoryHeaps[h].size;
            best = h;
        }
    free_ = total;
    if (budget_ext && bp.heapBudget[best] > 0) free_ = bp.heapBudget[best] > bp.heapUsage[best] ? bp.heapBudget[best] - bp.heapUsage[best] : 0;
    free_ = std::min(free_, total);
}
}  // namespace

std::vector<GpuDeviceInfo> gpu_enumerate() {
    std::vector<GpuDeviceInfo> out;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vox"; app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return out;
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());
    for (uint32_t i = 0; i < n; ++i) {
        GpuDeviceInfo d;
        d.index = int(i);
        VkPhysicalDeviceSubgroupProperties sp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sp;
        vkGetPhysicalDeviceProperties2(devs[i], &p2);
        const VkPhysicalDeviceProperties &p = p2.properties;
        d.name = p.deviceName;
        d.type = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "discrete" : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "integrated"
               : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? "virtual" : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? "cpu" : "other";
        d.software = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU || d.name.find("llvmpipe") != std::string::npos || d.name.find("lavapipe") != std::string::npos;
        d.unified_memory = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || d.software;
        d.api_version = p.apiVersion; d.vendor_id = p.vendorID; d.driver_version = p.driverVersion;
        d.max_storage_range = p.limits.maxStorageBufferRange;
        d.subgroup_arith = (sp.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) && (sp.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(devs[i], nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> ext(ne);
        vkEnumerateDeviceExtensionProperties(devs[i], nullptr, &ne, ext.data());
        for (const auto &e : ext) {
            if (!std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) d.memory_budget = true;
            if (!std::strcmp(e.extensionName, "VK_KHR_cooperative_matrix") || !std::strcmp(e.extensionName, "VK_NV_cooperative_matrix")) d.coop_matrix = true;
        }
        heap_info(devs[i], d.memory_budget, d.vram_total, d.vram_free);
        out.push_back(std::move(d));
    }
    vkDestroyInstance(inst, nullptr);
    return out;
}

std::string gpu_list_devices() {
    auto devs = gpu_enumerate();
    if (devs.empty()) return "no Vulkan devices\n";
    std::string out;
    for (const auto &d : devs) {
        char buf[640];
        const char *t = d.type == "cpu" ? "cpu(software)" : d.type.c_str();
        std::snprintf(buf, sizeof buf, "vulkan:%d  %s  [%s]  VRAM %.1f GiB, free %.1f GiB%s%s\n", d.index, d.name.c_str(), t,
                      double(d.vram_total) / (1024.0 * 1024 * 1024), double(d.vram_free) / (1024.0 * 1024 * 1024),
                      d.coop_matrix ? ", coopmat" : "", d.memory_budget ? "" : " (free: estimate)");
        out += buf;
    }
    return out;
}

void gpu_set_vram_limit(uint64_t bytes) { std::lock_guard<std::mutex> lk(g.mu); g.vram_limit = bytes; }
GpuMemInfo gpu_mem_info() {
    std::lock_guard<std::mutex> lk(g.mu);
    GpuMemInfo m;
    if (!g.ok) return m;
    m.used = g.vram_used; m.limit = g.vram_limit; m.device_index = g.dev_index;
    uint32_t ne = 0;
    bool budget = false;
    vkEnumerateDeviceExtensionProperties(g.phys, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> ext(ne);
    vkEnumerateDeviceExtensionProperties(g.phys, nullptr, &ne, ext.data());
    for (const auto &e : ext) budget = budget || !std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    heap_info(g.phys, budget, m.device_total, m.device_free);
    return m;
}
std::string gpu_device_name() { return g.name; }

namespace vk {
// Uploads one GPU matrix made of the rows of `parts` stacked (same type and width). Caller holds g.mu. -1 on failure.
int upload_parts(const std::vector<const Mat *> &parts) {
    if (!g.ok || parts.empty()) return -1;
    const Mat &P0 = *parts[0];
    int64_t rows = 0;
    bool all_bias = true, any_bias = false;
    for (const Mat *p : parts) {
        if (p->type != P0.type || p->cols != P0.cols || !p->valid()) return -1;
        rows += p->rows;
        all_bias = all_bias && p->bias; any_bias = any_bias || p->bias;
    }
    if (any_bias && !all_bias) return -1;
    if ((P0.type == WType::Q8 || P0.type == WType::Q4) && P0.cols % 32) return -1;
    const size_t row_bytes = mat_row_bytes(P0.type, P0.cols);
    const uint64_t bytes = uint64_t(rows) * uint64_t(row_bytes);
    if (g.mats.size() >= 4000 || bytes > g.max_range || rows > 4 * 65535) {  // не влезает в лимиты устройства → считаем на CPU
        g.stats.mats_failed++; g.stats.bytes_failed += bytes;
        g.stats.last_error = g.mats.size() >= 4000 ? "too many matrices" : "matrix exceeds maxStorageBufferRange / row limit";
        return -1;
    }
    try {
        GpuMat m;
        m.type = P0.type; m.rows = rows; m.cols = P0.cols; m.bias = all_bias;
        auto cat = [&](auto get, size_t row_bytes) {
            if (parts.size() == 1) return std::vector<uint8_t>();
            std::vector<uint8_t> v;
            v.reserve(size_t(rows) * row_bytes);
            for (const Mat *p : parts) { const uint8_t *src = get(*p); v.insert(v.end(), src, src + size_t(p->rows) * row_bytes); }
            return v;
        };
        const size_t wrow = row_bytes;
        auto wd = cat([](const Mat &p) { return p.data; }, wrow);
        upload_buffer(parts.size() == 1 ? P0.data : wd.data(), bytes, m.w, m.mw);
        if (P0.type == WType::Q8 || P0.type == WType::Q4) {
            const size_t srow = size_t(P0.cols / 32) * 4;
            auto sd = cat([](const Mat &p) { return reinterpret_cast<const uint8_t *>(p.scale); }, srow);
            upload_buffer(parts.size() == 1 ? reinterpret_cast<const void *>(P0.scale) : sd.data(), uint64_t(rows) * srow, m.s, m.ms);
        }
        if (all_bias) {
            auto bd = cat([](const Mat &p) { return reinterpret_cast<const uint8_t *>(p.bias); }, 4);
            upload_buffer(parts.size() == 1 ? reinterpret_cast<const void *>(P0.bias) : bd.data(), uint64_t(rows) * 4, m.b, m.mb);
        }
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = g.dpool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &g.dsl;
        VKC(vkAllocateDescriptorSets(g.dev, &ai, &m.ds));
        VkDescriptorBufferInfo bi[6] = {{m.w, 0, VK_WHOLE_SIZE}, {m.s ? m.s : g.dummy, 0, VK_WHOLE_SIZE},
                                        {m.b ? m.b : g.dummy, 0, VK_WHOLE_SIZE}, {g.xb, 0, VK_WHOLE_SIZE}, {g.yb, 0, VK_WHOLE_SIZE}, {g.dummy, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet wr[6];
        for (uint32_t i = 0; i < 6; ++i)
            wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m.ds, i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[i], nullptr};
        vkUpdateDescriptorSets(g.dev, 6, wr, 0, nullptr);
        g.mats.push_back(m);
        g.stats.mats_ok += int64_t(parts.size()); g.stats.bytes += bytes;
        return int(g.mats.size()) - 1;
    } catch (const std::exception &e) {
        g.stats.mats_failed += int64_t(parts.size()); g.stats.bytes_failed += bytes; g.stats.last_error = e.what();
        if (!g.warned) { std::fprintf(stderr, "vox: GPU upload failed (%s); some weights will stay on CPU\n", e.what()); g.warned = true; }
        return -1;
    }
}
}  // namespace vk

int gpu_upload(const Mat &W) {
    std::lock_guard<std::mutex> lk(g.mu);
    return upload_parts({&W});
}

bool gpu_matmul(const Mat &W, const float *x, int64_t m, float *y) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok || W.gpu < 0 || size_t(W.gpu) >= g.mats.size()) return false;
    try {
        const GpuMat &gm = g.mats[size_t(W.gpu)];
        const int64_t rows = gm.rows, cols = gm.cols;
        prof_stage("matmul(single)");
        const int64_t chunk = std::max<int64_t>(1, std::min<int64_t>(int64_t(kScratchFloats) / std::max(rows, cols), 65535 * 4));
        for (int64_t i0 = 0; i0 < m; i0 += chunk) {
            const int64_t mc = std::min(chunk, m - i0);
            std::memcpy(g.xp, x + i0 * cols, size_t(mc * cols) * sizeof(float));
            begin_cmd();
            vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe[int(gm.type)]);
            vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &gm.ds, 0, nullptr);
            const uint32_t pc[6] = {uint32_t(rows), uint32_t(cols), uint32_t(mc), (gm.bias ? 1u : 0u) | vec_flag(gm), 0u, 0u};
            vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 24, pc);
            vkCmdDispatch(g.cmd, uint32_t((rows + 3) / 4), uint32_t((mc + 3) / 4), 1);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
            vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            VKC(vkEndCommandBuffer(g.cmd));
            submit_wait(g.cmd);
            std::memcpy(y + i0 * rows, g.yp, size_t(mc * rows) * sizeof(float));
        }
        return true;
    } catch (const std::exception &e) {
        if (!g.warned) { std::fprintf(stderr, "vox: GPU matmul failed (%s); falling back to CPU\n", e.what()); g.warned = true; }
        return false;
    }
}


// ====================== трансформер целиком на GPU ======================
namespace vk {
VkDescriptorSet dyn_set(const VkBuffer (&b)[6]) {
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = g.dyn;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &g.dsl;
    VkDescriptorSet ds;
    VKC(vkAllocateDescriptorSets(g.dev, &ai, &ds));
    VkDescriptorBufferInfo bi[6];
    VkWriteDescriptorSet wr[6];
    for (uint32_t i = 0; i < 6; ++i) {
        bi[i] = {b[i] ? b[i] : g.dummy, 0, VK_WHOLE_SIZE};
        wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[i], nullptr};
    }
    vkUpdateDescriptorSets(g.dev, 6, wr, 0, nullptr);
    return ds;
}
void barrier_cc() {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}
const char *kPipeNames[] = {"matmul.f32", "matmul.f16", "matmul.bf16", "matmul.q8", "matmul.q4", "rmsnorm", "qkrope", "attention",
                            "silumul", "gemm.f32", "gemm.f16", "gemm.bf16", "gemm.q8", "gemm.q4", "snake", "dwconv", "layernorm",
                            "overlap_add", "history", "attn_window", "ring_put", "qkv_split", "swiglu", "sample",
                            "gather.f32", "gather.f16", "gather.bf16", "f5.attention", "f5.conv", "f5.layernorm", "f5.elementwise",
                            "f5.rope", "gemm8x8.f32", "gemm8x8.f16", "gemm8x8.bf16", "gemm8x8.q8"};
static_assert(sizeof(kPipeNames) / sizeof(kPipeNames[0]) == kNumPipes, "kPipeNames must list every pipeline in enum Pipe order");
void disp(int pipe, const VkBuffer (&b)[6], const uint32_t *pc, uint32_t pc_words, uint32_t gx, uint32_t gy) {
    disp3(pipe, b, pc, pc_words, gx, gy, 1);
}
void disp3(int pipe, const VkBuffer (&b)[6], const uint32_t *pc, uint32_t pc_words, uint32_t gx, uint32_t gy, uint32_t gz) {
    VkDescriptorSet ds = dyn_set(b);
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe[pipe]);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_words * 4, pc);
    vkCmdDispatch(g.cmd, gx, gy, gz);
    barrier_cc();
    if (g.prof && g.qn < g.qcap) {
        vkCmdWriteTimestamp(g.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g.qpool, g.qn++);
        g.qlabels.push_back(g.prof_stage + "/" + kPipeNames[pipe]);
    }
}
uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
void ensure_scratch(const GpuXfmr &X) {
    const size_t qd = size_t(X.heads) * size_t(X.hd), kd = size_t(X.kv) * size_t(X.hd);
    const size_t H = size_t(kMaxM) * size_t(X.H) * 4, Q = size_t(kMaxM) * qd * 4,
                 I = size_t(kMaxM) * std::max(2 * size_t(X.I), qd + 2 * kd) * 4;  // слитые выходы gate|up и q|k|v
    if (g.sh && H <= g.scratch_h && Q <= g.scratch_q && I <= g.scratch_i) return;
    vkDeviceWaitIdle(g.dev);
    for (VkBuffer b : {g.sh, g.sxn, g.sq, g.sao, g.sg, g.su}) if (b) vkDestroyBuffer(g.dev, b, nullptr);
    for (VkDeviceMemory m : {g.mh, g.mxn, g.mq, g.mao, g.mg, g.mup}) if (m) dev_mem_free(m);
    g.scratch_h = std::max(H, g.scratch_h); g.scratch_q = std::max(Q, g.scratch_q); g.scratch_i = std::max(I, g.scratch_i);
    const VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    const VkMemoryPropertyFlags dl = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    make_buffer(g.scratch_h, u, dl, g.sh, g.mh);
    make_buffer(g.scratch_h, u, dl, g.sxn, g.mxn);
    make_buffer(g.scratch_q, u, dl, g.sq, g.mq);
    make_buffer(g.scratch_q, u, dl, g.sao, g.mao);
    make_buffer(g.scratch_i, u, dl, g.sg, g.mg);
    make_buffer(g.scratch_i, u, dl, g.su, g.mup);
}

const VkBuffer kNoBuf = VK_NULL_HANDLE;

// y (+)= W x for m rows. m <= 8: GEMV (weights streamed once per 4 rows, best for decoding);
// m > 8: tiled GEMM (weights read once per 64 rows) — the prefill path.
void rec_mm(int mat, VkBuffer in, VkBuffer out, int64_t m, bool accum, int64_t y_off) {
    const GpuMat &gm = g.mats[size_t(mat)];
    if (m > 8 && y_off == 0) {
        const VkBuffer b[6] = {gm.w, gm.s ? gm.s : kNoBuf, gm.b ? gm.b : kNoBuf, in, out, kNoBuf};
        const uint32_t pc[8] = {uint32_t(m), uint32_t(gm.rows), uint32_t(gm.cols), 1u, 1u, 0u, (gm.bias ? 1u : 0u) | (accum ? 8u : 0u),
                                uint32_t(gm.rows)};
        disp(P_GEMM + int(gm.type), b, pc, 8, uint32_t((m + 63) / 64), uint32_t((gm.rows + 63) / 64));
        return;
    }
    const VkBuffer b[6] = {gm.w, gm.s ? gm.s : kNoBuf, gm.b ? gm.b : kNoBuf, in, out, kNoBuf};
    const uint32_t pc[6] = {uint32_t(gm.rows), uint32_t(gm.cols), uint32_t(m), (gm.bias ? 1u : 0u) | (accum ? 2u : 0u) | vec_flag(gm), 0u,
                            uint32_t(y_off)};
    disp(P_MM + int(gm.type), b, pc, 6, uint32_t((gm.rows + 3) / 4), uint32_t((m + 3) / 4));
}

void rec_rms(const GpuXfmr &X, size_t w_off, VkBuffer in, int64_t x_off, VkBuffer out, int64_t rows) {
    const VkBuffer b[6] = {X.norms, kNoBuf, kNoBuf, in, out, kNoBuf};
    const uint32_t pc[6] = {uint32_t(X.H), uint32_t(rows), fbits(X.eps), uint32_t(x_off), 0u, uint32_t(w_off)};
    disp(P_RMS, b, pc, 6, uint32_t(rows), 1);
}

// All layers for mc rows already in g.sh (positions pos .. pos+mc-1); RoPE rows cs_row0.. of table csbuf.
void rec_layers(const GpuXfmr &X, GpuSeq &S, int64_t pos, int64_t mc, VkBuffer csbuf, uint32_t cs_row0) {
    const int64_t qd = int64_t(X.heads) * X.hd, kd = int64_t(X.kv) * X.hd;
    const float scale = 1.0f / std::sqrt(float(X.hd));
    for (int li = 0; li < X.layers; ++li) {
        const GpuXfmr::L &L = X.L_[size_t(li)];
        rec_rms(X, L.ln1, g.sh, 0, g.sxn, mc);
        rec_mm(L.qkv, g.sxn, g.su, mc, false);
        {
            const VkBuffer b[6] = {S.k[size_t(li)], S.v[size_t(li)], kNoBuf, g.su, g.sq, kNoBuf};
            const uint32_t pc[4] = {uint32_t(mc), uint32_t(qd), uint32_t(kd), uint32_t(pos)};
            disp(P_QKV, b, pc, 4, uint32_t((mc * (qd + 2 * kd) + 255) / 256), 1);
        }
        {
            const VkBuffer bq[6] = {X.norms, csbuf, kNoBuf, g.sq, kNoBuf, kNoBuf};
            const uint32_t pq[8] = {uint32_t(X.heads), uint32_t(X.hd), uint32_t(mc), L.qk ? 1u : 0u, fbits(X.eps), 0u, uint32_t(L.qn), cs_row0};
            disp(P_ROPE, bq, pq, 8, uint32_t(X.heads), uint32_t(mc));
            const VkBuffer bk[6] = {X.norms, csbuf, kNoBuf, S.k[size_t(li)], kNoBuf, kNoBuf};
            const uint32_t pk[8] = {uint32_t(X.kv), uint32_t(X.hd), uint32_t(mc), L.qk ? 1u : 0u, fbits(X.eps), uint32_t(pos * kd), uint32_t(L.kn),
                                    cs_row0};
            disp(P_ROPE, bk, pk, 8, uint32_t(X.kv), uint32_t(mc));
        }
        {
            const VkBuffer ba[6] = {S.k[size_t(li)], S.v[size_t(li)], kNoBuf, g.sq, g.sao, kNoBuf};
            const uint32_t pa[7] = {uint32_t(X.heads), uint32_t(X.kv), uint32_t(X.hd), uint32_t(mc), uint32_t(pos), fbits(scale), uint32_t(kd)};
            disp(P_ATTN, ba, pa, 7, uint32_t(X.heads), uint32_t(mc));
        }
        rec_mm(L.o, g.sao, g.sh, mc, true);
        rec_rms(X, L.ln2, g.sh, 0, g.sxn, mc);
        rec_mm(L.gu, g.sxn, g.su, mc, false);
        {
            const VkBuffer bs[6] = {kNoBuf, kNoBuf, kNoBuf, g.su, g.sg, kNoBuf};
            const uint32_t ps[2] = {uint32_t(mc), uint32_t(X.I)};
            disp(P_SWIGLU, bs, ps, 2, uint32_t((mc * X.I + 255) / 256), 1);
        }
        rec_mm(L.down, g.sg, g.sh, mc, true);
    }
}

void host_barrier() {
    VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0,
                         nullptr, 0, nullptr);
}
}  // namespace vk
using namespace vk;

int gpu_xfmr_create(const TransformerW &w, const char *name, int n_layers, const std::function<void(const Mat &)> &uploaded) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok || w.window != 0 || w.head_dim > 128 || w.head_dim % 2 || w.layers.empty()) return -1;
    try {
        GpuXfmr X;
        X.name = name ? name : "xfmr";
        const int use_layers = n_layers < 0 || n_layers > int(w.layers.size()) ? int(w.layers.size()) : n_layers;
        if (use_layers == 0) return -1;
        X.layers = use_layers; X.H = w.hidden; X.heads = w.heads; X.kv = w.kv_heads; X.hd = w.head_dim; X.I = w.inter;
        X.eps = w.eps; X.inv_freq = w.inv_freq;
        std::vector<float> nv;
        auto push = [&](const float *p, int n) { size_t off = nv.size(); nv.insert(nv.end(), p, p + n); return off; };
        for (int li = 0; li < use_layers; ++li) {
            const LayerW &L = w.layers[size_t(li)];
            if (L.attn_scale || L.mlp_scale) return -1;
            GpuXfmr::L l;
            // weights are uploaded here, stacked: one q|k|v and one gate|up matrix per layer (no separate copies in VRAM)
            l.qkv = upload_parts({&L.q, &L.k, &L.v});
            l.o = upload_parts({&L.o});
            l.gu = upload_parts({&L.gate, &L.up});
            l.down = upload_parts({&L.down});
            if (l.qkv < 0 || l.o < 0 || l.gu < 0 || l.down < 0) return -1;
            if (uploaded) for (const Mat *m : {&L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down}) uploaded(*m);
            l.ln1 = push(L.ln1, w.hidden); l.ln2 = push(L.ln2, w.hidden);
            l.qk = L.qn && L.kn;
            if (l.qk) { l.qn = push(L.qn, w.head_dim); l.kn = push(L.kn, w.head_dim); }
            X.L_.push_back(l);
        }
        X.final_off = push(w.final_norm, w.hidden);
        upload_buffer(nv.data(), nv.size() * 4, X.norms, X.nm);
        g.xf.push_back(std::move(X));
        return int(g.xf.size()) - 1;
    } catch (const std::exception &e) {
        if (!g.warned) { std::fprintf(stderr, "vox: GPU transformer setup failed (%s); using CPU attention\n", e.what()); g.warned = true; }
        return -1;
    }
}

int gpu_seq_create(int xfmr, int64_t capacity) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok || xfmr < 0 || size_t(xfmr) >= g.xf.size() || capacity > kMaxCtx || capacity <= 0) return -1;
    const GpuXfmr &X = g.xf[size_t(xfmr)];
    try {
        const VkDeviceSize bytes = VkDeviceSize(capacity) * VkDeviceSize(X.kv * X.hd) * 4;
        if (bytes > g.max_range) return -1;
        size_t id = g.seqs.size();
        for (size_t i = 0; i < g.seqs.size(); ++i)
            if (!g.seqs[i].used && g.seqs[i].k.size() == size_t(X.layers) && g.seqs[i].cap >= capacity) { id = i; break; }
        if (id == g.seqs.size()) {
            GpuSeq S;
            S.cap = capacity;
            S.k.resize(size_t(X.layers)); S.v.resize(size_t(X.layers)); S.km.resize(size_t(X.layers)); S.vm.resize(size_t(X.layers));
            for (int l = 0; l < X.layers; ++l) {
                make_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, S.k[size_t(l)], S.km[size_t(l)]);
                make_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, S.v[size_t(l)], S.vm[size_t(l)]);
            }
            g.seqs.push_back(std::move(S));
        }
        g.seqs[id].used = true;
        return int(id);
    } catch (const std::exception &e) {
        if (!g.warned) { std::fprintf(stderr, "vox: GPU KV cache allocation failed (%s); using CPU attention\n", e.what()); g.warned = true; }
        return -1;
    }
}

void gpu_seq_free(int seq) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (seq >= 0 && size_t(seq) < g.seqs.size()) g.seqs[size_t(seq)].used = false;  // буферы остаются в пуле для повторного использования
}

bool gpu_xfmr_forward(const TransformerW &w, int seq, int64_t pos0, const float *x, int64_t m, float *out_last, const Mat *in_proj,
                      const Mat *head, float *head_out) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok || w.gpu_id < 0 || size_t(w.gpu_id) >= g.xf.size() || seq < 0 || size_t(seq) >= g.seqs.size()) return false;
    try {
        const GpuXfmr &X = g.xf[size_t(w.gpu_id)];
        GpuSeq &S = g.seqs[size_t(seq)];
        const int64_t H = X.H, half = X.hd / 2;
        if (pos0 + m > S.cap) throw std::runtime_error("context exceeds GPU KV capacity");
        ensure_scratch(X);
        prof_stage(X.name.c_str());
        for (int64_t c0 = 0; c0 < m; c0 += kMaxM) {
            const int64_t mc = std::min(kMaxM, m - c0), pos = pos0 + c0;
            const bool last = c0 + mc == m;
            const int64_t xc = in_proj ? in_proj->cols : H;  // ширина входной строки
            std::memcpy(g.xp, x + c0 * xc, size_t(mc * xc) * 4);
            for (int64_t i = 0; i < mc; ++i)
                for (int64_t p = 0; p < half; ++p) {
                    const float f = X.inv_freq[size_t(p)] * float(pos + i);
                    g.csp[(i * half + p) * 2] = std::cos(f);
                    g.csp[(i * half + p) * 2 + 1] = std::sin(f);
                }
            VKC(vkResetDescriptorPool(g.dev, g.dyn, 0));
            begin_cmd();
            if (in_proj) {
                rec_mm(in_proj->gpu, g.xb, g.sh, mc, false);  // проекция входа прямо в рабочий буфер
            } else {
                VkBufferCopy bc{0, 0, VkDeviceSize(mc * H * 4)};
                vkCmdCopyBuffer(g.cmd, g.xb, g.sh, 1, &bc);
                barrier_cc();
            }
            rec_layers(X, S, pos, mc, g.csb, 0);
            if (last) {
                rec_rms(X, X.final_off, g.sh, (mc - 1) * H, g.yb, 1);
                if (head) rec_mm(head->gpu, g.yb, g.yb, 1, false, H);  // логиты: yb[H .. H + rows)
                host_barrier();
            }
            VKC(vkEndCommandBuffer(g.cmd));
            submit_wait(g.cmd);
            if (last && out_last) std::memcpy(out_last, g.yp, size_t(H) * 4);
            if (last && head && head_out) std::memcpy(head_out, g.yp + H, size_t(g.mats[size_t(head->gpu)].rows) * 4);
        }
        return true;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "vox: GPU transformer failed: %s\n", e.what());
        return false;
    }
}

bool gpu_xfmr_forward_hidden(const TransformerW &w, int seq, int64_t pos0, const float *x, int64_t m, float *hidden_out) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok || w.gpu_id < 0 || size_t(w.gpu_id) >= g.xf.size() || seq < 0 || size_t(seq) >= g.seqs.size()) return false;
    try {
        const GpuXfmr &X = g.xf[size_t(w.gpu_id)];
        GpuSeq &S = g.seqs[size_t(seq)];
        const int64_t H = X.H, half = X.hd / 2;
        if (pos0 + m > S.cap) throw std::runtime_error("context exceeds GPU KV capacity");
        ensure_scratch(X);
        prof_stage(X.name.c_str());
        for (int64_t c0 = 0; c0 < m; c0 += kMaxM) {
            const int64_t mc = std::min(kMaxM, m - c0), pos = pos0 + c0;
            std::memcpy(g.xp, x + c0 * H, size_t(mc * H) * 4);
            for (int64_t i = 0; i < mc; ++i)
                for (int64_t p = 0; p < half; ++p) {
                    const float f = X.inv_freq[size_t(p)] * float(pos + i);
                    g.csp[(i * half + p) * 2] = std::cos(f);
                    g.csp[(i * half + p) * 2 + 1] = std::sin(f);
                }
            VKC(vkResetDescriptorPool(g.dev, g.dyn, 0));
            begin_cmd();
            VkBufferCopy bc{0, 0, VkDeviceSize(mc * H * 4)};
            vkCmdCopyBuffer(g.cmd, g.xb, g.sh, 1, &bc);
            barrier_cc();
            rec_layers(X, S, pos, mc, g.csb, 0);
            vkCmdCopyBuffer(g.cmd, g.sh, g.yb, 1, &bc);  // скрытые состояния — в читаемый хостом буфер
            host_barrier();
            VKC(vkEndCommandBuffer(g.cmd));
            submit_wait(g.cmd);
            std::memcpy(hidden_out + c0 * H, g.yp, size_t(mc * H) * 4);
        }
        return true;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "vox: GPU transformer failed: %s\n", e.what());
        return false;
    }
}

// ====================== code predictor: весь цикл кадра одной отправкой ======================
bool gpu_pred_frame(const TransformerW &w, int seq, const float *in2, const Mat *in_proj, const std::vector<Mat> &heads,
                    const std::vector<Mat> &embs, const GpuSampleCfg &sc, int32_t *codes) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.ok || w.gpu_id < 0 || size_t(w.gpu_id) >= g.xf.size() || seq < 0 || size_t(seq) >= g.seqs.size()) return false;
    const int groups = int(heads.size()) + 1;
    if (in_proj && in_proj->gpu < 0) return false;
    for (const Mat &h : heads) if (h.gpu < 0 || h.rows > 4096) return false;
    for (const Mat &e : embs) if (e.gpu < 0 || e.type == WType::Q8 || e.type == WType::Q4) return false;
    if (int(embs.size()) < groups - 2 || groups > 64) return false;
    try {
        const GpuXfmr &X = g.xf[size_t(w.gpu_id)];
        GpuSeq &S = g.seqs[size_t(seq)];
        if (groups + 1 > S.cap) return false;
        ensure_scratch(X);
        const int64_t H = X.H, half = X.hd / 2;
        const int64_t din = in_proj ? in_proj->cols : H;
        const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if (!g.pcodes) {  // buffers on first use: codes (host-visible), logits, gathered input, RoPE table for positions 0..63
            make_buffer(64 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, g.pcodes, g.mpcodes);
            VKC(vkMapMemory(g.dev, g.mpcodes, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&g.pcodes_p)));
            make_buffer(4096 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, g.plog, g.mplog);
            make_buffer(VkDeviceSize(std::max<int64_t>(din, H)) * 4 * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, g.pin, g.mpin);
            make_buffer(64 * 64 * 2 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, g.pcs, g.mpcs);
            VKC(vkMapMemory(g.dev, g.mpcs, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&g.pcs_p)));
        }
        if (half > 64) return false;
        for (int64_t i = 0; i < 64; ++i)
            for (int64_t p = 0; p < half; ++p) {
                const float f = X.inv_freq[size_t(p)] * float(i);
                g.pcs_p[(i * half + p) * 2] = std::cos(f);
                g.pcs_p[(i * half + p) * 2 + 1] = std::sin(f);
            }
        std::memcpy(g.xp, in2, size_t(2 * din) * 4);
        prof_stage("predictor");
        VKC(vkResetDescriptorPool(g.dev, g.dyn, 0));
        begin_cmd();
        int64_t pos = 0;
        for (int cb = 1; cb < groups; ++cb) {
            const int64_t m = cb == 1 ? 2 : 1;
            VkBuffer in = g.xb;
            if (cb >= 2) {  // input = embedding of the code just sampled, looked up on the GPU
                const Mat &E = embs[size_t(cb - 2)];
                const GpuMat &ge = g.mats[size_t(E.gpu)];
                const VkBuffer b[6] = {ge.w, g.pcodes, kNoBuf, kNoBuf, g.pin, kNoBuf};
                const uint32_t pc[4] = {uint32_t(ge.cols), uint32_t(cb - 1), uint32_t(ge.rows), 0u};
                disp(P_GATHER + int(ge.type), b, pc, 4, uint32_t((ge.cols + 255) / 256), 1);
                in = g.pin;
            }
            if (in_proj) {
                rec_mm(in_proj->gpu, in, g.sh, m, false);
            } else {
                VkBufferCopy bc{0, 0, VkDeviceSize(m * H * 4)};
                vkCmdCopyBuffer(g.cmd, in, g.sh, 1, &bc);
                barrier_cc();
            }
            rec_layers(X, S, pos, m, g.pcs, uint32_t(pos));
            rec_rms(X, X.final_off, g.sh, (m - 1) * H, g.sxn, 1);
            const Mat &hd = heads[size_t(cb - 1)];
            rec_mm(hd.gpu, g.sxn, g.plog, 1, false);
            {
                const VkBuffer b[6] = {kNoBuf, g.pcodes, kNoBuf, g.plog, kNoBuf, kNoBuf};
                const uint32_t pc[8] = {uint32_t(hd.rows), fbits(sc.temperature), uint32_t(sc.top_k), fbits(sc.top_p), sc.key0, sc.key1,
                                        uint32_t(sc.frame), uint32_t(cb)};
                disp(P_SAMPLE, b, pc, 8, 1, 1);
            }
            pos += m;
        }
        host_barrier();
        VKC(vkEndCommandBuffer(g.cmd));
        submit_wait(g.cmd);
        for (int cb = 1; cb < groups; ++cb) codes[cb] = g.pcodes_p[cb];
        return true;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "vox: GPU predictor loop failed: %s\n", e.what());
        return false;
    }
}

}  // namespace vox
