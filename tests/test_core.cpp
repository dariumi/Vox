// Тесты ядра без внешних зависимостей. Запуск: ./vox_tests
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/backend.hpp"
#include "../src/f5.hpp"
#include "../src/f5_ops.hpp"
#include "../src/formats.hpp"
#include "../src/kernels.hpp"
#include "../src/resources.hpp"
#include "../src/sampling.hpp"
#include "../src/voice.hpp"
#include "vox.h"

static int g_fail = 0, g_pass = 0;
#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (cond) ++g_pass;                                                            \
        else { ++g_fail; std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)
#define CHECK_OK(expr)                                                                 \
    do {                                                                               \
        vox_status _s = (expr);                                                        \
        if (_s >= 0) ++g_pass;                                                         \
        else { ++g_fail; std::fprintf(stderr, "FAIL %s:%d: %s -> %d (%s)\n", __FILE__, __LINE__, #expr, _s, vox_last_error()); } \
    } while (0)

using namespace vox;

static void test_philox_kat() {
    // Известные векторы Random123 для Philox4x32-10.
    auto r = Philox4x32::block({0, 0, 0, 0}, {0, 0});
    CHECK(r[0] == 0x6627e8d5u && r[1] == 0xe169c58du && r[2] == 0xbc57ac4cu && r[3] == 0x9b00dbd8u);
    r = Philox4x32::block({0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, {0xffffffffu, 0xffffffffu});
    CHECK(r[0] == 0x408f276du && r[1] == 0x41c83b0eu && r[2] == 0xa20bc7c6u && r[3] == 0x6d5451fdu);
    r = Philox4x32::block({0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u});
    CHECK(r[0] == 0xd16cfe09u && r[1] == 0x94fdccebu && r[2] == 0x5001e420u && r[3] == 0x24126ea1u);
}

static void test_sampler() {
    DetRng a(42), b(42), c(43);
    CHECK(a.uniform(5, 3) == b.uniform(5, 3));
    CHECK(a.uniform(5, 3) != c.uniform(5, 3));
    CHECK(a.uniform(5, 3) != a.uniform(5, 4));
    float u = a.uniform(0, 0);
    CHECK(u >= 0.f && u < 1.f);

    std::vector<float> l = {0.1f, 3.0f, 0.2f, 3.0f};
    SampleParams greedy;
    greedy.temperature = 0;
    auto l2 = l;
    CHECK(sample_logits(l2, greedy, 0.5f) == 1);  // ничья → меньший индекс

    SampleParams top1;
    top1.top_k = 1;
    for (float uu : {0.0f, 0.3f, 0.99f}) { l2 = l; CHECK(sample_logits(l2, top1, uu) == 1); }

    // repetition penalty гасит уже выбранный токен
    std::vector<int32_t> hist = {1};
    SampleParams rp;
    rp.temperature = 0;
    rp.repetition_penalty = 10.f;
    l2 = l;
    CHECK(sample_logits(l2, rp, 0.f, &hist) == 3);

    // распределение примерно соответствует softmax
    std::vector<float> two = {0.f, std::log(3.f)};  // p = 0.25 / 0.75
    SampleParams sp;
    sp.temperature = 1;
    sp.top_k = 0;
    int ones = 0;
    for (int i = 0; i < 20000; ++i) { auto t = two; ones += sample_logits(t, sp, a.uniform(uint64_t(i), 0)) == 1; }
    CHECK(std::fabs(ones / 20000.0 - 0.75) < 0.02);
}

static void test_formats() {
    SafetensorsFile f;
    f.metadata["k"] = "значение";
    f.tensors["a"] = Tensor::from(DType::F32, {2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    f.tensors["b"] = Tensor::from(DType::I16, {4}, std::vector<int16_t>{-1, 2, -3, 4});
    safetensors_save("t.safetensors", f);
    auto g = safetensors_load("t.safetensors");
    CHECK(g.metadata["k"] == "значение");
    CHECK(g.tensors["a"].shape == (std::vector<int64_t>{2, 3}));
    CHECK(g.tensors["a"].as<float>()[5] == 6.f);
    CHECK(g.tensors["b"].as<int16_t>()[2] == -3);

    std::map<std::string, Tensor> blobs;
    blobs["w1"] = Tensor::from(DType::F32, {3}, std::vector<float>{1.5f, 2.5f, 3.5f});
    blobs["w2"] = Tensor::from(DType::U8, {5}, std::vector<uint8_t>{1, 2, 3, 4, 5});
    tmod_write("t.tmod", {{"model", {{"family", "x"}}}}, blobs);
    TmodFile t("t.tmod");
    CHECK(t.manifest()["model"]["family"] == "x");
    CHECK(reinterpret_cast<uintptr_t>(t.get("w2").data) % TMOD_ALIGN == 0 ||
          (t.get("w2").data - t.get("w1").data) % TMOD_ALIGN == 0);
    CHECK(reinterpret_cast<const float *>(t.get("w1").data)[2] == 3.5f);
    CHECK(t.get("w2").data[4] == 5);

    bool threw = false;
    try { TmodFile bad("t.safetensors"); } catch (const Error &e) { threw = e.status == VOX_ERR_FORMAT; }
    CHECK(threw);
}

static std::vector<float> run(vox_engine *e, vox_voice *v, uint64_t seed, const std::vector<std::string> &parts,
                              size_t pull_cap = 4096) {
    vox_gen_params p;
    vox_gen_params_default(&p);
    p.seed = seed;
    vox_session *s = nullptr;
    CHECK_OK(vox_session_create(e, v, &p, &s));
    std::vector<float> out(pull_cap), all;
    auto drain = [&](bool until_done) {
        for (;;) {
            size_t n = 0;
            vox_status st = vox_session_pull(s, out.data(), out.size(), &n);
            if (st == VOX_OK) { all.insert(all.end(), out.begin(), out.begin() + long(n)); continue; }
            if (st == VOX_AGAIN && !until_done) return;
            if (st == VOX_DONE) return;
            ++g_fail;
            std::fprintf(stderr, "FAIL pull -> %d (%s)\n", st, vox_last_error());
            return;
        }
    };
    for (auto &t : parts) { CHECK_OK(vox_session_feed_text(s, t.c_str())); drain(false); }
    CHECK_OK(vox_session_end_text(s));
    drain(true);
    vox_session_free(s);
    return all;
}

static void test_engine() {
    vox::write_test_model("test.tmod");
    vox_engine *e = nullptr;
    CHECK_OK(vox_engine_open("test.tmod", nullptr, &e));
    if (!e) return;
    CHECK(vox_engine_sample_rate(e) == 24000);

    vox_engine_opts o;
    vox_engine_opts_default(&o);
    o.device = "cuda:0";
    vox_engine *bad = nullptr;
    CHECK(vox_engine_open("test.tmod", &o, &bad) == VOX_ERR_DEVICE);

    std::vector<float> ref(24000 * 3);
    for (size_t i = 0; i < ref.size(); ++i) ref[i] = 0.3f * std::sin(float(i) * 0.05f) * float(i % 4000) / 4000.f;
    vox_voice *v = nullptr;
    CHECK(vox_voice_create(e, ref.data(), ref.size(), 24000, "", VOX_VOICE_ICL, &v) == VOX_ERR_ARG);
    CHECK_OK(vox_voice_create(e, ref.data(), ref.size(), 24000, "Пример референса.", VOX_VOICE_ICL, &v));
    vox_gen_params d;
    vox_gen_params_default(&d);
    d.seed = 777;
    d.temperature = 0.7f;
    CHECK_OK(vox_voice_set_defaults(v, &d));
    CHECK_OK(vox_voice_save(v, "test.tvoice"));

    vox_voice *v2 = nullptr;
    CHECK_OK(vox_voice_load(e, "test.tvoice", &v2));
    VoiceData vd = voice_load("test.tvoice");
    CHECK(vd.ref_frames == 38 && vd.ref_codebooks == 4);
    CHECK(vd.sampling["seed"] == 777);
    CHECK(!vd.ref_audio.empty());

    const std::string text = "Привет! Это проверка движка, слово за словом.";
    auto a1 = run(e, v2, 42, {text});
    auto a2 = run(e, v2, 42, {text});
    auto a3 = run(e, v2, 7, {text});
    CHECK(!a1.empty());
    CHECK(a1 == a2);  // тот же seed → побитово тот же звук
    CHECK(a1 != a3);
    CHECK(a1.size() % 1920 == 0);

    // Текст кусками (как из LLM) и маленький буфер чтения → тот же результат.
    auto a4 = run(e, v2, 42, {"При", "вет! Это про", "верка движка, сло", "во за словом."}, 777);
    CHECK(a4 == a1);

    // seed = 0 → берётся из профиля, результат стабилен
    auto p1 = run(e, v2, 0, {text});
    auto p2 = run(e, v2, 777, {text});
    CHECK(p1 == p2);

    // Отмена
    vox_session *s = nullptr;
    CHECK_OK(vox_session_create(e, v2, nullptr, &s));
    CHECK_OK(vox_session_feed_text(s, "длинный текст "));
    vox_session_cancel(s);
    float buf[16];
    size_t n = 0;
    CHECK(vox_session_pull(s, buf, 16, &n) == VOX_ERR_CANCELLED);
    vox_session_free(s);

    // Профиль без текста референса (x-vector)
    vox_voice *xv = nullptr;
    CHECK_OK(vox_voice_create(e, ref.data(), ref.size(), 24000, nullptr, VOX_VOICE_XVECTOR_ONLY, &xv));
    CHECK_OK(vox_voice_save(xv, "x.tvoice"));
    CHECK(voice_load("x.tvoice").mode == VOX_VOICE_XVECTOR_ONLY);

    char info[64];
    size_t len = 0;
    CHECK_OK(vox_engine_info(e, info, sizeof info, &len));
    CHECK(len > sizeof info && std::strlen(info) == sizeof info - 1);

    vox_voice_free(xv);
    vox_voice_free(v);
    vox_voice_free(v2);
    vox_engine_close(e);
}


// ---- планировщик ресурсов на выдуманных устройствах ----
static const uint64_t MiB = 1024ull * 1024, GiB = 1024 * MiB;
static ModelFootprint fake_model() {
    ModelFootprint fp;
    fp.family = "fake";
    fp.always_ram_bytes = 100 * MiB;
    auto st = [](const char *n, uint64_t w, double bpf, uint64_t kv) {
        StageDesc d; d.name = n; d.weight_bytes = w; d.bytes_per_frame = bpf; d.flops_per_frame = bpf * 2;
        d.session_vram_bytes = kv; d.session_ram_bytes = kv; return d;
    };
    fp.stages = {st("talker", 1200 * MiB, 1200.0 * MiB, 100 * MiB), st("predictor", 300 * MiB, 4500.0 * MiB, 10 * MiB), st("codec", 400 * MiB, 400.0 * MiB, 0)};
    return fp;
}
static SystemInfo fake_sys(int cores = 8, uint64_t ram = 16 * GiB) {
    SystemInfo s; s.cpu_logical = cores; s.cpu_allowed = cores; s.ram_total = ram; s.ram_available = ram; s.os = "test"; return s;
}
static GpuDeviceInfo fake_gpu(int idx, uint64_t vram, bool software = false, bool unified = false) {
    GpuDeviceInfo g; g.index = idx; g.name = "gpu" + std::to_string(idx); g.type = software ? "cpu" : "discrete";
    g.software = software; g.unified_memory = unified; g.vram_total = vram; g.vram_free = vram; return g;
}
static void test_planner() {
    const auto fp = fake_model();
    const auto sys = fake_sys();
    ResourceConfig c;
    // auto выбирает GPU с большей свободной памятью и кладёт всё
    auto p = make_plan(c, fp, sys, {fake_gpu(0, 2 * GiB), fake_gpu(1, 8 * GiB)});
    CHECK(p.feasible && p.backend == "vulkan" && p.gpu_index == 1);
    for (auto &st : p.stages) CHECK(st.place == Place::Gpu);
    CHECK(p.vram_estimate > 1900 * MiB && p.vram_estimate < 3 * GiB);
    // программный Vulkan в auto не берётся
    p = make_plan(c, fp, sys, {fake_gpu(0, 8 * GiB, true)});
    CHECK(p.backend == "cpu" && p.feasible);
    // явный cpu
    c.device = "cpu";
    p = make_plan(c, fp, sys, {fake_gpu(0, 8 * GiB)});
    CHECK(p.backend == "cpu" && !p.uses_gpu());
    // vram_limit: влезает только часть стадий, приоритет — плотности выигрыша (predictor читает 15× байт на вес)
    c = ResourceConfig(); c.vram_limit = 700 * MiB;
    p = make_plan(c, fp, sys, {fake_gpu(0, 8 * GiB)});
    CHECK(p.place_of("predictor") == Place::Gpu);
    CHECK(p.place_of("talker") == Place::Cpu);
    CHECK(p.vram_estimate <= 700 * MiB);
    // strict без GPU — невозможно
    c = ResourceConfig(); c.device = "vulkan"; c.strict = true;
    p = make_plan(c, fp, sys, {});
    CHECK(!p.feasible);
    // принудительное размещение стадии
    c = ResourceConfig(); c.place["codec"] = Place::Cpu;
    p = make_plan(c, fp, sys, {fake_gpu(0, 8 * GiB)});
    CHECK(p.place_of("codec") == Place::Cpu && p.place_of("talker") == Place::Gpu);
    // лимит RAM: на CPU ничего не помещается → infeasible при enforce, предупреждение иначе
    c = ResourceConfig(); c.device = "cpu"; c.ram_limit = 512 * MiB;
    p = make_plan(c, fp, sys, {});
    CHECK(!p.feasible && p.enforce);
    c.enforce = 0;
    p = make_plan(c, fp, sys, {});
    CHECK(p.feasible && !p.warnings.empty());
    // тот же лимит, но веса на GPU: RAM почти свободен
    c = ResourceConfig(); c.ram_limit = 512 * MiB;
    p = make_plan(c, fp, sys, {fake_gpu(0, 8 * GiB)});
    CHECK(p.feasible && p.ram_estimate <= 512 * MiB);
    // гибрид: talker делим по слоям, когда целиком не влезает
    {
        ModelFootprint h = fp;
        for (auto &st : h.stages)
            if (st.name == "talker") { st.splittable = true; st.layers = 12; st.layer_bytes = 100 * MiB; st.session_vram_bytes = 120 * MiB; st.session_ram_bytes = 120 * MiB; }
        c = ResourceConfig(); c.vram_limit = 900 * MiB;
        p = make_plan(c, h, sys, {fake_gpu(0, 8 * GiB)});
        int k = -2;
        for (auto &st : p.stages) if (st.stage == "talker") k = st.gpu_layers;
        CHECK(p.feasible && k >= 1 && k < 12);
        CHECK(p.vram_estimate <= 900 * MiB);
        c = ResourceConfig(); c.gpu_layers["talker"] = 5;
        p = make_plan(c, h, sys, {fake_gpu(0, 8 * GiB)});
        CHECK(p.gpu_layers_of("talker") == 5 && p.place_of("talker") == Place::Gpu);
        c = ResourceConfig(); c.gpu_layers["predictor"] = 1;  // не делится
        bool threw2 = false;
        try { make_plan(c, h, sys, {fake_gpu(0, 8 * GiB)}); } catch (const Error &) { threw2 = true; }
        CHECK(threw2);
    }
    // режимы: потоки
    c = ResourceConfig(); c.device = "cpu";
    c.mode = Mode::Speed;      CHECK(make_plan(c, fp, sys, {}).threads == 8);
    c.mode = Mode::Balanced;   CHECK(make_plan(c, fp, sys, {}).threads == 7);
    c.mode = Mode::Memory;     CHECK(make_plan(c, fp, sys, {}).threads == 4);
    c.mode = Mode::Background; { auto b = make_plan(c, fp, sys, {}); CHECK(b.threads == 2 && b.tuning.nice > 0 && !b.tuning.spin); }
    c.mode = Mode::Speed; c.threads = 3; CHECK(make_plan(c, fp, sys, {}).threads == 3);
    // background не уходит на GPU в auto
    c = ResourceConfig(); c.mode = Mode::Background;
    CHECK(make_plan(c, fp, sys, {fake_gpu(0, 8 * GiB)}).backend == "cpu");
    // JSON-конфиг: размеры строкой, неизвестный ключ — ошибка
    auto jc = ResourceConfig::from_json(json::parse(R"({"mode":"memory","ram_limit":"2GiB","vram_limit":"1GiB","place":{"codec":"cpu"}})", nullptr, false));
    CHECK(jc.mode == Mode::Memory && jc.ram_limit == 2 * GiB && jc.vram_limit == GiB && jc.place["codec"] == Place::Cpu);
    CHECK(parse_size(json("1.5GiB")) == 1536 * MiB);
    CHECK(parse_size(json("50%"), 8 * GiB) == 4 * GiB);
    bool threw = false;
    try { ResourceConfig::from_json(json::parse(R"({"nonsense":1})")); } catch (const Error &) { threw = true; }
    CHECK(threw);
    // строка устройства
    ResourceConfig d; d.apply_device_string("vulkan:1,codec=cpu,strict,mode=memory,threads=3,ram=4GiB");
    CHECK(d.gpu == 1 && d.strict && d.mode == Mode::Memory && d.threads == 3 && d.ram_limit == 4 * GiB && d.place["codec"] == Place::Cpu);
    // MemoryBudget
    MemoryBudget mb(100);
    CHECK(mb.try_reserve(60) && !mb.try_reserve(60) && mb.used() == 60);
    mb.release(60); CHECK(mb.try_reserve(100) && mb.peak() == 100);
}

// ---- Q4: квантование, распаковка, матричное произведение ----
static void test_q4() {
    const int64_t rows = 37, cols = 160, m = 5;
    std::vector<float> w(size_t(rows * cols)), x(size_t(m * cols));
    uint32_t st = 12345;
    auto rnd = [&] { st = st * 1664525u + 1013904223u; return float(int32_t(st >> 8) % 2001 - 1000) / 1000.0f; };
    for (auto &v : w) v = rnd() * (1.0f + float(&v - w.data()) / float(w.size()));
    for (auto &v : x) v = rnd();
    std::vector<uint8_t> q;
    std::vector<float> sc;
    quantize_q4(w.data(), rows, cols, q, sc);
    CHECK(q.size() == size_t(rows * cols / 2) && sc.size() == size_t(rows * cols / 32));
    Mat M;
    M.type = WType::Q4; M.data = q.data(); M.scale = sc.data(); M.rows = rows; M.cols = cols;
    std::vector<float> deq(static_cast<size_t>(cols)), ref(static_cast<size_t>(m * rows)), y(static_cast<size_t>(m * rows));
    bool bound_ok = true;
    std::vector<float> wd(size_t(rows * cols));
    for (int64_t r = 0; r < rows; ++r) {
        mat_gather(M, r, deq.data());
        for (int64_t c = 0; c < cols; ++c) {
            wd[size_t(r * cols + c)] = deq[size_t(c)];
            const float d = std::fabs(sc[size_t(r * (cols / 32) + c / 32)]);
            if (std::fabs(deq[size_t(c)] - w[size_t(r * cols + c)]) > d * 2.001f) bound_ok = false;  // подбор масштаба допускает срез края до 2 шагов
        }
    }
    CHECK(bound_ok);  // ошибка квантования не больше двух шагов
    {   // подбор масштаба не хуже прежней схемы d = mx/−8 по сумме квадратов ошибки
        double e_new = 0, e_old = 0;
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t b = 0; b < cols / 32; ++b) {
                const float *p = &w[size_t(r * cols + b * 32)];
                float mx = 0;
                for (int j = 0; j < 32; ++j) if (std::fabs(p[j]) > std::fabs(mx)) mx = p[j];
                const float d = mx / -8.0f, inv = d != 0.0f ? 1.0f / d : 0.0f;
                for (int j = 0; j < 32; ++j) {
                    const float c = std::min(15.0f, std::max(0.0f, std::floor(p[j] * inv + 8.5f)));
                    e_old += double(p[j] - (c - 8.0f) * d) * double(p[j] - (c - 8.0f) * d);
                    const double en = double(p[j]) - double(wd[size_t(r * cols + b * 32 + j)]);
                    e_new += en * en;
                }
            }
        CHECK(e_new <= e_old * 1.0000001);
    }
    for (int64_t i = 0; i < m; ++i)
        for (int64_t r = 0; r < rows; ++r) {
            double a = 0;
            for (int64_t c = 0; c < cols; ++c) a += double(wd[size_t(r * cols + c)]) * double(x[size_t(i * cols + c)]);
            ref[size_t(i * rows + r)] = float(a);
        }
    matmul(M, x.data(), m, y.data());
    double num = 0, den = 0;
    for (size_t i = 0; i < y.size(); ++i) { num += double(y[i] - ref[i]) * double(y[i] - ref[i]); den += double(ref[i]) * double(ref[i]); }
    CHECK(std::sqrt(num / den) < 0.01);  // остаётся только квантование входа в int8 (≈0.4% на блок)
    // тот же результат при другом числе потоков
    std::vector<float> y1(y.size());
    set_num_threads(1);
    matmul(M, x.data(), m, y1.data());
    set_num_threads(0);
    bool same = true;
    for (size_t i = 0; i < y.size(); ++i) if (y[i] != y1[i]) same = false;
    CHECK(same);
}

// ---- release_pages: резидентная память mmap действительно возвращается системе ----
static void test_release_pages() {
#if defined(__linux__)
    const size_t bytes = 64u << 20;
    char path[] = "/tmp/vox_relpages_XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd < 0) return;
    unlink(path);
    std::vector<char> blk(1 << 20, 7);
    for (int i = 0; i < 64; ++i) CHECK(write(fd, blk.data(), blk.size()) == ssize_t(blk.size()));
    void *p = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
    CHECK(p != MAP_FAILED);
    if (p == MAP_FAILED) { close(fd); return; }
    volatile unsigned sum = 0;
    for (size_t i = 0; i < bytes; i += 4096) sum += static_cast<const unsigned char *>(p)[i];
    const SystemInfo a = system_info();
    release_pages(p, bytes);
    const SystemInfo b = system_info();
    CHECK(a.rss >= (48u << 20));                 // страницы были резидентными
    CHECK(b.rss + (32u << 20) < a.rss);          // и вернулись системе
    CHECK(static_cast<const unsigned char *>(p)[4096 * 5] == 7);  // данные по-прежнему читаются (из файла)
    munmap(p, bytes);
    close(fd);
#endif
}


// ---- F5-TTS: операции без внешних эталонов (с официальным пакетом сверяет tests/test_f5_tiny.py) ----
static void test_f5_ops() {
    // свёртка: сгруппированная, нечётное ядро, "same"-дополнение нулями — против прямого расчёта
    {
        const int cin = 6, cout = 6, K = 5, G = 2, T = 9, cig = cin / G, cog = cout / G;
        std::vector<float> w(size_t(cout * cig * K)), b(cout), x(size_t(T * cin)), y(size_t(T * cout)), want(y.size());
        for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin(0.37f * float(i)) * 0.5f;
        for (int i = 0; i < cout; ++i) b[size_t(i)] = 0.1f * float(i);
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos(0.21f * float(i));
        vox::Conv1d c;
        c.load(w.data(), cin, cout, K, G, b.data());
        c.forward(x.data(), T, y.data());
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < cout; ++o) {
                double acc = b[size_t(o)];
                const int g = o / cog;
                for (int i = 0; i < cig; ++i)
                    for (int k = 0; k < K; ++k) {
                        const int tt = t + k - K / 2;
                        if (tt >= 0 && tt < T) acc += double(w[(size_t(o) * size_t(cig) + size_t(i)) * size_t(K) + size_t(k)]) * double(x[size_t(tt) * size_t(cin) + size_t(g * cig + i)]);
                    }
                want[size_t(t) * size_t(cout) + size_t(o)] = float(acc);
            }
        double md = 0;
        for (size_t i = 0; i < y.size(); ++i) md = std::max(md, std::fabs(double(y[i] - want[i])));
        CHECK(md < 1e-5);
        // depthwise
        vox::Conv1d d;
        std::vector<float> wd(size_t(cin * 3));
        for (size_t i = 0; i < wd.size(); ++i) wd[i] = 0.3f * std::sin(float(i));
        d.load(wd.data(), cin, cin, 3, cin, nullptr);
        CHECK(d.depthwise());
        std::vector<float> yd(size_t(T * cin));
        d.forward(x.data(), T, yd.data());
        md = 0;
        for (int t = 0; t < T; ++t)
            for (int c2 = 0; c2 < cin; ++c2) {
                double acc = 0;
                for (int k = 0; k < 3; ++k) { const int tt = t + k - 1; if (tt >= 0 && tt < T) acc += double(wd[size_t(c2) * 3 + size_t(k)]) * double(x[size_t(tt) * size_t(cin) + size_t(c2)]); }
                md = std::max(md, std::fabs(acc - double(yd[size_t(t) * size_t(cin) + size_t(c2)])));
            }
        CHECK(md < 1e-5);
    }
    // mel: синус на частоте бина k даёт пик энергии в нужном mel-фильтре; число кадров = 1 + n / hop
    {
        vox::MelCfg c;
        c.n_fft = 256; c.hop = 64; c.win = 256; c.n_mels = 8;
        const int nb = c.n_fft / 2 + 1;
        std::vector<float> fb(size_t(c.n_mels * nb), 0.f);
        for (int m = 0; m < c.n_mels; ++m)
            for (int k = m * (nb - 1) / c.n_mels; k < (m + 1) * (nb - 1) / c.n_mels; ++k) fb[size_t(m * nb + k)] = 1.f;  // 8 непересекающихся полос
        std::vector<float> pcm(2000);
        for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = std::sin(2.0 * 3.14159265358979 * 40.0 * double(i) / 256.0);  // бин 40 → полоса 40/16 = 2
        std::vector<float> mel;
        int frames = 0;
        vox::mel_log_spectrogram(c, fb.data(), pcm.data(), pcm.size(), mel, frames);
        CHECK(frames == 1 + 2000 / 64);
        int best = 0;
        const size_t mid = size_t(frames / 2) * size_t(c.n_mels);
        for (int m = 1; m < c.n_mels; ++m) if (mel[mid + size_t(m)] > mel[mid + size_t(best)]) best = m;
        CHECK(best == 2);
    }
    // iSTFT "same": косинус в одном бине с постоянной фазой восстанавливается как синусоида нужной частоты и длины T * hop
    {
        const int n_fft = 64, hop = 16, T = 20, nb = n_fft / 2 + 1, k0 = 4;
        std::vector<float> mag(size_t(T * nb), -30.f), ph(size_t(T * nb), 0.f);   // exp(-30) ≈ 0
        for (int t = 0; t < T; ++t) {
            mag[size_t(t * nb + k0)] = std::log(8.0f);
            ph[size_t(t * nb + k0)] = float(2.0 * 3.14159265358979 * double(k0) * double(t * hop) / double(n_fft));  // когерентная фаза между кадрами
        }
        std::vector<float> out;
        vox::istft_same(mag.data(), ph.data(), T, n_fft, hop, out);
        CHECK(out.size() == size_t(T * hop));
        double md = 0;
        // кадр = (2 * 8 / n_fft) * cos(...) = 0.25 * cos(...); Vocos делит сумму w·кадр на сумму w² (для Hann и перекрытия 75% это 2 / 1.5), поэтому амплитуда 1/3
        for (size_t n = 64; n + 64 < out.size(); ++n)
            md = std::max(md, std::fabs(double(out[n]) - (1.0 / 3.0) * std::cos(2.0 * 3.14159265358979 * k0 * double(n + (n_fft - hop) / 2) / n_fft)));
        CHECK(md < 1e-4);
    }
    // шум: детерминирован, не зависит от разбиения, имеет нормальные моменты
    {
        std::vector<float> a(100001), b(100001);
        vox::F5Model::noise(42, 3, a.size(), a.data());
        vox::F5Model::noise(42, 3, b.size(), b.data());
        CHECK(a == b);
        std::vector<float> c(a.size());
        vox::F5Model::noise(42, 4, c.size(), c.data());
        CHECK(a != c);
        double m = 0, v = 0;
        for (float x : a) m += x;
        m /= double(a.size());
        for (float x : a) v += (x - m) * (x - m);
        v /= double(a.size());
        CHECK(std::fabs(m) < 0.02 && std::fabs(v - 1.0) < 0.03);
    }
    // сетка времени: EPSS для 5/6/7/10/12/16, linspace для остальных; sway сдвигает внутренние точки, концы остаются
    {
        vox::F5Sampler s;
        s.sway = 0.f;
        s.nfe = 5;
        auto t = vox::F5Model::timesteps(s);
        CHECK(t.size() == 6 && std::fabs(t[3] - 8.f / 32.f) < 1e-7f && t[5] == 1.f);
        s.nfe = 8;
        t = vox::F5Model::timesteps(s);
        CHECK(t.size() == 9 && std::fabs(t[4] - 0.5f) < 1e-7f);
        s.epss = false; s.nfe = 5; s.sway = -1.f;
        t = vox::F5Model::timesteps(s);
        CHECK(t.size() == 6 && std::fabs(t[0]) < 1e-7f && std::fabs(t[5] - 1.f) < 1e-6f && t[2] < 0.4f);   // sway=-1 тянет точки к началу
    }
    // токенизатор: слоги, правило пробелов, замены
    {
        vox::json tok = {{"type", "pinyin"}, {"vocab", {" ", "a", "b", "c", "shi4", "jie4", ","}}, {"pinyin", {{"\u4e16", "shi4"}, {"\u754c", "jie4"}}}};
        vox::F5Tokenizer t;
        t.load(tok);
        auto sy = t.symbols("ab, cd");
        std::vector<std::string> want = {"a", "b", ",", " ", "c", "d"};
        CHECK(sy == want);
        auto sy2 = t.symbols("ab,cd");   // слово из 2+ букв после запятой получает пробел
        std::vector<std::string> want2 = {"a", "b", ",", " ", "c", "d"};
        CHECK(sy2 == want2);
        auto sy3 = t.symbols("a;b");     // ; → ,
        std::vector<std::string> want3 = {"a", ",", "b"};
        CHECK(sy3 == want3);
        auto sy4 = t.symbols("\xe4\xb8\x96\xe7\x95\x8c");   // 世界
        std::vector<std::string> want4 = {" ", "shi4", " ", "jie4"};
        CHECK(sy4 == want4);
        auto ids = t.encode("a?");
        CHECK(ids.size() == 2 && ids[0] == 1 && ids[1] == 0);   // неизвестный символ → 0
    }
}

int main() {
    test_f5_ops();
    test_release_pages();
    test_q4();
    test_planner();
    test_philox_kat();
    test_sampler();
    test_formats();
    test_engine();
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
