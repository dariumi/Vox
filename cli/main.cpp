// vox — командная строка движка. Синтез и голоса идут только через C ABI (vox.h);
// внутренние модули используются лишь для служебных команд (info по файлу, тестовая модель).
#include <algorithm>
#include <chrono>
#include <fstream>
#include <mutex>
#include <thread>
#if defined(__unix__) || defined(__APPLE__)
#  include <sys/resource.h>
#endif
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#if !defined(_WIN32)
#  include <unistd.h>
#endif

#include "../src/backend.hpp"
#include "../src/formats.hpp"
#include "../src/gpu.hpp"
#include "../src/tokenizer.hpp"
#include "../src/voice.hpp"
#include "../src/wav.hpp"
#include "vox.h"

#if defined(_WIN32)
#  include <fcntl.h>
#  include <io.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Args {
    std::vector<std::string> pos;
    std::map<std::string, std::string> opt;
    bool has(const std::string &k) const { return opt.count(k) != 0; }
    std::string get(const std::string &k, const std::string &d = "") const {
        auto it = opt.find(k);
        return it == opt.end() ? d : it->second;
    }
};

// Флаги без значения; всё остальное в виде --key value.
const char *kFlags[] = {"--no-argmax", "--xvector-only", "--stream", "--stdin", "--float", "--help", "-h", "--strict", "--json", "--no-release", "--default",
                        "--show", "--forget", "--dry-run"};

Args parse(int argc, char **argv, int from) {
    Args a;
    for (int i = from; i < argc; ++i) {
        std::string s = argv[i];
        bool flag = false;
        for (auto f : kFlags) flag |= s == f;
        if (flag) a.opt[s] = "1";
        else if (s.size() > 1 && s[0] == '-' && i + 1 < argc) a.opt[s] = argv[++i];
        else a.pos.push_back(s);
    }
    return a;
}

[[noreturn]] void die(const std::string &msg, int code = 1) {
    std::fprintf(stderr, "vox: %s\n", msg.c_str());
    std::exit(code);
}

void check(vox_status st, const char *what) {
    if (st < 0) die(std::string(what) + ": " + vox_last_error());
}

void usage() {
    std::printf("vox %s — движок синтеза речи\n\n", vox_version());
    std::puts(
        "  vox info <файл.tmod|файл.tvoice>\n"
        "  vox phonemize --model kokoro.tmod [--lang en-gb] \"текст\"   (фонемы, которые озвучит модель с фонемным входом)\n"
        "  vox devices [--json]   (процессор, память, GPU с индексами и свободной видеопамятью)\n"
        "  vox voice speaker --model M.tmod [--speaker NAME] [--instruct ТЕКСТ] -o v.tvoice   (диктор CustomVoice / описание голоса VoiceDesign)\n"
        "  vox say \"текст\" --model M.tmod --speaker NAME [--instruct ТЕКСТ] -o out.wav        (без файла голоса)\n"
        "  vox plan --model M.tmod [ресурсы]   (куда что будет размещено, сколько памяти нужно — без загрузки весов)\n"
        "  vox voice create <ref.wav> --model M.tmod --text \"текст референса\" -o voice.tvoice\n"
        "                   [--xvector-only] [--name N] [--lang ru] [--seed 42] [--temperature 0.7]\n"
        "  vox say \"текст\" --model M.tmod --voice V.tvoice -o out.wav [--seed 42] [--float]\n"
        "  vox say --stdin --stream --model M.tmod --voice V.tvoice   (сырой float32 PCM в stdout)\n"
        "      профиль: --voice-id ИМЯ --style СТИЛЬ --markup auto|strict|off; разметка в тексте: [anna] / \"anna: …\", [happy], [pause 500ms], [speed 1.2], [laugh], [/]\n"
        "  vox profile new P.tvoice [--name N]                       (профиль: несколько голосов, стили-эмоции, любые модели)\n"
        "  vox profile add P.tvoice --voice anna [--style happy] REC.wav --text \"…\" [--lang ru]   |   --from V.tvoice [--from-voice X]\n"
        "  vox profile set P.tvoice --voice anna [--style happy] [--instruct …] [--description …] [--speaker \"kokoro=af_heart;qwen3-tts=serena\"]\n"
        "                  [--speed 1.1] [--options JSON] [--language ru] [--default-style S] [--default]\n"
        "  vox profile rm|show|patch P.tvoice …;  vox profile prepare P.tvoice --model M.tmod   (собрать голоса под модель и сохранить)\n"
        "  vox live --model M.tmod --voice V.tvoice (-o out.wav | --raw) [--commit auto|sentence|manual] [--idle-ms N] [--flush-nl]\n"
        "                                                   (текст из stdin на лету → звук на лету; двусторонний стриминг)\n"
        "  vox tokenize --model M.tmod   (JSON-строки из stdin → id токенов)\n"
        "  vox bench --model M.tmod --voice V.tvoice [--text T] [--runs 5]\n"
        "  vox tune --model M.tmod [--voice V.tvoice | --speaker S] [--goal speed|latency] [--text T] [--runs 2] [--show | --forget]\n"
        "                (замерить устройства и размещение стадий на этом компьютере; потом \"auto\" открывает модель так, как быстрее)\n"
        "  vox make-test-model -o test.tmod   (модель тестового бэкенда для разработки)\n\n"
        "ресурсы: --device auto|cpu|gpu[:N]|vulkan[:N][,codec=cpu][,strict]  --mode speed|balanced|memory|background\n"
        "         --threads N  --affinity 0-3,6  --priority low|normal|idle|<nice>  --ram-limit 6GiB  --vram-limit 3GiB\n"
        "         --place talker=gpu,predictor=gpu,codec=cpu  --gpu-layers talker=12  --max-sessions N  --strict\n"
        "         --overlap on|off  --no-release  --config resources.json (JSON; флаги командной строки его перекрывают)\n"
        "общие параметры: --temperature --top-k --top-p\n"
        "                 --repetition-penalty --sub-temperature --sub-top-k --chunk-frames\n"
        "                 --decoder stream|reference (reference = нарезка декодера как в qwen_tts, для сверки)\n"
        "                 --context N|all (кадры референса для прогрева декодера в режиме stream)\n\n"
        "  vox build --model <папка HF> --dtype bf16 -o model.tmod   (через Python-пакет vox)\n"
        "  vox serve --model M.tmod --voices ./voices --port 8080     (через Python-пакет vox)\n"        "  vox web [--models DIR] [--voices DIR] [--host 0.0.0.0 --tls]   (движок для Vox Studio: печатает ссылку с секретом; Python-пакет)\n"
        "  vox verify --model M.tmod --dump case.safetensors          (сверка с эталонным дампом)");
}

// Флаги ресурсов → JSON конфигурации (vox_engine_opts.config_json). --config file.json — основа, флаги её перекрывают.
std::string resource_json(const Args &a) {
    vox::json j = vox::json::object();
    if (a.has("--config")) {
        std::ifstream in(a.get("--config"));
        if (!in) die("cannot read " + a.get("--config"));
        try { j = vox::json::parse(in); } catch (const std::exception &e) { die(a.get("--config") + ": " + e.what()); }
    }
    auto str = [&](const char *flag, const char *key) { if (a.has(flag)) j[key] = a.get(flag); };
    auto num = [&](const char *flag, const char *key) { if (a.has(flag)) j[key] = std::stoul(a.get(flag)); };
    str("--mode", "mode"); str("--ram-limit", "ram_limit"); str("--vram-limit", "vram_limit"); str("--affinity", "cpu_affinity"); str("--priority", "priority");
    num("--threads", "threads"); num("--max-sessions", "max_sessions");
    if (a.has("--gpu")) j["gpu"] = std::stoi(a.get("--gpu"));
    if (a.has("--strict")) j["strict"] = true;
    if (a.has("--no-release")) j["release_host_weights"] = false;
    if (a.has("--overlap")) j["overlap"] = a.get("--overlap") != "off";
    auto kv = [&](const char *flag, const char *key, bool number) {
        if (!a.has(flag)) return;
        vox::json o = j.value(key, vox::json::object());
        std::string s = a.get(flag);
        size_t p = 0;
        while (p < s.size()) {
            size_t q = s.find(',', p), e = s.find('=', p);
            if (e == std::string::npos || (q != std::string::npos && e > q)) die(std::string(flag) + " expects stage=value[,stage=value]");
            const std::string k = s.substr(p, e - p), v = s.substr(e + 1, q == std::string::npos ? std::string::npos : q - e - 1);
            if (number) o[k] = std::stoi(v); else o[k] = v;
            if (q == std::string::npos) break;
            p = q + 1;
        }
        j[key] = o;
    };
    kv("--place", "place", false);
    kv("--gpu-layers", "gpu_layers", true);
    return j.empty() ? std::string() : j.dump();
}

vox_engine *open_engine(const Args &a) {
    if (!a.has("--model")) die("--model is required");
    vox_engine_opts o;
    vox_engine_opts_default(&o);
    std::string dev = a.get("--device", "auto");
    o.device = dev.c_str();
    const std::string cfg = resource_json(a);
    if (!cfg.empty()) o.config_json = cfg.c_str();
    vox_engine *e = nullptr;
    check(vox_engine_open(a.get("--model").c_str(), &o, &e), "open model");
    return e;
}

// Size query, then the call; the text can grow in between (free memory, timings), so a result that did not fit is
// fetched again with room to spare instead of being cut.
template <typename F>
std::string fetch(F &&fn, const char *what) {
    size_t n = 0;
    fn(nullptr, size_t(0), &n);
    for (int attempt = 0;; ++attempt) {
        std::string s(n + 256, '\0');
        size_t m = 0;
        check(fn(s.data(), s.size(), &m), what);
        if (m < s.size() || attempt >= 4) { s.resize(std::min(m, s.size() - 1)); return s; }
        n = m;
    }
}

std::string json_call(vox_status (*fn)(char *, size_t, size_t *)) { return fetch(fn, "query"); }

template <typename F>
std::string json_of(F &&fn) {
    return fetch(fn, "query");
}

vox_gen_params gen_params(const Args &a, std::string &lang_store) {
    static std::string extra_store;  // на неё указывает extra_json
    vox_gen_params p;
    vox_gen_params_default(&p);
    auto f = [&](const char *k, float &dst) { if (a.has(k)) dst = std::stof(a.get(k)); };
    auto i = [&](const char *k, int32_t &dst) { if (a.has(k)) dst = std::stoi(a.get(k)); };
    f("--temperature", p.temperature);
    f("--top-p", p.top_p);
    f("--repetition-penalty", p.repetition_penalty);
    f("--sub-temperature", p.sub_temperature);
    f("--sub-top-p", p.sub_top_p);
    i("--top-k", p.top_k);
    i("--sub-top-k", p.sub_top_k);
    if (a.has("--seed")) p.seed = std::stoull(a.get("--seed"));
    if (a.has("--chunk-frames")) p.chunk_frames = uint32_t(std::stoul(a.get("--chunk-frames")));
    if (a.has("--context")) p.decoder_context_frames = a.get("--context") == "all" ? 0xFFFFFFFFu : uint32_t(std::stoul(a.get("--context")));
    if (a.has("--decoder")) {
        const std::string d = a.get("--decoder");
        if (d == "stream") p.decoder_mode = VOX_DECODER_STREAM;
        else if (d == "reference") p.decoder_mode = VOX_DECODER_REFERENCE;
        else die("--decoder must be stream or reference");
    }
    if (a.has("--extra")) { extra_store = a.get("--extra"); }
    {   // параметры моделей без покадровых кодов: --nfe --cfg --sway --speed складываются в extra_json
        vox::json ex = extra_store.empty() ? vox::json::object() : vox::json::parse(extra_store);
        if (a.has("--nfe")) ex["nfe_step"] = std::stoi(a.get("--nfe"));
        if (a.has("--cfg")) ex["cfg_strength"] = std::stod(a.get("--cfg"));
        if (a.has("--sway")) ex["sway_sampling_coef"] = std::stod(a.get("--sway"));
        if (a.has("--speed")) ex["speed"] = std::stod(a.get("--speed"));
        if (a.has("--voice-id")) ex["voice"] = a.get("--voice-id");   // голос и стиль профиля, разметка (0.15)
        if (a.has("--style")) ex["style"] = a.get("--style");
        if (a.has("--markup")) ex["markup"] = a.get("--markup");
        if (a.has("--commit") || a.has("--idle-ms")) {   // двусторонний стриминг: политика коммита текста
            vox::json st = ex.value("stream", vox::json::object());
            if (a.has("--commit")) st["commit"] = a.get("--commit");
            if (a.has("--idle-ms")) st["idle_ms"] = std::stoul(a.get("--idle-ms"));
            ex["stream"] = st;
        }
        if (!ex.empty()) { extra_store = ex.dump(); p.extra_json = extra_store.c_str(); }
    }
    if (a.has("--max-frames")) p.max_frames = uint32_t(std::stoul(a.get("--max-frames")));
    if (a.has("--lang")) { lang_store = a.get("--lang"); p.language = lang_store.c_str(); }
    return p;
}

std::string engine_json(vox_engine *e) {
    return fetch([&](char *b, size_t c, size_t *l) { return vox_engine_info(e, b, c, l); }, "engine info");
}

int cmd_devices(const Args &a) {
    const std::string js = json_call(vox_devices_json);
    if (a.has("--json")) { std::puts(js.c_str()); return 0; }
    const vox::json j = vox::json::parse(js);
    const auto &c = j["cpu"];
    auto gib = [](double b) { return b / (1024.0 * 1024 * 1024); };
    std::printf("cpu  %d logical cores (%d usable by this process), RAM %.1f GiB total, %.1f GiB available%s\n", c["cpu_logical"].get<int>(),
                c["cpu_allowed"].get<int>(), gib(c["ram_total"].get<double>()), gib(c["ram_available"].get<double>()),
                c["cgroup_limit"].get<uint64_t>() ? (", container limit leaves " + std::to_string(gib(c["cgroup_limit"].get<double>())).substr(0, 4) + " GiB").c_str() : "");
    for (const auto &g : j["gpus"])
        std::printf("vulkan:%d  %s  [%s]  VRAM %.1f GiB, free %.1f GiB%s%s%s\n", g["index"].get<int>(), g["name"].get<std::string>().c_str(),
                    g["software"].get<bool>() ? "cpu(software)" : g["type"].get<std::string>().c_str(), gib(g["vram_total"].get<double>()),
                    gib(g["vram_free"].get<double>()), g["unified_memory"].get<bool>() ? ", shares system RAM" : "",
                    g["cooperative_matrix"].get<bool>() ? ", coopmat" : "", g["memory_budget"].get<bool>() ? "" : " (free: estimate)");
    if (j["gpus"].empty()) std::puts("no Vulkan devices");
    std::string fams;
    for (const auto &f : j["families"]) fams += (fams.empty() ? "" : ", ") + f.get<std::string>();
    std::printf("model families: %s\n", fams.c_str());
    return 0;
}

int cmd_plan(const Args &a) {
    if (!a.has("--model")) die("--model is required");
    vox::json cfg = resource_json(a).empty() ? vox::json::object() : vox::json::parse(resource_json(a));
    if (a.has("--device")) cfg["device"] = a.get("--device");
    const std::string cfg_s = cfg.dump();
    const std::string s = fetch([&](char *b, size_t c, size_t *l) { return vox_plan_json(a.get("--model").c_str(), cfg_s.c_str(), b, c, l); }, "plan");
    const vox::json j = vox::json::parse(s);
    if (a.has("--json")) std::puts(j.dump(2).c_str());
    else std::fputs(j["summary"].get<std::string>().c_str(), stdout);
    return j["feasible"].get<bool>() ? 0 : 3;
}

int cmd_info(const Args &a) {
    if (a.pos.empty()) die("usage: vox info <file>");
    const std::string &path = a.pos[0];
    if (path.size() > 7 && path.substr(path.size() - 7) == ".tvoice") {
        vox_voice *v = nullptr;
        check(vox_profile_open(path.c_str(), &v), "open profile");
        std::string s = json_of([&](char *b, size_t c, size_t *l) { return vox_voice_info(v, b, c, l); });
        std::cout << vox::json::parse(s).dump(2) << "\n";
        vox_voice_free(v);
        return 0;
    }
    vox_engine_opts o;
    vox_engine_opts_default(&o);
    std::string dev = a.get("--device", "auto");
    o.device = dev.c_str();
    vox_engine *e = nullptr;
    if (vox_engine_open(path.c_str(), &o, &e) < 0) {
        // Модель не запускается здесь — показываем хотя бы манифест.
        std::string why = vox_last_error();
        vox::TmodFile f(path);
        vox::json m = f.manifest();
        m.erase("blobs");
        std::cout << m.dump(2) << "\n";
        std::fprintf(stderr, "vox: this build cannot run on this machine: %s\n", why.c_str());
        return 3;
    }
    std::cout << vox::json::parse(engine_json(e)).dump(2) << "\n";
    vox_engine_close(e);
    return 0;
}

vox_voice *load_voice(vox_engine *e, const Args &a);

int cmd_voice_speaker(const Args &a) {
    if (!a.has("-o")) die("-o is required");
    vox_engine *e = open_engine(a);
    vox_voice *v = load_voice(e, a);
    if (a.has("--name")) check(vox_voice_set_name(v, a.get("--name").c_str()), "set name");
    std::string lang;
    vox_gen_params p = gen_params(a, lang);
    check(vox_voice_set_defaults(v, &p), "set defaults");
    check(vox_voice_save(v, a.get("-o").c_str()), "save voice");
    std::fprintf(stderr, "voice saved: %s\n", a.get("-o").c_str());
    vox_voice_free(v);
    vox_engine_close(e);
    return 0;
}

// ---- vox profile: voice profiles (several voices, styles, any model) ----
//   vox profile new P.tvoice [--name NAME]
//   vox profile add P.tvoice --voice anna [--style happy] (REC.wav --text "…" [--lang ru] | --from V.tvoice [--from-voice X])
//   vox profile set P.tvoice --voice anna [--style happy] [--instruct …] [--description …] [--speaker "kokoro=af_heart;qwen3-tts=serena"]
//                    [--speed 1.1] [--options '{"temperature":0.8}'] [--language ru] [--default]
//   vox profile rm P.tvoice --voice anna [--style happy]
//   vox profile patch P.tvoice '{JSON merge patch}'
//   vox profile show P.tvoice [--json]
//   vox profile prepare P.tvoice --model M.tmod [--voice anna] [--style happy] [-o OUT.tvoice]
std::string profile_json(vox_voice *v) {
    return json_of([&](char *b, size_t c, size_t *l) { return vox_voice_info(v, b, c, l); });
}

void print_profile(const vox::json &j) {
    std::printf("profile \"%s\": %zu voice(s), default %s\n", j.value("name", "").c_str(), j["voices"].size(),
                j.value("default_voice", "-").c_str());
    for (auto it = j["voices"].begin(); it != j["voices"].end(); ++it) {
        const vox::json &v = it.value();
        std::printf("  %s%s  language %s", it.key().c_str(), it.key() == j.value("default_voice", "") ? " (default)" : "",
                    v.value("language", "auto").c_str());
        if (!v.value("description", "").empty()) std::printf("  \"%s\"", v["description"].get<std::string>().c_str());
        std::printf("\n");
        for (auto sp = v["speakers"].begin(); sp != v["speakers"].end(); ++sp)
            std::printf("      speaker on %s: %s\n", sp.key().c_str(), sp.value().get<std::string>().c_str());
        for (auto st = v["styles"].begin(); st != v["styles"].end(); ++st) {
            const vox::json &s = st.value();
            std::printf("    style %s%s", st.key().c_str(), st.key() == v.value("default_style", "") ? " (default)" : "");
            const std::string ref = s.value("ref", "");
            if (!ref.empty() && j["recordings"].contains(ref)) {
                const vox::json &r = j["recordings"][ref];
                std::printf("  recording %s (%.1f s): \"%s\"", ref.c_str(), r.value("seconds", 0.0), r.value("text", "").c_str());
            }
            if (!s.value("instruct", "").empty()) std::printf("  instruct \"%s\"", s["instruct"].get<std::string>().c_str());
            if (!s["options"].empty()) std::printf("  options %s", s["options"].dump().c_str());
            std::printf("\n");
        }
    }
    if (!j["cache"].empty()) {
        std::printf("  built for:\n");
        for (const auto &c : j["cache"])
            std::printf("    %s/%s on %s/%s (%s)\n", c.value("voice", "").c_str(), c.value("style", "").c_str(),
                        c["model"].value("family", "").c_str(), c["model"].value("variant", "").c_str(), c.value("mode", "").c_str());
    }
}

int cmd_profile(const Args &a) {
    const char *use = "usage: vox profile new|add|set|rm|patch|show|prepare FILE.tvoice … (vox --help)";
    if (a.pos.size() < 2) die(use);
    const std::string sub = a.pos[0], path = a.pos[1];
    vox_voice *v = nullptr;
    auto save = [&](const std::string &out) {
        check(vox_voice_save(v, out.c_str()), "save profile");
        std::fprintf(stderr, "profile saved: %s\n", out.c_str());
    };
    if (sub == "new") {
        check(vox_profile_new(a.get("--name", "profile").c_str(), &v), "new profile");
        save(path);
        vox_voice_free(v);
        return 0;
    }
    check(vox_profile_open(path.c_str(), &v), "open profile");
    const std::string voice = a.get("--voice"), style = a.get("--style");
    if (sub == "show") {
        const vox::json j = vox::json::parse(profile_json(v));
        if (a.has("--json")) std::puts(j.dump(2).c_str());
        else print_profile(j);
    } else if (sub == "add") {
        if (voice.empty()) die("vox profile add FILE --voice NAME [--style S] (REC.wav --text \"…\" | --from V.tvoice)");
        if (a.has("--from")) {
            vox_voice *src = nullptr;
            check(vox_profile_open(a.get("--from").c_str(), &src), "open source voice");
            check(vox_voice_merge(v, src, a.has("--from-voice") ? a.get("--from-voice").c_str() : nullptr, voice.c_str(),
                                  style.empty() ? nullptr : style.c_str()), "add voice");
            vox_voice_free(src);
        } else {
            if (a.pos.size() < 3) die("vox profile add FILE --voice NAME [--style S] REC.wav --text \"what is said in it\"");
            const vox::Audio rec = vox::wav_read(a.pos[2]);
            check(vox_voice_add_recording(v, voice.c_str(), style.empty() ? nullptr : style.c_str(), rec.pcm.data(), rec.pcm.size(),
                                          rec.sample_rate, a.get("--text").c_str(), a.has("--lang") ? a.get("--lang").c_str() : nullptr), "add recording");
        }
        save(a.get("-o", path));
    } else if (sub == "set" || sub == "rm") {
        if (voice.empty()) die(std::string("vox profile ") + sub + " FILE --voice NAME [--style S] …");
        vox::json pv = vox::json::object();
        if (sub == "rm") {
            if (style.empty()) pv = nullptr;
            else pv["styles"][style] = nullptr;
        } else {
            vox::json st = vox::json::object();
            if (a.has("--instruct")) st["instruct"] = a.get("--instruct");
            if (a.has("--speed")) st["options"]["speed"] = std::stod(a.get("--speed"));
            if (a.has("--options")) {
                vox::json o = vox::json::parse(a.get("--options"));
                for (auto it = o.begin(); it != o.end(); ++it) st["options"][it.key()] = it.value();
            }
            if (!style.empty()) pv["styles"][style] = st;
            else if (!st.empty()) {   // no --style: voice-level options; an instruction goes to the default style
                if (st.contains("options")) pv["options"] = st["options"];
                if (st.contains("instruct")) {
                    const vox::json j = vox::json::parse(profile_json(v));
                    std::string ds = "default";
                    if (j["voices"].contains(voice)) ds = j["voices"][voice].value("default_style", "default");
                    pv["styles"][ds]["instruct"] = st["instruct"];
                }
            }
            if (a.has("--description")) pv["description"] = a.get("--description");
            if (a.has("--language")) pv["language"] = a.get("--language");
            if (a.has("--default-style")) pv["default_style"] = a.get("--default-style");
            if (a.has("--speaker")) {   // "kokoro=af_heart;qwen3-tts/custom_voice=serena", empty value removes
                std::string spec = a.get("--speaker");
                size_t i0 = 0;
                while (i0 <= spec.size()) {
                    size_t i1 = spec.find(';', i0);
                    if (i1 == std::string::npos) i1 = spec.size();
                    const std::string item = spec.substr(i0, i1 - i0);
                    const size_t eq = item.find('=');
                    if (eq == std::string::npos) die("--speaker wants MODEL=NAME[;MODEL=NAME…] (MODEL: family or family/variant, e.g. kokoro=af_heart)");
                    const std::string val = item.substr(eq + 1);
                    pv["speakers"][item.substr(0, eq)] = val.empty() ? vox::json(nullptr) : vox::json(val);
                    i0 = i1 + 1;
                }
            }
        }
        vox::json patch = {{"voices", {{voice, pv}}}};
        if (a.has("--default")) patch["default_voice"] = voice;
        check(vox_voice_update(v, patch.dump().c_str()), "update profile");
        save(a.get("-o", path));
    } else if (sub == "patch") {
        if (a.pos.size() < 3) die("vox profile patch FILE '{\"voices\": {…}}'");
        check(vox_voice_update(v, a.pos[2].c_str()), "update profile");
        save(a.get("-o", path));
    } else if (sub == "prepare") {
        if (!a.has("--model")) die("vox profile prepare FILE --model M.tmod [--voice V] [--style S] [-o OUT]");
        vox_engine *e = open_engine(a);
        std::string r(1 << 20, '\0');   // one call: preparing builds voices, a second call would report them as already built
        size_t n = 0;
        check(vox_voice_prepare(e, v, voice.empty() ? nullptr : voice.c_str(), style.empty() ? nullptr : style.c_str(), r.data(), r.size(), &n),
              "prepare profile");
        r.resize(std::min(n, r.size() - 1));
        const vox::json j = vox::json::parse(r);
        for (const auto &x : j["voices"])
            std::fprintf(stderr, "  %s/%s: %s\n", x.value("voice", "").c_str(), x.value("style", "").c_str(), x.value("built", false) ? "built" : "already built");
        for (const auto &n : j["notes"]) std::fprintf(stderr, "  note: %s\n", n.get<std::string>().c_str());
        save(a.get("-o", path));
        vox_engine_close(e);
    } else {
        die(use);
    }
    vox_voice_free(v);
    return 0;
}

int cmd_voice(const Args &a) {
    if (!a.pos.empty() && a.pos[0] == "speaker") return cmd_voice_speaker(a);
    if (a.pos.size() < 2 || a.pos[0] != "create") die("usage: vox voice create <ref.wav> --model M --text T -o out.tvoice\n       vox voice speaker --model M [--speaker NAME] [--instruct TEXT] -o out.tvoice");
    if (!a.has("-o")) die("-o is required");
    vox::Audio ref = vox::wav_read(a.pos[1]);
    vox_engine *e = open_engine(a);
    vox_voice_mode mode = a.has("--xvector-only") ? VOX_VOICE_XVECTOR_ONLY : VOX_VOICE_ICL;
    std::string text = a.get("--text");
    vox_voice *v = nullptr;
    check(vox_voice_create(e, ref.pcm.data(), ref.pcm.size(), ref.sample_rate, text.c_str(), mode, &v), "create voice");
    if (a.has("--name")) check(vox_voice_set_name(v, a.get("--name").c_str()), "set name");
    std::string lang;
    vox_gen_params p = gen_params(a, lang);
    check(vox_voice_set_defaults(v, &p), "set defaults");
    check(vox_voice_save(v, a.get("-o").c_str()), "save voice");
    std::fprintf(stderr, "voice saved: %s (%.1f s reference)\n", a.get("-o").c_str(),
                 double(ref.pcm.size()) / ref.sample_rate);
    vox_voice_free(v);
    vox_engine_close(e);
    return 0;
}

struct RunStats {
    double first_chunk_ms = -1, total_ms = 0;
    size_t samples = 0;
};

// Один синтез: текст целиком, чанки отдаются в on_chunk.
template <typename F>
RunStats synth(vox_engine *e, vox_voice *v, const vox_gen_params &p, const std::string &text, F &&on_chunk) {
    RunStats r;
    auto t0 = Clock::now();
    vox_session *s = nullptr;
    check(vox_session_create(e, v, &p, &s), "create session");
    check(vox_session_feed_text(s, text.c_str()), "feed text");
    check(vox_session_end_text(s), "end text");
    std::vector<float> buf(8192);
    for (;;) {
        size_t n = 0;
        vox_status st = vox_session_pull(s, buf.data(), buf.size(), &n);
        if (st == VOX_DONE) break;
        check(st, "synthesis");
        if (st == VOX_AGAIN) die("internal: session wants text after end_text");
        if (r.first_chunk_ms < 0) r.first_chunk_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        r.samples += n;
        on_chunk(buf.data(), n);
    }
    vox_session_free(s);
    r.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return r;
}

vox_voice *load_voice(vox_engine *e, const Args &a) {
    vox_voice *v = nullptr;
    if (!a.has("--voice") && (a.has("--speaker") || a.has("--instruct"))) {  // готовый диктор / описание голоса без файла
        check(vox_voice_from_speaker(e, a.has("--speaker") ? a.get("--speaker").c_str() : nullptr,
                                     a.has("--instruct") ? a.get("--instruct").c_str() : nullptr, &v), "speaker voice");
        return v;
    }
    if (!a.has("--voice")) die("--voice (or --speaker / --instruct) is required");
    check(vox_voice_load(e, a.get("--voice").c_str(), &v), "load voice");
    return v;
}

int cmd_say(const Args &a) {
    std::string text;
    if (a.has("--stdin")) {
        text.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
    } else if (!a.pos.empty()) {
        text = a.pos[0];
    } else {
        die("text or --stdin is required");
    }
    bool stream = a.has("--stream");
    if (!stream && !a.has("-o")) die("-o out.wav or --stream is required");
    vox_engine *e = open_engine(a);
    vox_voice *v = load_voice(e, a);
    std::string lang;
    vox_gen_params p = gen_params(a, lang);
#if defined(_WIN32)
    if (stream) _setmode(_fileno(stdout), _O_BINARY);
#endif
    std::vector<float> all;
    RunStats r = synth(e, v, p, text, [&](const float *d, size_t n) {
        if (stream) { std::fwrite(d, sizeof(float), n, stdout); std::fflush(stdout); }
        else all.insert(all.end(), d, d + n);
    });
    uint32_t sr = vox_engine_sample_rate(e);
    if (!stream) vox::wav_write(a.get("-o"), all.data(), all.size(), sr, !a.has("--float"));
    double secs = double(r.samples) / sr;
    std::fprintf(stderr, "%.2f s audio, first chunk %.1f ms, RTF %.3f\n", secs, r.first_chunk_ms,
                 secs > 0 ? r.total_ms / 1000.0 / secs : 0.0);
    vox_voice_free(v);
    vox_engine_close(e);
    return 0;
}


// Живой режим: текст читается из stdin по мере поступления (печатаете, или пайп из LLM), звук идёт по мере готовности.
//   vox live --model M --voice V -o out.wav     текст на лету → одна итоговая запись
//   vox live --model M --voice V --raw          текст на лету → сырой float32 PCM в stdout (ffplay -f f32le -ar 24000 -i -)
// --commit auto|sentence|manual  --idle-ms N (автокоммит после паузы в наборе)  --flush-nl (Enter = flush)
int cmd_live(const Args &a) {
    const bool raw = a.has("--raw");
    if (!raw && !a.has("-o")) die("usage: vox live --model M.tmod --voice V.tvoice (-o out.wav | --raw) [--commit auto|sentence|manual] [--idle-ms N] [--flush-nl]");
    vox_engine *e = open_engine(a);
    vox_voice *v = load_voice(e, a);
    std::string lang;
    vox_gen_params p = gen_params(a, lang);
    vox_session *s = nullptr;
    check(vox_session_create(e, v, &p, &s), "create session");
#if defined(_WIN32)
    if (raw) _setmode(_fileno(stdout), _O_BINARY);
#endif
    std::mutex mu;
    std::string inbox;
    bool eof = false;
    std::thread reader([&] {
        for (;;) {
            const int c = std::fgetc(stdin);   // блокирует только этот поток; stdio отдаёт байт, как только он пришёл
            std::lock_guard<std::mutex> lk(mu);
            if (c == EOF) { eof = true; return; }
            inbox.push_back(char(c));
        }
    });
    reader.detach();
    const bool flush_nl = a.has("--flush-nl");
    const auto t0 = Clock::now();
    double first_text_ms = -1, first_audio_ms = -1;
    std::vector<float> buf(4096), all;
    size_t samples = 0;
    bool ended = false;
    for (;;) {
        std::string take;
        bool at_eof = false;
        {
            std::lock_guard<std::mutex> lk(mu);
            take.swap(inbox);
            at_eof = eof;
        }
        if (!take.empty()) {
            if (first_text_ms < 0) first_text_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            const bool nl = flush_nl && take.find('\n') != std::string::npos;
            check(vox_session_feed_text(s, take.c_str()), "feed text");
            if (nl) check(vox_session_flush(s), "flush");
        }
        if (at_eof && !ended) { check(vox_session_end_text(s), "end text"); ended = true; }
        bool got = false;
        for (;;) {
            size_t n = 0;
            const vox_status st = vox_session_pull(s, buf.data(), buf.size(), &n);
            if (st == VOX_DONE) goto finished;
            check(st, "synthesis");
            if (st == VOX_AGAIN) break;
            if (first_audio_ms < 0) first_audio_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            samples += n;
            got = true;
            if (raw) { std::fwrite(buf.data(), sizeof(float), n, stdout); std::fflush(stdout); }
            else all.insert(all.end(), buf.begin(), buf.begin() + long(n));
        }
        if (!got) std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
finished:
    const uint32_t sr = vox_engine_sample_rate(e);
    if (!raw) vox::wav_write(a.get("-o"), all.data(), all.size(), sr, !a.has("--float"));
    const std::string info = fetch([&](char *b, size_t c, size_t *l) { return vox_session_info(s, b, c, l); }, "session info");
    const vox::json ij = vox::json::parse(info);
    std::fprintf(stderr, "%.2f s audio; first text at %.0f ms, first audio at %.0f ms (+%.0f ms); %lld flushes (%lld by idle)\n",
                 double(samples) / sr, first_text_ms, first_audio_ms, first_audio_ms - first_text_ms, (long long)ij.value("flushes", uint64_t(0)),
                 (long long)ij.value("idle_flushes", uint64_t(0)));
    vox_session_free(s);
    vox_voice_free(v);
    vox_engine_close(e);
    return 0;
}

double peak_rss_mb() {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
#  if defined(__APPLE__)
    return double(ru.ru_maxrss) / 1048576.0;
#  else
    return double(ru.ru_maxrss) / 1024.0;
#  endif
#else
    return 0.0;
#endif
}

// Бенчмарк: первый запуск («холодный»: прогрев, кэши) и последующие («тёплые») считаются отдельно.
int cmd_bench(const Args &a) {
    vox_engine *e = open_engine(a);
    vox_voice *v = load_voice(e, a);
    std::string lang;
    vox_gen_params p = gen_params(a, lang);
    std::string text = a.get("--text", "Это тестовая фраза для замера скорости движка синтеза речи.");
    if (a.has("--text-file")) {
        std::ifstream in(a.get("--text-file"));
        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    int runs = std::max(1, std::stoi(a.get("--runs", "3")));
    uint32_t sr = vox_engine_sample_rate(e);
    std::vector<RunStats> rs;
    for (int i = 0; i < runs; ++i) {
        rs.push_back(synth(e, v, p, text, [](const float *, size_t) {}));
        const auto &r = rs.back();
        std::fprintf(stderr, "run %d%s: %.2f s audio, first chunk %.0f ms, total %.0f ms, RTF %.3f\n", i + 1, i == 0 ? " (cold)" : "",
                     double(r.samples) / sr, r.first_chunk_ms, r.total_ms, r.total_ms / 1000.0 / (double(r.samples) / sr));
    }
    auto summarize = [&](size_t from, size_t to) {
        double first = 0, rtf = 0, audio = 0, total = 0;
        size_t n = to - from;
        for (size_t i = from; i < to; ++i) {
            first += rs[i].first_chunk_ms;
            audio += double(rs[i].samples) / sr;
            total += rs[i].total_ms / 1000.0;
            rtf += rs[i].total_ms / 1000.0 / (double(rs[i].samples) / sr);
        }
        return vox::json{{"first_chunk_ms", first / double(n)}, {"rtf", rtf / double(n)}, {"audio_s", audio / double(n)}, {"total_s", total / double(n)}};
    };
    vox::json out = {{"backend", vox::json::parse(engine_json(e))["backend"]}, {"runs", runs}, {"cold", summarize(0, 1)},
                     {"peak_rss_mb", peak_rss_mb()}, {"threads", vox::json(a.get("--threads", "auto"))}};
    if (runs > 1) out["warm"] = summarize(1, size_t(runs));
    {
        out["stats"] = vox::json::parse(fetch([&](char *b, size_t c, size_t *l) { return vox_engine_stats(e, b, c, l); }, "stats"));
    }
    std::cout << out.dump() << "\n";
    vox_voice_free(v);
    vox_engine_close(e);
    return 0;
}

// Читает JSON-строки из stdin, печатает JSON-массивы id (для проверки токенизатора).
// vox phonemize --model kokoro.tmod [--lang en-gb] "text"   (or lines of JSON strings on stdin with --json)
int cmd_phonemize(const Args &a) {
    vox_engine *e = open_engine(a);
    const std::string lang = a.get("--lang", "");
    auto one = [&](const std::string &text) {
        return fetch([&](char *b, size_t c, size_t *l) { return vox_engine_phonemize(e, text.c_str(), lang.c_str(), b, c, l); }, "phonemize");
    };
    if (a.has("--json")) {
        std::string line;
        while (std::getline(std::cin, line)) std::cout << one(vox::json::parse(line).get<std::string>()) << "\n" << std::flush;
    } else {
        std::string text;
        for (const auto &p : a.pos) text += (text.empty() ? "" : " ") + p;
        const vox::json r = vox::json::parse(one(text));
        if (r.contains("phonemes") && r["phonemes"].is_string()) std::cout << r["phonemes"].get<std::string>() << "\n";
        else std::cout << r.dump(1) << "\n";   // models without phonemes (CosyVoice): the pieces of text they will speak
    }
    vox_engine_close(e);
    return 0;
}

int cmd_tokenize(const Args &a) {
    if (!a.has("--model")) die("--model is required");
    vox::TmodFile f(a.get("--model"));
    vox::Tokenizer tok = vox::Tokenizer::from_tmod(f);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::string text = vox::json::parse(line).get<std::string>();
        std::cout << vox::json(tok.encode(text)).dump() << "\n";
    }
    return 0;
}


// ---------------------------------------------------------------------------------------------
// vox verify: сверка движка с эталонным дампом (tools/dump_reference.py — настоящая модель в PyTorch,
// tools/ref_numpy.py — независимая numpy-реализация). Две проверки:
//   1) «с учителем»: на каждом шаге в движок подаются эталонные коды, сравниваются логики talker'а,
//      логиты code predictor'а и итоговое аудио декодера кодека;
//   2) «свободный полёт»: жадная генерация через публичный C ABI и сравнение аудио.
// ---------------------------------------------------------------------------------------------
struct Cmp {
    double max_abs = 0, max_rel = 0, sum_abs = 0;
    size_t n = 0, argmax_ok = 0, argmax_n = 0;
    // распределительные метрики для квантованных моделей: KL(ref‖ours) и попадание эталонного argmax в наш top-5
    double kl_sum = 0, kl_max = 0;
    size_t top5_ok = 0, dist_n = 0;
    void add_dist(const float *ours, const float *ref, size_t len) {
        auto lse = [&](const float *x) {
            double m = x[0];
            for (size_t i = 1; i < len; ++i) m = std::max<double>(m, x[i]);
            double s = 0;
            for (size_t i = 0; i < len; ++i) s += std::exp(double(x[i]) - m);
            return m + std::log(s);
        };
        const double la = lse(ours), lb = lse(ref);
        double kl = 0;
        for (size_t i = 0; i < len; ++i) {
            const double lp = double(ref[i]) - lb, lq = double(ours[i]) - la;
            kl += std::exp(lp) * (lp - lq);
        }
        kl_sum += kl; kl_max = std::max(kl_max, kl); ++dist_n;
        const size_t ib = size_t(std::max_element(ref, ref + len) - ref);
        size_t better = 0;
        for (size_t i = 0; i < len; ++i) better += ours[i] > ours[ib];
        top5_ok += better < 5;
    }
    double mean_kl() const { return dist_n ? kl_sum / double(dist_n) : 0.0; }
    double top5_frac() const { return dist_n ? double(top5_ok) / double(dist_n) : 1.0; }
    bool ok_quant(double kl_tol, double top5_min) const { return mean_kl() <= kl_tol && top5_frac() >= top5_min; }
    void print_quant(const char *name, double kl_tol, double top5_min) const {
        std::printf("  %-18s mean KL=%.4f (max %.4f)  ref-argmax∈top5 %.2f%%  argmax %zu/%zu  %s\n", name, mean_kl(), kl_max, 100 * top5_frac(),
                    argmax_ok, argmax_n, ok_quant(kl_tol, top5_min) ? "OK" : "FAIL");
    }
    void add(const float *a, const float *b, size_t len, bool argmax = false) {
        double mb = 0;
        for (size_t i = 0; i < len; ++i) mb = std::max(mb, std::fabs(double(b[i])));
        double scale = std::max(1.0, mb);
        for (size_t i = 0; i < len; ++i) {
            double d = std::fabs(double(a[i]) - double(b[i]));
            max_abs = std::max(max_abs, d);
            max_rel = std::max(max_rel, d / scale);
            sum_abs += d;
        }
        n += len;
        if (argmax) {
            ++argmax_n;
            size_t ia = size_t(std::max_element(a, a + len) - a), ib = size_t(std::max_element(b, b + len) - b);
            argmax_ok += ia == ib;
        }
    }
    void print(const char *name, double tol) const {
        std::printf("  %-18s max|Δ|=%.3e  max rel=%.3e  mean|Δ|=%.3e", name, max_abs, max_rel, n ? sum_abs / double(n) : 0.0);
        if (argmax_n) std::printf("  argmax %zu/%zu", argmax_ok, argmax_n);
        std::printf("  %s\n", ok(tol) ? "OK" : "FAIL");
    }
    bool check_argmax = true;
    bool ok(double tol) const { return max_rel <= tol && (!check_argmax || argmax_ok == argmax_n); }
};

int cmd_verify(const Args &a) {
    if (!a.has("--model") || !a.has("--dump")) die("usage: vox verify --model M.tmod --dump case.safetensors [--voice V.tvoice] [--tol 2e-3]");
    const double tol = std::stod(a.get("--tol", "2e-3"));
    // --quant: режим для q8/квантованных моделей — вместо argmax и max-rel сравниваем распределения (KL, top-5) и аудио с мягким допуском
    const bool quant = a.get("--quant", "0") != "0";
    const double kl_tol = std::stod(a.get("--kl-tol", "0.02")), top5_min = std::stod(a.get("--top5-min", "0.97"));
    const double wav_tol = std::stod(a.get("--wav-tol", "0.2"));
    {   // модели без покадровых кодов (F5-TTS…) сверяют свои стадии сами: см. Backend::debug_check
        auto tm0 = std::make_shared<vox::TmodFile>(a.get("--model"));
        auto be0 = vox::make_backend(tm0, a.get("--device", "auto"), 0);
        if (be0->direct()) {
            std::string rep;
            const bool ok = be0->debug_check(a.get("--dump"), tol, rep);
            std::printf("verify: %s, backend %s, tol %.1e\n%s%s\n", a.get("--dump").c_str(), be0->name().c_str(), tol, rep.c_str(), ok ? "PASS" : "FAIL");
            return ok ? 0 : 1;
        }
    }
    vox::SafetensorsFile d = vox::safetensors_load(a.get("--dump"));
    std::string voice_path = a.get("--voice", d.metadata.count("voice") ? d.metadata["voice"] : "");
    // дамп CustomVoice/VoiceDesign хранит диктора и инструкцию вместо пути к голосу
    std::string spk = a.get("--speaker", d.metadata.count("speaker") ? d.metadata["speaker"] : ""),
                ins = a.get("--instruct", d.metadata.count("instruct") ? d.metadata["instruct"] : "");
    const bool named_voice = voice_path.empty() && (!spk.empty() || !ins.empty());
    if (voice_path.empty() && !named_voice) die("--voice is required (the dump has no voice path)");
    std::string text = d.metadata["text"], lang = d.metadata.count("language") ? d.metadata["language"] : "auto";

    auto tl = d.tensors.at("talker_logits");
    auto codes_t = d.tensors.at("codes");
    auto wav_t = d.tensors.at("wav");
    const size_t steps = size_t(tl.shape[0]), V = size_t(tl.shape[1]), frames = size_t(codes_t.shape[0]);
    const size_t Q = size_t(codes_t.shape[1]);
    auto tlv = tl.as<float>();
    auto codes = codes_t.as<int32_t>();
    auto wav = wav_t.as<float>();
    std::vector<float> plv;
    size_t PV = 0;
    if (d.tensors.count("predictor_logits")) { plv = d.tensors["predictor_logits"].as<float>(); PV = size_t(d.tensors["predictor_logits"].shape[2]); }

    auto tmod = std::make_shared<vox::TmodFile>(a.get("--model"));
    auto be = vox::make_backend(tmod, a.get("--device", "auto"), 0);
    vox::VoiceData v;
    if (named_voice) {
        v.mode = VOX_VOICE_SPEAKER;
        v.speaker = spk;
        v.instruct = ins;
    } else {
        v = vox::voice_load(voice_path);
    }
    std::printf("verify: %s, %zu steps, %zu frames, backend %s\n  text: %s\n", a.get("--dump").c_str(), steps, frames,
                be->name().c_str(), text.c_str());
    if (be->info().num_codebooks != Q) die("dump has " + std::to_string(Q) + " codebooks, model has " + std::to_string(be->info().num_codebooks));

    // ---- 1) с учителем ----
    Cmp ct, cp, cw;
    if (a.get("--no-argmax", "0") != "0") ct.check_argmax = cp.check_argmax = false;
    auto st = be->begin(v, lang);
    { vox::SessionOptions so; so.decoder_context_frames = 0xFFFFFFFFu; so.max_frames = uint32_t(steps) + 16;
      so.decoder_mode = VOX_DECODER_REFERENCE; be->configure(*st, so); }
    be->append_text(*st, be->tokenize(text));
    std::vector<float> lg, pg;
    std::vector<float> pcm;
    for (size_t s = 0; s < steps; ++s) {
        be->talker_logits(*st, true, lg);
        if (lg.size() != V) die("talker logits size " + std::to_string(lg.size()) + " != dump " + std::to_string(V));
        ct.add(lg.data(), &tlv[s * V], V, true);
        if (quant) ct.add_dist(lg.data(), &tlv[s * V], V);
        if (s >= frames) break;  // шаг eos: дальше кадра нет
        be->accept_code0(*st, codes[s * Q]);
        for (size_t cb = 1; cb < Q; ++cb) {
            be->predictor_logits(*st, uint32_t(cb), pg);
            if (!plv.empty()) {
                if (pg.size() != PV) die("predictor logits size mismatch");
                cp.add(pg.data(), &plv[(s * (Q - 1) + (cb - 1)) * PV], PV, true);
                if (quant) cp.add_dist(pg.data(), &plv[(s * (Q - 1) + (cb - 1)) * PV], PV);
            }
            be->accept_code(*st, uint32_t(cb), codes[s * Q + cb]);
        }
        be->end_frame(*st);
    }
    // декодер: по 4 кадра, как в стриминге
    std::vector<int16_t> c16(codes.begin(), codes.end());
    for (size_t f = 0; f < frames; f += 4) be->decode(*st, &c16[f * Q], std::min<size_t>(4, frames - f), pcm);
    if (a.has("--dump-wav")) vox::wav_write(a.get("--dump-wav"), pcm.data(), pcm.size(), be->info().sample_rate, false);
    bool shape_ok = pcm.size() == wav.size();
    if (!shape_ok) std::printf("  WAV LENGTH MISMATCH: ours %zu, reference %zu\n", pcm.size(), wav.size());
    else cw.add(pcm.data(), wav.data(), wav.size());
    std::puts(quant ? "teacher-forced (quantized metrics):" : "teacher-forced:");
    bool ok;
    if (quant) {
        ct.print_quant("talker logits", kl_tol, top5_min);
        if (cp.n) cp.print_quant("predictor logits", kl_tol, top5_min);
        if (shape_ok) std::printf("  %-18s max rel=%.3e (допуск %.2f)  mean|Δ|=%.3e\n", "codec decoder wav", cw.max_rel, wav_tol, cw.n ? cw.sum_abs / double(cw.n) : 0.0);
        ok = shape_ok && ct.ok_quant(kl_tol, top5_min) && (!cp.n || cp.ok_quant(kl_tol, top5_min)) && cw.max_rel <= wav_tol;
    } else {
        ct.print("talker logits", tol);
        if (cp.n) cp.print("predictor logits", tol);
        const double wt = a.has("--wav-tol") ? wav_tol : tol * 10;  // свёрточная цепочка накапливает ошибку сильнее
        if (shape_ok) cw.print("codec decoder wav", wt);
        ok = shape_ok && ct.ok(tol) && (!cp.n || cp.ok(tol)) && cw.max_rel <= wt;
    }

    // ---- 2) свободный полёт (жадный, через C ABI) ----
    // в --quant свободный полёт расходится с эталоном после первого другого токена, сравнивать аудио бессмысленно
    if (a.get("--free", quant ? "0" : "1") != "0") {
        vox_engine_opts o; vox_engine_opts_default(&o);
        std::string dev = a.get("--device", "auto"); o.device = dev.c_str();
        vox_engine *e = nullptr; check(vox_engine_open(a.get("--model").c_str(), &o, &e), "open model");
        vox_voice *vv = nullptr;
        if (named_voice) check(vox_voice_from_speaker(e, spk.empty() ? nullptr : spk.c_str(), ins.empty() ? nullptr : ins.c_str(), &vv), "speaker voice");
        else check(vox_voice_load(e, voice_path.c_str(), &vv), "load voice");
        vox_gen_params p; vox_gen_params_default(&p);
        p.temperature = 0; p.sub_temperature = 0;
        p.max_frames = uint32_t(frames);  // при упоре в max_new_tokens оригинал отдаёт steps-1 полных кадров
        p.decoder_mode = VOX_DECODER_REFERENCE;  // нарезка декодера как в qwen_tts (куски по 300 кадров)
        p.language = lang.c_str();
        std::vector<float> got;
        synth(e, vv, p, text, [&](const float *x, size_t n) { got.insert(got.end(), x, x + n); });
        std::puts("free-running greedy (C ABI):");
        Cmp cf;
        bool len_ok = got.size() == wav.size();
        if (!len_ok) std::printf("  WAV LENGTH MISMATCH: ours %zu, reference %zu\n", got.size(), wav.size());
        else cf.add(got.data(), wav.data(), wav.size());
        if (len_ok) cf.print("wav", tol * 10);
        ok = ok && len_ok && cf.max_rel <= tol * 10;
        vox_voice_free(vv);
        vox_engine_close(e);
    }
    std::puts(ok ? "RESULT: PASS" : "RESULT: FAIL");
    return ok ? 0 : 1;
}

// ---- vox tune: measure the candidate configurations of a model on this computer, keep the fastest ----
std::string self_exe(const char *argv0) {
#if defined(__linux__)
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n > 0) return std::string(buf, size_t(n));
#endif
    return argv0;
}
std::string shq(const std::string &s) {
#if defined(_WIN32)
    return "\"" + s + "\"";
#else
    std::string q = "'";
    for (char c : s) q += (c == '\'') ? std::string("'\\''") : std::string(1, c);
    return q + "'";
#endif
}
const char *g_argv0 = "vox";

int cmd_tune(const Args &a) {
    if (!a.has("--model")) die("--model is required");
    const std::string model = a.get("--model");
    if (a.has("--forget")) {
        check(vox_tune_forget(model.c_str()), "forget");
        std::fprintf(stderr, "the tuning of %s on this computer is forgotten\n", model.c_str());
        return 0;
    }
    const vox::json info = vox::json::parse(fetch([&](char *b, size_t c, size_t *l) { return vox_tune_info(model.c_str(), b, c, l); }, "tune info"));
    if (a.has("--show")) { std::cout << info.dump(1) << "\n"; return 0; }
    const vox::json insp = vox::json::parse(fetch([&](char *b, size_t c, size_t *l) { return vox_model_inspect(model.c_str(), b, c, l); }, "inspect"));
    const vox::json caps = insp.value("capabilities", vox::json::object());
    // the voice every candidate speaks with
    std::string vargs;
    if (a.has("--voice")) vargs = " --voice " + shq(a.get("--voice"));
    else if (a.has("--speaker")) vargs = " --speaker " + shq(a.get("--speaker"));
    else if (caps.value("named_speakers", false) && caps.contains("speakers") && !caps["speakers"].empty())
        vargs = " --speaker " + shq(caps["speakers"][0].get<std::string>());
    else if (caps.value("requires_instruct", false)) vargs = " --instruct " + shq("A calm, clear adult voice.");
    else die("this model speaks with a cloned voice: give one with --voice V.tvoice (vox voice create …)");
    if (a.has("--instruct") && !a.has("--voice")) vargs += " --instruct " + shq(a.get("--instruct"));
    std::string text = a.get("--text", "");
    if (text.empty()) {
        const vox::json langs = caps.value("languages", vox::json::array());
        auto has = [&](const char *l) { for (const auto &x : langs) if (x.is_string() && x.get<std::string>().rfind(l, 0) == 0) return true; return false; };
        if (!langs.empty() && !has("en") && !has("auto") && has("zh")) text = "今天天气很好，我们一起去公园散步吧。这是一个测试，看看这台电脑说话有多快。";
        else if (!langs.empty() && !has("en") && !has("auto") && has("ru")) text = "Привет! Это короткая проверка движка. Она показывает, как быстро этот компьютер говорит.";
        else text = "Hello! This is a short test of the speech engine. It shows how fast this computer can speak, one sentence after another.";
    }
    const std::string goal = a.get("--goal", "speed");
    if (goal != "speed" && goal != "latency") die("--goal is speed (the lowest RTF) or latency (the earliest first sound)");
    const int runs = std::max(1, std::stoi(a.get("--runs", "2")));
    const std::string exe = self_exe(g_argv0);
    const vox::json cands = info.value("candidates", vox::json::array());
    std::fprintf(stderr, "tuning %s: %zu configuration(s), %d run(s) each, goal: %s\n", model.c_str(), cands.size(), runs, goal.c_str());
    vox::json results = vox::json::array();
    int best = -1;
    double best_score = 1e300;
    for (size_t i = 0; i < cands.size(); ++i) {
        const std::string dev = cands[i].value("device", "cpu"), label = cands[i].value("label", dev);
        vox::json r = {{"label", label}, {"device", dev}};
        if (a.has("--dry-run")) { results.push_back(r); continue; }
        std::fprintf(stderr, "  [%zu/%zu] %-44s ", i + 1, cands.size(), label.c_str());
        std::fflush(stderr);
        const std::string cmd = shq(exe) + " bench --model " + shq(model) + " --device " + shq(dev + ",strict") + vargs + " --text " + shq(text) +
                                " --runs " + std::to_string(runs) + " 2>&1";
#if defined(_WIN32)
        FILE *pp = _popen(cmd.c_str(), "r");
#else
        FILE *pp = popen(cmd.c_str(), "r");
#endif
        std::string outp;
        if (pp) {
            char buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, pp)) > 0) outp.append(buf, n);
#if defined(_WIN32)
            _pclose(pp);
#else
            pclose(pp);
#endif
        }
        vox::json bj;
        std::istringstream ls(outp);
        for (std::string line; std::getline(ls, line);)
            if (!line.empty() && line[0] == '{') try { bj = vox::json::parse(line); } catch (const std::exception &) {}
        if (!bj.is_object()) {
            std::string why = outp.substr(outp.size() > 300 ? outp.size() - 300 : 0);
            for (char &c : why) if (c == '\n') c = ' ';
            r["error"] = why;
            std::fprintf(stderr, "failed: %s\n", why.c_str());
            results.push_back(r);
            continue;
        }
        const vox::json &w = bj.contains("warm") ? bj["warm"] : bj["cold"];
        r["rtf"] = w.value("rtf", 0.0);
        r["first_chunk_ms"] = w.value("first_chunk_ms", 0.0);
        r["peak_rss_mb"] = bj.value("peak_rss_mb", 0.0);
        const vox::json be = bj.value("stats", vox::json::object()).value("backend", vox::json::object());
        if (be.contains("gemm") && be["gemm"].is_string() && !be["gemm"].get<std::string>().empty()) {
            std::string gm = be["gemm"].get<std::string>();
            r["gemm"] = gm.substr(0, gm.find(' '));
        }
        r["backend"] = bj.value("backend", "");
        const double score = goal == "speed" ? r["rtf"].get<double>() : r["first_chunk_ms"].get<double>();
        std::fprintf(stderr, "RTF %.3f, first sound %.0f ms\n", r["rtf"].get<double>(), r["first_chunk_ms"].get<double>());
        if (score > 0 && score < best_score) { best_score = score; best = int(results.size()); }
        results.push_back(r);
    }
    if (a.has("--dry-run")) { std::cout << results.dump(1) << "\n"; return 0; }
    if (best < 0) die("no configuration worked; nothing saved");
    const vox::json &b = results[size_t(best)];
    vox::json entry = {{"device", b["device"]}, {"goal", goal}, {"rtf", b["rtf"]}, {"first_chunk_ms", b["first_chunk_ms"]}, {"label", b["label"]},
                       {"candidates", results}, {"text_chars", text.size()}};
    if (b.contains("gemm")) entry["gemm"] = b["gemm"];
    check(vox_tune_save(model.c_str(), entry.dump().c_str()), "save");
    std::fprintf(stderr, "best: %s (RTF %.3f, first sound %.0f ms); saved to %s\n", b["label"].get<std::string>().c_str(), b["rtf"].get<double>(),
                 b["first_chunk_ms"].get<double>(), info.value("cache", "").c_str());
    std::fprintf(stderr, "engines opened with the default resources (device auto) now use it; VOX_TUNE=0 or --device … overrides\n");
    std::cout << entry.dump() << "\n";
    return 0;
}

int cmd_make_test_model(const Args &a) {
    const std::string out = a.get("-o", "test.tmod");
    vox::write_test_model(out);
    std::fprintf(stderr, "test model written: %s\n", out.c_str());
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];
    Args a = parse(argc, argv, 2);
    if (cmd == "-web" || cmd == "--web") cmd = "web";   // vox -web: the engine for Vox Studio
    if (cmd == "-h" || cmd == "--help" || cmd == "help" || a.has("--help")) { usage(); return 0; }
    try {
        if (cmd == "devices") return cmd_devices(a);
        if (cmd == "plan") return cmd_plan(a);
        if (cmd == "info") return cmd_info(a);
        if (cmd == "voice") return cmd_voice(a);
        if (cmd == "profile") return cmd_profile(a);
        if (cmd == "say") return cmd_say(a);
        if (cmd == "live") return cmd_live(a);
        if (cmd == "bench") return cmd_bench(a);
        if (cmd == "tune") { g_argv0 = argv[0]; return cmd_tune(a); }
        if (cmd == "tokenize") return cmd_tokenize(a);
        if (cmd == "phonemize") return cmd_phonemize(a);
        if (cmd == "verify") return cmd_verify(a);
        if (cmd == "make-test-model") return cmd_make_test_model(a);
        if (cmd == "build" || cmd == "serve" || cmd == "web") {
            // сборка модели и сервер живут в Python-пакете vox: пробрасываем аргументы как есть
            std::string line = "python3 -m vox " + cmd;
            for (int i = 2; i < argc; ++i) {
                std::string arg = argv[i], q = "'";
                for (char c : arg) q += (c == '\'') ? std::string("'\\''") : std::string(1, c);
                line += " " + q + "'";
            }
            int rc = std::system(line.c_str());
            if (rc != 0) std::fprintf(stderr, "vox: нужен Python-пакет vox (pip install .); команда: %s\n", line.c_str());
            return rc == 0 ? 0 : 1;
        }
    } catch (const std::exception &ex) {
        die(ex.what());
    }
    usage();
    return 1;
}
