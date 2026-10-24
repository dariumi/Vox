#pragma once
#include "sampling.hpp"
// Интерфейс бэкенда модели.
//
// Разделение ответственности:
//   бэкенд  — вычисления стадий (talker, code predictor, decoder, speaker encoder)
//   рантайм — сэмплинг, RNG, стриминг, сессии, форматы файлов
// Благодаря этому сэмплинг и детерминизм одинаковы на любом железе,
// а новый бэкенд реализует только математику модели.
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "formats.hpp"
#include "resources.hpp"

namespace vox {

// Переносимая часть профиля голоса + кэши под конкретные сборки.
struct VoiceData {
    std::string name = "voice";
    vox_voice_mode mode = VOX_VOICE_ICL;
    std::string ref_text;
    std::string speaker, instruct;         // режим SPEAKER: имя готового диктора (может быть пустым) и инструкция о стиле
    std::string language = "auto";
    std::string model_family, model_variant, weights_sha256;
    std::vector<float> speaker_emb;        // [D]
    std::vector<int16_t> ref_codes;        // [frames * codebooks], row-major
    int64_t ref_frames = 0, ref_codebooks = 0;
    std::vector<int32_t> ref_text_tokens;
    json sampling = json::object();        // значения по умолчанию для этого голоса
    std::map<std::string, Tensor> kv_cache; // ключи вида "kv.<fingerprint>.<имя>"
    std::map<std::string, Tensor> features; // признаки записи для моделей без кодов (ключи "feat.<имя>"): ссылочный звук, mel и т.п.
    std::vector<uint8_t> ref_audio;        // опционально: исходная запись (любой кодек)
    std::string created;
};

// Состояние одной генерации внутри бэкенда (KV-cache, позиции, состояние декодера).
struct GenState {
    virtual ~GenState() = default;
};

struct ModelInfo {
    uint32_t sample_rate = 24000;
    uint32_t samples_per_frame = 1920;
    uint32_t num_codebooks = 16;
    uint32_t codebook_size = 2048;
    int32_t eos_code = -1;                 // код первого кодбука, означающий конец речи
    // Служебные токены, которые нельзя сэмплировать (кроме eos): [suppress_from, suppress_to)
    int32_t suppress_from = 0, suppress_to = 0;
    uint32_t min_eos_frames = 0;           // сколько первых кадров eos запрещён
};

// Что умеет модель (семейство + вариант); показывается в vox_engine_info / Engine.capabilities.
struct Capabilities {
    bool clone_icl = false;        // клонирование по записи + тексту записи
    bool clone_xvector = false;    // клонирование только по эмбеддингу диктора
    bool named_speakers = false;   // готовые дикторы по имени
    bool instruct = false;         // управление стилем текстовой инструкцией
    bool streaming_text = true;    // текст можно подавать по частям
    bool requires_instruct = false;
    // Двусторонний стриминг: как модель принимает текст и отдаёт звук (движок подстраивает под это политику коммита текста).
    //   text_in: "token" — модель читает текст по мере поступления (слова), звук идёт, пока есть непрочитанный текст;
    //            "chunk" — модель озвучивает куски целиком (предложения/фразы), частичный текст ждёт, пока закончится кусок;
    //            "none"  — текст только целиком (до end_text звука нет).
    //   audio_out: "frames" — звук мелкими порциями, как только готов кадр; "chunks" — звук приходит куском на каждую фразу.
    //   exact: стриминговый вывод при commit=auto совпадает с одноразовым синтезом всего текста (F5 — да; AR-модели читают текст «вживую»).
    std::string text_in = "token", audio_out = "frames";
    bool stream_exact = false;
    std::vector<std::string> speakers, languages;
    std::string default_speaker;   // the speaker to start with (the model's own recommendation), "" = the first one
    bool speed = false;            // the "speed" option changes the speaking rate (F5, Kokoro): markup [speed] and emotions use it
    std::vector<std::string> tags; // non-verbal tags the model reads natively ("laugh", "breath", …), passed through as [tag]
    json to_json() const {
        return {{"clone_icl", clone_icl}, {"clone_xvector", clone_xvector}, {"named_speakers", named_speakers}, {"instruct", instruct},
                {"streaming_text", streaming_text}, {"requires_instruct", requires_instruct}, {"speakers", speakers}, {"languages", languages},
                {"speed", speed}, {"tags", tags}, {"default_speaker", default_speaker.empty() && !speakers.empty() ? speakers.front() : default_speaker},
                {"stream", {{"text_in", text_in}, {"audio_out", audio_out}, {"exact", stream_exact}, {"commit", {"auto", "sentence", "manual"}}}}};
    }
};

struct SessionOptions {
    uint64_t seed = 1;                    // зерно сессии (то же, что у DetRng); модели без покадровых кодов берут из него шум
    json options = json::object();        // параметры генерации: manifest "generation" < значения голоса < extra_json сессии
    uint32_t decoder_context_frames = 0;  // 0 = по умолчанию модели, UINT32_MAX = все
    uint32_t max_frames = 0;              // предел длины генерации (для размера KV-кэша на GPU), 0 = 2048
    uint32_t decoder_mode = 0;            // vox_decoder_mode
};

// Во что обходится одна сессия: память хоста и видеопамять (KV-кэши, состояние декодера).
struct SessionCost { uint64_t ram = 0, vram = 0; };

class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string name() const = 0;
    virtual const ModelInfo &info() const = 0;
    // Отпечаток сборки: веса + точность стадий + цель. Ключ для KV-cache в .tvoice.
    virtual std::string fingerprint() const = 0;

    virtual std::vector<int32_t> tokenize(const std::string &utf8) = 0;
    // Токены текста референса в том виде, как они хранятся в .tvoice
    virtual std::vector<int32_t> tokenize_ref(const std::string &utf8) { return tokenize(utf8); }
    // Сколько байт накопленного текста можно токенизировать уже сейчас (остальное ждёт продолжения).
    virtual size_t text_safe_cut(const std::string &pending) const {
        for (size_t i = pending.size(); i > 0; --i) {
            unsigned char c = static_cast<unsigned char>(pending[i - 1]);
            if (c == ' ' || c == '\n' || c == '\t' || c == '.' || c == ',' || c == '!' || c == '?' || c == ';' || c == ':') return i;
        }
        return 0;
    }

    // Текст → фонемы (модели с фонемным входом, например Kokoro): JSON {"phonemes": "...", "chunks": [...]}.
    virtual std::string phonemize(const std::string &, const std::string & /*lang*/) {
        fail(VOX_ERR_UNSUPPORTED, "this model does not use phonemes");
    }

    // --- клонирование голоса ---
    virtual std::vector<float> speaker_embedding(const float *pcm, size_t n, uint32_t sr) = 0;
    // Возвращает коды [frames * num_codebooks] и число кадров.
    virtual std::vector<int16_t> encode_codes(const float *pcm, size_t n, uint32_t sr, int64_t &frames) = 0;

    // --- генерация ---
    virtual std::unique_ptr<GenState> begin(const VoiceData &voice, const std::string &language) = 0;
    virtual void configure(GenState &, const SessionOptions &) {}
    virtual void append_text(GenState &st, const std::vector<int32_t> &tokens) = 0;
    // true, если talker'у нужно больше текста, чтобы сделать следующий шаг.
    virtual bool needs_text(const GenState &st, bool text_ended) const = 0;
    // Логиты первого кодбука для следующего кадра.
    virtual void talker_logits(GenState &st, bool text_ended, std::vector<float> &logits) = 0;
    virtual void accept_code0(GenState &st, int32_t code) = 0;
    // Логиты кодбука cb (1..N-1) для текущего кадра; предыдущие коды кадра уже приняты.
    virtual void predictor_logits(GenState &st, uint32_t cb, std::vector<float> &logits) = 0;
    virtual void accept_code(GenState &st, uint32_t cb, int32_t code) = 0;
    // Все кодбуки 1..N-1 текущего кадра сразу, с сэмплингом на устройстве (Philox с ключом сессии, счётчик (frame, cb)).
    // codes[0] уже принят. Заполняет codes[1..N-1] и принимает их. false → движок делает это по одному кодбуку.
    virtual bool predict_frame(GenState &, const SampleParams &, uint32_t /*key0*/, uint32_t /*key1*/, uint64_t /*frame*/,
                               int32_t * /*codes*/) { return false; }
    // Завершает кадр (все кодбуки приняты): talker продвигается на одну позицию.
    virtual void end_frame(GenState &st) = 0;
    // Потоковое декодирование кадров в PCM. Состояние декодера живёт в GenState.
    virtual void decode(GenState &st, const int16_t *codes, size_t n_frames, std::vector<float> &pcm) = 0;
    virtual void flush(GenState &st, std::vector<float> &pcm) = 0;
    // true, если decode() можно запускать в отдельном потоке параллельно с talker/predictor (они на GPU, декодер на CPU).
    virtual bool decode_overlappable() const { return false; }

    // --- модели без покадровых кодов (flow matching и т.п.): текст целиком → звук кусками ---
    // true → рантайм не вызывает talker_logits/predictor_logits/decode, а ведёт сессию через direct_*.
    virtual bool direct() const { return false; }
    // Признаки голоса из записи вместо speaker_embedding/encode_codes (ссылочный звук, mel…). Кладутся в VoiceData::features.
    virtual void prepare_voice(VoiceData &, const float * /*pcm*/, size_t /*n*/, uint32_t /*sr*/) {}
    // Текст приходит как есть (UTF-8), токенизацию делает сама модель; вызывается по мере поступления кусков.
    virtual void direct_text(GenState &, const std::string & /*utf8*/) {}
    // Один возобновляемый шаг: дописывает звук в pcm (может ничего не дописать, если ждёт текст). true → озвучено всё.
    virtual bool direct_step(GenState &, bool /*text_ended*/, std::vector<float> & /*pcm*/) { return true; }
    // Движок объявляет: весь поданный до сих пор текст — законченные куски (flush или конец предложения при commit=sentence);
    // ждать продолжения последнего куска не нужно.
    virtual void direct_seal(GenState &) {}
    // What the model did in this generation (pieces of text, mode…): the "model" field of vox_session_info; null = nothing.
    virtual json state_info(const GenState &) const { return nullptr; }
    // Сверка внутренних стадий с эталонным дампом (safetensors) для `vox verify`; пишет отчёт, true → в допуске.
    virtual bool debug_check(const std::string & /*dump*/, double /*tol*/, std::string &report) {
        report = "this model has no reference check";
        return false;
    }

    // --- ресурсы и самоописание ---
    virtual Capabilities capabilities() const { return {}; }
    virtual SessionCost session_cost(const VoiceData &, const SessionOptions &) const { return {}; }
    // Накопленная статистика работы (время по стадиям, число кадров, размещение); попадает в vox_engine_stats.
    virtual json stats() const { return json::object(); }
};

// Фабрики встроенных семейств (регистрируются в family.cpp).
class Family;
std::shared_ptr<Family> make_qwen3_family();
std::shared_ptr<Family> make_test_family();
std::shared_ptr<Family> make_f5_family();
std::shared_ptr<Family> make_kokoro_family();
std::shared_ptr<Family> make_cosyvoice_family();
std::shared_ptr<Family> make_chatterbox_family();

void write_test_model(const std::string &path);
// Короткий путь: строка устройства → план → бэкенд (CLI verify, тесты). Движок использует plan_for_model + create_backend.
std::unique_ptr<Backend> make_backend(std::shared_ptr<TmodFile> tmod, const std::string &device, uint32_t n_threads);

}  // namespace vox
