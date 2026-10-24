/*
 * vox.h — публичный C ABI движка vox, версия 0.1
 *
 * Это единственный контракт движка. CLI, Python, JS/WASM и сервер работают
 * только через эти функции.
 *
 * Правила ABI:
 *   1. Каждая структура параметров начинается с struct_size = sizeof(структуры).
 *      Новые поля добавляются только в конец. Старые клиенты продолжают работать.
 *   2. Аудио-буферы выделяет вызывающий код. Библиотека не отдаёт указатели,
 *      которые надо освобождать чужим free().
 *   3. Аудио на выходе: float32, mono, частота из vox_engine_sample_rate().
 *   4. vox_session_pull не блокирует ожиданием текста: VOX_AGAIN = нужно больше
 *      текста или чанк ещё не готов.
 *   5. Engine потокобезопасен. Session — нет (кроме vox_session_cancel).
 */
#ifndef VOX_H
#define VOX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(VOX_BUILD)
#    define VOX_API __declspec(dllexport)
#  else
#    define VOX_API __declspec(dllimport)
#  endif
#else
#  define VOX_API __attribute__((visibility("default")))
#endif

#define VOX_ABI_VERSION_MAJOR 0
#define VOX_ABI_VERSION_MINOR 4

typedef struct vox_engine  vox_engine;
typedef struct vox_voice   vox_voice;
typedef struct vox_session vox_session;

typedef enum {
    VOX_OK = 0,          /* чанк записан в буфер */
    VOX_AGAIN = 1,       /* чанк не готов: подайте ещё текст или вызовите снова */
    VOX_DONE = 2,        /* генерация закончена, аудио больше не будет */
    VOX_ERR_IO = -1,
    VOX_ERR_FORMAT = -2, /* файл повреждён или несовместимая версия формата */
    VOX_ERR_DEVICE = -3, /* устройство недоступно или сборка не под это железо */
    VOX_ERR_OOM = -4,
    VOX_ERR_CANCELLED = -5,
    VOX_ERR_ARG = -6,
    VOX_ERR_STATE = -7,  /* вызов в неправильном состоянии (текст после end и т.п.) */
    VOX_ERR_UNSUPPORTED = -8
} vox_status;

typedef enum {
    VOX_VOICE_ICL = 0,         /* эмбеддинг + коды и текст референса (точнее тембр) */
    VOX_VOICE_XVECTOR_ONLY = 1, /* только эмбеддинг спикера (короче префикс, быстрее) */
    VOX_VOICE_SPEAKER = 2       /* готовый диктор модели по имени и/или текстовая инструкция о стиле (ABI 0.2) */
} vox_voice_mode;

typedef struct {
    uint32_t struct_size;
    const char *device;      /* "auto" (лучшее из доступного), "cpu", "gpu", "gpu:N", "vulkan[:N]"; после запятой опции:
                                strict, codec=cpu|gpu, talker=…, predictor=…, layers=N, mode=…, threads=N, ram=…, vram=… */
    uint32_t max_sessions;   /* на сколько сессий резервировать память при планировании, 0 = 1 */
    uint32_t n_threads;      /* потоки CPU, 0 = по режиму */
    uint64_t memory_limit;   /* байт RAM, 0 = без лимита (то же, что ram_limit в config_json) */
    /* добавлено в ABI 0.2 */
    const char *config_json; /* NULL или JSON ресурсов: {"mode":"balanced","device":"auto","gpu":1,"threads":4,
                                "ram_limit":"6GiB","vram_limit":"3GiB","place":{"codec":"cpu"},"gpu_layers":{"talker":12},
                                "cpu_affinity":"0-3","priority":"low","strict":false,"max_sessions":2, ...};
                                перекрывает остальные поля, неизвестные ключи — ошибка */
} vox_engine_opts;

/* Значение < 0 в любом поле сэмплинга = взять из профиля голоса, затем из .tmod */
typedef struct {
    uint32_t struct_size;
    /* talker (первый кодбук) */
    float temperature;
    float top_p;
    float repetition_penalty;
    int32_t top_k;
    /* code predictor (остальные кодбуки) */
    float sub_temperature;
    float sub_top_p;
    int32_t sub_top_k;
    /* общее */
    uint64_t seed;           /* 0 = из профиля голоса */
    uint32_t chunk_frames;   /* кадров кодека в одном чанке, 0 = по умолчанию */
    uint32_t max_frames;     /* предел длины, 0 = по умолчанию */
    const char *language;    /* "auto", "ru", "en", ... NULL = из профиля */
    /* Сколько последних кадров референса прогревают декодер кодека перед речью (ICL).
       0 = значение модели (по умолчанию 50), UINT32_MAX = все кадры (точно как в оригинале). */
    uint32_t decoder_context_frames;
    /* Режим декодера кодека (vox_decoder_mode). 0 = по умолчанию (VOX_DECODER_STREAM). */
    uint32_t decoder_mode;
    /* ABI 0.3. JSON-объект с параметрами модели, которых нет выше (NULL = нет): для моделей без покадровых кодов,
       например F5-TTS {"nfe_step":32,"cfg_strength":2.0,"sway_sampling_coef":-1.0,"speed":1.0}. Перекрывает значения
       из манифеста модели и профиля голоса; неизвестные ключи игнорируются моделью. */
    const char *extra_json;
} vox_gen_params;

/* VOX_DECODER_STREAM: непрерывное потоковое декодирование со скользящим состоянием; прогрев последними
     decoder_context_frames кадрами референса. Низкая задержка, нет шва на длинных фразах.
   VOX_DECODER_REFERENCE: побитово та же схема, что у оригинального qwen_tts: референс + речь режутся на куски
     по 300 кадров, каждый кусок декодируется заново с 25 кадрами левого контекста (chunked_decode). Прогрев
     идёт от начала текущего куска (до 299 кадров референса), поэтому первый чанк дольше. Нужен для сверки
     с PyTorch/qwentts и A/B-сравнений; decoder_context_frames игнорируется. */
typedef enum {
    VOX_DECODER_DEFAULT = 0,
    VOX_DECODER_STREAM = 1,
    VOX_DECODER_REFERENCE = 2
} vox_decoder_mode;

/* ---- служебное ---- */
VOX_API const char *vox_version(void);
VOX_API const char *vox_last_error(void);         /* текст последней ошибки в этом потоке */
VOX_API void vox_engine_opts_default(vox_engine_opts *opts);
VOX_API void vox_gen_params_default(vox_gen_params *params);

/* ---- Engine ---- */
VOX_API vox_status vox_engine_open(const char *tmod_path, const vox_engine_opts *opts, vox_engine **out);
VOX_API void       vox_engine_close(vox_engine *e);
VOX_API uint32_t   vox_engine_sample_rate(const vox_engine *e);
/* JSON с манифестом модели и отпечатком. len получает полную длину; при cap < len строка обрезается. */
VOX_API vox_status vox_engine_info(const vox_engine *e, char *json_buf, size_t cap, size_t *len);
/* JSON с планом ресурсов (куда что размещено и почему), памятью (RSS, бюджет, VRAM), статистикой стадий и пропускной
   способностью. Можно вызывать в любой момент из любого потока. */
VOX_API vox_status vox_engine_stats(const vox_engine *e, char *json_buf, size_t cap, size_t *len);
/* Текст → фонемы для моделей с фонемным входом (Kokoro): {"phonemes": "...", "chunks": ["...", ...]} — то, что модель
   озвучит (куски не длиннее 510 фонем). lang: NULL / "" — по умолчанию модели, "en-us", "en-gb". Иначе VOX_ERR_UNSUPPORTED. */
VOX_API vox_status vox_engine_phonemize(vox_engine *e, const char *utf8, const char *lang, char *json_buf, size_t cap, size_t *len);

/* ---- без загрузки весов ---- */
/* Устройства и система: {"cpu":{…ядра, RAM, лимит контейнера…},"gpus":[{index,name,type,vram_total,vram_free,…}],"families":[…]} */
VOX_API vox_status vox_devices_json(char *json_buf, size_t cap, size_t *len);
/* Описание модели: семейство, вариант, стадии и их вес, возможности (клонирование, дикторы, инструкции, языки). */
VOX_API vox_status vox_model_inspect(const char *tmod_path, char *json_buf, size_t cap, size_t *len);
/* План размещения для модели при данных ресурсах (config_json как в vox_engine_opts; NULL = по умолчанию). */
VOX_API vox_status vox_plan_json(const char *tmod_path, const char *config_json, char *json_buf, size_t cap, size_t *len);

/* Пишет модель тестового семейства "vox-test" (крошечная, без весов Qwen) — для разработки, тестов и примеров. */
VOX_API vox_status vox_write_test_model(const char *path);

/* Голос без записи: готовый диктор модели (CustomVoice) и/или инструкция о стиле (VoiceDesign, CustomVoice 1.7B).
   speaker и instruct могут быть NULL, но не оба; допустимые комбинации определяет модель (vox_engine_info →
   capabilities: named_speakers, speakers, instruct, requires_instruct). Нарушение — VOX_ERR_ARG с объяснением. */
VOX_API vox_status vox_voice_from_speaker(vox_engine *e, const char *speaker, const char *instruct, vox_voice **out);

/* ---- Voice ---- */
VOX_API vox_status vox_voice_create(vox_engine *e, const float *pcm, size_t n_samples, uint32_t sample_rate,
                                    const char *ref_text, vox_voice_mode mode, vox_voice **out);
VOX_API vox_status vox_voice_load(vox_engine *e, const char *path, vox_voice **out);
VOX_API vox_status vox_voice_save(const vox_voice *v, const char *path);
VOX_API vox_status vox_voice_set_name(vox_voice *v, const char *name);
/* Сохраняет в профиль значения сэмплинга по умолчанию (поля < 0 и seed = 0 не трогаются). */
VOX_API vox_status vox_voice_set_defaults(vox_voice *v, const vox_gen_params *p);
VOX_API vox_status vox_voice_info(const vox_voice *v, char *json_buf, size_t cap, size_t *len);
VOX_API void       vox_voice_free(vox_voice *v);

/* ---- Session ---- */
VOX_API vox_status vox_session_create(vox_engine *e, const vox_voice *v, const vox_gen_params *p, vox_session **out);
VOX_API vox_status vox_session_feed_text(vox_session *s, const char *utf8);
VOX_API vox_status vox_session_end_text(vox_session *s);

/* ---- Двусторонний стриминг (ABI 0.4) ----
   Текст можно подавать порциями в любой момент, а звук забирать по мере готовности (vox_session_pull возвращает VOX_AGAIN,
   когда нужно больше текста). Что именно и когда уходит в модель, решает политика коммита; её задаёт extra_json сессии:
     {"stream": {"commit": "auto" | "sentence" | "manual", "idle_ms": 600}}
   auto     — как умеет модель: AR-модели (Qwen3) читают текст по словам сразу, F5 озвучивает все законченные куски и держит
              последний (результат совпадает с синтезом всего текста сразу);
   sentence — в модель уходит только законченное предложение (знак конца + пробел, либо 。！？…);
   manual   — ничего не уходит до vox_session_flush / vox_session_end_text («набираю, потом получаю одну запись»).
   idle_ms  — если текст не поступал столько мс, всё накопленное коммитится само (при очередном vox_session_pull).
   Что модель умеет — vox_engine_info → capabilities.stream (text_in, audio_out, exact). */

/* Передать модели всё накопленное (даже слово без пробела) и закрыть текущий кусок. Сессия остаётся открытой. */
VOX_API vox_status vox_session_flush(vox_session *s);
/* Заменить ещё не переданный модели хвост текста (правка при наборе). Переданное назад не берётся: смотрите committed_bytes. */
VOX_API vox_status vox_session_set_pending(vox_session *s, const char *utf8);
/* JSON: {committed_bytes, pending, pending_bytes, text_ended, finished, frames, commit, idle_ms, flushes, idle_flushes}. */
VOX_API vox_status vox_session_info(const vox_session *s, char *json_buf, size_t cap, size_t *len);
/* Пишет до cap сэмплов в pcm_out. Возвращает VOX_OK / VOX_AGAIN / VOX_DONE или ошибку. */
VOX_API vox_status vox_session_pull(vox_session *s, float *pcm_out, size_t cap, size_t *written);
VOX_API void       vox_session_cancel(vox_session *s);  /* можно звать из другого потока */
VOX_API void       vox_session_free(vox_session *s);

#ifdef __cplusplus
}
#endif
#endif /* VOX_H */
