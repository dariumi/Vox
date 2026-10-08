#pragma once
// Реестр семейств моделей. Движок (engine.cpp), планировщик ресурсов и интерфейсы не знают о Qwen3-TTS:
// каждое семейство — класс Family, который по манифесту .tmod
//   - узнаёт «свои» файлы (matches),
//   - описывает, что и сколько стоит (footprint → планировщик ресурсов),
//   - говорит, что модель умеет (capabilities: клонирование, готовые дикторы, инструкции, языки),
//   - создаёт Backend под готовый план размещения (create).
// Новая модель = новый Family + Backend; рантайм (сессии, потоковый текст, детерминированный сэмплинг, ресурсы,
// C ABI, Python, сервер) остаётся тем же. Семейство можно зарегистрировать и из Python (vox.register_family).
#include <memory>
#include <string>
#include <vector>

#include "backend.hpp"
#include "resources.hpp"

namespace vox {

class Family {
public:
    virtual ~Family() = default;
    virtual std::string name() const = 0;                      // "qwen3-tts"
    virtual std::string description() const { return ""; }
    virtual bool matches(const json &manifest) const {         // по умолчанию — model.family в манифесте
        return manifest.contains("model") && manifest["model"].value("family", "") == name();
    }
    virtual ModelFootprint footprint(const TmodFile &f) const = 0;
    virtual Capabilities capabilities(const TmodFile &f) const = 0;
    virtual std::unique_ptr<Backend> create(std::shared_ptr<TmodFile> f, const ResourcePlan &plan) const = 0;
};

void unregister_family(const std::string &name);               // убрать семейство (Python делает это при выходе, пока жив интерпретатор)
void register_family(std::shared_ptr<Family> f);               // повторная регистрация того же имени заменяет семейство
std::vector<std::string> family_names();
const Family &find_family(const json &manifest);               // VOX_ERR_UNSUPPORTED с перечнем известных семейств

// Сумма размеров блобов .tmod, имя которых начинается с одного из prefixes и не начинается ни с одного из exclude.
uint64_t blob_bytes(const TmodFile &f, const std::vector<std::string> &prefixes, const std::vector<std::string> &exclude = {});

// План для модели без загрузки весов (то, что делает `vox plan`): система и устройства опрашиваются здесь.
ResourcePlan plan_for_model(const TmodFile &f, const ResourceConfig &cfg);
// Поднимает выбранное планом устройство и создаёт бэкенд.
std::unique_ptr<Backend> create_backend(std::shared_ptr<TmodFile> f, const ResourcePlan &plan);
// Описание модели без загрузки весов: семейство, вариант, стадии, возможности.
json inspect_model(const TmodFile &f);

}  // namespace vox
