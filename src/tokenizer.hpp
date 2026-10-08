#pragma once
// Токенизатор Qwen (byte-level BPE) — повторяет претокенизатор Qwen2/Qwen3:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Ограничение: нормализация NFC не выполняется (на вход ожидается текст уже в NFC;
// Python-обёртка нормализует сама).
#include <string>
#include <unordered_map>
#include <vector>

#include "formats.hpp"

namespace vox {

class Tokenizer {
public:
    // vocab_text: строки "id\tтокен" (токен в byte-level представлении GPT-2), merges_text: строки "a b" по рангу.
    // specials: токены, которые ищутся в тексте буквально (<|im_start|> и т.п.).
    Tokenizer(const std::string &vocab_text, const std::string &merges_text,
              const std::vector<std::string> &specials);
    static Tokenizer from_tmod(const TmodFile &f);

    std::vector<int32_t> encode(const std::string &utf8) const;
    int32_t token_id(const std::string &token) const;  // -1, если токена нет
    size_t vocab_size() const { return id_to_tok_.size(); }

    // До какого байта буфер можно токенизировать сейчас, не меняя результат относительно
    // токенизации всего текста целиком (см. комментарий в tokenizer.cpp).
    size_t safe_cut(const std::string &pending) const;

    // Разбиение на претокены (для тестов).
    static std::vector<std::string> pretokenize(const std::string &utf8);

private:
    void bpe(const std::string &mapped, std::vector<int32_t> &out) const;
    void encode_plain(const std::string &utf8, std::vector<int32_t> &out) const;

    std::unordered_map<std::string, int32_t> tok_to_id_;
    std::vector<std::string> id_to_tok_;
    std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges_;  // (a,b) → (rank, merged id)
    std::vector<std::pair<std::string, int32_t>> specials_;             // длинные первыми
    std::string byte_to_uni_[256];
    mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;
};

}  // namespace vox
