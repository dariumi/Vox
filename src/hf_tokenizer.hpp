#pragma once
// A tokenizer.json of Hugging Face `tokenizers` with a character-level BPE model (Chatterbox's English tokenizer and the like):
// added tokens matched literally first, then the normalizer (Sequence / Replace / Lowercase / NFC / Strip), the pre-tokenizer
// (Whitespace = \w+|[^\w\s]+, WhitespaceSplit, Sequence), BPE by merge rank with unk / fuse_unk / continuing_subword_prefix /
// end_of_word_suffix. Anything else is refused when the file is loaded, so a tokenizer is never approximated silently.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vox {

class HfTokenizer {
public:
    explicit HfTokenizer(const std::string &json_text);
    std::vector<int32_t> encode(const std::string &utf8) const;
    int32_t token_id(const std::string &t) const;   // -1 if absent
    size_t vocab_size() const { return vocab_.size(); }

private:
    struct Rule { int kind; std::string a, b; };   // normalizer steps: 0 replace(a → b), 1 replace \s+ → b, 2 lowercase, 3 strip
    std::vector<Rule> norm_;
    std::vector<int> pre_;                         // 0 Whitespace, 1 WhitespaceSplit
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<std::string, int> ranks_;   // "a b" → rank
    std::vector<std::pair<std::string, int32_t>> added_;   // longest first
    int32_t unk_ = -1;
    bool fuse_unk_ = false;
    std::string prefix_, suffix_;
    std::string normalize(const std::string &s) const;
    std::vector<std::string> pretokenize(const std::string &s) const;
    void bpe(const std::string &word, std::vector<int32_t> &out) const;
};

}  // namespace vox
