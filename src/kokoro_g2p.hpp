#pragma once
// English G2P for Kokoro: a port of misaki.en (the G2P Kokoro was trained with) that runs without spaCy and espeak.
// What is the same as misaki: the gold/silver lexicons (embedded in the .tmod), stress rules (apply_stress), -s / -ed / -ing
// stems, numbers (num2words: cardinals, ordinals, years, decimals), currencies, special words (a, an, the, to, I, am, used…),
// sub-word splitting of hyphenated / mixed tokens and the compound stress resolution.
// What differs: there is no part-of-speech tagger, so heteronyms take the DEFAULT entry (misaki picks by POS: "read" VBD
// vs present), proper nouns are not detected from context, and words missing from both lexicons are spelled letter by letter
// (misaki asks espeak-ng). tests/test_kokoro_tiny.py reports the agreement with misaki on a set of sentences.
#include <memory>
#include <string>
#include <vector>

namespace vox {

struct KToken {
    std::string text, phonemes, whitespace;
};

class KokoroG2P {
public:
    // lexicons: JSON texts of {us|gb}_gold.json and {us|gb}_silver.json
    KokoroG2P(const std::string &gold_json, const std::string &silver_json, bool british);
    ~KokoroG2P();
    // text → tokens with phonemes (misaki's G2P.__call__ with unk = "")
    std::vector<KToken> tokens(const std::string &text) const;
    // phoneme string of the whole text
    std::string phonemize(const std::string &text) const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// KPipeline.en_tokenize: tokens → chunks of at most 510 phonemes (split at the last sentence / clause / comma that fits),
// 'ɾ' → 'T' for American English. Returns the phoneme string of every chunk.
std::vector<std::string> kokoro_chunks(std::vector<KToken> tokens, bool british, size_t limit = 510);
// Phoneme input: split one line of phonemes into tokens (spaces and punctuation) for kokoro_chunks.
std::vector<KToken> kokoro_phoneme_tokens(const std::string &ps);

}  // namespace vox
