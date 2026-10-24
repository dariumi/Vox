#pragma once
// Inline markup in the text a session speaks (0.15):
//   [voice:anna] [anna] "anna: …" at the start of a line   switch the voice of the profile (dialogue lines: profiles with ≥ 2 voices)
//   [style:happy] [happy] [whisper] [радостно]               a style of the voice, or an emotion the engine maps for the model
//   [/] [neutral]                                            back to the voice's default style and speed
//   [pause 500ms] [pause 1.5s] [break] [пауза 300мс]         silence
//   [speed 1.2] [скорость 0.9]                               speaking rate (models that have one)
//   [laugh] [sigh] [cough] [смех] …                          non-verbal events: the model's own tag if it has one, else a short pause
//   \[ … ]                                                   a literal bracket
// markup "auto" (default): a bracket that is none of the above stays text; "strict": unknown brackets are dropped; "off": no markup.
#include <functional>
#include <string>
#include <vector>

namespace vox {

struct MarkupOp {
    enum Kind { Text, Voice, Style, Reset, Pause, Speed, Event, Unknown } kind = Text;
    std::string s;     // text / name / canonical keyword / the original tag
    double num = 0;    // pause ms, speed factor
};

// emotion keyword (any language alias) → canonical name ("happy", "whisper", …); "" if it is not one
std::string markup_emotion(const std::string &word);
// non-verbal event keyword → canonical ("laugh", "sigh", …); "" if it is not one
std::string markup_event(const std::string &word);

class MarkupParser {
public:
    enum Mode { Off = 0, Auto = 1, Strict = 2 };
    Mode mode = Auto;
    bool dialogue = false;                                   // "name: …" at line start switches voice
    std::function<bool(const std::string &)> is_voice, is_style;
    // Splits a committed piece of text into operations. Pieces must not end inside a tag (see safe_cut).
    std::vector<MarkupOp> parse(const std::string &piece);
    // How much of pending can be committed now without cutting a tag or a dialogue prefix in two (≤ cut).
    size_t safe_cut(const std::string &pending, size_t cut) const;
    bool at_line_start = true;
};

}  // namespace vox
