#include "markup.hpp"

#include <cstdlib>
#include <map>

#include "profile.hpp"

namespace vox {
namespace {

// lower-case for ASCII and Cyrillic (the two scripts the keyword tables use)
std::string lower_u8(const std::string &s) {
    std::string o;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) { o += char(std::tolower(c)); ++i; continue; }
        if (c == 0xD0 && i + 1 < s.size()) {
            const unsigned char d = static_cast<unsigned char>(s[i + 1]);
            if (d >= 0x90 && d <= 0x9F) { o += char(0xD0); o += char(d + 0x20); i += 2; continue; }        // А-П → а-п
            if (d >= 0xA0 && d <= 0xAF) { o += char(0xD1); o += char(d - 0x20); i += 2; continue; }        // Р-Я → р-я
            if (d == 0x81) { o += char(0xD1); o += char(0x91); i += 2; continue; }                          // Ё → ё
        }
        o += s[i++];
    }
    // ё → е, so "шёпот" and "шепот" are the same keyword
    std::string r;
    for (size_t i = 0; i < o.size(); ++i) {
        if (static_cast<unsigned char>(o[i]) == 0xD1 && i + 1 < o.size() && static_cast<unsigned char>(o[i + 1]) == 0x91) {
            r += char(0xD0); r += char(0xB5); ++i;
        } else r += o[i];
    }
    return r;
}

std::string trim(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) --b;
    return s.substr(a, b - a);
}

const std::map<std::string, std::string> &emotions() {
    static const std::map<std::string, std::string> m = [] {
        std::map<std::string, std::string> t;
        auto add = [&](const char *canon, std::initializer_list<const char *> al) {
            t[canon] = canon;
            for (const char *a : al) t[lower_u8(a)] = canon;
        };
        add("happy", {"happiness", "joy", "joyful", "cheerful", "glad", "радостно", "радость", "весело", "весёлый", "счастливо"});
        add("sad", {"sadness", "sorrow", "upset", "грустно", "грусть", "печально", "печаль"});
        add("angry", {"anger", "mad", "furious", "злобно", "злость", "сердито", "гнев", "зло"});
        add("excited", {"excitement", "enthusiastic", "взволнованно", "восторг", "воодушевленно", "восторженно"});
        add("calm", {"calmly", "relaxed", "спокойно", "спокойный", "размеренно"});
        add("whisper", {"whispering", "whispers", "шепот", "шепотом", "шепчет"});
        add("shout", {"shouting", "shouts", "loud", "yell", "крик", "кричит", "громко"});
        add("fear", {"scared", "afraid", "fearful", "страх", "испуганно", "боязливо"});
        add("surprised", {"surprise", "amazed", "удивленно", "удивление"});
        add("serious", {"seriously", "stern", "серьезно", "строго"});
        add("tender", {"gentle", "gently", "soft", "softly", "нежно", "ласково", "мягко"});
        add("tired", {"sleepy", "exhausted", "устало", "сонно"});
        add("sarcastic", {"sarcasm", "ironic", "ironically", "саркастично", "иронично"});
        return t;
    }();
    return m;
}

const std::map<std::string, std::string> &events() {
    static const std::map<std::string, std::string> m = [] {
        std::map<std::string, std::string> t;
        auto add = [&](const char *canon, std::initializer_list<const char *> al) {
            t[canon] = canon;
            for (const char *a : al) t[lower_u8(a)] = canon;
        };
        add("laugh", {"laughs", "laughing", "laughter", "chuckle", "giggle", "смех", "смеется", "смеясь", "хихикает"});
        add("sigh", {"sighs", "вздох", "вздыхает"});
        add("cough", {"coughs", "кашель", "кашляет"});
        add("breath", {"breathe", "inhale", "вдох", "дыхание"});
        add("gasp", {"gasps", "ах", "ахает"});
        add("sniff", {"sniffs", "шмыгает"});
        add("yawn", {"yawns", "зевает", "зевок"});
        add("groan", {"groans", "стон", "стонет"});
        add("clear_throat", {"clears throat", "clear throat", "откашливается"});
        return t;
    }();
    return m;
}

bool is_cmd(const std::string &k, std::initializer_list<const char *> names) {
    for (const char *n : names)
        if (k == lower_u8(n)) return true;
    return false;
}

// "500ms" "1.5s" "2" "300 мс" → ms; bare numbers ≥ 20 are ms, smaller ones seconds; -1 = not a duration
double parse_ms(const std::string &v0) {
    const std::string v = lower_u8(trim(v0));
    if (v.empty()) return -1;
    char *end = nullptr;
    const double x = std::strtod(v.c_str(), &end);
    if (end == v.c_str() || x < 0) return -1;
    const std::string unit = trim(std::string(end));
    if (unit.empty()) return x >= 20 ? x : x * 1000.0;
    if (unit == "ms" || unit == lower_u8("мс")) return x;
    if (unit == "s" || unit == "sec" || unit == lower_u8("с") || unit == lower_u8("сек")) return x * 1000.0;
    return -1;
}

}  // namespace

std::string markup_emotion(const std::string &w) {
    auto it = emotions().find(lower_u8(trim(w)));
    return it == emotions().end() ? std::string() : it->second;
}
std::string markup_event(const std::string &w) {
    auto it = events().find(lower_u8(trim(w)));
    return it == events().end() ? std::string() : it->second;
}

std::vector<MarkupOp> MarkupParser::parse(const std::string &t) {
    std::vector<MarkupOp> ops;
    if (mode == Off) {
        if (!t.empty()) ops.push_back({MarkupOp::Text, t, 0});
        if (!t.empty()) at_line_start = t.back() == '\n';
        return ops;
    }
    std::string text;
    auto flush_text = [&] { if (!text.empty()) { ops.push_back({MarkupOp::Text, text, 0}); text.clear(); } };
    size_t i = 0;
    while (i < t.size()) {
        // dialogue: "name: " at the start of a line
        if (at_line_start && dialogue && is_voice) {
            size_t a = i;
            while (a < t.size() && (t[a] == ' ' || t[a] == '\t')) ++a;
            size_t c = t.find(':', a);
            const size_t nl = t.find('\n', a);
            if (c != std::string::npos && c > a && c - a <= 64 && (nl == std::string::npos || c < nl)) {
                const std::string name = t.substr(a, c - a);
                if (name.find(' ') == std::string::npos && name.find('[') == std::string::npos && is_voice(name)) {
                    text.append(t, i, a - i);
                    flush_text();
                    ops.push_back({MarkupOp::Voice, name, 0});
                    i = c + 1;
                    while (i < t.size() && (t[i] == ' ' || t[i] == '\t')) ++i;
                    at_line_start = false;
                    continue;
                }
            }
        }
        const char ch = t[i];
        if (ch == '\\' && i + 1 < t.size() && (t[i + 1] == '[' || t[i + 1] == ']')) {
            text += t[i + 1];
            i += 2;
            at_line_start = false;
            continue;
        }
        if (ch == '[') {
            const size_t close = t.find(']', i + 1);
            const size_t nl = t.find('\n', i + 1);
            if (close != std::string::npos && close - i - 1 <= 64 && (nl == std::string::npos || close < nl)) {
                const std::string raw = t.substr(i, close - i + 1);
                const std::string body = trim(t.substr(i + 1, close - i - 1));
                const std::string lb = lower_u8(body);
                MarkupOp op;
                op.kind = MarkupOp::Unknown;
                op.s = raw;
                // key: value | key=value | key value
                std::string key = lb, val;
                size_t sep = body.find_first_of(":= ");
                if (sep != std::string::npos) { key = lower_u8(trim(body.substr(0, sep))); val = trim(body.substr(sep + 1)); }
                if (lb == "/" || (!lb.empty() && lb[0] == '/') || is_cmd(lb, {"neutral", "default", "normal", "нейтрально", "обычно"})) {
                    op = {MarkupOp::Reset, "", 0};
                } else if (is_cmd(key, {"voice", "speaker", "голос", "диктор"}) && !val.empty()) {
                    op = {MarkupOp::Voice, val, 0};
                } else if (is_cmd(key, {"style", "emotion", "mood", "стиль", "эмоция", "тон"}) && !val.empty()) {
                    op = {MarkupOp::Style, val, 0};
                } else if (is_cmd(key, {"pause", "break", "wait", "silence", "пауза", "тишина"})) {
                    const double ms = val.empty() ? (key == "break" ? 300.0 : 400.0) : parse_ms(val);
                    if (ms >= 0) op = {MarkupOp::Pause, "", std::min(ms, 10000.0)};
                } else if (is_cmd(key, {"speed", "rate", "tempo", "скорость", "темп"}) && !val.empty()) {
                    char *e = nullptr;
                    const double x = std::strtod(val.c_str(), &e);
                    if (e != val.c_str() && x > 0) op = {MarkupOp::Speed, "", std::max(0.5, std::min(x, 2.0))};
                } else if (is_voice && is_voice(body)) {
                    op = {MarkupOp::Voice, body, 0};
                } else if (is_style && is_style(body)) {
                    op = {MarkupOp::Style, body, 0};
                } else if (!markup_emotion(body).empty()) {
                    op = {MarkupOp::Style, body, 0};
                } else if (!markup_event(body).empty()) {
                    op = {MarkupOp::Event, markup_event(body), 0};
                }
                if (op.kind == MarkupOp::Unknown && mode == Auto) {   // not ours: plain text
                    text += raw;
                } else {
                    flush_text();
                    ops.push_back(op);
                }
                i = close + 1;
                at_line_start = false;
                continue;
            }
        }
        text += ch;
        at_line_start = ch == '\n';
        ++i;
    }
    flush_text();
    return ops;
}

size_t MarkupParser::safe_cut(const std::string &p, size_t cut) const {
    if (mode == Off || cut == 0) return cut;
    // an unclosed '[' (a tag that may still be typed) holds the text from it on
    for (size_t k = cut; k > 0; --k) {
        const char c = p[k - 1];
        if (c == ']' || c == '\n') break;
        if (c == '[' && !(k >= 2 && p[k - 2] == '\\')) {
            if (cut - (k - 1) <= 66) return k - 1;
            break;
        }
    }
    // dialogue: a first word of a line without ':' or a space yet may become "name:"
    if (dialogue) {
        const size_t ls = p.rfind('\n', cut ? cut - 1 : 0);
        const bool line_start = ls != std::string::npos || at_line_start;
        const size_t from = ls == std::string::npos ? 0 : ls + 1;
        if (line_start && from < cut) {
            const std::string tail = p.substr(from, cut - from);
            if (tail.find(':') == std::string::npos && tail.find(' ') == std::string::npos && tail.size() <= 64 &&
                tail.find('[') == std::string::npos)
                return from;
        }
    }
    return cut;
}

}  // namespace vox
