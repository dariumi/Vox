// English G2P for Kokoro — a port of misaki.en (misaki 0.7) without spaCy and espeak. See kokoro_g2p.hpp for what differs.
// Function and variable names follow misaki/en.py so the two can be read side by side.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "common.hpp"
#include "formats.hpp"
#include "kokoro_g2p.hpp"

namespace vox {
namespace {

// ------------------------------------------------------------------------------------------------ UTF-8 helpers
using U = std::u32string;

U dec(const std::string &s) {
    U o;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        char32_t cp;
        size_t n;
        if (c < 0x80) { cp = c; n = 1; }
        else if ((c >> 5) == 6 && i + 1 < s.size()) { cp = char32_t(((c & 0x1F) << 6) | (s[i + 1] & 0x3F)); n = 2; }
        else if ((c >> 4) == 14 && i + 2 < s.size()) { cp = char32_t(((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F)); n = 3; }
        else if ((c >> 3) == 30 && i + 3 < s.size()) { cp = char32_t(((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) | ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F)); n = 4; }
        else { cp = c; n = 1; }
        o.push_back(cp);
        i += n;
    }
    return o;
}
std::string enc(const U &u) {
    std::string o;
    for (char32_t c : u) {
        if (c < 0x80) o += char(c);
        else if (c < 0x800) { o += char(0xC0 | (c >> 6)); o += char(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { o += char(0xE0 | (c >> 12)); o += char(0x80 | ((c >> 6) & 0x3F)); o += char(0x80 | (c & 0x3F)); }
        else { o += char(0xF0 | (c >> 18)); o += char(0x80 | ((c >> 12) & 0x3F)); o += char(0x80 | ((c >> 6) & 0x3F)); o += char(0x80 | (c & 0x3F)); }
    }
    return o;
}
bool is_upper_cp(char32_t c) {
    if (c < 0x80) return c >= 'A' && c <= 'Z';
    if (c >= 0xC0 && c <= 0xDE) return c != 0xD7;
    if (c >= 0x100 && c <= 0x17F) return (c % 2) == 0;
    if (c >= 0x391 && c <= 0x3A9) return true;
    if (c >= 0x410 && c <= 0x42F) return true;
    return false;
}
bool is_lower_cp(char32_t c) {
    if (c < 0x80) return c >= 'a' && c <= 'z';
    if (c >= 0xDF && c <= 0xFF) return c != 0xF7;
    if (c >= 0x100 && c <= 0x17F) return (c % 2) == 1;
    if (c >= 0x3B1 && c <= 0x3C9) return true;
    if (c >= 0x430 && c <= 0x44F) return true;
    return false;
}
bool is_letter(char32_t c) {
    if (c < 0x80) return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    if (c == 0xD7 || c == 0xF7) return false;
    if (c >= 0xC0 && c < 0x2000) return true;          // Latin, Greek, Cyrillic, IPA… (good enough for \p{L})
    if (c >= 0x3040 && c < 0xA000) return true;         // CJK
    if (c >= 0xAC00 && c < 0xD7B0) return true;         // Hangul
    return false;
}
bool is_digit(char32_t c) { return c >= '0' && c <= '9'; }
bool is_space(char32_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v' || c == 0xA0; }
char32_t lower_cp(char32_t c) {
    if (c < 0x80) return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return c + 32;
    if (c >= 0x100 && c <= 0x17F && (c % 2) == 0) return c + 1;
    if (c >= 0x391 && c <= 0x3A9) return c + 32;
    if (c >= 0x410 && c <= 0x42F) return c + 32;
    return c;
}
char32_t upper_cp(char32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') ? c - 32 : c;
    if (c >= 0xE0 && c <= 0xFE && c != 0xF7) return c - 32;
    if (c >= 0x100 && c <= 0x17F && (c % 2) == 1) return c - 1;
    if (c >= 0x3B1 && c <= 0x3C9 && c != 0x3C2) return c - 32;
    if (c >= 0x430 && c <= 0x44F) return c - 32;
    return c;
}
std::string lower(const std::string &s) { U u = dec(s); for (auto &c : u) c = lower_cp(c); return enc(u); }
std::string upper(const std::string &s) { U u = dec(s); for (auto &c : u) c = upper_cp(c); return enc(u); }
std::string capitalize(const std::string &s) {   // Python str.capitalize: first upper, rest lower
    U u = dec(s);
    for (size_t i = 0; i < u.size(); ++i) u[i] = i == 0 ? upper_cp(u[i]) : lower_cp(u[i]);
    return enc(u);
}
bool isalpha_s(const std::string &s) { U u = dec(s); if (u.empty()) return false; for (auto c : u) if (!is_letter(c)) return false; return true; }
bool isdigit_s(const std::string &s) { if (s.empty()) return false; for (char c : s) if (c < '0' || c > '9') return false; return true; }
bool ends(const std::string &s, const std::string &x) { return s.size() >= x.size() && s.compare(s.size() - x.size(), x.size(), x) == 0; }
bool starts(const std::string &s, const std::string &x) { return s.compare(0, x.size(), x) == 0; }
size_t ulen(const std::string &s) { return dec(s).size(); }
bool contains_cp(const std::string &set, char32_t c) { for (char32_t x : dec(set)) if (x == c) return true; return false; }
std::string replace_all(std::string s, const std::string &a, const std::string &b) {
    for (size_t p = s.find(a); p != std::string::npos; p = s.find(a, p + b.size())) s.replace(p, a.size(), b);
    return s;
}
std::string strip_chars(const std::string &s, char ch) {
    size_t a = 0, b = s.size();
    while (a < b && s[a] == ch) ++a;
    while (b > a && s[b - 1] == ch) --b;
    return s.substr(a, b - a);
}
std::vector<std::string> split_on(const std::string &s, char ch) {   // Python str.split(ch)
    std::vector<std::string> out;
    size_t a = 0;
    for (;;) {
        size_t b = s.find(ch, a);
        if (b == std::string::npos) { out.push_back(s.substr(a)); break; }
        out.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ constants
const std::string DIPHTHONGS = "AIOQWYʤʧ";
const std::string PUNCTS = ";:,.!?—…\"“”";
const std::string NON_QUOTE_PUNCTS = ";:,.!?—…";
const std::string SUBTOKEN_JUNKS = "',-._‘’/";
const std::string CONSONANTS = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const std::string US_TAUS = "AIOWYiuæɑəɛɪɹʊʌ";
const std::string VOWELS = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
const std::string PRIMARY = "ˈ", SECONDARY = "ˌ";
const char32_t PRIMARY_C = 0x2C8, SECONDARY_C = 0x2CC;
bool in_lexicon_ords(char32_t c) { return c == 39 || c == 45 || (c >= 65 && c <= 90) || (c >= 97 && c <= 122); }
bool all_lexicon_ords(const std::string &w) { for (char32_t c : dec(w)) if (!in_lexicon_ords(c)) return false; return true; }
bool is_vowel(char32_t c) { return contains_cp(VOWELS, c); }

using OptS = std::optional<std::string>;
using OptD = std::optional<double>;
struct PS { OptS ps; int rating = -1; };   // (phonemes, rating); rating -1 = None

// ------------------------------------------------------------------------------------------------ apply_stress
std::string restress(const std::string &ps) {
    const U u = dec(ps);
    std::vector<std::pair<double, char32_t>> ips;
    for (size_t i = 0; i < u.size(); ++i) ips.push_back({double(i), u[i]});
    for (size_t i = 0; i < u.size(); ++i)
        if (u[i] == PRIMARY_C || u[i] == SECONDARY_C) {
            size_t j = i;
            while (j < u.size() && !is_vowel(u[j])) ++j;
            if (j < u.size()) ips[i].first = double(j) - 0.5;
        }
    std::sort(ips.begin(), ips.end());
    U o;
    for (auto &p : ips) o.push_back(p.second);
    return enc(o);
}
bool has_vowel(const std::string &ps) { for (char32_t c : dec(ps)) if (is_vowel(c)) return true; return false; }
bool has_stress(const std::string &ps) { return ps.find(PRIMARY) != std::string::npos || ps.find(SECONDARY) != std::string::npos; }

OptS apply_stress(const OptS &ps_in, OptD stress) {
    if (!stress) return ps_in;
    if (!ps_in) return ps_in;
    const std::string &ps = *ps_in;
    const double s = *stress;
    if (s < -1) return replace_all(replace_all(ps, PRIMARY, ""), SECONDARY, "");
    if (s == -1 || ((s == 0 || s == -0.5) && ps.find(PRIMARY) != std::string::npos))
        return replace_all(replace_all(ps, SECONDARY, ""), PRIMARY, SECONDARY);
    if ((s == 0 || s == 0.5 || s == 1) && !has_stress(ps)) {
        if (!has_vowel(ps)) return ps;
        return restress(SECONDARY + ps);
    }
    if (s >= 1 && ps.find(PRIMARY) == std::string::npos && ps.find(SECONDARY) != std::string::npos) return replace_all(ps, SECONDARY, PRIMARY);
    if (s > 1 && !has_stress(ps)) {
        if (!has_vowel(ps)) return ps;
        return restress(PRIMARY + ps);
    }
    return ps;
}

// ------------------------------------------------------------------------------------------------ num2words (English)
const char *kOnes[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                       "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
const char *kTens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
const char *kScales[] = {"", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion"};

std::string below_1000(int n) {   // num2words style: "one hundred and five", "forty-two"
    std::string o;
    if (n >= 100) {
        o = std::string(kOnes[n / 100]) + " hundred";
        n %= 100;
        if (n) o += " and ";
    }
    if (n >= 20) {
        o += kTens[n / 10];
        if (n % 10) o += std::string("-") + kOnes[n % 10];
    } else if (n > 0 || o.empty()) {
        o += kOnes[n];
    }
    return o;
}
std::string cardinal(long long n) {
    if (n < 0) return "minus " + cardinal(-n);
    if (n < 1000) return below_1000(int(n));
    std::vector<int> groups;
    while (n > 0) { groups.push_back(int(n % 1000)); n /= 1000; }
    std::string o;
    for (int i = int(groups.size()) - 1; i >= 0; --i) {
        const int g = groups[size_t(i)];
        if (!g) continue;
        std::string part = below_1000(g) + (i > 0 ? std::string(" ") + kScales[i] : "");
        if (!o.empty()) o += (i == 0 && g < 100) ? " and " : ", ";
        o += part;
    }
    return o;
}
std::string ordinal_word(const std::string &w) {
    static const std::unordered_map<std::string, std::string> irr = {
        {"one", "first"}, {"two", "second"}, {"three", "third"}, {"five", "fifth"}, {"eight", "eighth"}, {"nine", "ninth"}, {"twelve", "twelfth"}};
    auto it = irr.find(w);
    if (it != irr.end()) return it->second;
    if (ends(w, "y")) return w.substr(0, w.size() - 1) + "ieth";
    return w + "th";
}
std::string ordinal(long long n) {
    std::string c = cardinal(n);
    size_t p = c.find_last_of(" -");
    const std::string head = p == std::string::npos ? "" : c.substr(0, p + 1), last = p == std::string::npos ? c : c.substr(p + 1);
    return head + ordinal_word(last);
}
std::string year(long long v) {
    if (v < 0) return year(-v) + " BC";
    const long long high = v / 100, low = v % 100;
    if (high == 0 || (high % 10 == 0 && low < 10) || high >= 100) return cardinal(v);
    const std::string ht = cardinal(high);
    std::string lt;
    if (low == 0) lt = "hundred";
    else if (low < 10) lt = "oh-" + cardinal(low);
    else lt = cardinal(low);
    return ht + " " + lt;
}
// num2words(float(s)): "three point five"; s is a decimal string without commas
std::string float_words(const std::string &s) {
    std::string ip = s, fp;
    const size_t dot = s.find('.');
    if (dot != std::string::npos) { ip = s.substr(0, dot); fp = s.substr(dot + 1); }
    while (!fp.empty() && fp.back() == '0') fp.pop_back();   // float("3.50") == 3.5
    if (fp.empty()) fp = "0";
    long long iv = ip.empty() ? 0 : std::stoll(ip);
    std::string o = cardinal(iv) + " point";
    for (char c : fp) o += std::string(" ") + kOnes[c - '0'];
    return o;
}

// ------------------------------------------------------------------------------------------------ lexicon
struct Entry {
    bool is_dict = false;
    std::string s;                              // plain entry
    std::unordered_map<std::string, OptS> d;    // POS → phonemes (or None)
};
struct TokenContext { std::optional<bool> future_vowel; bool future_to = false; };

struct MTok {
    std::string text, tag, whitespace;
    bool is_head = true;
    OptS alias, phonemes;
    OptD stress;
    std::string currency, num_flags;
    bool prespace = false;
    int rating = -1;
    bool is_to() const { return text == "to" || text == "To" || (text == "TO" && (tag == "TO" || tag == "IN")); }
    int stress_weight() const {
        if (!phonemes) return 0;
        int w = 0;
        for (char32_t c : dec(*phonemes)) w += contains_cp(DIPHTHONGS, c) ? 2 : 1;
        return w;
    }
};

std::string parent_tag(const std::string &tag) {
    if (starts(tag, "VB")) return "VERB";
    if (starts(tag, "NN")) return "NOUN";
    if (starts(tag, "ADV") || starts(tag, "RB")) return "ADV";
    if (starts(tag, "ADJ") || starts(tag, "JJ")) return "ADJ";
    return tag;
}

class Lexicon {
public:
    Lexicon(const std::string &gold, const std::string &silver, bool british) : british_(british) {
        load(gold, golds_);
        load(silver, silvers_);
    }
    bool british() const { return british_; }
    bool in_golds(const std::string &w) const { return golds_.count(w) != 0; }
    bool has_word(const std::string &w) const {   // a real lexicon word with a plain (not POS-split) entry
        auto it = golds_.find(w);
        if (it == golds_.end()) { it = silvers_.find(w); if (it == silvers_.end()) return false; }
        return !it->second.is_dict && !it->second.s.empty();
    }

    PS get_NNP(const std::string &word) const {
        std::string ps;
        for (char32_t c : dec(word)) {
            if (!is_letter(c)) continue;
            auto it = golds_.find(enc(U(1, upper_cp(c))));
            if (it == golds_.end() || it->second.is_dict) return {};
            ps += it->second.s;
        }
        OptS r = apply_stress(ps, 0.0);
        std::string x = *r;
        const size_t p = x.rfind(SECONDARY);
        if (p != std::string::npos) x = x.substr(0, p) + PRIMARY + x.substr(p + SECONDARY.size());
        return {x, 3};
    }

    PS get_special_case(const std::string &word, const std::string &tag, OptD stress, const TokenContext &ctx) const {
        static const std::unordered_map<std::string, std::string> ADD = {{".", "dot"}, {"/", "slash"}};
        static const std::unordered_map<std::string, std::string> SYM = {{"%", "percent"}, {"&", "and"}, {"+", "plus"}, {"@", "at"}};
        if (tag == "ADD" && ADD.count(word)) return lookup(ADD.at(word), "", -0.5, &ctx);
        if (SYM.count(word)) return lookup(SYM.at(word), "", std::nullopt, &ctx);
        const std::string inner = strip_chars(word, '.');
        if (inner.find('.') != std::string::npos && isalpha_s(replace_all(word, ".", ""))) {
            size_t mx = 0;
            for (const auto &part : split_on(word, '.')) mx = std::max(mx, ulen(part));
            if (mx < 3) return get_NNP(word);
        }
        if (word == "a" || (word == "A" && tag == "DT")) return {std::string("ɐ"), 4};
        if (word == "am" || word == "Am" || word == "AM") {
            if (starts(tag, "NN")) return get_NNP(word);
            if (!ctx.future_vowel.has_value() || word != "am" || (stress && *stress > 0)) return {plain("am"), 4};
            return {std::string("ɐm"), 4};
        }
        if (word == "an" || word == "An" || word == "AN") {
            if (word == "AN" && starts(tag, "NN")) return get_NNP(word);
            return {std::string("ɐn"), 4};
        }
        if (word == "I" && tag == "PRP") return {SECONDARY + "I", 4};
        if ((word == "by" || word == "By" || word == "BY") && parent_tag(tag) == "ADV") return {std::string("bˈI"), 4};
        if (word == "to" || word == "To" || (word == "TO" && (tag == "TO" || tag == "IN"))) {
            if (!ctx.future_vowel.has_value()) return {plain("to"), 4};
            return {std::string(*ctx.future_vowel ? "tʊ" : "tə"), 4};
        }
        if (word == "the" || word == "The" || (word == "THE" && tag == "DT"))
            return {std::string(ctx.future_vowel.has_value() && *ctx.future_vowel ? "ði" : "ðə"), 4};
        if (tag == "IN") {
            const std::string lw = lower(word);
            if (lw == "vs" || lw == "vs.") return lookup("versus", "", std::nullopt, &ctx);
        }
        if (word == "used" || word == "Used" || word == "USED") {
            auto it = golds_.find("used");
            if (it != golds_.end() && it->second.is_dict) {
                const auto &d = it->second.d;
                if ((tag == "VBD" || tag == "JJ") && ctx.future_to && d.count("VBD") && d.at("VBD")) return {*d.at("VBD"), 4};
                if (d.count("DEFAULT") && d.at("DEFAULT")) return {*d.at("DEFAULT"), 4};
            }
        }
        return {};
    }

    bool is_known(const std::string &word, const std::string &) const {
        static const std::unordered_set<std::string> SYM = {"%", "&", "+", "@"};
        if (golds_.count(word) || SYM.count(word) || silvers_.count(word)) return true;
        if (!isalpha_s(word) || !all_lexicon_ords(word)) return false;
        if (ulen(word) == 1) return true;
        if (word == upper(word) && golds_.count(lower(word))) return true;
        const U u = dec(word);
        const std::string rest = enc(U(u.begin() + 1, u.end()));
        return rest == upper(rest);
    }

    PS lookup(std::string word, std::string tag, OptD stress, const TokenContext *ctx) const {
        std::optional<bool> is_NNP;
        if (word == upper(word) && !golds_.count(word)) {
            word = lower(word);
            is_NNP = tag == "NNP";
        }
        const Entry *e = nullptr;
        int rating = 4;
        auto it = golds_.find(word);
        if (it != golds_.end()) e = &it->second;
        if (!e && !(is_NNP && *is_NNP)) {
            auto is = silvers_.find(word);
            if (is != silvers_.end()) e = &is->second;
            rating = 3;
        }
        OptS ps;
        if (e) {
            if (e->is_dict) {
                if (ctx && !ctx->future_vowel.has_value() && e->d.count("None")) tag = "None";
                else if (!e->d.count(tag)) tag = parent_tag(tag);
                auto dt = e->d.find(tag);
                if (dt != e->d.end()) ps = dt->second;
                else { auto df = e->d.find("DEFAULT"); if (df != e->d.end()) ps = df->second; }
            } else {
                ps = e->s;
            }
        }
        if (!ps || (is_NNP && *is_NNP && ps->find(PRIMARY) == std::string::npos)) {
            PS n = get_NNP(word);
            if (n.ps) return n;
        }
        if (!ps) return {};
        return {apply_stress(ps, stress), rating};
    }

    std::string _s(const std::string &stem) const {
        const U u = dec(stem);
        if (u.empty()) return stem;
        const char32_t last = u.back();
        if (contains_cp("ptkfθ", last)) return stem + "s";
        if (contains_cp("szʃʒʧʤ", last)) return stem + (british_ ? "ɪ" : "ᵻ") + "z";
        return stem + "z";
    }
    PS stem_s(const std::string &word, const std::string &tag, OptD stress, const TokenContext *ctx) const {
        std::string stem;
        if (word.size() > 2 && ends(word, "s") && !ends(word, "ss") && is_known(word.substr(0, word.size() - 1), tag)) stem = word.substr(0, word.size() - 1);
        else if ((ends(word, "'s") || (word.size() > 4 && ends(word, "es"))) && is_known(word.substr(0, word.size() - 2), tag)) stem = word.substr(0, word.size() - 2);
        else if (word.size() > 4 && ends(word, "ies") && is_known(word.substr(0, word.size() - 3) + "y", tag)) stem = word.substr(0, word.size() - 3) + "y";
        else return {};
        PS r = lookup(stem, tag, stress, ctx);
        if (!r.ps) return {std::nullopt, r.rating};
        return {_s(*r.ps), r.rating};
    }
    OptS _ed(const std::string &stem) const {
        const U u = dec(stem);
        if (u.empty()) return std::nullopt;
        const char32_t last = u.back();
        if (contains_cp("pkfθʃsʧ", last)) return stem + "t";
        if (last == 'd') return stem + (british_ ? "ɪ" : "ᵻ") + "d";
        if (last != 't') return stem + "d";
        if (british_ || u.size() < 2) return stem + "ɪd";
        if (contains_cp(US_TAUS, u[u.size() - 2])) return enc(U(u.begin(), u.end() - 1)) + "ɾᵻd";
        return stem + "ᵻd";
    }
    PS stem_ed(const std::string &word, const std::string &tag, OptD stress, const TokenContext *ctx) const {
        std::string stem;
        if (ends(word, "d") && !ends(word, "dd") && is_known(word.substr(0, word.size() - 1), tag)) stem = word.substr(0, word.size() - 1);
        else if (ends(word, "ed") && !ends(word, "eed") && is_known(word.substr(0, word.size() - 2), tag)) stem = word.substr(0, word.size() - 2);
        else return {};
        PS r = lookup(stem, tag, stress, ctx);
        if (!r.ps) return {std::nullopt, r.rating};
        return {_ed(*r.ps), r.rating};
    }
    OptS _ing(const std::string &stem) const {
        const U u = dec(stem);
        if (u.empty()) return std::nullopt;
        if (british_) {
            if (contains_cp("əː", u.back())) return std::nullopt;
        } else if (u.size() > 1 && u.back() == 't' && contains_cp(US_TAUS, u[u.size() - 2])) {
            return enc(U(u.begin(), u.end() - 1)) + "ɾɪŋ";
        }
        return stem + "ɪŋ";
    }
    PS stem_ing(const std::string &word, const std::string &tag, OptD stress, const TokenContext *ctx) const {
        std::string stem;
        auto doubled = [&]() {   // ([bcdgklmnprstvxz])\1ing$|cking$
            if (ends(word, "cking")) return true;
            if (word.size() < 5 || !ends(word, "ing")) return false;
            const char a = word[word.size() - 5], b = word[word.size() - 4];
            return a == b && std::strchr("bcdgklmnprstvxz", a) != nullptr;
        };
        if (ends(word, "ing") && is_known(word.substr(0, word.size() - 3), tag)) stem = word.substr(0, word.size() - 3);
        else if (ends(word, "ing") && is_known(word.substr(0, word.size() - 3) + "e", tag)) stem = word.substr(0, word.size() - 3) + "e";
        else if (doubled() && is_known(word.substr(0, word.size() - 4), tag)) stem = word.substr(0, word.size() - 4);
        else return {};
        PS r = lookup(stem, tag, stress, ctx);
        if (!r.ps) return {std::nullopt, r.rating};
        return {_ing(*r.ps), r.rating};
    }

    PS get_word(const std::string &word, const std::string &tag, OptD stress, const TokenContext &ctx) const {
        PS sc = get_special_case(word, tag, stress, ctx);
        if (sc.ps) return sc;
        if (is_known(word, tag)) return lookup(word, tag, stress, &ctx);
        if (ends(word, "s'") && is_known(word.substr(0, word.size() - 2) + "'s", tag)) return lookup(word.substr(0, word.size() - 2) + "'s", tag, stress, &ctx);
        if (ends(word, "'") && is_known(word.substr(0, word.size() - 1), tag)) return lookup(word.substr(0, word.size() - 1), tag, stress, &ctx);
        PS r = stem_s(word, tag, stress, &ctx);
        if (r.ps) return r;
        r = stem_ed(word, tag, stress, &ctx);
        if (r.ps) return r;
        r = stem_ing(word, tag, stress ? stress : OptD(0.5), &ctx);
        if (r.ps) return r;
        return {};
    }

    static bool is_currency(const std::string &word) {
        if (word.find('.') == std::string::npos) return true;
        if (std::count(word.begin(), word.end(), '.') > 1) return false;
        const std::string cents = split_on(word, '.')[1];
        return cents.size() < 3;   // (set(cents) == {0} is never true for a string)
    }

    PS get_number(std::string word, const std::string &currency, bool is_head, const std::string &num_flags) const {
        std::string suffix;
        {
            size_t p = word.size();
            while (p > 0 && ((word[p - 1] >= 'a' && word[p - 1] <= 'z') || word[p - 1] == '\'')) --p;
            if (p < word.size()) { suffix = word.substr(p); word = word.substr(0, p); }
        }
        std::vector<PS> result;
        if (starts(word, "-")) {
            result.push_back(lookup("minus", "", std::nullopt, nullptr));
            word = word.substr(1);
        }
        auto extend_num = [&](const std::string &num, bool first = true, bool escape = false) {
            const std::string text = escape ? num : cardinal(std::stoll(num));
            std::vector<std::string> splits;
            {   // re.split(r'[^a-z]+', text)
                std::string cur;
                bool any = false;
                for (char c : text) {
                    if (c >= 'a' && c <= 'z') { cur += c; any = true; }
                    else { splits.push_back(cur); cur.clear(); while (false) {} }
                }
                splits.push_back(cur);
                (void)any;
                // collapse empty pieces produced by runs of separators (re.split joins a run into one separator)
                std::vector<std::string> sp;
                bool prev_sep = false;
                for (size_t i = 0; i < splits.size(); ++i) {
                    if (splits[i].empty() && i > 0 && i + 1 < splits.size()) { prev_sep = true; continue; }
                    sp.push_back(splits[i]);
                }
                (void)prev_sep;
                splits = sp;
            }
            for (size_t i = 0; i < splits.size(); ++i) {
                const std::string &w = splits[i];
                if (w != "and" || num_flags.find('&') != std::string::npos) {
                    if (first && i == 0 && splits.size() > 1 && w == "one" && num_flags.find('a') != std::string::npos) result.push_back({std::string("ə"), 4});
                    else result.push_back(lookup(w, "", w == "point" ? OptD(-2) : std::nullopt, nullptr));
                } else if (w == "and" && num_flags.find('n') != std::string::npos && !result.empty() && result.back().ps) {
                    result.back().ps = *result.back().ps + "ən";
                }
            }
        };
        if (!is_head && word.find('.') == std::string::npos) {
            const std::string num = replace_all(word, ",", "");
            if (!num.empty() && (num[0] == '0' || num.size() > 3)) {
                for (char c : num) extend_num(std::string(1, c), false);
            } else if (num.size() == 3 && !ends(num, "00")) {
                extend_num(num.substr(0, 1));
                if (num[1] == '0') {
                    result.push_back(lookup("O", "", -2.0, nullptr));
                    extend_num(num.substr(2, 1), false);
                } else {
                    extend_num(num.substr(1), false);
                }
            } else if (!num.empty()) {
                extend_num(num);
            }
        } else if (std::count(word.begin(), word.end(), '.') > 1 || !is_head) {
            bool first = true;
            for (const std::string &num : split_on(replace_all(word, ",", ""), '.')) {
                if (num.empty()) {
                } else if (num[0] == '0' || (num.size() != 2 && std::any_of(num.begin() + 1, num.end(), [](char c) { return c != '0'; }))) {
                    for (char c : num) extend_num(std::string(1, c), false);
                } else {
                    extend_num(num, first);
                }
                first = false;
            }
        } else if ((currency == "$" || currency == "£" || currency == "€") && is_currency(word)) {
            static const std::unordered_map<std::string, std::pair<std::string, std::string>> CUR = {
                {"$", {"dollar", "cent"}}, {"£", {"pound", "pence"}}, {"€", {"euro", "cent"}}};
            const auto &units = CUR.at(currency);
            const auto parts = split_on(replace_all(word, ",", ""), '.');
            std::vector<std::pair<long long, std::string>> pairs;
            for (size_t i = 0; i < parts.size() && i < 2; ++i)
                pairs.push_back({parts[i].empty() ? 0 : std::stoll(parts[i]), i == 0 ? units.first : units.second});
            if (pairs.size() > 1) {
                if (pairs[1].first == 0) pairs.resize(1);
                else if (pairs[0].first == 0) pairs.erase(pairs.begin());
            }
            for (size_t i = 0; i < pairs.size(); ++i) {
                if (i > 0) result.push_back(lookup("and", "", std::nullopt, nullptr));
                extend_num(std::to_string(pairs[i].first), i == 0);
                if (std::llabs(pairs[i].first) != 1 && pairs[i].second != "pence") result.push_back(stem_s(pairs[i].second + "s", "", std::nullopt, nullptr));
                else result.push_back(lookup(pairs[i].second, "", std::nullopt, nullptr));
            }
        } else {
            static const std::unordered_set<std::string> ORD = {"st", "nd", "rd", "th"};
            std::string w;
            if (isdigit_s(word)) {
                const long long v = std::stoll(word);
                w = ORD.count(suffix) ? ordinal(v) : ((result.empty() && word.size() == 4) ? year(v) : cardinal(v));
            } else if (word.find('.') == std::string::npos) {
                const long long v = std::stoll(replace_all(word, ",", ""));
                w = ORD.count(suffix) ? ordinal(v) : cardinal(v);
            } else {
                const std::string x = replace_all(word, ",", "");
                if (!x.empty() && x[0] == '.') {
                    w = "point";
                    for (size_t i = 1; i < x.size(); ++i) w += std::string(" ") + kOnes[x[i] - '0'];
                } else {
                    w = float_words(x);
                }
            }
            extend_num(w, true, true);
        }
        if (result.empty()) return {};
        std::string joined;
        int rating = 99;
        for (const auto &r : result) {
            if (!r.ps) return {};   // a missing piece (misaki would fail joining None)
            if (!joined.empty()) joined += " ";
            joined += *r.ps;
            rating = std::min(rating, r.rating);
        }
        if (suffix == "s" || suffix == "'s") return {_s(joined), rating};
        if (suffix == "ed" || suffix == "'d") return {_ed(joined), rating};
        if (suffix == "ing") return {_ing(joined), rating};
        return {joined, rating};
    }

    std::string append_currency(const std::string &ps, const std::string &currency) const {
        if (currency.empty()) return ps;
        static const std::unordered_map<std::string, std::string> CUR = {{"$", "dollar"}, {"£", "pound"}, {"€", "euro"}};
        auto it = CUR.find(currency);
        if (it == CUR.end()) return ps;
        PS c = stem_s(it->second + "s", "", std::nullopt, nullptr);
        return c.ps ? ps + " " + *c.ps : ps;
    }

    static bool is_number(const std::string &word_in, bool is_head) {
        bool any = false;
        for (char c : word_in) if (c >= '0' && c <= '9') any = true;
        if (!any) return false;
        std::string word = word_in;
        for (const char *s : {"ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"})
            if (ends(word, s)) { word = word.substr(0, word.size() - std::strlen(s)); break; }
        for (size_t i = 0; i < word.size(); ++i) {
            const char c = word[i];
            if (!((c >= '0' && c <= '9') || c == ',' || c == '.' || (is_head && i == 0 && c == '-'))) return false;
        }
        return true;
    }

    // Lexicon.__call__
    PS operator()(const MTok &t, const TokenContext &ctx) const {
        std::string word = t.alias ? *t.alias : t.text;
        word = replace_all(replace_all(word, "‘", "'"), "’", "'");
        OptD stress;
        if (word != lower(word)) stress = (word == upper(word)) ? 2.0 : 0.5;
        PS r = get_word(word, t.tag, stress, ctx);
        if (r.ps) return {apply_stress(append_currency(*r.ps, t.currency), t.stress), r.rating};
        if (is_number(word, t.is_head)) {
            PS n = get_number(word, t.currency, t.is_head, t.num_flags);
            return {apply_stress(n.ps, t.stress), n.rating};
        }
        if (!all_lexicon_ords(word)) return {};
        if (word != lower(word)) {
            const U u = dec(word);
            const std::string rest = enc(U(u.begin() + (u.empty() ? 0 : 1), u.end()));
            if (word == upper(word) || rest == lower(rest)) {
                PS q = get_word(lower(word), t.tag, stress, ctx);
                if (q.ps) return {apply_stress(append_currency(*q.ps, t.currency), t.stress), q.rating};
            }
        }
        return {};
    }

private:
    std::string plain(const std::string &w) const {
        auto it = golds_.find(w);
        if (it == golds_.end()) return w;
        if (!it->second.is_dict) return it->second.s;
        auto d = it->second.d.find("DEFAULT");
        return d != it->second.d.end() && d->second ? *d->second : w;
    }
    void load(const std::string &js, std::unordered_map<std::string, Entry> &dst) {
        if (js.empty()) return;
        const json j = json::parse(js);
        std::unordered_map<std::string, Entry> d;
        for (const auto &kv : j.items()) {
            Entry e;
            if (kv.value().is_string()) e.s = kv.value().get<std::string>();
            else if (kv.value().is_object()) {
                e.is_dict = true;
                for (const auto &p : kv.value().items()) e.d[p.key()] = p.value().is_null() ? OptS() : OptS(p.value().get<std::string>());
            } else continue;
            d[kv.key()] = std::move(e);
        }
        // grow_dictionary: lowercase ↔ Capitalized variants (the original entries win)
        for (const auto &kv : d) {
            const std::string &k = kv.first;
            if (ulen(k) < 2) continue;
            if (k == lower(k)) {
                const std::string c = capitalize(k);
                if (k != c) dst.emplace(c, kv.second);
            } else if (k == capitalize(lower(k))) {
                dst.emplace(lower(k), kv.second);
            }
        }
        for (auto &kv : d) dst[kv.first] = std::move(kv.second);
    }
    bool british_;
    std::unordered_map<std::string, Entry> golds_, silvers_;
};

// ------------------------------------------------------------------------------------------------ subtokenize (misaki's regex)
std::vector<std::string> subtokenize(const std::string &word) {
    const U s = dec(word);
    std::vector<std::string> out;
    auto is_apos = [](char32_t c) { return c == '\'' || c == 0x2018 || c == 0x2019; };
    size_t p = 0;
    const size_t n = s.size();
    while (p < n) {
        size_t e = 0;
        // ^['‘’]+
        if (p == 0 && is_apos(s[0])) { e = 0; while (e < n && is_apos(s[e])) ++e; }
        // \p{Lu}(?=\p{Lu}\p{Ll})
        else if (is_upper_cp(s[p]) && p + 2 < n + 0 && p + 2 <= n - 1 + 1 && p + 1 < n && is_upper_cp(s[p + 1]) && p + 2 < n && is_lower_cp(s[p + 2])) e = p + 1;
        else {
            // (?:^-)?(?:\d?[,.]?\d)+
            size_t q = p;
            if (p == 0 && s[0] == '-') q = 1;
            size_t reps = 0, end = q;
            for (;;) {
                size_t r = end;
                // greedy options of one repetition: (d)(sep)(d), (d)(d), (sep)(d), (d)
                auto dg = [&](size_t i) { return i < n && is_digit(s[i]); };
                auto sp = [&](size_t i) { return i < n && (s[i] == ',' || s[i] == '.'); };
                if (dg(r) && sp(r + 1) && dg(r + 2)) r += 3;
                else if (dg(r) && dg(r + 1)) r += 2;
                else if (sp(r) && dg(r + 1)) r += 2;
                else if (dg(r)) r += 1;
                else break;
                end = r;
                ++reps;
            }
            if (reps > 0) e = end;
            else if (s[p] == '-' || s[p] == '_') { e = p; while (e < n && (s[e] == '-' || s[e] == '_')) ++e; }              // [-_]+
            else if (is_apos(s[p]) && p + 1 < n && is_apos(s[p + 1])) { e = p; while (e < n && is_apos(s[e])) ++e; }        // ['‘’]{2,}
            else {
                // \p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu}) — the shortest run of letters ending in lower case before an upper case
                size_t camel = 0;
                for (size_t k = p; k < n; ++k) {
                    if (is_lower_cp(s[k]) && k + 1 < n && is_upper_cp(s[k + 1])) { camel = k + 1; break; }
                    if (!is_letter(s[k]) && !(is_apos(s[k]) && k + 1 < n && is_letter(s[k + 1]) && k > p)) break;
                }
                if (camel) e = camel;
                else if (is_letter(s[p])) {   // \p{L}+(?:['‘’]\p{L})*
                    e = p;
                    while (e < n && is_letter(s[e])) ++e;
                    while (e + 1 < n && is_apos(s[e]) && is_letter(s[e + 1])) e += 2;
                } else if (!is_apos(s[p]) && !is_digit(s[p])) {
                    e = p + 1;   // [^-_\p{L}'‘’\d]
                } else if (is_apos(s[p])) {   // ['‘’]+$
                    size_t k = p;
                    while (k < n && is_apos(s[k])) ++k;
                    if (k == n) e = n;
                }
            }
        }
        if (e > p) { out.push_back(enc(U(s.begin() + long(p), s.begin() + long(e)))); p = e; }
        else ++p;
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ tokenizer (instead of spaCy)
const std::unordered_set<std::string> DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its",
                                                     "our", "their", "some", "any", "no", "every", "each", "another", "which", "whose"};
// words that usually follow a transitive verb ("present your report", "object to", "close the door")
const std::unordered_set<std::string> OBJECT_CUES = {"the", "a", "an", "your", "my", "his", "her", "their", "our", "its", "this", "these",
                                                     "those", "it", "them", "him", "me", "us", "to", "that"};
const std::unordered_set<std::string> VERB_CUES = {"to", "will", "would", "can", "could", "shall", "should", "may", "might", "must", "i", "you",
                                                   "we", "they", "he", "she", "it", "don't", "doesn't", "didn't", "won't", "can't", "not", "please"};

std::string punct_tag(char32_t c, bool &open_quote) {
    switch (c) {
        case '.': case '!': case '?': return ".";
        case ',': return ",";
        case ';': case ':': case 0x2026: return ":";
        case 0x2014: case 0x2013: return ":";
        case '(': case '[': case '{': return "-LRB-";
        case ')': case ']': case '}': return "-RRB-";
        case 0x201C: return "``";
        case 0x201D: return "''";
        case '"': { const std::string t = open_quote ? "''" : "``"; open_quote = !open_quote; return t; }
        default: return "";
    }
}

class Tokenizer {
public:
    explicit Tokenizer(const Lexicon &lex) : lex_(lex) {}
    // text → spaCy-like tokens with a coarse tag ("" = unknown word)
    std::vector<MTok> operator()(const std::string &text) const {
        std::vector<MTok> out;
        const U u = dec(text);
        bool open_quote = false;
        size_t i = 0;
        while (i < u.size()) {
            if (is_space(u[i])) { ++i; continue; }
            size_t j = i;
            while (j < u.size() && !is_space(u[j])) ++j;
            size_t k = j;
            while (k < u.size() && is_space(u[k])) ++k;
            const std::string ws = k > j ? " " : "";
            chunk(U(u.begin() + long(i), u.begin() + long(j)), ws, out, open_quote);
            i = k;
        }
        // tags: digits → CD, currency symbols → $, "I" → PRP, "A" before a lowercase word → DT; heteronym hints from the word before
        for (size_t t = 0; t < out.size(); ++t) {
            MTok &m = out[t];
            if (!m.tag.empty()) continue;
            bool digit = false;
            for (char c : m.text) if (c >= '0' && c <= '9') digit = true;
            if (digit) { m.tag = "CD"; continue; }
            if (m.text == "I") { m.tag = "PRP"; continue; }
            if (m.text == "A" && t + 1 < out.size() && !out[t + 1].text.empty() && is_lower_cp(dec(out[t + 1].text)[0])) { m.tag = "DT"; continue; }
            // a rough part of speech for heteronyms (misaki takes it from spaCy): by the neighbouring words
            const std::string prev = (t > 0 && out[t - 1].tag != "." && out[t - 1].tag != ",") ? lower(out[t - 1].text) : "";
            const std::string next = t + 1 < out.size() ? lower(out[t + 1].text) : "";
            const bool prev_num = t > 0 && (out[t - 1].tag == "CD" || out[t - 1].text == "%");
            auto ends_any = [](const std::string &w, std::initializer_list<const char *> sfx) {
                for (const char *x : sfx) if (ends(w, x)) return true;
                return false;
            };
            if (DETERMINERS.count(prev) || prev_num) m.tag = "NN";
            else if (prev == "too" || prev == "so" || prev == "very" || prev == "quite" || prev == "more" || prev == "most" || prev == "less") m.tag = "JJ";
            else if ((prev == "he" || prev == "she" || prev == "it") && OBJECT_CUES.count(next)) m.tag = "VBD";   // "she read the book"
            else if (VERB_CUES.count(prev) || ends_any(prev, {"'ll", "'d", "'re", "'ve", "n't", "’ll", "’re", "’ve"})) m.tag = "VB";
            else if (OBJECT_CUES.count(next)) m.tag = "VB";
        }
        return out;
    }

private:
    void chunk(U c, const std::string &ws, std::vector<MTok> &out, bool &open_quote) const {
        std::vector<MTok> lead, trail;
        auto punct_tok = [&](char32_t ch) {
            MTok m;
            m.text = enc(U(1, ch));
            m.tag = punct_tag(ch, open_quote);
            if (m.tag.empty()) m.tag = "NFP";
            return m;
        };
        static const std::u32string LEAD = U"\"“(['‘«¿¡{", TRAIL = U"\"”)]’'.,!?;:…»}";
        // leading punctuation
        while (!c.empty() && LEAD.find(c[0]) != std::u32string::npos) {
            if ((c[0] == '\'' || c[0] == 0x2018) && c.size() > 1 && is_letter(c[1]) && lex_.in_golds(enc(c))) break;   // 'cause
            lead.push_back(punct_tok(c[0]));
            c.erase(c.begin());
        }
        // trailing punctuation ("..." stays a single ellipsis token)
        while (!c.empty() && TRAIL.find(c.back()) != std::u32string::npos) {
            const std::string whole = enc(c);
            if (c.back() == '.') {
                // abbreviations and initials keep their period: "Dr.", "U.S.", "e.g."
                if (lex_.in_golds(whole) || (c.size() >= 2 && c[c.size() - 2] != '.' && whole.find('.') < whole.size() - 1)) break;
                if (c.size() >= 3 && c[c.size() - 2] == '.' && c[c.size() - 3] == '.') {
                    MTok m = punct_tok(0x2026);
                    m.text = "...";
                    m.tag = ":";
                    trail.insert(trail.begin(), m);
                    c.resize(c.size() - 3);
                    continue;
                }
            }
            if ((c.back() == '\'' || c.back() == 0x2019) && c.size() > 1 && (c[c.size() - 2] == 's' || c[c.size() - 2] == 'S')) break;   // dogs'
            trail.insert(trail.begin(), punct_tok(c.back()));
            c.pop_back();
        }
        for (auto &m : lead) out.push_back(m);
        if (!c.empty()) {
            // a lone dash between spaces is a clause dash
            if (c.size() == 1 && (c[0] == '-' || c[0] == 0x2013 || c[0] == 0x2014)) {
                MTok m;
                m.text = enc(c);
                m.tag = ":";
                out.push_back(m);
            } else {
                // currency symbol and percent sign are their own tokens ("$3.50" → "$", "3.50"; "42%" → "42", "%")
                size_t a = 0, b = c.size();
                if (b - a > 1 && (c[0] == '$' || c[0] == 0xA3 || c[0] == 0x20AC) && is_digit(c[1])) {
                    MTok m;
                    m.text = enc(U(1, c[0]));
                    m.tag = "$";
                    out.push_back(m);
                    a = 1;
                }
                std::vector<MTok> tailpct;
                if (b - a > 1 && c[b - 1] == '%') {
                    MTok m;
                    m.text = "%";
                    m.tag = "NN";
                    tailpct.push_back(m);
                    --b;
                }
                MTok w;
                w.text = enc(U(c.begin() + long(a), c.begin() + long(b)));
                out.push_back(w);
                for (auto &m : tailpct) out.push_back(m);
            }
        }
        for (auto &m : trail) out.push_back(m);
        if (!out.empty()) out.back().whitespace = ws;
    }
    const Lexicon &lex_;
};

// MToken.merge_tokens
MTok merge_tokens(const std::vector<MTok> &ts, const OptS &unk) {
    MTok r;
    std::optional<double> st;
    bool st_set = false, st_multi = false;
    for (const auto &t : ts)
        if (t.stress) {
            if (st_set && *st != *t.stress) st_multi = true;
            st = t.stress;
            st_set = true;
        }
    if (unk) {
        std::string ps;
        for (const auto &t : ts) {
            if (t.prespace && !ps.empty() && !is_space(dec(ps).back()) && t.phonemes && !t.phonemes->empty()) ps += " ";
            ps += t.phonemes ? *t.phonemes : *unk;
        }
        r.phonemes = ps;
    }
    for (size_t i = 0; i + 1 < ts.size(); ++i) r.text += ts[i].text + ts[i].whitespace;
    r.text += ts.back().text;
    // tag of the token with the most "capital weight"
    int best = -1;
    for (const auto &t : ts) {
        int w = 0;
        for (char32_t c : dec(t.text)) w += (c == lower_cp(c)) ? 1 : 2;
        if (w > best) { best = w; r.tag = t.tag; }
    }
    r.whitespace = ts.back().whitespace;
    r.is_head = ts.front().is_head;
    r.stress = (st_set && !st_multi) ? st : std::nullopt;
    for (const auto &t : ts) if (!t.currency.empty() && t.currency > r.currency) r.currency = t.currency;
    r.prespace = ts.front().prespace;
    int rating = 99;
    for (const auto &t : ts) { if (t.rating < 0) { rating = -1; break; } rating = std::min(rating, t.rating); }
    r.rating = rating;
    return r;
}

using Word = std::vector<MTok>;   // size 1 = plain token, else a group without spaces
struct WordItem { bool list = false; Word toks; };

TokenContext token_context(const TokenContext &ctx, const OptS &ps, const MTok &token) {
    std::optional<bool> vowel = ctx.future_vowel;
    if (ps && !ps->empty())
        for (char32_t c : dec(*ps))
            if (contains_cp(VOWELS, c) || contains_cp(CONSONANTS, c) || contains_cp(NON_QUOTE_PUNCTS, c)) {
                vowel = contains_cp(NON_QUOTE_PUNCTS, c) ? std::optional<bool>() : std::optional<bool>(contains_cp(VOWELS, c));
                break;
            }
    TokenContext r;
    r.future_vowel = vowel;
    r.future_to = token.is_to();
    return r;
}

void resolve_tokens(Word &tokens) {
    std::string text;
    for (size_t i = 0; i + 1 < tokens.size(); ++i) text += tokens[i].text + tokens[i].whitespace;
    text += tokens.back().text;
    bool prespace = text.find(' ') != std::string::npos || text.find('/') != std::string::npos;
    if (!prespace) {
        std::unordered_set<int> kinds;
        for (char32_t c : dec(text)) {
            if (contains_cp(SUBTOKEN_JUNKS, c)) continue;
            kinds.insert(is_letter(c) ? 0 : (is_digit(c) ? 1 : 2));
        }
        prespace = kinds.size() > 1;
    }
    for (size_t i = 0; i < tokens.size(); ++i) {
        MTok &t = tokens[i];
        if (!t.phonemes) {
            bool junk = true;
            for (char32_t c : dec(t.text)) if (!contains_cp(SUBTOKEN_JUNKS, c)) junk = false;
            if (i == tokens.size() - 1 && ulen(t.text) == 1 && contains_cp(NON_QUOTE_PUNCTS, dec(t.text)[0])) { t.phonemes = t.text; t.rating = 3; }
            else if (junk) { t.phonemes = std::string(); t.rating = 3; }
        } else if (i > 0) {
            t.prespace = prespace;
        }
    }
    if (prespace) return;
    struct Ix { bool primary; int weight; size_t i; };
    std::vector<Ix> idx;
    for (size_t i = 0; i < tokens.size(); ++i)
        if (tokens[i].phonemes && !tokens[i].phonemes->empty())
            idx.push_back({tokens[i].phonemes->find(PRIMARY) != std::string::npos, tokens[i].stress_weight(), i});
    if (idx.size() == 2 && ulen(tokens[idx[0].i].text) == 1) {
        MTok &t = tokens[idx[1].i];
        t.phonemes = apply_stress(t.phonemes, -0.5);
        return;
    }
    int sum_primary = 0;
    for (auto &x : idx) sum_primary += x.primary ? 1 : 0;
    if (idx.size() < 2 || sum_primary <= int((idx.size() + 1) / 2)) return;
    std::sort(idx.begin(), idx.end(), [](const Ix &a, const Ix &b) {
        if (a.primary != b.primary) return !a.primary && b.primary;
        if (a.weight != b.weight) return a.weight < b.weight;
        return a.i < b.i;
    });
    idx.resize(idx.size() / 2);
    for (auto &x : idx) tokens[x.i].phonemes = apply_stress(tokens[x.i].phonemes, -0.5);
}

// G2P.retokenize
std::vector<WordItem> retokenize(std::vector<MTok> &tokens) {
    std::vector<WordItem> words;
    std::string currency;
    static const std::unordered_set<std::string> PUNCT_TAGS = {".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP"};
    for (size_t i = 0; i < tokens.size(); ++i) {
        MTok &token = tokens[i];
        std::vector<MTok> ts;
        if (!token.alias && !token.phonemes) {
            for (const auto &x : subtokenize(token.text)) { MTok m = token; m.text = x; m.whitespace = ""; ts.push_back(m); }
        } else {
            ts.push_back(token);
        }
        if (ts.empty()) { MTok m = token; m.whitespace = ""; ts.push_back(m); }
        ts.back().whitespace = token.whitespace;
        for (size_t j = 0; j < ts.size(); ++j) {
            MTok &t = ts[j];
            if (t.alias || t.phonemes) {
            } else if (t.tag == "$" && (t.text == "$" || t.text == "£" || t.text == "€")) {
                currency = t.text;
                t.phonemes = std::string();
                t.rating = 4;
            } else if (t.tag == ":" && (t.text == "-" || t.text == "–")) {
                t.phonemes = std::string("—");
                t.rating = 3;
            } else if (PUNCT_TAGS.count(t.tag)) {
                if (t.tag == "-LRB-") t.phonemes = std::string("(");
                else if (t.tag == "-RRB-") t.phonemes = std::string(")");
                else if (t.tag == "``") t.phonemes = std::string("“");
                else if (t.tag == "\"\"" || t.tag == "''") t.phonemes = std::string("”");
                else {
                    std::string p;
                    for (char32_t c : dec(t.text)) if (contains_cp(PUNCTS, c)) p += enc(U(1, c));
                    if (t.text == "...") p = "…";
                    t.phonemes = p;
                }
                t.rating = 4;
            } else if (!currency.empty()) {
                if (t.tag != "CD") currency.clear();
                else if (j + 1 == ts.size() && (i + 1 == tokens.size() || tokens[i + 1].tag != "CD")) t.currency = currency;
            } else if (j > 0 && j + 1 < ts.size() && t.text == "2") {
                const U a = dec(ts[j - 1].text), b = dec(ts[j + 1].text);
                if (!a.empty() && !b.empty() && is_letter(a.back()) && is_letter(b.front())) t.alias = std::string("to");
            }
            if (t.alias || t.phonemes) {
                words.push_back({false, {t}});
            } else if (!words.empty() && words.back().list && words.back().toks.back().whitespace.empty()) {
                t.is_head = false;
                words.back().toks.push_back(t);
            } else {
                words.push_back({t.whitespace.empty(), {t}});
            }
        }
    }
    for (auto &w : words) if (w.list && w.toks.size() == 1) w.list = false;
    return words;
}

OptS spell_letters(const Lexicon &lex, const std::string &word);
// Fallback for words outside the lexicons (misaki asks espeak-ng here): split into known lexicon words, longest first
// ("Orwell" → "Or" + "well"), keep only the first primary stress; acronyms and short words are spelled letter by letter.
OptS spell(const Lexicon &lex, const std::string &word_in) {
    std::string word = word_in, suffix_ps;
    if (ends(word, "'s") || ends(word, "’s")) { word = word.substr(0, word.size() - (ends(word, "'s") ? 2 : 4)); suffix_ps = "s"; }
    const U u = dec(word);
    bool letters = !u.empty();
    for (char32_t c : u) if (!(c < 0x80 && is_letter(c))) letters = false;
    if (letters && word != upper(word) && u.size() > 3) {
        const std::string lw = lower(word);
        const U lu = dec(lw);
        // best split by dynamic programming: fewest pieces, every piece (length ≥ 2) a lexicon word
        const size_t n = lu.size();
        std::vector<int> best(n + 1, 1 << 20), from(n + 1, -1);
        best[0] = 0;
        for (size_t i = 0; i < n; ++i) {
            if (best[i] >= (1 << 20)) continue;
            for (size_t j = std::min(n, i + 14); j >= i + 2; --j) {
                const std::string piece = enc(U(lu.begin() + long(i), lu.begin() + long(j)));
                static const std::unordered_set<std::string> SHORT = {"or", "in", "on", "at", "up", "an", "as", "be", "by", "do", "go",
                                                                      "he", "if", "is", "it", "me", "my", "no", "of", "so", "us", "we"};
                const bool ok = (j - i >= 3 || SHORT.count(piece)) && lex.has_word(piece);
                if (ok && best[i] + 1 < best[j]) { best[j] = best[i] + 1; from[j] = int(i); }
            }
        }
        if (best[n] <= 3) {
            std::vector<std::string> pieces;
            for (size_t j = n; j > 0; j = size_t(from[j])) pieces.insert(pieces.begin(), enc(U(lu.begin() + from[j], lu.begin() + long(j))));
            std::string ps;
            for (size_t k = 0; k < pieces.size(); ++k) {
                OptS p = lex.lookup(pieces[k], "", std::nullopt, nullptr).ps;
                p = apply_stress(p, k == 0 ? 2.0 : -1.0);   // the first piece carries the primary stress, the rest secondary
                ps += *p;
            }
            if (!suffix_ps.empty()) ps = lex._s(ps);
            return ps;
        }
    }
    OptS r = spell_letters(lex, word);
    if (r && !suffix_ps.empty()) r = lex._s(*r);
    return r;
}
// letter by letter
OptS spell_letters(const Lexicon &lex, const std::string &word) {
    std::string ps;
    bool any = false;
    for (char32_t c : dec(word)) {
        if (!is_letter(c) || c >= 0x80) continue;
        PS p = lex.lookup(enc(U(1, upper_cp(c))), "", std::nullopt, nullptr);
        if (!p.ps) continue;
        if (any) ps += " ";
        ps += *p.ps;
        any = true;
    }
    if (!any) return std::nullopt;
    return ps;
}

}  // namespace

// ------------------------------------------------------------------------------------------------ G2P
struct KokoroG2P::Impl {
    Lexicon lex;
    Impl(const std::string &g, const std::string &s, bool b) : lex(g, s, b) {}
};

KokoroG2P::KokoroG2P(const std::string &gold, const std::string &silver, bool british) : p_(new Impl(gold, silver, british)) {}
KokoroG2P::~KokoroG2P() = default;

std::vector<KToken> KokoroG2P::tokens(const std::string &text_in) const {
    const Lexicon &lex = p_->lex;
    // preprocess: [word](/phonemes/), [word](-1 | 0.5 | 2 …) stress, [word](#flags#) — misaki's link syntax
    std::string text;
    struct Feat { std::string ph; OptD stress; std::string flags; };
    std::vector<std::pair<size_t, Feat>> feats;   // by byte offset of the word in text
    {
        size_t i = 0;
        const std::string &s = text_in;
        while (i < s.size()) {
            if (s[i] == '[') {
                const size_t rb = s.find(']', i + 1), lp = rb == std::string::npos ? rb : rb + 1;
                if (rb != std::string::npos && lp < s.size() && s[lp] == '(') {
                    const size_t rp = s.find(')', lp + 1);
                    if (rp != std::string::npos && s.find('[', i + 1) > rb) {
                        const std::string word = s.substr(i + 1, rb - i - 1), f = s.substr(lp + 1, rp - lp - 1);
                        Feat ft;
                        bool ok = false;
                        if (f.size() > 1 && f[0] == '/' ) { ft.ph = strip_chars(f.substr(1), '/'); ok = true; }
                        else if (f.size() > 1 && f[0] == '#' && f.back() == '#') { ft.flags = strip_chars(f, '#'); ok = true; }
                        else {
                            try { size_t used = 0; const double v = std::stod(f, &used); if (used == f.size()) { ft.stress = v; ok = true; } } catch (...) {}
                        }
                        if (ok) feats.push_back({text.size(), ft});
                        text += word;
                        i = rp + 1;
                        continue;
                    }
                }
            }
            text += s[i++];
        }
    }
    // left-strip like misaki's preprocess
    {
        size_t a = 0;
        while (a < text.size() && (text[a] == ' ' || text[a] == '\t' || text[a] == '\n' || text[a] == '\r')) ++a;
        if (a) {
            text = text.substr(a);
            for (auto &f : feats) f.first = f.first >= a ? f.first - a : 0;
        }
    }
    Tokenizer tok(lex);
    std::vector<MTok> toks = tok(text);
    // apply features to the tokens that start at a feature's offset (the whole run until the next space)
    if (!feats.empty()) {
        size_t off = 0;
        std::vector<size_t> starts_;
        for (const auto &t : toks) { const size_t at = text.find(t.text, off); starts_.push_back(at); if (at != std::string::npos) off = at + t.text.size(); }
        for (const auto &f : feats) {
            for (size_t k = 0; k < toks.size(); ++k) {
                if (starts_[k] != f.first) continue;
                size_t end = k;
                while (end + 1 < toks.size() && toks[end].whitespace.empty()) ++end;
                for (size_t m = k; m <= end; ++m) {
                    if (!f.second.ph.empty()) { toks[m].is_head = m == k; toks[m].phonemes = m == k ? f.second.ph : std::string(); toks[m].rating = 5; }
                    if (f.second.stress) toks[m].stress = f.second.stress;
                    if (!f.second.flags.empty()) toks[m].num_flags = f.second.flags;
                }
                break;
            }
        }
    }
    std::vector<WordItem> words = retokenize(toks);
    TokenContext ctx;
    for (long wi = long(words.size()) - 1; wi >= 0; --wi) {
        WordItem &w = words[size_t(wi)];
        if (!w.list) {
            MTok &t = w.toks[0];
            if (!t.phonemes) {
                PS r = lex(t, ctx);
                t.phonemes = r.ps;
                t.rating = r.rating;
            }
            if (!t.phonemes) {
                t.phonemes = spell(lex, t.alias ? *t.alias : t.text);
                t.rating = 1;
            }
            ctx = token_context(ctx, t.phonemes, t);
            continue;
        }
        Word &ts = w.toks;
        size_t left = 0, right = ts.size();
        bool fallback = false;
        while (left < right) {
            bool has_fixed = false;
            for (size_t k = left; k < right; ++k) if (ts[k].alias || ts[k].phonemes) has_fixed = true;
            PS r;
            MTok merged;
            if (!has_fixed) {
                merged = merge_tokens(Word(ts.begin() + long(left), ts.begin() + long(right)), std::nullopt);
                r = lex(merged, ctx);
            }
            if (r.ps) {
                ts[left].phonemes = r.ps;
                ts[left].rating = r.rating;
                for (size_t k = left + 1; k < right; ++k) { ts[k].phonemes = std::string(); ts[k].rating = r.rating; }
                ctx = token_context(ctx, r.ps, merged);
                right = left;
                left = 0;
            } else if (left + 1 < right) {
                ++left;
            } else {
                --right;
                MTok &t = ts[right];
                if (!t.phonemes) {
                    bool junk = true;
                    for (char32_t c : dec(t.text)) if (!contains_cp(SUBTOKEN_JUNKS, c)) junk = false;
                    if (junk) { t.phonemes = std::string(); t.rating = 3; }
                    else fallback = true;
                }
                left = 0;
            }
        }
        if (fallback) {
            // spell what is still unknown (misaki: the espeak fallback on the whole group)
            for (auto &t : ts)
                if (!t.phonemes) { t.phonemes = spell(lex, t.text); t.rating = 1; }
        }
        resolve_tokens(ts);
    }
    std::vector<KToken> out;
    for (auto &w : words) {
        const MTok m = w.list ? merge_tokens(w.toks, std::string()) : w.toks[0];
        KToken k;
        k.text = m.text;
        k.phonemes = m.phonemes ? *m.phonemes : std::string();
        k.whitespace = m.whitespace;
        out.push_back(k);
    }
    return out;
}

std::string KokoroG2P::phonemize(const std::string &text) const {
    std::string r;
    for (const auto &t : tokens(text)) r += t.phonemes + t.whitespace;
    return r;
}

}  // namespace vox
