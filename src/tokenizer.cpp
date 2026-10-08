#include "tokenizer.hpp"

#include <algorithm>
#include <sstream>

namespace vox {
namespace {

#include "unicode_tables.inc"

bool in_table(uint32_t cp, const uint32_t (*t)[2], size_t n) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < t[mid][0]) hi = mid;
        else if (cp > t[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}
bool is_letter(uint32_t c) { return in_table(c, kLetter, kLetter_N); }
bool is_number(uint32_t c) { return in_table(c, kNumber, kNumber_N); }
bool is_space(uint32_t c) { return in_table(c, kSpace, kSpace_N); }
bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }

// Декодирует UTF-8 в кодпоинты с байтовыми смещениями. Некорректные байты → U+FFFD (1 байт).
struct Cps {
    std::vector<uint32_t> cp;
    std::vector<size_t> off;  // off[i] — начало кодпоинта i, off[n] — конец строки
};
Cps decode_utf8(const std::string &s) {
    Cps r;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        uint32_t v = 0xFFFD;
        if (len && i + size_t(len) <= s.size()) {
            v = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
            bool ok = true;
            for (int k = 1; k < len; ++k) {
                unsigned char d = static_cast<unsigned char>(s[i + size_t(k)]);
                if ((d & 0xC0) != 0x80) { ok = false; break; }
                v = (v << 6) | (d & 0x3F);
            }
            if (!ok) { v = 0xFFFD; len = 1; }
        } else {
            len = 1;
        }
        r.cp.push_back(v);
        r.off.push_back(i);
        i += size_t(len);
    }
    r.off.push_back(s.size());
    return r;
}

void put_utf8(std::string &o, uint32_t c) {
    if (c < 0x80) o += char(c);
    else if (c < 0x800) { o += char(0xC0 | (c >> 6)); o += char(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { o += char(0xE0 | (c >> 12)); o += char(0x80 | ((c >> 6) & 0x3F)); o += char(0x80 | (c & 0x3F)); }
    else { o += char(0xF0 | (c >> 18)); o += char(0x80 | ((c >> 12) & 0x3F)); o += char(0x80 | ((c >> 6) & 0x3F)); o += char(0x80 | (c & 0x3F)); }
}

uint64_t pair_key(int32_t a, int32_t b) { return (uint64_t(uint32_t(a)) << 32) | uint32_t(b); }

}  // namespace

std::vector<std::string> Tokenizer::pretokenize(const std::string &s) {
    Cps u = decode_utf8(s);
    const auto &c = u.cp;
    const size_t n = c.size();
    std::vector<std::string> out;
    auto ci = [&](size_t i, uint32_t ch) {  // сравнение без учёта регистра для ASCII
        return i < n && (c[i] | 0x20) == (ch | 0x20) && ((c[i] | 0x20) >= 'a' && (c[i] | 0x20) <= 'z');
    };
    size_t i = 0;
    while (i < n) {
        size_t end = 0;  // 0 = пока нет совпадения
        // 1) сокращения
        if (c[i] == '\'' && i + 1 < n) {
            if (ci(i + 1, 's') || ci(i + 1, 't') || ci(i + 1, 'm') || ci(i + 1, 'd')) end = i + 2;
            else if ((ci(i + 1, 'r') && ci(i + 2, 'e')) || (ci(i + 1, 'v') && ci(i + 2, 'e')) ||
                     (ci(i + 1, 'l') && ci(i + 2, 'l'))) end = i + 3;
        }
        // 2) [^\r\n\p{L}\p{N}]?\p{L}+
        if (!end) {
            size_t j = i;
            if (!is_nl(c[j]) && !is_letter(c[j]) && !is_number(c[j]) && j + 1 < n && is_letter(c[j + 1])) j = i + 1;
            if (is_letter(c[j])) {
                while (j < n && is_letter(c[j])) ++j;
                end = j;
            }
        }
        // 3) один \p{N}
        if (!end && is_number(c[i])) end = i + 1;
        // 4)  ?[^\s\p{L}\p{N}]+[\r\n]*
        if (!end) {
            size_t j = i;
            if (c[j] == ' ' && j + 1 < n && !is_space(c[j + 1]) && !is_letter(c[j + 1]) && !is_number(c[j + 1])) j = i + 1;
            if (!is_space(c[j]) && !is_letter(c[j]) && !is_number(c[j])) {
                while (j < n && !is_space(c[j]) && !is_letter(c[j]) && !is_number(c[j])) ++j;
                while (j < n && is_nl(c[j])) ++j;
                end = j;
            }
        }
        // 5)-7) пробельные: \s*[\r\n]+ | \s+(?!\S) | \s+
        if (!end && is_space(c[i])) {
            size_t run = i;
            while (run < n && is_space(c[run])) ++run;  // [i, run) — максимальный пробельный отрезок
            size_t after_nl = 0;  // позиция сразу после последнего \r/\n в отрезке
            for (size_t k = run; k > i; --k)
                if (is_nl(c[k - 1])) { after_nl = k; break; }
            if (after_nl) end = after_nl;          // \s*[\r\n]+
            else if (run == n) end = run;          // \s+(?!\S) в конце текста
            else if (run - i >= 2) end = run - 1;  // последний пробел уходит следующему токену
            else end = run;                        // \s+
        }
        if (!end) end = i + 1;  // недостижимо, но страхуемся от зацикливания
        out.emplace_back(s.substr(u.off[i], u.off[end] - u.off[i]));
        i = end;
    }
    return out;
}

Tokenizer::Tokenizer(const std::string &vocab_text, const std::string &merges_text,
                     const std::vector<std::string> &specials) {
    // таблица байт → символ (GPT-2 bytes_to_unicode)
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);
    std::vector<int> cs(bs.begin(), bs.end());
    int extra = 0;
    for (int b = 0; b < 256; ++b)
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + extra++); }
    for (size_t k = 0; k < bs.size(); ++k) { std::string s; put_utf8(s, uint32_t(cs[k])); byte_to_uni_[bs[k]] = s; }

    std::istringstream vs(vocab_text);
    std::string line;
    int32_t max_id = -1;
    std::vector<std::pair<int32_t, std::string>> items;
    while (std::getline(vs, line)) {
        if (line.empty()) continue;
        auto tab = line.find('\t');
        if (tab == std::string::npos) fail(VOX_ERR_FORMAT, "bad tokenizer.vocab line");
        int32_t id = std::stoi(line.substr(0, tab));
        items.emplace_back(id, line.substr(tab + 1));
        max_id = std::max(max_id, id);
    }
    id_to_tok_.assign(size_t(max_id + 1), std::string());
    for (auto &[id, t] : items) { id_to_tok_[size_t(id)] = t; tok_to_id_[t] = id; }

    std::istringstream ms(merges_text);
    int32_t rank = 0;
    while (std::getline(ms, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto sp = line.find(' ');
        if (sp == std::string::npos) continue;
        std::string a = line.substr(0, sp), b = line.substr(sp + 1);
        auto ia = tok_to_id_.find(a), ib = tok_to_id_.find(b), im = tok_to_id_.find(a + b);
        if (ia != tok_to_id_.end() && ib != tok_to_id_.end() && im != tok_to_id_.end())
            merges_.emplace(pair_key(ia->second, ib->second), std::make_pair(rank, im->second));
        ++rank;
    }
    for (auto &s : specials) {
        auto it = tok_to_id_.find(s);
        if (it != tok_to_id_.end()) specials_.emplace_back(s, it->second);
    }
    std::sort(specials_.begin(), specials_.end(), [](auto &x, auto &y) { return x.first.size() > y.first.size(); });
}

Tokenizer Tokenizer::from_tmod(const TmodFile &f) {
    auto text = [&](const char *name) {
        const TensorView &v = f.get(name);
        return std::string(reinterpret_cast<const char *>(v.data), v.nbytes);
    };
    std::vector<std::string> sp;
    const json &m = f.manifest();
    if (m.contains("tokenizer") && m["tokenizer"].contains("specials"))
        for (auto &s : m["tokenizer"]["specials"]) sp.push_back(s.get<std::string>());
    return Tokenizer(text("tokenizer.vocab"), text("tokenizer.merges"), sp);
}

int32_t Tokenizer::token_id(const std::string &t) const {
    auto it = tok_to_id_.find(t);
    return it == tok_to_id_.end() ? -1 : it->second;
}

void Tokenizer::bpe(const std::string &word, std::vector<int32_t> &out) const {
    auto hit = cache_.find(word);
    if (hit != cache_.end()) { out.insert(out.end(), hit->second.begin(), hit->second.end()); return; }
    // байты → символы byte-level → начальные токены
    std::vector<int32_t> sym;
    for (unsigned char b : word) {
        auto it = tok_to_id_.find(byte_to_uni_[b]);
        if (it == tok_to_id_.end()) fail(VOX_ERR_FORMAT, "tokenizer vocab lacks a single-byte token");
        sym.push_back(it->second);
    }
    while (sym.size() > 1) {
        int32_t best = INT32_MAX, bi = -1, bid = 0;
        for (size_t k = 0; k + 1 < sym.size(); ++k) {
            auto m = merges_.find(pair_key(sym[k], sym[k + 1]));
            if (m != merges_.end() && m->second.first < best) { best = m->second.first; bi = int32_t(k); bid = m->second.second; }
        }
        if (bi < 0) break;
        sym[size_t(bi)] = bid;
        sym.erase(sym.begin() + bi + 1);
    }
    if (cache_.size() > 50000) cache_.clear();
    cache_[word] = sym;
    out.insert(out.end(), sym.begin(), sym.end());
}

void Tokenizer::encode_plain(const std::string &s, std::vector<int32_t> &out) const {
    for (auto &p : pretokenize(s)) bpe(p, out);
}

std::vector<int32_t> Tokenizer::encode(const std::string &s) const {
    std::vector<int32_t> out;
    size_t seg = 0, i = 0;
    while (i < s.size()) {
        bool matched = false;
        if (s[i] == '<') {
            for (auto &[tok, id] : specials_) {
                if (s.compare(i, tok.size(), tok) == 0) {
                    if (i > seg) encode_plain(s.substr(seg, i - seg), out);
                    out.push_back(id);
                    i += tok.size();
                    seg = i;
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) ++i;
    }
    if (seg < s.size()) encode_plain(s.substr(seg), out);
    return out;
}

// Безопасная граница потоковой токенизации.
// Токенизация целого текста == конкатенация токенизаций частей, если разрез проходит
// в начале пробельного отрезка, перед которым стоит буква или цифра: буквенный претокен
// заканчивается на пробеле независимо от продолжения, а пробельный отрезок целиком
// уходит в следующую часть (там его судьба решается так же, как в полном тексте).
size_t Tokenizer::safe_cut(const std::string &pending) const {
    Cps u = decode_utf8(pending);
    const size_t n = u.cp.size();
    for (size_t i = n; i-- > 1;) {
        if (is_space(u.cp[i]) && !is_space(u.cp[i - 1]) && (is_letter(u.cp[i - 1]) || is_number(u.cp[i - 1])))
            return u.off[i];
    }
    return 0;
}

}  // namespace vox
