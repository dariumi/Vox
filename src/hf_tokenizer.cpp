#include "hf_tokenizer.hpp"

#include <algorithm>
#include <climits>

#include "common.hpp"
#include "formats.hpp"

namespace vox {
namespace {

#include "unicode_word.inc"

bool in_ranges(uint32_t cp, const uint32_t (*t)[2], size_t n) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < t[mid][0]) hi = mid;
        else if (cp > t[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}
bool is_word(uint32_t c) { return in_ranges(c, kWordCh, kWordCh_N); }
bool is_space(uint32_t c) {   // White_Space
    return (c >= 9 && c <= 13) || c == ' ' || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
           c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// UTF-8 → (code point, byte offset) list
struct Cps {
    std::vector<uint32_t> cp;
    std::vector<size_t> off;
};
Cps decode(const std::string &s) {
    Cps r;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if (i + size_t(len) > s.size()) len = 1;
        uint32_t v = len == 1 ? c : len == 2 ? (c & 0x1Fu) : len == 3 ? (c & 0x0Fu) : (c & 0x07u);
        for (int k = 1; k < len; ++k) v = (v << 6) | (static_cast<unsigned char>(s[i + size_t(k)]) & 0x3Fu);
        r.cp.push_back(v);
        r.off.push_back(i);
        i += size_t(len);
    }
    r.off.push_back(s.size());
    return r;
}

std::string lower_utf8(const std::string &s) {   // ASCII and Latin-1 / Cyrillic / Greek basic ranges (str.lower for common text)
    std::string out;
    const Cps d = decode(s);
    for (size_t i = 0; i < d.cp.size(); ++i) {
        uint32_t c = d.cp[i];
        if (c >= 'A' && c <= 'Z') c += 32;
        else if ((c >= 0xC0 && c <= 0xDE && c != 0xD7)) c += 32;
        else if (c >= 0x410 && c <= 0x42F) c += 32;
        else if (c >= 0x400 && c <= 0x40F) c += 80;
        else if (c >= 0x391 && c <= 0x3A9 && c != 0x3A2) c += 32;
        if (c < 0x80) out += char(c);
        else if (c < 0x800) { out += char(0xC0 | (c >> 6)); out += char(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { out += char(0xE0 | (c >> 12)); out += char(0x80 | ((c >> 6) & 0x3F)); out += char(0x80 | (c & 0x3F)); }
        else { out += char(0xF0 | (c >> 18)); out += char(0x80 | ((c >> 12) & 0x3F)); out += char(0x80 | ((c >> 6) & 0x3F)); out += char(0x80 | (c & 0x3F)); }
    }
    return out;
}

}  // namespace

HfTokenizer::HfTokenizer(const std::string &text) {
    const json j = json::parse(text);
    const json &model = j.at("model");
    if (model.value("type", std::string("BPE")) != "BPE") fail(VOX_ERR_UNSUPPORTED, "tokenizer.json: only BPE models are supported");
    if (model.value("byte_fallback", false)) fail(VOX_ERR_UNSUPPORTED, "tokenizer.json: byte_fallback is not supported");
    for (auto it = model.at("vocab").begin(); it != model.at("vocab").end(); ++it) vocab_[it.key()] = it.value().get<int32_t>();
    int rank = 0;
    for (const auto &m : model.at("merges")) {
        std::string a, b;
        if (m.is_string()) {
            const std::string s = m.get<std::string>();
            const size_t sp = s.find(' ');
            if (sp == std::string::npos) fail(VOX_ERR_FORMAT, "tokenizer.json: bad merge '" + s + "'");
            a = s.substr(0, sp);
            b = s.substr(sp + 1);
        } else {
            a = m.at(0).get<std::string>();
            b = m.at(1).get<std::string>();
        }
        ranks_.emplace(a + " " + b, rank++);
    }
    if (model.contains("unk_token") && model["unk_token"].is_string()) {
        auto it = vocab_.find(model["unk_token"].get<std::string>());
        if (it != vocab_.end()) unk_ = it->second;
    }
    fuse_unk_ = model.value("fuse_unk", false);
    if (model.contains("continuing_subword_prefix") && model["continuing_subword_prefix"].is_string()) prefix_ = model["continuing_subword_prefix"];
    if (model.contains("end_of_word_suffix") && model["end_of_word_suffix"].is_string()) suffix_ = model["end_of_word_suffix"];
    if (j.contains("added_tokens"))
        for (const auto &t : j["added_tokens"]) {
            added_.push_back({t.at("content").get<std::string>(), t.at("id").get<int32_t>()});
            vocab_.emplace(t.at("content").get<std::string>(), t.at("id").get<int32_t>());
        }
    std::stable_sort(added_.begin(), added_.end(), [](const auto &x, const auto &y) { return x.first.size() > y.first.size(); });
    // normalizer
    std::vector<json> steps;
    auto flat = [&](const json &n, auto &&self) -> void {
        if (n.is_null()) return;
        const std::string ty = n.value("type", std::string());
        if (ty == "Sequence") { for (const auto &x : n.at("normalizers")) self(x, self); return; }
        steps.push_back(n);
    };
    if (j.contains("normalizer")) flat(j["normalizer"], flat);
    for (const auto &n : steps) {
        const std::string ty = n.at("type").get<std::string>();
        if (ty == "Replace") {
            const json &p = n.at("pattern");
            const std::string to = n.at("content").get<std::string>();
            if (p.contains("String")) norm_.push_back({0, p["String"].get<std::string>(), to});
            else if (p.contains("Regex") && p["Regex"].get<std::string>() == "\\s+") norm_.push_back({1, "", to});
            else fail(VOX_ERR_UNSUPPORTED, "tokenizer.json: unsupported Replace pattern " + p.dump());
        } else if (ty == "Lowercase") {
            norm_.push_back({2, "", ""});
        } else if (ty == "Strip") {
            norm_.push_back({3, "", ""});
        } else if (ty == "NFC") {
            // the text is expected in NFC already (the Python wrapper normalizes)
        } else {
            fail(VOX_ERR_UNSUPPORTED, "tokenizer.json: unsupported normalizer " + ty);
        }
    }
    // pre-tokenizer
    std::vector<json> pres;
    auto flatp = [&](const json &n, auto &&self) -> void {
        if (n.is_null()) return;
        if (n.value("type", std::string()) == "Sequence") { for (const auto &x : n.at("pretokenizers")) self(x, self); return; }
        pres.push_back(n);
    };
    if (j.contains("pre_tokenizer")) flatp(j["pre_tokenizer"], flatp);
    for (const auto &n : pres) {
        const std::string ty = n.at("type").get<std::string>();
        if (ty == "Whitespace") pre_.push_back(0);
        else if (ty == "WhitespaceSplit") pre_.push_back(1);
        else fail(VOX_ERR_UNSUPPORTED, "tokenizer.json: unsupported pre-tokenizer " + ty);
    }
}

int32_t HfTokenizer::token_id(const std::string &t) const {
    auto it = vocab_.find(t);
    return it == vocab_.end() ? -1 : it->second;
}

std::string HfTokenizer::normalize(const std::string &s0) const {
    std::string s = s0;
    for (const Rule &r : norm_) {
        if (r.kind == 0) {
            if (r.a.empty()) continue;
            std::string o;
            for (size_t i = 0; i < s.size();) {
                if (s.compare(i, r.a.size(), r.a) == 0) { o += r.b; i += r.a.size(); }
                else o += s[i++];
            }
            s.swap(o);
        } else if (r.kind == 1) {
            const Cps d = decode(s);
            std::string o;
            for (size_t i = 0; i < d.cp.size();) {
                if (is_space(d.cp[i])) {
                    while (i < d.cp.size() && is_space(d.cp[i])) ++i;
                    o += r.b;
                } else {
                    o.append(s, d.off[i], d.off[i + 1] - d.off[i]);
                    ++i;
                }
            }
            s.swap(o);
        } else if (r.kind == 2) {
            s = lower_utf8(s);
        } else {
            const Cps d = decode(s);
            size_t a = 0, b = d.cp.size();
            while (a < b && is_space(d.cp[a])) ++a;
            while (b > a && is_space(d.cp[b - 1])) --b;
            s = s.substr(d.off[a], d.off[b] - d.off[a]);
        }
    }
    return s;
}

std::vector<std::string> HfTokenizer::pretokenize(const std::string &s) const {
    std::vector<std::string> words{s};
    for (int p : pre_) {
        std::vector<std::string> out;
        for (const std::string &w : words) {
            const Cps d = decode(w);
            size_t i = 0;
            const size_t n = d.cp.size();
            while (i < n) {
                if (is_space(d.cp[i])) { ++i; continue; }
                size_t j = i;
                if (p == 1) {
                    while (j < n && !is_space(d.cp[j])) ++j;
                } else if (is_word(d.cp[i])) {
                    while (j < n && is_word(d.cp[j])) ++j;
                } else {
                    while (j < n && !is_word(d.cp[j]) && !is_space(d.cp[j])) ++j;
                }
                out.push_back(w.substr(d.off[i], d.off[j] - d.off[i]));
                i = j;
            }
        }
        words.swap(out);
    }
    return words;
}

void HfTokenizer::bpe(const std::string &word, std::vector<int32_t> &out) const {
    const Cps d = decode(word);
    std::vector<std::string> sym;
    for (size_t i = 0; i < d.cp.size(); ++i) {
        std::string c = word.substr(d.off[i], d.off[i + 1] - d.off[i]);
        if (i > 0) c = prefix_ + c;
        if (i + 1 == d.cp.size()) c += suffix_;
        sym.push_back(c);
    }
    for (;;) {
        int best = INT_MAX;
        size_t at = 0;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            auto it = ranks_.find(sym[i] + " " + sym[i + 1]);
            if (it != ranks_.end() && it->second < best) { best = it->second; at = i; }
        }
        if (best == INT_MAX) break;
        std::string merged = sym[at];
        std::string b = sym[at + 1];
        if (!prefix_.empty() && b.compare(0, prefix_.size(), prefix_) == 0) b = b.substr(prefix_.size());
        merged += b;
        sym[at] = merged;
        sym.erase(sym.begin() + long(at) + 1);
    }
    bool prev_unk = false;
    for (const std::string &x : sym) {
        auto it = vocab_.find(x);
        if (it != vocab_.end()) { out.push_back(it->second); prev_unk = false; continue; }
        if (unk_ < 0) fail(VOX_ERR_ARG, "tokenizer: '" + x + "' is not in the vocabulary and the model has no unk token");
        if (!(fuse_unk_ && prev_unk)) out.push_back(unk_);
        prev_unk = true;
    }
}

std::vector<int32_t> HfTokenizer::encode(const std::string &text) const {
    std::vector<int32_t> out;
    auto plain = [&](const std::string &seg) {
        if (seg.empty()) return;
        for (const std::string &w : pretokenize(normalize(seg))) bpe(w, out);
    };
    size_t start = 0, i = 0;
    while (i < text.size()) {
        bool hit = false;
        for (const auto &a : added_) {
            if (!a.first.empty() && text.compare(i, a.first.size(), a.first) == 0) {
                plain(text.substr(start, i - start));
                out.push_back(a.second);
                i += a.first.size();
                start = i;
                hit = true;
                break;
            }
        }
        if (!hit) ++i;
    }
    plain(text.substr(start));
    return out;
}

}  // namespace vox
