#include "kokoro_g2p.hpp"

#include <algorithm>
#include <cstring>

#include "common.hpp"

namespace vox {
namespace {

std::vector<std::string> cps(const std::string &s) {   // UTF-8 code points
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        n = std::min(n, s.size() - i);
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}
size_t ulen(const std::string &s) { return cps(s).size(); }
bool is_space_cp(const std::string &c) { return c == " " || c == "\t" || c == "\n" || c == "\r" || c == "\f" || c == "\v"; }
std::string ustrip(const std::string &s) {
    auto v = cps(s);
    size_t a = 0, b = v.size();
    while (a < b && is_space_cp(v[a])) ++a;
    while (b > a && is_space_cp(v[b - 1])) --b;
    std::string o;
    for (size_t i = a; i < b; ++i) o += v[i];
    return o;
}
std::string rstrip(const std::string &s) {
    auto v = cps(s);
    size_t b = v.size();
    while (b > 0 && is_space_cp(v[b - 1])) --b;
    std::string o;
    for (size_t i = 0; i < b; ++i) o += v[i];
    return o;
}
std::string lstrip(const std::string &s) {
    auto v = cps(s);
    size_t a = 0;
    while (a < v.size() && is_space_cp(v[a])) ++a;
    std::string o;
    for (size_t i = a; i < v.size(); ++i) o += v[i];
    return o;
}
std::string replace_all(std::string s, const std::string &a, const std::string &b) {
    for (size_t p = s.find(a); p != std::string::npos; p = s.find(a, p + b.size())) s.replace(p, a.size(), b);
    return s;
}
std::string tokens_to_ps(const std::vector<KToken> &t, size_t from, size_t to) {
    std::string s;
    for (size_t i = from; i < to; ++i) s += t[i].phonemes + (t[i].whitespace.empty() ? "" : " ");
    return ustrip(s);
}
bool single_of(const std::string &ps, const char *set) {
    const auto v = cps(ps);
    if (v.size() != 1) return false;
    for (const auto &c : cps(set))
        if (c == v[0]) return true;
    return false;
}
size_t waterfall_last(const std::vector<KToken> &tks, size_t next_count, size_t limit) {
    static const char *kFalls[] = {"!.?…", ":;", ",—"};
    for (const char *w : kFalls) {
        long z = -1;
        for (long i = long(tks.size()) - 1; i >= 0; --i)
            if (single_of(tks[size_t(i)].phonemes, w)) { z = i; break; }
        if (z < 0) continue;
        size_t zz = size_t(z) + 1;
        if (zz < tks.size() && single_of(tks[zz].phonemes, ")”")) ++zz;
        if (next_count - ulen(tokens_to_ps(tks, 0, zz)) <= limit) return zz;
    }
    return tks.size();
}

}  // namespace

std::vector<std::string> kokoro_chunks(std::vector<KToken> tokens, bool keep_flap, size_t limit) {
    std::vector<std::string> out;
    std::vector<KToken> tks;
    size_t pcount = 0;
    auto emit = [&](const std::string &ps) {
        if (ps.empty()) return;
        auto v = cps(ps);
        if (v.size() > limit) {   // KPipeline truncates (with a warning)
            std::string t;
            for (size_t i = 0; i < limit; ++i) t += v[i];
            out.push_back(t);
        } else {
            out.push_back(ps);
        }
    };
    for (KToken t : tokens) {
        if (!keep_flap) t.phonemes = replace_all(t.phonemes, "ɾ", "T");
        std::string next_ps = t.phonemes + (t.whitespace.empty() ? "" : " ");
        const size_t next_pcount = pcount + ulen(rstrip(next_ps));
        if (next_pcount > limit) {
            const size_t z = waterfall_last(tks, next_pcount, limit);
            emit(tokens_to_ps(tks, 0, z));
            tks.erase(tks.begin(), tks.begin() + long(z));
            pcount = ulen(tokens_to_ps(tks, 0, tks.size()));
            if (tks.empty()) next_ps = lstrip(next_ps);
        }
        tks.push_back(t);
        pcount += ulen(next_ps);
    }
    if (!tks.empty()) emit(tokens_to_ps(tks, 0, tks.size()));
    return out;
}

std::vector<KToken> kokoro_phoneme_tokens(const std::string &ps) {
    static const std::vector<std::string> punct = cps(";:,.!?—…\"“”()");
    std::vector<KToken> out;
    const auto v = cps(ps);
    size_t i = 0;
    while (i < v.size()) {
        while (i < v.size() && is_space_cp(v[i])) ++i;
        if (i >= v.size()) break;
        std::string word;
        size_t j = i;
        while (j < v.size() && !is_space_cp(v[j])) {
            const bool p = std::find(punct.begin(), punct.end(), v[j]) != punct.end();
            if (p) {
                if (!word.empty()) { out.push_back({word, word, ""}); word.clear(); }
                out.push_back({v[j], v[j], ""});
            } else {
                word += v[j];
            }
            ++j;
        }
        if (!word.empty()) out.push_back({word, word, ""});
        if (j < v.size() && !out.empty()) out.back().whitespace = " ";
        i = j;
    }
    return out;
}

}  // namespace vox
