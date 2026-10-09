// ============================================================================
// Токенизатор Qwen2/Qwen3 (byte-level BPE) — порт src/tokenizer.cpp: тот же претокенизатор, те же ранги слияний,
// те же особые токены и та же «безопасная граница» для потокового ввода.
// ============================================================================
const _reLetter = /\p{L}/u, _reNumber = /\p{N}/u, _reSpace = /\p{White_Space}/u;
const _isLetter = (c) => _reLetter.test(c), _isNumber = (c) => _reNumber.test(c), _isSpace = (c) => _reSpace.test(c);
const _isNl = (c) => c === "\r" || c === "\n";

// Претокены: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
function qwenPretokenize(s) {
  const c = Array.from(s), n = c.length, out = [];
  const lower = (ch) => (ch >= "A" && ch <= "Z" ? String.fromCharCode(ch.charCodeAt(0) + 32) : ch);
  const ci = (i, ch) => i < n && lower(c[i]) === ch;
  let i = 0;
  while (i < n) {
    let end = 0;
    if (c[i] === "'" && i + 1 < n) {
      if (ci(i + 1, "s") || ci(i + 1, "t") || ci(i + 1, "m") || ci(i + 1, "d")) end = i + 2;
      else if ((ci(i + 1, "r") && ci(i + 2, "e")) || (ci(i + 1, "v") && ci(i + 2, "e")) || (ci(i + 1, "l") && ci(i + 2, "l"))) end = i + 3;
    }
    if (!end) {
      let j = i;
      if (!_isNl(c[j]) && !_isLetter(c[j]) && !_isNumber(c[j]) && j + 1 < n && _isLetter(c[j + 1])) j = i + 1;
      if (_isLetter(c[j])) { while (j < n && _isLetter(c[j])) j++; end = j; }
    }
    if (!end && _isNumber(c[i])) end = i + 1;
    if (!end) {
      let j = i;
      if (c[j] === " " && j + 1 < n && !_isSpace(c[j + 1]) && !_isLetter(c[j + 1]) && !_isNumber(c[j + 1])) j = i + 1;
      if (!_isSpace(c[j]) && !_isLetter(c[j]) && !_isNumber(c[j])) {
        while (j < n && !_isSpace(c[j]) && !_isLetter(c[j]) && !_isNumber(c[j])) j++;
        while (j < n && _isNl(c[j])) j++;
        end = j;
      }
    }
    if (!end && _isSpace(c[i])) {
      let run = i;
      while (run < n && _isSpace(c[run])) run++;
      let afterNl = 0;
      for (let k = run; k > i; k--) if (_isNl(c[k - 1])) { afterNl = k; break; }
      if (afterNl) end = afterNl;
      else if (run === n) end = run;
      else if (run - i >= 2) end = run - 1;
      else end = run;
    }
    if (!end) end = i + 1;
    out.push(c.slice(i, end).join(""));
    i = end;
  }
  return out;
}

class QwenTokenizer {
  // vocabText: строки "id\tтокен"; mergesText: строки "a b" по рангу; specials: токены, которые ищутся в тексте буквально.
  constructor(vocabText, mergesText, specials) {
    const bs = [];
    for (let b = 33; b <= 126; b++) bs.push(b);
    for (let b = 161; b <= 172; b++) bs.push(b);
    for (let b = 174; b <= 255; b++) bs.push(b);
    const cs = bs.slice();
    let extra = 0;
    for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + extra++); }
    this.byteUni = new Array(256);
    for (let k = 0; k < bs.length; k++) this.byteUni[bs[k]] = String.fromCodePoint(cs[k]);
    this.tokToId = new Map();
    let maxId = -1;
    for (const line of vocabText.split("\n")) {
      if (!line) continue;
      const tab = line.indexOf("\t");
      if (tab < 0) fail("токенизатор: плохая строка словаря", "format");
      const id = parseInt(line.slice(0, tab), 10);
      this.tokToId.set(line.slice(tab + 1), id);
      if (id > maxId) maxId = id;
    }
    this.vocabSize = maxId + 1;
    this.merges = new Map();
    let rank = 0;
    for (const line of mergesText.split("\n")) {
      if (!line || line[0] === "#") continue;
      const sp = line.indexOf(" ");
      if (sp < 0) continue;
      const a = line.slice(0, sp), b = line.slice(sp + 1);
      const ia = this.tokToId.get(a), ib = this.tokToId.get(b), im = this.tokToId.get(a + b);
      if (ia !== undefined && ib !== undefined && im !== undefined) this.merges.set(ia * 4294967296 + ib, [rank, im]);
      rank++;
    }
    this.specials = [];
    for (const s of specials || []) { const id = this.tokToId.get(s); if (id !== undefined) this.specials.push([s, id]); }
    this.specials.sort((x, y) => y[0].length - x[0].length);
    this.cache = new Map();
  }
  _bpe(word, out) {
    const hit = this.cache.get(word);
    if (hit) { for (const t of hit) out.push(t); return; }
    const sym = [];
    for (const b of new TextEncoder().encode(word)) {
      const id = this.tokToId.get(this.byteUni[b]);
      if (id === undefined) fail("в словаре токенизатора нет однобайтового токена", "format");
      sym.push(id);
    }
    while (sym.length > 1) {
      let best = Infinity, bi = -1, bid = 0;
      for (let k = 0; k + 1 < sym.length; k++) {
        const m = this.merges.get(sym[k] * 4294967296 + sym[k + 1]);
        if (m && m[0] < best) { best = m[0]; bi = k; bid = m[1]; }
      }
      if (bi < 0) break;
      sym[bi] = bid;
      sym.splice(bi + 1, 1);
    }
    if (this.cache.size > 50000) this.cache.clear();
    this.cache.set(word, sym.slice());
    for (const t of sym) out.push(t);
  }
  _plain(s, out) { for (const p of qwenPretokenize(s)) this._bpe(p, out); }
  encode(s) {
    const out = [];
    let seg = 0, i = 0;
    while (i < s.length) {
      let matched = false;
      if (s[i] === "<") {
        for (const [tok, id] of this.specials) {
          if (s.startsWith(tok, i)) {
            if (i > seg) this._plain(s.slice(seg, i), out);
            out.push(id);
            i += tok.length; seg = i; matched = true;
            break;
          }
        }
      }
      if (!matched) i++;
    }
    if (seg < s.length) this._plain(s.slice(seg), out);
    return out;
  }
  // Граница в строке JS (индекс UTF-16), до которой пока можно токенизировать, не меняя результат относительно целого текста;
  // 0 — пока нельзя. Порт Tokenizer::safe_cut.
  safeCut(pending) {
    const u = Array.from(pending), offs = [];
    let o = 0;
    for (const ch of u) { offs.push(o); o += ch.length; }
    for (let i = u.length; i-- > 1;) {
      if (_isSpace(u[i]) && !_isSpace(u[i - 1]) && (_isLetter(u[i - 1]) || _isNumber(u[i - 1]))) return offs[i];
    }
    return 0;
  }
}
