"""Text → phonemes for Kokoro with misaki, exactly as kokoro.KPipeline does it (English: misaki.en.G2P with the espeak-ng
fallback when available, then en_tokenize: pieces of at most 510 phonemes cut at the last sentence / clause / comma that fits,
'ɾ' → 'T'). Used by vox.Engine for Kokoro models when misaki is installed; otherwise the engine's built-in G2P
(misaki's lexicons without spaCy and espeak) is used. Choose explicitly with g2p="misaki" | "builtin"."""
from __future__ import annotations

import re
from typing import List, Optional

_WATERFALL = ["!.?…", ":;", ",—"]
_BUMPS = [")", "”"]


def available() -> bool:
    try:
        import misaki.en  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _tokens_to_ps(tokens) -> str:
    return "".join(t.phonemes + (" " if t.whitespace else "") for t in tokens).strip()


def _waterfall_last(tokens, next_count: int, limit: int) -> int:
    for w in _WATERFALL:
        z = next((i for i, t in reversed(list(enumerate(tokens))) if t.phonemes in set(w)), None)
        if z is None:
            continue
        z += 1
        if z < len(tokens) and tokens[z].phonemes in _BUMPS:
            z += 1
        if next_count - len(_tokens_to_ps(tokens[:z])) <= limit:
            return z
    return len(tokens)


def en_chunks(tokens, limit: int = 510) -> List[str]:
    """kokoro.KPipeline.en_tokenize → the phoneme string of every piece (empty pieces dropped, long ones truncated)."""
    out, tks, pcount = [], [], 0
    for t in tokens:
        t.phonemes = "" if t.phonemes is None else t.phonemes.replace("ɾ", "T")
        next_ps = t.phonemes + (" " if t.whitespace else "")
        next_pcount = pcount + len(next_ps.rstrip())
        if next_pcount > limit:
            z = _waterfall_last(tks, next_pcount, limit)
            out.append(_tokens_to_ps(tks[:z]))
            tks = tks[z:]
            pcount = len(_tokens_to_ps(tks))
            if not tks:
                next_ps = next_ps.lstrip()
        tks.append(t)
        pcount += len(next_ps)
    if tks:
        out.append(_tokens_to_ps(tks))
    return [p[:limit] for p in out if p]


class MisakiFrontend:
    def __init__(self, british: bool = False):
        from misaki import en
        fallback = None
        try:
            from misaki import espeak
            fallback = espeak.EspeakFallback(british=british)
        except Exception:  # noqa: BLE001 — no espeak-ng: words outside the lexicons are skipped, as in KPipeline
            fallback = None
        self.g2p = en.G2P(trf=False, british=british, fallback=fallback, unk="")

    def chunks(self, text: str) -> List[str]:
        """One text (any number of lines) → phoneme pieces, as KPipeline(text) yields them."""
        out = []
        for line in re.split(r"\n+", text.strip()):
            if not line.strip():
                continue
            _, tokens = self.g2p(line)
            out.extend(en_chunks(tokens))
        return out


_cache = {}


def frontend(british: bool) -> MisakiFrontend:
    if british not in _cache:
        _cache[british] = MisakiFrontend(british)
    return _cache[british]


def is_british(voice_name: Optional[str], lang: Optional[str]) -> bool:
    if lang in ("b", "en-gb", "british"):
        return True
    if lang in ("a", "en-us", "american"):
        return False
    return bool(voice_name) and voice_name.strip().lower().startswith("b")
