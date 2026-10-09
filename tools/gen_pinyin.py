#!/usr/bin/env python3
"""Таблица иероглиф → слог (pypinyin, TONE3) для vox.js: webgpu/web/dist/pinyin.json. Нужна только китайскому тексту
в моделях F5-TTS с токенайзером pinyin; остальным моделям файл не нужен."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
from vox.build_f5 import pinyin_table  # noqa: E402

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webgpu", "web", "dist", "pinyin.json")
t = pinyin_table()
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(t, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
print(out, len(t), "символов,", os.path.getsize(out) // 1024, "КиБ")
