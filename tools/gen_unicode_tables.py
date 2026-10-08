#!/usr/bin/env python3
"""Генерирует src/unicode_tables.inc: диапазоны \\p{L}, \\p{N} и White_Space для претокенизатора Qwen.
Запуск: python tools/gen_unicode_tables.py > src/unicode_tables.inc"""
import sys
import unicodedata

def ranges(pred):
    out, start = [], None
    for cp in range(0x110000):
        ok = pred(cp)
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1)); start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out

def emit(name, rs):
    print(f"static const uint32_t {name}[][2] = {{")
    line = "   "
    for lo, hi in rs:
        item = f" {{0x{lo:X},0x{hi:X}}},"
        if len(line) + len(item) > 110:
            print(line); line = "   "
        line += item
    print(line)
    print("};")
    print(f"static const size_t {name}_N = sizeof({name}) / sizeof({name}[0]);")

print(f"// Сгенерировано tools/gen_unicode_tables.py, Unicode {unicodedata.unidata_version}")
emit("kLetter", ranges(lambda c: unicodedata.category(chr(c)).startswith("L")))
emit("kNumber", ranges(lambda c: unicodedata.category(chr(c)).startswith("N")))
ws = [0x9,0xA,0xB,0xC,0xD,0x20,0x85,0xA0,0x1680,*range(0x2000,0x200B),0x2028,0x2029,0x202F,0x205F,0x3000]
emit("kSpace", [(c, c) for c in ws])
