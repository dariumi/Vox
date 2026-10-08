#!/usr/bin/env python3
"""Собирает шейдеры WGSL (webgpu/wgsl/*.wgsl) в webgpu/web/kernels.js (браузер).
Шаблоны подставляются при создании конвейера: {{WTYPE}} (формат весов gemm) и {{HD}} (размер головы внимания)."""
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob(os.path.join(ROOT, "webgpu", "wgsl", "*.wgsl")))
js = ["// Сгенерировано tools/gen_wgsl.py из webgpu/wgsl/*.wgsl — не править вручную.", "export const KERNELS = {"]
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    src = open(f, encoding="utf-8").read()
    js.append(f"  {name}: {src!r},".replace("\\n", "\\n") if False else "  %s: `%s`," % (name, src.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")))
js.append("};")
os.makedirs(os.path.join(ROOT, "webgpu", "web"), exist_ok=True)
open(os.path.join(ROOT, "webgpu", "web", "kernels.js"), "w", encoding="utf-8").write("\n".join(js) + "\n")
print("kernels:", [os.path.splitext(os.path.basename(f))[0] for f in files])
