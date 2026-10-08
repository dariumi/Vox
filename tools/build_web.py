#!/usr/bin/env python3
"""Собирает браузерную библиотеку: webgpu/web/src/part*.js + шейдеры -> webgpu/web/dist/vox.js (обычный <script>, глобальный Vox,
CommonJS) и vox.mjs (ES-модуль). Сначала запускает tools/gen_wgsl.py. Без зависимостей."""
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
subprocess.check_call([sys.executable, os.path.join(ROOT, "tools", "gen_wgsl.py")])
src = os.path.join(ROOT, "webgpu", "web", "src")
kern = open(os.path.join(ROOT, "webgpu", "web", "kernels.js"), encoding="utf-8").read().replace("export const KERNELS", "const KERNELS")
ver = re.search(r"project\(vox VERSION ([0-9.]+)", open(os.path.join(ROOT, "CMakeLists.txt")).read())
body = kern + "\n" + "\n".join(open(p, encoding="utf-8").read() for p in sorted(glob.glob(os.path.join(src, "part*.js"))))
head = "/*! vox.js %s — универсальный TTS-движок vox в браузере (WebGPU). Лицензия — как у проекта: https://github.com/dariumi/Vox */\n"
os.makedirs(os.path.join(ROOT, "webgpu", "web", "dist"), exist_ok=True)
v = ver.group(1) if ver else "0"
js = head % v + '(function (global) {\n"use strict";\n' + body + '\nif (typeof module === "object" && module.exports) module.exports = Vox;\nglobal.Vox = Vox;\n})(typeof globalThis !== "undefined" ? globalThis : window);\n'
mjs = head % v + '"use strict";\n' + body + "\nexport default Vox;\nexport { Vox, VoxError, Voice };\n"
open(os.path.join(ROOT, "webgpu", "web", "dist", "vox.js"), "w", encoding="utf-8").write(js)
open(os.path.join(ROOT, "webgpu", "web", "dist", "vox.mjs"), "w", encoding="utf-8").write(mjs)
print("dist: vox.js %d KiB, vox.mjs %d KiB" % (len(js.encode()) // 1024, len(mjs.encode()) // 1024))
