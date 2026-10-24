#!/usr/bin/env python3
"""GLSL → SPIR-V (glslangValidator) → src/vk_shaders.inc. Результат лежит в репозитории, поэтому
для сборки vox glslang не нужен. Запуск: python tools/gen_shaders.py"""
import os, subprocess, sys, tempfile
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
out = ["// Сгенерировано tools/gen_shaders.py — не править вручную"]
VARIANTS = {"matmul_nosg": ("matmul", ["-DNO_SUBGROUP"])}
for name in ("matmul", "rmsnorm", "qkrope", "attention", "silumul", "convgemm", "snake", "dwconv", "layernorm", "ola", "hist",
             "attnw", "ringput", "qkvsplit", "swiglu", "gather", "sample", "matmul_nosg",
             "f5attn", "f5conv", "f5ln", "f5ew", "f5rope"):
    spv = os.path.join(tempfile.mkdtemp(), name + ".spv")
    src, defs = VARIANTS.get(name, (name, []))
    subprocess.run(["glslangValidator", "-V", "--target-env", "vulkan1.1"] + defs + ["-o", spv, os.path.join(root, "src", "shaders", src + ".comp")], check=True)
    data = open(spv, "rb").read()
    words = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]
    out.append(f"static const uint32_t k_{name}_spv[] = {{")
    for i in range(0, len(words), 8):
        out.append("    " + ", ".join(f"0x{w:08x}" for w in words[i:i + 8]) + ",")
    out.append("};")
open(os.path.join(root, "src", "vk_shaders.inc"), "w").write("\n".join(out) + "\n")
print("vk_shaders.inc written", file=sys.stderr)
