"""Builds the README / site infographics (docs/img/*.svg, a light and a dark variant of each) from measured numbers.

Every number here comes from a run that is cited next to it; nothing is estimated. Re-run after a new measurement:
    python tools/make_charts.py
"""
from __future__ import annotations

import os
from html import escape

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "img")

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", grid="#e4e3df", s1="#2a78d6", s2="#eb6834",
                  s3="#1baf7a", s4="#eda100"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", grid="#3a3a37", s1="#3987e5", s2="#d95926",
                 s3="#199e70", s4="#c98500"),
    # the project page (site/style.css: --panel, --text, --muted, --line); the dark slots pass the validator on #211641 too
    "site": dict(surface="#211641", text="#efe9fa", text2="#b3a7d3", grid="#3d2d6b", s1="#3987e5", s2="#d95926",
                 s3="#199e70", s4="#c98500"),
}
FONT = "font-family=\"-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif\""


def bar_path(x0, y, w, h, r=4):
    """A bar anchored at x0 with a rounded data end (right side) only."""
    r = min(r, w / 2, h / 2)
    if w <= 0:
        return ""
    return (f"M{x0:.1f},{y:.1f} H{x0 + w - r:.1f} Q{x0 + w:.1f},{y:.1f} {x0 + w:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x0 + w:.1f},{y + h:.1f} {x0 + w - r:.1f},{y + h:.1f} H{x0:.1f} Z")


def hbar_chart(title, subtitle, rows, series, xmax, ticks, fmt, unit, note, ref=None, width=760):
    """rows: [(group label, [(series index, value, bar label)])]; one bar per (group, series) pair."""
    def render(theme):
        t = THEMES[theme]
        colors = [t["s1"], t["s2"], t["s3"], t["s4"]]
        left, right, top = 210, 70, 92
        bar_h, gap_in, gap_grp = 18, 2, 14
        plot_w = width - left - right
        y = top
        body = []
        for label, bars in rows:
            gy = y
            for si, val, blabel in bars:
                w = plot_w * val / xmax
                body.append(f'<path d="{bar_path(left, y, w, bar_h)}" fill="{colors[si]}">'
                            f'<title>{escape(blabel or label)}: {fmt(val)}{unit}</title></path>')
                body.append(f'<text x="{left + w + 6:.1f}" y="{y + bar_h - 4:.1f}" fill="{t["text"]}" font-size="12" '
                            f'font-variant-numeric="tabular-nums">{fmt(val)}</text>')
                if blabel:
                    body.append(f'<text x="{left - 10}" y="{y + bar_h - 4:.1f}" fill="{t["text2"]}" font-size="11.5" '
                                f'text-anchor="end">{escape(blabel)}</text>')
                y += bar_h + gap_in
            body.insert(0, f'<text x="16" y="{gy + 13:.1f}" fill="{t["text"]}" font-size="13" font-weight="600">'
                           f'{escape(label)}</text>')
            y += gap_grp
        bottom = y + 8
        grid = []
        for v in ticks:
            x = left + plot_w * v / xmax
            grid.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{bottom}" stroke="{t["grid"]}" stroke-width="1"/>')
            grid.append(f'<text x="{x:.1f}" y="{bottom + 16}" fill="{t["text2"]}" font-size="11" text-anchor="middle">'
                        f'{v:g}</text>')
        if ref is not None:
            rv, rl = ref
            x = left + plot_w * rv / xmax
            grid.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 10}" y2="{bottom}" stroke="{t["text2"]}" '
                        f'stroke-width="1.5" stroke-dasharray="4 3"/>')
            grid.append(f'<text x="{x + 5:.1f}" y="{top - 12}" fill="{t["text2"]}" font-size="11">{escape(rl)}</text>')
        legend, lx = [], 16
        for i, name in enumerate(series):
            legend.append(f'<rect x="{lx}" y="56" width="12" height="12" rx="3" fill="{colors[i]}"/>'
                          f'<text x="{lx + 18}" y="66" fill="{t["text"]}" font-size="12.5">{escape(name)}</text>')
            lx += 30 + 7.2 * len(name)
        height = bottom + 52
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
                f'{FONT} role="img" aria-label="{escape(title)}">'
                f'<rect width="{width}" height="{height}" rx="10" fill="{t["surface"]}"/>'
                f'<text x="16" y="28" fill="{t["text"]}" font-size="16" font-weight="700">{escape(title)}</text>'
                f'<text x="16" y="46" fill="{t["text2"]}" font-size="12">{escape(subtitle)}</text>'
                + "".join(legend) + "".join(grid) + "".join(body) +
                f'<text x="16" y="{height - 14}" fill="{t["text2"]}" font-size="11">{escape(note)}</text></svg>')
    return render


def f2(v):
    return f"{v:.2f}"


def f1(v):
    return f"{v:.1f}"


CHARTS = {}

# Qwen3-TTS 1.7B-Base, ~13 s of speech, Ryzen 7 5700X + RTX 3050 6 GB — the user's 0.12.2 report (run_full_check.py).
for lang in ("en", "ru"):
    L = lang == "en"
    CHARTS[f"qwen-rtf.{lang}"] = hbar_chart(
        "Qwen3-TTS 1.7B: real-time factor" if L else "Qwen3-TTS 1.7B: коэффициент реального времени",
        ("RTF = synthesis time / audio length, lower is better · Ryzen 7 5700X + RTX 3050 6 GB, ~13 s of speech"
         if L else "RTF = время синтеза / длина аудио, меньше — лучше · Ryzen 7 5700X + RTX 3050 6 GB, ~13 с речи"),
        [("GPU" if L else "Видеокарта", [(0, 0.41, "vox q4 · Vulkan"), (0, 0.44, "vox q8 · Vulkan"),
                                          (0, 0.59, "vox bf16 · Vulkan"), (1, 0.429, "qwentts.cpp · CUDA")]),
         ("CPU" if L else "Процессор", [(0, 1.64, "vox q4"), (0, 1.96, "vox q8"), (0, 2.98, "vox bf16"),
                                        (1, 2.164, "qwentts.cpp")])],
        ["vox", "qwentts.cpp"], 3.2, [0, 0.5, 1, 1.5, 2, 2.5, 3], f2, "",
        ("vox 0.12.2, Vulkan with codec=cpu (best of the two modes); measured by the user with tools/run_full_check.py"
         if L else "vox 0.12.2, Vulkan с codec=cpu (лучший из двух режимов); замер пользователя, tools/run_full_check.py"),
        ref=(1.0, "real time" if L else "реальное время"))
    CHARTS[f"qwen-memory.{lang}"] = hbar_chart(
        "Qwen3-TTS 1.7B: peak process memory" if L else "Qwen3-TTS 1.7B: пиковая память процесса",
        "RSS in GB, lower is better · same run as the RTF chart" if L else "RSS в ГБ, меньше — лучше · тот же прогон",
        [("GPU" if L else "Видеокарта", [(0, 0.47, "vox q8 · Vulkan"), (1, 2.97, "qwentts.cpp · CUDA")]),
         ("CPU" if L else "Процессор", [(0, 2.4, "vox q8"), (1, 7.1, "qwentts.cpp")])],
        ["vox", "qwentts.cpp"], 7.6, [0, 2, 4, 6], f2, " GB",
        ("On the GPU vox returns the mmap pages of the weights to the system after upload (release_host_weights)"
         if L else "На GPU vox после загрузки отдаёт системе mmap-страницы весов (release_host_weights)"))
    # F5-TTS v1 Base layer sizes, CPU benchmark of this repository, 0.12 vs 0.13 (development box, 2 cores).
    CHARTS[f"f5-cpu.{lang}"] = hbar_chart(
        "F5-TTS on the CPU: 0.12 → 0.13" if L else "F5-TTS на процессоре: 0.12 → 0.13",
        ("Seconds, lower is better · DiT with real F5 layer widths (2 layers), 2 steps, 12.8 s of audio, 2 cores"
         if L else "Секунды, меньше — лучше · DiT с настоящей шириной слоёв F5 (2 слоя), 2 шага, 12,8 с звука, 2 ядра"),
        [("f32", [(1, 16.6, "0.12"), (0, 7.7, "0.13")]),
         ("bf16", [(1, 17.4, "0.12"), (0, 8.0, "0.13")]),
         ("q8", [(1, 17.1, "0.12"), (0, 7.2, "0.13")])],
        ["0.13", "0.12"], 19, [0, 5, 10, 15], f1, " s",
        ("Tiled attention per head and q8 with weights dequantized once; absolute numbers depend on the machine"
         if L else "Тайловое внимание по головам и q8 с однократной распаковкой весов; абсолютные числа зависят от машины"))


def banner(lang):
    import math
    L = lang == "en"
    tag = ("Native speech synthesis · C++17 · CPU, Vulkan, WebGPU" if L
           else "Нативный синтез речи · C++17 · CPU, Vulkan, WebGPU")
    sub = ("Qwen3-TTS · F5-TTS · Kokoro-82M — one runtime, one file format, streaming in and out" if L
           else "Qwen3-TTS · F5-TTS · Kokoro-82M — один рантайм, один формат, потоковый ввод и вывод")

    def render(theme):
        t = THEMES[theme]
        w, h = 900, 200
        bars = []
        for i in range(56):   # a deterministic "waveform": two beating sines under a speech-like envelope
            x = 520 + i * 6.5
            env = math.sin(math.pi * i / 55) ** 0.7
            a = env * (0.55 + 0.45 * math.sin(i * 0.9) * math.cos(i * 0.31))
            bh = max(4.0, abs(a) * 120)
            col = t["s1"] if i < 40 else t["s3"]
            bars.append(f'<rect x="{x:.1f}" y="{100 - bh / 2:.1f}" width="3.5" height="{bh:.1f}" rx="1.75" fill="{col}"/>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" {FONT} '
                f'role="img" aria-label="vox"><rect width="{w}" height="{h}" rx="14" fill="{t["surface"]}"/>'
                f'<text x="40" y="104" fill="{t["text"]}" font-size="76" font-weight="800" letter-spacing="-3">vox</text>'
                f'<circle cx="183" cy="97" r="7" fill="{t["s2"]}"/>'
                f'<text x="42" y="138" fill="{t["text"]}" font-size="15" font-weight="600">{escape(tag)}</text>'
                f'<text x="42" y="162" fill="{t["text2"]}" font-size="12.5">{escape(sub)}</text>'
                + "".join(bars) + "</svg>")
    return render


for _lang in ("en", "ru"):
    CHARTS[f"banner.{_lang}"] = banner(_lang)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, render in CHARTS.items():
        for theme in THEMES:
            stem, lang = name.split(".")
            if theme == "site" and stem == "banner":
                continue
            path = os.path.join(OUT, f"{stem}-{theme}.{lang}.svg")
            with open(path, "w", encoding="utf-8") as f:
                f.write(render(theme))
            print(path)


if __name__ == "__main__":
    main()
