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


def fg(v):
    return f"{v:.3g}"


def f1(v):
    return f"{v:.1f}"


CHARTS = {}

# All numbers below: the user's report on vox 0.14.0 (tools/run_full_check.py and check_f5_real.py), 16-thread CPU,
# RTX 3050 6 GB (vulkan:1) and RTX 5050 8 GB (vulkan:0); qwentts.cpp (Q8_0) on the same machine and text.
for lang in ("en", "ru"):
    L = lang == "en"
    CHARTS[f"qwen-rtf.{lang}"] = hbar_chart(
        "Qwen3-TTS 1.7B: real-time factor" if L else "Qwen3-TTS 1.7B: коэффициент реального времени",
        ("RTF = synthesis time / audio length, lower is better · 16-thread CPU, RTX 3050 6 GB, ~13 s of speech"
         if L else "RTF = время синтеза / длина аудио, меньше — лучше · 16 потоков CPU, RTX 3050 6 ГБ, ~13 с речи"),
        [("GPU" if L else "Видеокарта", [(0, 0.42, "vox q4 · Vulkan"), (0, 0.44, "vox q8 · Vulkan"),
                                          (0, 0.59, "vox bf16 · Vulkan"), (1, 0.426, "qwentts.cpp · CUDA")]),
         ("CPU" if L else "Процессор", [(0, 1.35, "vox q4"), (0, 1.65, "vox q8"), (0, 2.61, "vox bf16"),
                                        (1, 1.839, "qwentts.cpp")])],
        ["vox", "qwentts.cpp"], 2.9, [0, 0.5, 1, 1.5, 2, 2.5], f2, "",
        ("vox 0.14.0 (warm), Vulkan with codec=cpu; qwentts.cpp Q8_0, synthesis time from its own log; WER 0 everywhere"
         if L else "vox 0.14.0 (тёплый старт), Vulkan с codec=cpu; qwentts.cpp Q8_0, время синтеза по его логу; WER 0 у всех"),
        ref=(1.0, "real time" if L else "реальное время"))
    CHARTS[f"qwen-memory.{lang}"] = hbar_chart(
        "Qwen3-TTS 1.7B: peak process memory" if L else "Qwen3-TTS 1.7B: пиковая память процесса",
        "RSS in GB, lower is better · q8, same run as the RTF chart" if L else "RSS в ГБ, меньше — лучше · q8, тот же прогон",
        [("GPU" if L else "Видеокарта", [(0, 0.476, "vox · Vulkan"), (1, 2.97, "qwentts.cpp · CUDA")]),
         ("CPU" if L else "Процессор", [(0, 2.444, "vox"), (1, 7.137, "qwentts.cpp")])],
        ["vox", "qwentts.cpp"], 7.8, [0, 2, 4, 6], f2, " GB",
        ("On the GPU vox returns the mmap pages of the weights to the system after upload (release_host_weights)"
         if L else "На GPU vox после загрузки отдаёт системе mmap-страницы весов (release_host_weights)"))
    CHARTS[f"f5-speed.{lang}"] = hbar_chart(
        "F5-TTS: 0.12.2 → 0.14.0" if L else "F5-TTS: 0.12.2 → 0.14.0",
        ("RTF, lower is better · hotstone228/F5-TTS-Russian, bf16, 4 steps, a 2.3 s phrase"
         if L else "RTF, меньше — лучше · hotstone228/F5-TTS-Russian, bf16, 4 шага, фраза 2,3 с"),
        [("CPU" if L else "Процессор", [(1, 12.5, "0.12.2"), (0, 6.71, "0.14.0")]),
         ("RTX 3050", [(1, 35.1, "0.12.2"), (0, 1.39, "0.14.0")]),
         ("RTX 5050", [(0, 0.76, "0.14.0")])],
        ["0.14.0", "0.12.2"], 37, [0, 10, 20, 30], fg, "",
        ("At 32 steps: CPU 96.6 → 53.2, RTX 3050 9.3, RTX 5050 5.0. 0.13 put the whole DiT step and Vocos on the GPU"
         if L else "При 32 шагах: CPU 96,6 → 53,2, RTX 3050 9,3, RTX 5050 5,0. В 0.13 шаг DiT и Vocos целиком ушли на GPU"),
        ref=(1.0, "real time" if L else "реальное время"))
    CHARTS[f"kokoro-cpu.{lang}"] = hbar_chart(
        "Kokoro-82M on the CPU" if L else "Kokoro-82M на процессоре",
        ("RTF, lower is better · real weights, af_heart, 8.1 s of speech, 16 threads"
         if L else "RTF, меньше — лучше · настоящие веса, af_heart, 8,1 с речи, 16 потоков"),
        [("vox 0.14.0", [(0, 0.39, "f32"), (0, 0.35, "bf16")]),
         ("PyTorch", [(1, 0.21, "kokoro")])],
        ["vox", "PyTorch (kokoro 0.7.16)"], 0.45, [0, 0.1, 0.2, 0.3, 0.4], f2, "",
        ("Measured before the 0.14.1 matmul changes (2D split, 4×3 tiles): on 2 cores those took Kokoro from RTF 0.89 to 0.56"
         if L else "Замер до изменений умножения в 0.14.1 (2D-деление, плитка 4×3): на 2 ядрах они дали Kokoro RTF 0,89 → 0,56"))


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
