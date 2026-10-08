#!/usr/bin/env python3
"""Полный прогон: конвертация → голос → эталонные дампы → верификация → бенчмарк → сравнение с внешним движком.

    python tools/run_full_check.py --model-dir ./Qwen3-TTS-12Hz-1.7B-Base --ref-audio ref.wav \
        --ref-text "Текст референса" --lang russian --dtypes bf16,q8 --devices cpu,vulkan --threads 8 \
        --external "qwentts.cpp=./qwentts-cli --model m.gguf --text-file {text_file} --out {out}" \
        --out work/

Шаги можно пропускать/ограничивать: --reference torch|numpy|none, --skip-bench, --voice готовый.tvoice.
Результат: таблицы в консоль и work/report.md + work/report.json.
Требуется torch + qwen_tts только для --reference torch и экспорта голоса.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "python"))

DEFAULT_PHRASES = [
    "Привет! Это проверка движка синтеза речи.",
    "В 2026 году склад обработал 1 250 000 заказов, на 14% больше, чем год назад.",
    "Короткая фраза.",
    "Hello, this is a mixed language test, и немного русского текста.",
    "Вопрос? Восклицание! Перечисление: раз, два, три… и пауза.",
]
QUANTS = ("q8", "q4", "q4mlp", "q4full")
TOL = {"f32": "2e-3", "bf16": "2e-2", "f16": "1e-2", "q8": "1e-1", "q4": "1e-1", "q4mlp": "1e-1", "q4full": "1e-1"}


def split_devices(spec: str):
    """'cpu,vulkan:1,strict,vulkan:0' → ['cpu', 'vulkan:1,strict', 'vulkan:0']; можно и через ';'."""
    if ";" in spec:
        return [s.strip() for s in spec.split(";") if s.strip()]
    out = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        if out and (tok == "strict" or tok.startswith("codec=")):
            out[-1] += "," + tok
        else:
            out.append(tok)
    return out


def run(cmd, check=True, **kw):
    print("$", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if check and r.returncode != 0:
        print(r.stdout + r.stderr)
        raise SystemExit(f"command failed ({r.returncode})")
    return r


def wav_seconds(path):
    from vox.audio import load_wav
    a, sr = load_wav(path)
    return len(a) / sr


_RU_UNITS = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять", "одиннадцать",
             "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
_RU_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
_RU_HUND = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]
_EN_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
             "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _ru_triple(n, fem=False):
    """0 < n < 1000 → слова; fem: «одна/две» (тысячи)."""
    out = []
    if n >= 100:
        out.append(_RU_HUND[n // 100]); n %= 100
    if n >= 20:
        out.append(_RU_TENS[n // 10]); n %= 10
    if n:
        out.append(("одна" if n == 1 else "две") if fem and n in (1, 2) else _RU_UNITS[n])
    return out


def _ru_plural(n, forms):
    n %= 100
    if 11 <= n <= 19:
        return forms[2]
    return forms[0] if n % 10 == 1 else forms[1] if n % 10 in (2, 3, 4) else forms[2]


def num_to_words(n: int, lang: str) -> str:
    """Количественное числительное (именительный падеж) — чтобы «5» у whisper совпало с «пять» в тексте."""
    if lang.startswith("ru") or lang == "russian" or lang == "auto":
        if n == 0:
            return "ноль"
        parts = []
        for div, forms, fem in ((10**9, ("миллиард", "миллиарда", "миллиардов"), False),
                                (10**6, ("миллион", "миллиона", "миллионов"), False),
                                (10**3, ("тысяча", "тысячи", "тысяч"), True)):
            q, n = divmod(n, div)
            if q:  # «тысяча», «миллион» без «одна/один» — как говорят и как пишет текст
                parts += ([] if q == 1 else _ru_triple(q, fem)) + [_ru_plural(q, forms)]
        if n:
            parts += _ru_triple(n)
        return " ".join(parts)
    def en(n):
        if n < 20:
            return [_EN_UNITS[n]]
        if n < 100:
            return [_EN_TENS[n // 10]] + ([_EN_UNITS[n % 10]] if n % 10 else [])
        if n < 1000:
            return [_EN_UNITS[n // 100], "hundred"] + (en(n % 100) if n % 100 else [])
        for div, name in ((10**9, "billion"), (10**6, "million"), (10**3, "thousand")):
            if n >= div:
                return en(n // div) + [name] + (en(n % div) if n % div else [])
        return []
    return " ".join(en(n))


def normalize_text(s: str, lang: str) -> list:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}\b)", "", s)  # «1 250 000» → «1250000»
    s = re.sub(r"\d+", lambda m: " " + num_to_words(int(m.group(0)), lang) + " " if len(m.group(0)) <= 12 else m.group(0), s)
    return re.sub(r"[^\w\s]", " ", s).split()


def _edit(r, h):
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(h)]


def wer(ref: str, hyp: str, lang: str = "auto") -> float:
    r, h = normalize_text(ref, lang), normalize_text(hyp, lang)
    return _edit(r, h) / max(1, len(r))


def cer(ref: str, hyp: str, lang: str = "auto") -> float:
    r, h = list(" ".join(normalize_text(ref, lang))), list(" ".join(normalize_text(hyp, lang)))
    return _edit(r, h) / max(1, len(r))


ASR_CMD = None  # --asr-cmd: shell-команда, печатающая расшифровку {wav} в stdout (например whisper.cpp)


def transcribe(path, lang):
    if ASR_CMD:
        r = subprocess.run(["sh", "-c", ASR_CMD.format(wav=shlex.quote(str(path)), lang=(lang or "ru")[:2])], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel("small", device="cpu", compute_type="int8")
        segs, _ = m.transcribe(str(path), language=(lang or "ru")[:2])
        return " ".join(s.text for s in segs)
    except Exception:
        pass
    try:
        import whisper
        return whisper.load_model("small").transcribe(str(path))["text"]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default="work")
    ap.add_argument("--vox", default=str(ROOT / "build" / "vox"))
    ap.add_argument("--voice", help="готовый .tvoice (иначе создаётся из --ref-audio/--ref-text)")
    ap.add_argument("--voice-tool", default="native", choices=["native", "torch"],
                    help="native: `vox voice create` (без torch); torch: tools/export_voice.py (qwen_tts)")
    ap.add_argument("--ref-audio")
    ap.add_argument("--ref-text")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--dtypes", default="bf16,q8")
    ap.add_argument("--devices", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--reference", default="torch", choices=["torch", "numpy", "none"])
    ap.add_argument("--phrases", help="файл с фразами по одной на строку")
    ap.add_argument("--max-frames", type=int, default=120, help="длина эталонных дампов в кадрах")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--resources", action="append", default=[],
                    help='метка=JSON ресурсов для замера через Python-библиотеку (vox.Engine(resources=…)), напр. '
                         '\'mem={"mode":"memory","ram_limit":"3GiB"}\' или \'hybrid={"gpu_layers":{"talker":14}}\'. Без параметра: '
                         'default и mode=memory (устройство берётся из --devices)')
    ap.add_argument("--skip-library", action="store_true", help="не мерить через Python-библиотеку")
    ap.add_argument("--lib-python", default="", help="интерпретатор с установленным пакетом vox для замера через библиотеку "
                    "(по умолчанию ищется среди текущего, python3 и /usr/bin/python3 тот, где импортируется vox с нативным модулем)")
    ap.add_argument("--bench-text", help="текст для бенчмарка (по умолчанию ~14 с речи)")
    ap.add_argument("--bench-runs", type=int, default=3)
    ap.add_argument("--context", default="", help="decoder_context_frames для бенчмарка: число или all")
    ap.add_argument("--external", action="append", default=[], help='метка=команда; плейсхолдеры {text_file} {out}')
    ap.add_argument("--wer", action="store_true", help="распознать аудио whisper'ом и посчитать WER/CER (нужен faster_whisper/whisper "
                    "или --asr-cmd)")
    ap.add_argument("--asr-cmd", default="", help="своя ASR-команда: печатает текст {wav} в stdout, напр. "
                    "'whisper-cli -m ggml-small.bin -l {lang} -nt -np -f {wav}'")
    ap.add_argument("--external-time", action="append", default=[],
                    help="метка=regex: время синтеза из вывода внешнего движка (группа 1 — миллисекунды), напр. "
                         "'qwentts.cpp=\\[Perf\\] Total:\\s*([\\d.]+)\\s*ms'. Без него RTF внешнего движка считается по всему "
                         "процессу, включая загрузку модели")
    a = ap.parse_args()
    global ASR_CMD
    ASR_CMD = a.asr_cmd or None

    # все пути абсолютные: конвертер и внешние команды запускаются из других папок
    work = Path(a.out).resolve()
    a.model_dir = str(Path(a.model_dir).resolve())
    a.vox = str(Path(a.vox).resolve())
    for k in ("voice", "ref_audio", "phrases"):
        if getattr(a, k):
            setattr(a, k, str(Path(getattr(a, k)).resolve()))
    work.mkdir(parents=True, exist_ok=True)
    phrases = [l.strip() for l in open(a.phrases, encoding="utf-8") if l.strip()] if a.phrases else DEFAULT_PHRASES
    report = {"verify": [], "bench": [], "external": [], "library": [], "plans": []}
    py = sys.executable

    # 1. модели
    tmods = {}
    for dt in a.dtypes.split(","):
        out = work / f"model.{dt}.tmod"
        if not out.exists():
            cmd = [py, "-m", "vox.build", "--model", a.model_dir, "-o", out]
            cmd += ["--dtype", "f32" if dt == "f32" else "f16" if dt == "f16" else "bf16"] if dt not in QUANTS else ["--dtype", "bf16", "--quant", dt]
            run(cmd, cwd=ROOT / "python", env=dict(os.environ, PYTHONPATH=str(ROOT / "python")))
        tmods[dt] = out

    # 2. голос (нативно; при наличии torch — ещё и официальным кодом, с поэлементным сравнением)
    voice = Path(a.voice) if a.voice else work / "voice.tvoice"
    if not voice.exists():
        if not (a.ref_audio and a.ref_text):
            raise SystemExit("нужен --voice или пара --ref-audio и --ref-text")
        if a.voice_tool == "native":
            run([a.vox, "voice", "create", a.ref_audio, "--model", next(iter(tmods.values())), "--text", a.ref_text,
                 "--lang", a.lang, "--name", "check", "-o", voice])
        else:
            run([py, ROOT / "tools" / "export_voice.py", "--model", a.model_dir, "--ref", a.ref_audio, "--text", a.ref_text,
                 "--lang", a.lang, "--name", "check", "-o", voice])
    if a.reference == "torch" and a.ref_audio and a.ref_text and not a.voice:
        off = work / "voice.official.tvoice"
        if not off.exists():
            run([py, ROOT / "tools" / "export_voice.py", "--model", a.model_dir, "--ref", a.ref_audio, "--text", a.ref_text,
                 "--lang", a.lang, "--name", "check", "-o", off], check=False)
        if off.exists():
            import numpy as np
            from vox.formats import VoiceProfile
            va, vb = VoiceProfile.load(off), VoiceProfile.load(voice)
            same = va.ref_codes is not None and vb.ref_codes is not None and va.ref_codes.shape == vb.ref_codes.shape
            report["voice"] = {
                "speaker_emb_rel_err": float(np.abs(va.speaker_emb - vb.speaker_emb).max() / max(1e-9, np.abs(va.speaker_emb).max())),
                "speaker_emb_cos": float(va.speaker_emb @ vb.speaker_emb / (np.linalg.norm(va.speaker_emb) * np.linalg.norm(vb.speaker_emb))),
                "ref_codes_identical": float((va.ref_codes == vb.ref_codes).mean()) if same else 0.0,
                "ref_codes_shape": [list(va.ref_codes.shape) if va.ref_codes is not None else None,
                                    list(vb.ref_codes.shape) if vb.ref_codes is not None else None]}

    # 3. эталонные дампы
    dumps = []
    if a.reference != "none":
        refdir = work / f"ref_{a.reference}"
        refdir.mkdir(exist_ok=True)
        if a.reference == "torch":
            pf = work / "phrases.txt"
            pf.write_text("\n".join(phrases), encoding="utf-8")
            if not list(refdir.glob("case_*.safetensors")):
                run([py, ROOT / "tools" / "dump_reference.py", "--model", a.model_dir, "--voice", voice, "--lang", a.lang,
                     "--phrases", pf, "--seeds", "1", "--max-frames", str(a.max_frames), "--out", refdir])
            dumps = sorted(refdir.glob("case_*.safetensors"))
        else:
            for i, ph in enumerate(phrases):
                d = refdir / f"case_{i:02d}.safetensors"
                if not d.exists():
                    run([py, ROOT / "tools" / "ref_numpy.py", "--model", a.model_dir, "--voice", voice, "--text", ph,
                         "--lang", a.lang, "--max-frames", str(min(a.max_frames, 40)), "--out", d])
                dumps.append(d)

    # 4. верификация
    devices = split_devices(a.devices)
    for dt, tm in tmods.items():
        for dev in devices:
            for d in dumps:
                cmd = [a.vox, "verify", "--model", tm, "--dump", d, "--voice", voice, "--device", dev, "--tol", TOL.get(dt, "2e-2")]
                if dt in QUANTS:
                    cmd += ["--quant", "1"]  # распределения (KL, top-5) вместо argmax; свободный полёт для q8/q4 не сравнивается
                    # штатный допуск (KL ≤ 0.02, top-5 ≥ 97%) один для всех квантованных форматов; ослабленный (0.15 / 85%)
                    # применяется ниже только как вторая, отдельно помеченная проверка для q4
                elif dt != "f32":
                    cmd += ["--no-argmax", "--free", "0"]  # квантование законно сдвигает argmax у близких логитов
                if a.threads:
                    cmd += []
                r = run(cmd, check=False)
                strict_ok = r.returncode == 0
                relaxed_ok = strict_ok
                if not strict_ok and dt.startswith("q4"):
                    r2 = run(cmd + ["--kl-tol", "0.15", "--top5-min", "0.85"], check=False)
                    relaxed_ok = r2.returncode == 0
                txt = r.stdout + r.stderr
                g = lambda name: (re.search(name + r"\s+max\|Δ\|=\S+\s+max rel=(\S+)", txt) or re.search(name + r"\s+max rel=(\S+)", txt) or [None, None])[1]
                am = re.search(r"talker logits.*argmax (\d+)/(\d+)", txt)
                kl = re.search(r"talker logits\s+mean KL=(\S+)", txt)
                pkl = re.search(r"predictor logits\s+mean KL=(\S+)", txt)
                report["verify"].append({"dtype": dt, "device": dev, "case": d.name, "pass": strict_ok, "pass_relaxed": relaxed_ok,
                                         "talker_rel": g("talker logits"), "pred_rel": g("predictor logits"),
                                         "wav_rel": g("codec decoder wav"), "argmax": f"{am.group(1)}/{am.group(2)}" if am else "",
                                         "talker_kl": kl.group(1) if kl else "", "pred_kl": pkl.group(1) if pkl else ""})

    # 4b. устройства и план размещения для каждой пары формат × устройство (без загрузки весов)
    r = run([a.vox, "devices", "--json"], check=False)
    try:
        report["devices"] = json.loads(r.stdout)
    except Exception:
        report["devices"] = None
    for dt, tm in tmods.items():
        for dev in devices:
            r = run([a.vox, "plan", "--model", tm, "--device", dev, "--json"], check=False)
            try:
                report["plans"].append({"dtype": dt, "device": dev, "plan": json.loads(r.stdout)})
            except Exception:
                report["plans"].append({"dtype": dt, "device": dev, "plan": None, "error": (r.stdout + r.stderr)[-500:]})

    # 5. бенчмарк
    bench_text = a.bench_text or " ".join(phrases) + " " + " ".join(phrases[:2])
    (work / "bench_text.txt").write_text(bench_text, encoding="utf-8")
    if not a.skip_bench:
        for dt, tm in tmods.items():
            for dev in devices:
                cmd = [a.vox, "bench", "--model", tm, "--voice", voice, "--device", dev, "--text-file", work / "bench_text.txt",
                       "--runs", str(a.bench_runs), "--seed", "42"]
                if a.lang != "auto":
                    cmd += ["--lang", a.lang]
                if a.threads:
                    cmd += ["--threads", str(a.threads)]
                if a.context:
                    cmd += ["--context", a.context]
                env = dict(os.environ, VOX_PROFILE="1,gpu" if dev.startswith("vulkan") else "1", VOX_VERBOSE="1")
                r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env)
                try:
                    j = json.loads(r.stdout.strip().splitlines()[-1])
                except Exception:
                    print(r.stdout + r.stderr)
                    continue
                prof = [l for l in r.stderr.splitlines() if "[vox profile]" in l]
                el = r.stderr.splitlines()
                gi = [i for i, l in enumerate(el) if "[vox gpu profile]" in l]
                gpu_prof = el[gi[-1]:gi[-1] + 26] if gi else []
                vram = [l for l in el if "matrices" in l and "VRAM" in l]
                j.update({"dtype": dt, "device": dev, "profile": prof[-1] if prof else "", "gpu_profile": gpu_prof,
                          "vram": vram[-1] if vram else ""})
                report["bench"].append(j)

    # 5b. то же через Python-библиотеку (отдельным процессом, без CLI) с разными ограничениями ресурсов
    lib_py = None
    if not a.skip_bench and not a.skip_library:
        for cand in [a.lib_python, sys.executable, "python3", "/usr/bin/python3"]:
            if cand and subprocess.run([cand, "-I", "-c", "import sys, vox; sys.exit(0 if getattr(vox, '_vox', None) else 1)"],
                                       capture_output=True, cwd=work).returncode == 0:
                lib_py = cand
                break
        if lib_py is None:
            print("предупреждение: нет интерпретатора с установленным vox (pip install .): секция «Через Python-библиотеку» пропущена; "
                  "укажите --lib-python /путь/к/python", file=sys.stderr)
    if lib_py and not a.skip_bench and not a.skip_library:
        specs = [("default", {}), ("mode=memory", {"mode": "memory"})]
        if a.resources:
            specs = []
            for sp in a.resources:
                label, _, js = sp.partition("=")
                specs.append((label, json.loads(js) if js else {}))
        for dt, tm in tmods.items():
            for dev in devices:
                for label, res in specs:
                    res = dict(res, device=res.get("device", dev))
                    if a.threads and "threads" not in res:
                        res["threads"] = a.threads
                    r = run([lib_py, "-I", HERE / "library_bench.py", "--model", tm, "--voice", voice, "--resources", json.dumps(res),
                             "--text-file", work / "bench_text.txt", "--runs", str(a.bench_runs), "--lang", a.lang],
                            check=False, cwd=work, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
                    try:
                        item = json.loads(r.stdout.strip().splitlines()[-1])
                    except Exception:
                        item = {"ok": False, "error": (r.stdout + r.stderr)[-400:]}
                    item.update({"dtype": dt, "device": dev, "resources": label, "config": res})
                    report["library"].append(item)

    # 6. внешние движки: команда через sh -c (можно < {text_file}, пайпы); --bench-runs запусков, берётся лучший;
    #    RTF двух видов — по всему процессу (с загрузкой модели) и, если задан --external-time, по времени синтеза самого движка
    ext_time = dict(e.partition("=")[::2] for e in a.external_time)
    for ext in a.external:
        label, _, tpl = ext.partition("=")
        out = work / f"ext_{label}.wav"
        cmd = tpl.format(text_file=shlex.quote(str(work / "bench_text.txt")), out=shlex.quote(str(out)))
        best = None
        for _ in range(max(1, a.bench_runs)):
            if out.exists():
                out.unlink()
            t0 = time.time()
            logf = work / f"ext_{label}.log"
            with open(logf, "w") as lf:
                p = subprocess.Popen(["sh", "-c", cmd], stdout=lf, stderr=subprocess.STDOUT)
                _, status, ru = os.wait4(p.pid, 0)  # rusage именно этого процесса (пик памяти)
            dt_s = time.time() - t0
            log = logf.read_text(errors="replace")
            if status != 0 or not out.exists():
                best = best or {"label": label, "ok": False, "log": log[-2000:]}
                continue
            secs = wav_seconds(out)
            item = {"label": label, "ok": True, "audio_s": secs, "wall_s": dt_s, "rtf_process": dt_s / secs,
                    "peak_rss_mb": ru.ru_maxrss / 1024.0}
            if label in ext_time:
                mt = re.findall(ext_time[label], log)
                if mt:
                    item["engine_s"] = float(mt[-1]) / 1000.0
                    item["rtf_engine"] = item["engine_s"] / secs
            key = item.get("rtf_engine", item["rtf_process"])
            if not best or not best.get("ok") or key < best.get("rtf_engine", best.get("rtf_process", 1e9)):
                best = item
        report["external"].append(best)

    # 7. WER/CER (по желанию) для КАЖДОЙ пары формат×устройство и внешних движков; числа нормализуются в слова
    if a.wer:
        outs = {}
        for dt, tm in tmods.items():
            for dev in devices:
                o = work / f"vox_{dt}_{re.sub(r'[^A-Za-z0-9]+', '_', dev)}.wav"
                run([a.vox, "say", "--stdin", "--model", tm, "--voice", voice, "--device", dev, "--seed", "42", "-o", o] +
                    (["--lang", a.lang] if a.lang != "auto" else []), check=False, input=bench_text)
                outs[f"vox {dt} {dev}"] = o
        for e in report["external"]:
            if e and e.get("ok"):
                outs[e["label"]] = work / f"ext_{e['label']}.wav"
        report["wer"] = {}
        for k, pth in outs.items():
            hyp = transcribe(pth, a.lang) if pth.exists() else None
            report["wer"][k] = None if hyp is None else {"wer": round(wer(bench_text, hyp, a.lang), 3),
                                                          "cer": round(cer(bench_text, hyp, a.lang), 3), "text": hyp}

    # ---- отчёт ----
    lines = ["# vox: полный прогон", "", "## Верификация против эталона", "",
             "| формат | устройство | кейс | результат | talker (отн. ошибка или KL для q8) | predictor | аудио | argmax |", "|---|---|---|---|---|---|---|---|"]
    for v in report["verify"]:
        tl = v["talker_rel"] if v["talker_rel"] else f"KL {v.get('talker_kl', '')}"
        pl = v["pred_rel"] if v["pred_rel"] else f"KL {v.get('pred_kl', '')}"
        lines.append(f"| {v['dtype']} | {v['device']} | {v['case']} | {'PASS' if v['pass'] else ('PASS только по ослабленному допуску (KL ≤ 0.15, top-5 ≥ 85%)' if v.get('pass_relaxed') else '**FAIL**')} | {tl} | {pl} | {v['wav_rel']} | {v['argmax']} |")
    if report["bench"]:
        lines += ["", "## Скорость", "", "| формат | устройство | бэкенд | RTF (холодный) | RTF (тёплый) | до 1-го чанка, мс (холодный / тёплый) | пик памяти, МБ |", "|---|---|---|---|---|---|---|"]
        for b in report["bench"]:
            w = b.get("warm", b["cold"])
            lines.append(f"| {b['dtype']} | {b['device']} | {b['backend']} | {b['cold']['rtf']:.2f} | {w['rtf']:.2f} | {b['cold']['first_chunk_ms']:.0f} / {w['first_chunk_ms']:.0f} | {b['peak_rss_mb']:.0f} |")
        lines += ["", "Профиль по стадиям:", ""] + [f"- {b['dtype']}/{b['device']}: `{b['profile']}`" for b in report["bench"] if b.get("profile")]
        for b in report["bench"]:
            if b.get("vram"):
                lines += [f"- {b['dtype']}/{b['device']} размещение: `{b['vram'].strip()}`"]
            if b.get("gpu_profile"):
                lines += ["", f"GPU-профиль {b['dtype']}/{b['device']} (время по меткам GPU, VOX_PROFILE=gpu):", "", "```"] + b["gpu_profile"] + ["```"]
    if report.get("devices"):
        d = report["devices"]
        c = d["cpu"]
        lines += ["", "## Железо (vox devices)", "",
                  f"- CPU: {c['cpu_logical']} логических ядер ({c['cpu_allowed']} доступно процессу), RAM {c['ram_total'] / 2**30:.1f} ГиБ, "
                  f"доступно {c['ram_available'] / 2**30:.1f} ГиБ" + (f", лимит контейнера {c['cgroup_limit'] / 2**30:.1f} ГиБ" if c.get("cgroup_limit") else "")]
        for g in d.get("gpus", []):
            lines.append(f"- vulkan:{g['index']} {g['name']} [{'software' if g.get('software') else g['type']}], VRAM {g['vram_total'] / 2**30:.1f} ГиБ, "
                         f"свободно {g['vram_free'] / 2**30:.1f} ГиБ")
    if report["plans"]:
        lines += ["", "## План размещения (vox plan)", "", "| формат | устройство | допустим | размещение | RAM, оценка | VRAM, оценка | предупреждения |", "|---|---|---|---|---|---|---|"]
        for p in report["plans"]:
            pl = p.get("plan")
            if not pl:
                lines.append(f"| {p['dtype']} | {p['device']} | ошибка | {p.get('error', '')[:80]} | | | |")
                continue
            place = ", ".join(f"{s['stage']}={s['place']}" + (f"({s['gpu_layers']} сл.)" if s.get("gpu_layers") is not None else "") for s in pl["stages"])
            lines.append(f"| {p['dtype']} | {p['device']} | {'да' if pl['feasible'] else '**нет**'} | {place} | {pl['ram_estimate'] / 2**20:.0f} МиБ | "
                         f"{pl['vram_estimate'] / 2**20:.0f} МиБ | {'; '.join(pl.get('warnings', []))[:100]} |")
    if report["library"]:
        lines += ["", "## Через Python-библиотеку (vox.Engine(resources=…).benchmark)", "",
                  "| формат | устройство | ресурсы | потоки | размещение | RTF (тёплый) | до 1-го чанка, с | RSS пик, МиБ |", "|---|---|---|---|---|---|---|---|"]
        for it in report["library"]:
            if not it.get("ok"):
                lines.append(f"| {it['dtype']} | {it['device']} | {it['resources']} | | | ошибка: {(it.get('error', '').strip().splitlines() or [''])[-1][:140].replace('|', '/')} | | |")
                continue
            m = it.get("memory", {})
            peak = m.get("peak_rss") or m.get("rss_peak") or 0
            place = ", ".join(f"{s}={p}" + (f"({k})" if k is not None else "") for s, p, k in it["plan"])
            lines.append(f"| {it['dtype']} | {it['device']} | {it['resources']} | {it['plan_threads']} | {place} | {it['warm']['rtf']:.2f} | "
                         f"{it['warm']['first_chunk_s']:.2f} | {peak / 2**20 if peak else 0:.0f} |")
    if report.get("voice"):
        v = report["voice"]
        lines += ["", "## Голос: нативное создание против qwen_tts", "",
                  f"- эмбеддинг спикера: отн. ошибка {v['speaker_emb_rel_err']:.2e}, косинус {v['speaker_emb_cos']:.6f}",
                  f"- коды референса: совпадают {v['ref_codes_identical'] * 100:.2f}% (формы {v['ref_codes_shape']})",
                  "- точное совпадение ожидается для WAV 24 кГц; иначе ресемплеры (vox: sinc/Kaiser, qwen_tts: soxr) дают малое расхождение"]
    if report["external"]:
        lines += ["", "## Внешние движки (лучший из запусков)", "",
                  "| движок | RTF синтеза (по выводу движка) | RTF процесса (с загрузкой модели) | аудио, с | пик памяти, МБ |",
                  "|---|---|---|---|---|"]
        for e in report["external"]:
            if e and e.get("ok"):
                re_ = f"{e['rtf_engine']:.3f}" if "rtf_engine" in e else "— (задайте --external-time)"
                lines.append(f"| {e['label']} | {re_} | {e['rtf_process']:.2f} | {e['audio_s']:.1f} | {e['peak_rss_mb']:.0f} |")
            elif e:
                lines.append(f"| {e['label']} | не запустился | | | |")
        lines += ["", "У vox RTF в таблице «Скорость» считается внутри процесса (без загрузки модели) — сравнивайте с первой колонкой."]
    if report.get("wer"):
        lines += ["", "## Разборчивость (ASR; числа приведены к словам, ё→е; меньше — лучше)", "", "| вывод | WER | CER |", "|---|---|---|"]
        for k, v in report["wer"].items():
            lines.append(f"| {k} | {v['wer'] if v else '—'} | {v['cer'] if v else '—'} |")
    (work / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (work / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("\n" + "\n".join(lines))
    print(f"\nотчёт: {work / 'report.md'}")
    sys.exit(0 if all(v.get("pass_relaxed", v["pass"]) for v in report["verify"]) else 1)


if __name__ == "__main__":
    main()
