"""vox tune from Python: measure every candidate configuration of a model on this computer, keep the fastest.

    import vox
    best = vox.tune("cosyvoice2.tmod", "me.tvoice")            # or speaker="default" for a model with built-in voices
    vox.Engine("cosyvoice2.tmod")                              # opens with the measured configuration from now on

Each candidate runs in a child process (`python -m vox.tune --child …`): one Vulkan device per process, and a fresh process
measures what a fresh engine gets. The result goes to the cache of the C library (vox_tune_save), so the CLI, the server
and the Studio see it too."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Callable, Optional

TEXTS = {
    "en": "Hello! This is a short test of the speech engine. It shows how fast this computer can speak, one sentence after another.",
    "ru": "Привет! Это короткая проверка движка. Она показывает, как быстро этот компьютер говорит.",
    "zh": "今天天气很好，我们一起去公园散步吧。这是一个测试，看看这台电脑说话有多快。",
}


def default_text(caps: dict) -> str:
    langs = [l for l in caps.get("languages", []) if isinstance(l, str)]
    has = lambda p: any(l.startswith(p) for l in langs)
    if langs and not has("en") and not has("auto"):
        for k in ("zh", "ru"):
            if has(k):
                return TEXTS[k]
    return TEXTS["en"]


def _child(args: dict) -> dict:
    import vox
    eng = vox.Engine(args["model"], resources=args["device"] + ",strict")
    if args.get("voice"):
        v = eng.load_voice(args["voice"])
    else:
        v = eng.speaker_voice(args.get("speaker"), args.get("instruct"))
    if args.get("synth_deadline"):   # the synthesis alone passed twice the best one's: it cannot win
        import threading
        d = float(args["synth_deadline"])

        def stop():
            print(json.dumps({"error": f"slower than the best so far: the synthesis passed {d:.0f} s", "deadline": True}), flush=True)
            os._exit(3)
        t = threading.Timer(d, stop)
        t.daemon = True
        t.start()
    if os.environ.get("VOX_TUNE_TEST_SLOW") == args["device"]:   # tests of vox tune: this candidate pretends to be slow
        import time
        time.sleep(30)
    b = eng.benchmark(args["text"], v, runs=args["runs"])
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:   # Windows
        peak = None
    return {"warm": b["warm"], "runs": b["runs"], "stats": b["stats"], "backend": eng.info.get("backend", ""), "peak_rss_mb": peak}


def run(model, voice=None, *, speaker: Optional[str] = None, instruct: Optional[str] = None, text: Optional[str] = None,
        goal: str = "speed", runs: int = 2, candidates: Optional[list] = None, save: bool = True,
        progress: Optional[Callable[[dict], None]] = None, timeout: float = 1800) -> dict:
    """voice: a .tvoice path (a profile too: its default voice speaks); or speaker / instruct for models with built-in voices
    (by default the first built-in speaker, if the model has one). goal: "speed" (lowest RTF) or "latency" (earliest first
    sound). candidates: [{"label", "device"}] to measure instead of the library's list. progress(row) is called after each
    candidate. Returns the saved entry: {"device", "rtf", "first_chunk_ms", "label", "gemm"?, "candidates": [...]}."""
    import vox
    model = os.fspath(model)
    if goal not in ("speed", "latency"):
        raise ValueError("goal is 'speed' or 'latency'")
    info = vox.tune_info(model)
    caps = vox.inspect(model).get("capabilities", {})
    if voice is None and speaker is None and instruct is None:
        if caps.get("named_speakers") and caps.get("speakers"):
            speaker = caps.get("default_speaker") or caps["speakers"][0]
        elif caps.get("requires_instruct"):
            instruct = "A calm, clear adult voice."
        else:
            raise ValueError("this model speaks with a cloned voice: pass a .tvoice (vox.tune(model, 'me.tvoice'))")
    text = text or default_text(caps)
    cands = candidates if candidates is not None else info["candidates"]
    rows, best, best_wall, best_synth, slow_group = [], None, 0.0, 0.0, None
    for i, c in enumerate(cands):
        row = {"label": c.get("label", c["device"]), "device": c["device"], "index": i, "of": len(cands)}
        group = c.get("group")
        if group:
            row["group"] = group
        if group and group == slow_group:   # the other CPU thread counts differ by tens of percent, not by 2x
            row.update(error=f"not measured: another {group} configuration was already more than twice as slow as the best", skipped=True)
            rows.append(row)
            if progress:
                progress(dict(row))
            continue
        args = {"model": model, "device": c["device"], "voice": os.fspath(voice) if voice is not None else None, "speaker": speaker,
                "instruct": instruct, "text": text, "runs": max(1, int(runs)), "synth_deadline": 2.0 * best_synth + 3.0 if best_synth else None}
        env = dict(os.environ, VOX_TUNE="0")
        # a candidate running 3x as long as the best so far (load included, at least 90 s) cannot win: it is stopped
        limit = max(90.0, 3.0 * best_wall) if best_wall else timeout
        import time as _time
        t0 = _time.monotonic()
        try:
            try:
                r = subprocess.run([sys.executable, "-m", "vox.tune", "--child", json.dumps(args)], capture_output=True, text=True,
                                   timeout=limit, env=env)
            except subprocess.TimeoutExpired:
                row.update(error=f"slower than the best so far: stopped after {limit:.0f} s", skipped=True)
                slow_group = group or slow_group
                rows.append(row)
                if progress:
                    progress(dict(row))
                continue
            wall = _time.monotonic() - t0
            out = next((json.loads(l) for l in reversed(r.stdout.splitlines()) if l.startswith("{")), None)
            if out is not None and out.get("deadline"):
                row.update(error=out.get("error", "slower than the best so far"), skipped=True)
                slow_group = group or slow_group
                rows.append(row)
                if progress:
                    progress(dict(row))
                continue
            if r.returncode != 0 or out is None:
                raise RuntimeError((r.stderr or r.stdout)[-300:].strip() or f"exit code {r.returncode}")
            w = out["warm"]
            row.update(rtf=w.get("rtf"), first_chunk_ms=(w.get("first_chunk_s") or 0) * 1000.0, peak_rss_mb=out.get("peak_rss_mb"),
                       backend=out.get("backend", ""))
            gm = (out.get("stats", {}).get("backend", {}) or {}).get("gemm") or ""
            if gm:
                row["gemm"] = gm.split(" ")[0]
            score = row["rtf"] if goal == "speed" else row["first_chunk_ms"]
            if score and (best is None or score < best[0]):
                best = (score, row)
                best_wall = wall
                best_synth = sum(x.get("total_s") or 0.0 for x in out.get("runs", []))
        except Exception as e:   # noqa: BLE001 — a candidate that fails is reported, the others still run
            row["error"] = str(e)
        rows.append(row)
        if progress:
            progress(dict(row))
    if best is None:
        raise RuntimeError("no configuration worked: " + "; ".join(f"{r['label']}: {r.get('error')}" for r in rows))
    # within 3 % of the fastest is noise: the first such candidate wins (the list puts the plain configurations first)
    sc = lambda r: r.get("rtf") if goal == "speed" else r.get("first_chunk_ms")
    b = next((r for r in rows if sc(r) and sc(r) <= best[0] * 1.03), best[1])
    entry = {"device": b["device"], "goal": goal, "rtf": b["rtf"], "first_chunk_ms": b["first_chunk_ms"], "label": b["label"],
             "candidates": [{k: v for k, v in r.items() if k not in ("index", "of")} for r in rows], "text_chars": len(text)}
    if b.get("gemm"):
        entry["gemm"] = b["gemm"]
    if save:
        vox._native().tune_save(model, json.dumps(entry))
    return entry


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--child":
        print(json.dumps(_child(json.loads(argv[1]))), flush=True)
        return 0
    import argparse
    ap = argparse.ArgumentParser(prog="python -m vox.tune", description="measure the configurations of a model on this computer")
    ap.add_argument("model")
    ap.add_argument("--voice")
    ap.add_argument("--speaker")
    ap.add_argument("--goal", default="speed", choices=["speed", "latency"])
    ap.add_argument("--text")
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args(argv)
    show = lambda r: print(f"  [{r['index'] + 1}/{r['of']}] {r['label']:<44} " +
                           (f"RTF {r['rtf']:.3f}, first sound {r['first_chunk_ms']:.0f} ms" if "rtf" in r
                            else ("stopped: " if r.get("skipped") else "failed: ") + r.get("error", "")),
                           file=sys.stderr)
    e = run(a.model, a.voice, speaker=a.speaker, text=a.text, goal=a.goal, runs=a.runs, progress=show)
    print(json.dumps(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
