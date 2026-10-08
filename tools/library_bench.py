#!/usr/bin/env python3
"""Замер скорости через Python-библиотеку (без CLI): печатает одну строку JSON.
Запускается из run_full_check.py отдельным процессом, чтобы импортировался установленный пакет vox (pip install .),
а не папка python/ из исходников (в ней нет нативного модуля)."""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--resources", default="{}", help="JSON для vox.Resources")
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--lang", default="auto")
    a = ap.parse_args()
    import vox
    if getattr(vox, "_vox", None) is None:
        print(json.dumps({"ok": False, "error": "native module is not installed: run `pip install .` in the vox repository"}))
        return 1
    text = open(a.text_file, encoding="utf-8").read()
    try:
        with vox.Engine(a.model, resources=json.loads(a.resources)) as eng:
            v = eng.load_voice(a.voice)
            kw = {"seed": 42}
            if a.lang != "auto":
                kw["language"] = a.lang
            b = eng.benchmark(text, v, runs=max(2, a.runs), **kw)
            st = eng.stats()
            print(json.dumps({"ok": True, "version": vox.__version__, "warm": b["warm"], "first_run": b["runs"][0],
                              "plan_threads": eng.plan["threads"], "plan_mode": eng.plan["mode"],
                              "plan": [(s["stage"], s["place"], s.get("gpu_layers")) for s in eng.plan["stages"]],
                              "warnings": eng.plan.get("warnings", []), "memory": st.get("memory", {}),
                              "throughput": st.get("throughput", {})}))
    except vox.VoxError as e:
        print(json.dumps({"ok": False, "error": str(e)[:600], "status": e.status}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
