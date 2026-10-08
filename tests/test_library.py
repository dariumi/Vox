"""Python как библиотека: ни одного вызова CLI. Запуск: python3 tests/test_library.py"""
import os
import tempfile

import numpy as np

import vox


def ref_audio(sr=24000, secs=3):
    t = np.arange(sr * secs, dtype=np.float32)
    return (0.3 * np.sin(t * 0.05) * (t % 4000) / 4000).astype(np.float32), sr


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok:", msg)


def main():
    with tempfile.TemporaryDirectory() as d:
        model = vox.make_test_model(os.path.join(d, "t.tmod"))

        dev = vox.devices()
        check(dev["cpu"]["cpu_logical"] >= 1 and "vox-test" in dev["families"], "devices(): cpu + families")

        info = vox.inspect(model)
        check(info["family"] == "vox-test", "inspect(): family")

        p = vox.plan(model, vox.Resources(device="cpu", mode="memory", threads=1))
        check(p["feasible"] and p["backend"] == "cpu" and p["threads"] == 1 and p["mode"] == "memory", "plan(): cpu/memory/1 thread")
        p = vox.plan(model, "cpu")
        check(p["backend"] == "cpu" and "summary" in p, "plan(): device string")

        try:
            vox.plan(model, {"device": "cpu", "bogus": 1})
            check(False, "unknown resource key must fail")
        except TypeError:
            check(True, "Resources rejects unknown keys")
        try:
            vox.plan(model, vox.Resources(device="cpu", ram_limit="1KiB"))
            bad = False
        except vox.VoxError:
            bad = True
        # лимит меньше статической части: либо ошибка при плане, либо infeasible
        r = None if bad else vox.plan(model, vox.Resources(device="cpu", ram_limit="1KiB"))
        check(bad or (r is not None and not r["feasible"]), "plan(): tiny ram_limit is infeasible")

        # открытие с ресурсами
        with vox.Engine(model, resources=vox.Resources(device="cpu", threads=2, mode="balanced", max_sessions=2)) as eng:
            check(eng.plan["backend"] == "cpu" and eng.plan["threads"] == 2, "Engine.plan reflects resources")
            check(isinstance(eng.capabilities, dict), "Engine.capabilities")
            pcm, sr = ref_audio()
            v = eng.clone_voice((pcm, sr), ref_text="тест", name="t", seed=7)
            a1 = eng.synthesize("привет мир", v, seed=1)
            a2 = eng.synthesize("привет мир", v, seed=1)
            check(a1.size > 0 and np.array_equal(a1, a2), "deterministic by seed")
            outs = eng.synthesize_many(["привет мир", "привет мир", "другой текст"], v, seed=1)
            check(np.array_equal(outs[0], a1) and np.array_equal(outs[1], a1), "synthesize_many == sequential result")
            st = eng.stats()
            check("memory" in st and "plan" in st, "stats()")
            b = eng.benchmark("привет мир", v, runs=2, seed=1)
            check(b["warm"]["audio_s"] > 0, "benchmark() via library")

        # лимит одновременных сессий
        with vox.Engine(model, resources={"device": "cpu", "max_sessions": 1}) as eng:
            v = eng.clone_voice(ref_audio(), ref_text="тест")
            s1 = eng.session(v, seed=1)
            try:
                eng.session(v, seed=1)
                check(False, "second session must be refused")
            except vox.OutOfMemory:
                check(True, "max_sessions=1 -> OutOfMemory for 2nd session")
            s1.cancel()
            del s1
            s2 = eng.session(v, seed=1)
            check(s2 is not None, "slot is freed after the session is released")

        # два движка в одном процессе с разным числом потоков, работающие одновременно: результат не зависит от соседа
        import threading
        with vox.Engine(model, resources={"device": "cpu", "threads": 1}) as ea, vox.Engine(model, resources={"device": "cpu", "threads": 2}) as eb:
            va, vb = ea.clone_voice(ref_audio(), ref_text="тест"), eb.clone_voice(ref_audio(), ref_text="тест")
            out = {}

            def work(name, eng, vv):
                out[name] = [eng.synthesize("привет мир " * 3, vv, seed=5) for _ in range(3)]
            ts = [threading.Thread(target=work, args=("a", ea, va)), threading.Thread(target=work, args=("b", eb, vb))]
            [t.start() for t in ts]
            [t.join() for t in ts]
            base = ea.synthesize("привет мир " * 3, va, seed=5)
            check(all(np.array_equal(x, base) for x in out["a"] + out["b"]), "two engines in one process: results independent of threads and neighbour")
            check(ea.plan["threads"] == 1 and eb.plan["threads"] == 2, "each engine keeps its own thread pool size")

        # ошибки устройства
        try:
            vox.Engine(model, resources=vox.Resources(device="vulkan:99", strict=True))
            check(False, "missing device must fail")
        except vox.DeviceError:
            check(True, "missing GPU in strict mode -> DeviceError")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
