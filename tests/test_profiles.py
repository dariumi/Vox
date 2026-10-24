"""Voice profiles (0.15): several voices and styles in one .tvoice, any model, markup in the text.
Run: python tests/test_profiles.py [build/vox]

Checked on the test family (vox-test) always, on tiny F5-TTS / Kokoro models when torch + f5_tts / kokoro are installed:
  1. compatibility: a 1.0 voice opens as a one-voice profile; a profile with a built default voice is written as format 1.1 whose
     "voice" metadata and top-level tensors are that voice (vox ≤ 0.14 and vox.js read it); a single-voice profile speaks exactly
     like the 1.0 voice it came from (bit for bit); a profile without any build is format 2.0;
  2. CLI: profile new / add / set / rm / patch / show / prepare;
  3. markup: voice switches ([voice:x], "x: …" lines), styles, [pause] (exact silence length), [speed] (models with a speed),
     unknown brackets kept as text in "auto", dropped in "strict", read as text in "off"; tags split across feed() calls;
     the same text and seed give the same audio;
  4. one profile on several models: a voice cloned on one model is rebuilt from its recording on another (F5); on Kokoro the
     voice's speaker mapping is used, and without one the error says how to add it; prepare() builds and save() keeps the caches;
  5. Python API: vox.Profile, add_recording, add (merge), set_voice / set_style / remove / update, prepare, voice_id / style /
     markup parameters, session notes."""
import json
import os
import subprocess
import sys
import tempfile

import _cleanup  # noqa: F401  (temporary folders go away on exit; VOX_KEEP_TMP=1 keeps them)

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, r.stdout + r.stderr


def main(cli):
    import vox
    from safetensors import safe_open
    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            fails += 1

    work = tempfile.mkdtemp(prefix="voxprof_")
    tm = os.path.join(work, "t.tmod")
    vox.make_test_model(tm)
    sr = 24000
    t = np.arange(int(sr * 2.0)) / sr
    ref1, ref2 = os.path.join(work, "a.wav"), os.path.join(work, "b.wav")
    vox.save_wav(ref1, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    vox.save_wav(ref2, (0.3 * np.sin(2 * np.pi * 330 * t[: int(sr * 1.5)])).astype(np.float32), sr)

    # ---- 1. compatibility ----
    v1 = os.path.join(work, "v1.tvoice")
    rc, out = run([cli, "voice", "create", ref1, "--model", tm, "--text", "hello there", "--name", "old", "-o", v1])
    assert rc == 0, out
    with safe_open(v1, "numpy") as f:
        meta = f.metadata()
    eng = vox.Engine(tm)
    v = eng.load_voice(v1)
    check(list(v.voices) == ["old"] and v.info["mode"] == "icl" and v.info["model"]["family"] == "vox-test",
          f"1.0 voice opens as a one-voice profile ({list(v.voices)}, format of the file {meta['format']})")
    p11 = os.path.join(work, "re.tvoice")
    v.save(p11)
    with safe_open(p11, "numpy") as f:
        m2, keys = f.metadata(), set(f.keys())
    legacy = json.loads(m2["voice"])
    check(m2["format"] == "1.1" and {"speaker_emb", "ref_codes", "ref_text_tokens", "ref_audio"} <= keys and legacy["mode"] == "icl"
          and "profile" in m2, f"saved profile is format 1.1 with the 1.0 layout at the top ({sorted(keys)})")
    a = eng.synthesize("Hello world. This is a test.", v1, seed=5)
    b = eng.synthesize("Hello world. This is a test.", p11, seed=5)
    check(np.array_equal(a, b) and len(a) > 0, f"one-voice profile speaks exactly like the 1.0 voice ({len(a)} samples)")
    old = os.environ.get("VOX_OLD_CLI")   # optional: an older vox binary must read the 1.1 file as a voice
    if old:
        rc, out = run([old, "say", "Hello.", "--model", tm, "--voice", p11, "-o", os.path.join(work, "old.wav")])
        check(rc == 0, "an older vox reads the 1.1 profile as its default voice")
    empty = vox.Profile("empty")
    empty.add_recording("x", ref1, "hello")
    e2 = os.path.join(work, "e2.tvoice")
    empty.save(e2)
    with safe_open(e2, "numpy") as f:
        check(f.metadata()["format"] == "2.0" and "voice" not in f.metadata(), "a profile without any built voice is format 2.0")

    # ---- 2. CLI ----
    P = os.path.join(work, "p.tvoice")
    steps = [["new", P, "--name", "Cast"],
             ["add", P, "--voice", "anna", ref1, "--text", "hello there"],
             ["add", P, "--voice", "anna", "--style", "happy", ref2, "--text", "so happy"],
             ["add", P, "--voice", "boris", "--from", v1],
             ["set", P, "--voice", "anna", "--style", "whisper", "--instruct", "Whisper softly", "--speed", "0.9"],
             ["set", P, "--voice", "boris", "--speaker", "kokoro=am_adam", "--description", "a low calm voice"],
             ["set", P, "--voice", "anna", "--style", "tmp"], ["rm", P, "--voice", "anna", "--style", "tmp"],
             ["patch", P, json.dumps({"voices": {"anna": {"options": {"temperature": 0.8}}}})]]
    ok = True
    for st in steps:
        rc, out = run([cli, "profile"] + st)
        ok = ok and rc == 0
        if rc:
            print(out)
    rc, out = run([cli, "profile", "show", P, "--json"])
    j = json.loads(out[out.index("{"):])
    an, bo = j["voices"].get("anna", {}), j["voices"].get("boris", {})
    check(ok and set(an.get("styles", {})) == {"default", "happy", "whisper"} and an["styles"]["whisper"]["options"].get("speed") == 0.9
          and bo.get("speakers") == {"kokoro": "am_adam"} and an["options"].get("temperature") == 0.8 and len(j["recordings"]) == 2
          and any(c["voice"] == "boris" for c in j["cache"]), "CLI new / add / add --from / set / rm / patch / show")
    rc, out = run([cli, "profile", "prepare", P, "--model", tm])
    o = run([cli, "profile", "show", P, "--json"])[1]
    j = json.loads(o[o.index("{"):])
    built = {(c["voice"], c["style"]) for c in j["cache"]}
    check(rc == 0 and {("anna", "default"), ("anna", "happy"), ("anna", "whisper"), ("boris", "default")} <= built,
          f"prepare builds every voice and style for the model and saves them ({sorted(built)})")

    # ---- 3. markup ----
    p = vox.Voice.open(P)
    base = eng.synthesize("Hello there. Good.", p, seed=3)
    with_pause = eng.synthesize("Hello there. [pause 1s] Good.", p, seed=3)
    # the pause splits the text into two segments; each is spoken separately, so compare with the two parts spoken alone
    s1 = eng.synthesize("Hello there.", p, seed=3)
    check(len(with_pause) >= len(s1) + sr and np.all(with_pause[len(s1):len(s1) + sr] == 0),
          f"[pause 1s] inserts exactly 1 s of silence between the segments ({len(base)} → {len(with_pause)} samples)")
    d1 = eng.synthesize("anna: Hi there.\nboris: Hello. [happy] Fine.", p, seed=3)
    d2 = eng.synthesize("anna: Hi there.\nboris: Hello. [happy] Fine.", p, seed=3)
    s = eng.session(p, seed=3)
    s.feed("anna: Hi there.\nboris: Hello. [happy] Fine. [whisper] ok [nonsense] [laugh]")
    out = list(s)
    info = s.info
    check(np.array_equal(d1, d2) and len(d1) > 0, "dialogue markup is deterministic")
    check(info["segment"] >= 2 and info["voice"] == "boris" and any("[happy]" in n for n in info["notes"]) and any("[laugh]" in n for n in info["notes"]),
          f"segments: voice lines and tags switch voice / style, notes say what the model could not do ({info['segment']} segments, {info['notes']})")
    # voice selection without markup
    an = eng.synthesize("Hello.", p, seed=3, voice_id="anna", style="happy")
    an2 = eng.synthesize("[happy] Hello.", p, seed=3, voice_id="anna")
    check(np.array_equal(an, an2), "style= and a leading [happy] tag give the same audio")
    bo = eng.synthesize("Hello.", p, seed=3, voice_id="boris")
    old1 = eng.synthesize("Hello.", v1, seed=3)
    check(np.array_equal(bo, old1), "a voice merged from another file speaks like that file")
    # markup modes and tags split across feeds
    t_auto = eng.synthesize("Read [this] aloud.", p, seed=3)
    t_off = eng.synthesize("Read [this] aloud.", p, seed=3, markup="off")
    t_strict = eng.synthesize("Read [this] aloud.", p, seed=3, markup="strict")
    t_plain = eng.synthesize("Read  aloud.", p, seed=3)
    check(np.array_equal(t_auto, t_off) and np.array_equal(t_strict, t_plain),
          "unknown brackets: text in auto (= off), dropped in strict")
    whole = eng.session(p, seed=3)
    whole.feed("Hello. [happy] Fine. [pause 300ms] End.")
    w = list(whole)
    s = eng.session(p, seed=3)
    parts = []
    for piece in ["Hello. [ha", "ppy] Fine", ". [pau", "se 300ms] End."]:
        s.feed(piece)
        parts += list(s.ready())
    s.end()
    parts += list(s)
    check(np.array_equal(np.concatenate(parts), np.concatenate(w)), "tags split across feed() calls work like the whole text")
    try:
        eng.synthesize("x", p, voice_id="nobody")
        check(False, "unknown voice_id is an error")
    except vox.VoxError as e:
        check("anna" in str(e) and "boris" in str(e), f"unknown voice_id: the error lists the voices ({e})")
    try:
        eng.synthesize("x", p, voice_id="anna", style="nope")
        check(False, "unknown style is an error")
    except vox.VoxError as e:
        check("happy" in str(e), f"unknown style: the error lists the styles ({e})")

    # ---- 5. Python API ----
    q = vox.Profile("Py")
    q.add_recording("anna", ref1, "hello there", language="english")
    q.add_recording("anna", ref2, "so happy", style="happy")
    q.set_style("anna", "whisper", instruct="Whisper softly", speed=0.9)
    q.set_voice("anna", description="a bright voice", speakers={"kokoro": "af_heart"}, temperature=0.7)
    q.add(vox.Voice.open(v1), as_voice="boris")
    q.add(vox.Voice.open(v1), as_voice="anna", style="calm")
    q.remove("anna", "whisper")
    r = q.prepare(eng)
    qp = os.path.join(work, "q.tvoice")
    q.save(qp)
    q2 = vox.Voice.open(qp)
    vs = q2.voices
    check(set(vs) == {"anna", "boris"} and set(vs["anna"]["styles"]) == {"default", "happy", "calm"} and vs["anna"]["speakers"] == {"kokoro": "af_heart"}
          and vs["anna"]["options"].get("temperature") == 0.7 and len(q2.info["cache"]) >= 4 and q2.default_voice == "anna",
          f"Python: Profile / add_recording / add / set_voice / set_style / remove / prepare / save ({len(q2.info['cache'])} built, {len(r['voices'])} prepared)")

    # ---- 4. other models ----
    try:
        import torch  # noqa: F401
        import f5_tts  # noqa: F401
        have_f5 = True
    except Exception:  # noqa: BLE001
        have_f5 = False
    if have_f5:
        fw = os.path.join(work, "f5")
        rc, out = run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_f5.py"), "--out", fw])
        from vox.build_f5 import build
        f5m = os.path.join(work, "f5.tmod")
        build(os.path.join(fw, "ckpt", "model.safetensors"), os.path.join(fw, "ckpt", "vocab.txt"), os.path.join(fw, "vocos"), f5m,
              dtype="f32", tokenizer="char", sha=False)
        f5 = vox.Engine(f5m)
        out1 = f5.synthesize("Hello there, this is F5.", qp, seed=2, voice_id="anna", nfe_step=4)
        direct = f5.clone_voice(ref1, "hello there")
        out2 = f5.synthesize("Hello there, this is F5.", direct, seed=2, nfe_step=4)
        check(len(out1) > 0 and np.array_equal(out1, out2),
              "a profile made on another model is rebuilt from its recording on F5 (= cloning the recording on F5 directly)")
        fast = f5.synthesize("Hello there, this is F5.", qp, seed=2, voice_id="anna", nfe_step=4, speed=1.3)
        tag = f5.synthesize("[speed 1.3] Hello there, this is F5.", qp, seed=2, voice_id="anna", nfe_step=4)
        check(np.array_equal(fast, tag) and len(fast) < len(out1), f"[speed 1.3] = speed=1.3 on F5 ({len(out1)} → {len(fast)} samples)")
        q3 = vox.Voice.open(qp)
        q3.prepare(f5, "anna")
        fams = {c["model"]["family"] for c in q3.info["cache"]}
        check(fams == {"vox-test", "f5-tts"}, f"one profile holds builds for several models ({sorted(fams)})")
    else:
        print("SKIP  F5 part (needs torch and f5-tts)")
    try:
        import kokoro  # noqa: F401
        have_k = True
    except Exception:  # noqa: BLE001
        have_k = False
    if have_k:
        kw = os.path.join(work, "k")
        rc, out = run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_kokoro.py"), "--out", kw])
        from vox.build_kokoro import build as kbuild
        km = os.path.join(work, "k.tmod")
        kbuild(os.path.join(kw, "config.json"), os.path.join(kw, "kokoro-tiny.pth"), km, os.path.join(kw, "voices"), dtype="f32", sha=False)
        k = vox.Engine(km)
        kp = vox.Profile("K")
        kp.add_recording("anna", ref1, "hello there")
        kp.set_voice("anna", speakers={"kokoro": "af_tiny"})
        kp.add_recording("ben", ref1, "hello there")
        a1 = k.synthesize("Hello there.", kp, seed=1, voice_id="anna", g2p="builtin")
        a2 = k.synthesize("Hello there.", k.speaker_voice("af_tiny"), seed=1, g2p="builtin")
        check(len(a1) > 0 and np.array_equal(a1, a2), "Kokoro: the voice's speaker mapping is used (= the built-in voice)")
        try:
            k.synthesize("Hello.", kp, voice_id="ben", g2p="builtin")
            check(False, "Kokoro without a speaker mapping is an error")
        except vox.VoxError as e:
            check("kokoro=" in str(e), f"Kokoro without a mapping: the error says how to add one ({e})")
        h = k.synthesize("[happy] Hello there.", kp, seed=1, voice_id="anna", g2p="builtin")
        check(len(h) < len(a1), f"Kokoro: [happy] without a style in the profile becomes a faster speed ({len(a1)} → {len(h)})")
        if True:
            from vox import kokoro_frontend as kf
            if kf.available():
                m1 = k.synthesize("Hello there. [pause 500ms] Bye.", kp, seed=1, voice_id="anna", g2p="misaki")
                m0 = k.synthesize("Hello there.", kp, seed=1, voice_id="anna", g2p="misaki")
                check(len(m1) > len(m0) + 12000 and np.all(m1[len(m0):len(m0) + 12000] == 0),
                      "Kokoro with the misaki frontend keeps markup (the pause is there, the tag is not read)")
    else:
        print("SKIP  Kokoro part (needs kokoro)")

    print("profiles: " + ("ALL OK" if not fails else f"{fails} FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "vox")))
