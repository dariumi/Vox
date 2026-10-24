#!/usr/bin/env python3
"""Крошечная F5-TTS со случайными весами + эталонные результаты официального пакета f5_tts / vocos.

Зачем: настоящих весов в песочнице нет, а арифметику модели можно проверить и на случайных: официальный код
(f5_tts.model.DiT/CFM, vocos.Vocos) считает на тех же весах, vox — на своём C++, результаты сравниваются по стадиям.

Выход (--out DIR):
  ckpt/model.safetensors, ckpt/vocab.txt, vocos/pytorch_model.safetensors — вход конвертера vox.build_f5
  ref.wav                    — образец (24 кГц, 16 бит, тихий: проверяет подъём громкости до rms 0.1)
  dump.safetensors           — тензоры стадий для `vox verify` (см. F5Backend::debug_check)
  e2e.json                   — тексты и параметры сквозного прогона
  noise.safetensors          — шум по кускам (noise.0, noise.1…), который подставляется в vox вместо Philox
  expected.safetensors       — звук официального инференса (wave, float32)
Сквозной прогон идёт через настоящий f5_tts.infer.utils_infer.infer_batch_process (последовательно, с подменой torch.randn).
"""
import argparse
import json
import os
import sys
import wave

import numpy as np
import torch
from safetensors.numpy import save_file


def rand_init(module, rng, scale=1.0):
    """Все параметры случайные: у F5 AdaLN и proj_out при создании нулевые, и без этого проверять было бы нечего."""
    g = torch.Generator().manual_seed(rng)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim >= 3:      # свёртки
                fan = p.shape[1] * p.shape[2]
                p.copy_(torch.randn(p.shape, generator=g) / fan ** 0.5 * scale)
            elif p.ndim == 2:
                if "text_embed.text_embed" in name:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.5)
                else:
                    p.copy_(torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5 * scale * 0.7)
            elif name.endswith("norm.weight") or name.endswith("norm.linear.weight"):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            elif "grn" in name:
                p.copy_(0.3 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))


def write_wav16(path, x, sr=24000):
    x = np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())


def read_wav16(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0, w.getframerate()


class SerialExecutor:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *a, **k):
        class F:
            def __init__(s, v):
                s.v = v

            def result(s):
                return s.v
        return F(fn(*a, **k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--dim-head", type=int, default=32)
    ap.add_argument("--text-dim", type=int, default=32)
    ap.add_argument("--conv-layers", type=int, default=2)
    ap.add_argument("--v0", action="store_true", help="config F5TTS_Base (v0): text_mask_padding=False, pe_attn_head=1 (default: v1 config)")
    ap.add_argument("--vocos-center", action="store_true",
                    help="ISTFT head with padding=center, as in the real charactr/vocos-mel-24khz (default: same)")
    ap.add_argument("--vocos-dim", type=int, default=48)
    ap.add_argument("--vocos-inter", type=int, default=96)
    ap.add_argument("--vocos-layers", type=int, default=2)
    ap.add_argument("--crlf-vocab", action="store_true", help="write vocab.txt with CRLF line endings (as some fine-tunes do)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "vocos"), exist_ok=True)

    try:
        import datasets  # noqa: F401
    except Exception:  # обучающий код f5_tts тянет datasets; для инференса он не нужен — подменяем модуль набора данных
        import types
        stub = types.ModuleType("f5_tts.model.dataset")
        stub.DynamicBatchSampler = type("DynamicBatchSampler", (), {})
        stub.collate_fn = lambda *a, **k: None
        sys.modules["f5_tts.model.dataset"] = stub
    from f5_tts.infer import utils_infer
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import convert_char_to_pinyin, list_str_to_idx
    from vocos.heads import ISTFTHead
    from vocos.models import VocosBackbone

    torch.manual_seed(a.seed)
    # словарь: символы ASCII + слоги пиньиня, которые встретятся в тестах
    vocab = [" "] + [chr(c) for c in range(33, 127)] + ["shi4", "jie4", "ni3", "hao3", "wo3", "ai4", "ni2", "ma1", "是", "é"]
    vmap = {c: i for i, c in enumerate(vocab)}
    mel_dim = 100
    transformer = DiT(dim=a.dim, depth=a.depth, heads=a.heads, dim_head=a.dim_head, ff_mult=2, mel_dim=mel_dim,
                      text_num_embeds=len(vocab), text_dim=a.text_dim, text_mask_padding=not a.v0, conv_layers=a.conv_layers,
                      qk_norm=None, pe_attn_head=1 if a.v0 else None, attn_backend="torch", attn_mask_enabled=False)
    model = CFM(transformer=transformer, odeint_kwargs=dict(method="euler"), vocab_char_map=vmap,
                mel_spec_kwargs=dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=mel_dim, target_sample_rate=24000,
                                     mel_spec_type="vocos"))
    rand_init(model.transformer, a.seed + 1)
    model.eval()

    from vocos import Vocos
    from vocos.feature_extractors import MelSpectrogramFeatures
    voc = Vocos(MelSpectrogramFeatures(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=mel_dim, padding="center"),
                VocosBackbone(input_channels=mel_dim, dim=a.vocos_dim, intermediate_dim=a.vocos_inter, num_layers=a.vocos_layers),
                ISTFTHead(dim=a.vocos_dim, n_fft=1024, hop_length=256, padding="center" if a.vocos_center else "same"))
    rand_init(voc, a.seed + 2, 0.8)
    with torch.no_grad():
        for blk in voc.backbone.convnext:
            blk.gamma.copy_(0.5 + 0.1 * torch.randn_like(blk.gamma))
        nb = 513
        voc.head.out.bias[:nb].copy_(-2.0 + 0.5 * torch.randn(nb))   # логарифм модуля: не слишком громко
        voc.head.out.bias[nb:].copy_(torch.randn(nb))
    voc.eval()

    # --- вход конвертера ---
    ck = {("ema_model.transformer." + k): v.detach().float().numpy() for k, v in model.transformer.state_dict().items()}
    save_file(ck, os.path.join(a.out, "ckpt", "model.safetensors"))
    with open(os.path.join(a.out, "ckpt", "vocab.txt"), "w", encoding="utf-8", newline="") as f:
        nl = "\r\n" if a.crlf_vocab else "\n"
        f.write(nl.join(vocab) + nl)
    # папка вокодера как у charactr/vocos-mel-24khz: config.yaml + pytorch_model.bin (полный state_dict)
    torch.save(voc.state_dict(), os.path.join(a.out, "vocos", "pytorch_model.bin"))
    with open(os.path.join(a.out, "vocos", "config.yaml"), "w") as f:
        f.write("""feature_extractor:
  class_path: vocos.feature_extractors.MelSpectrogramFeatures
  init_args: {sample_rate: 24000, n_fft: 1024, hop_length: 256, n_mels: %d, padding: center}
backbone:
  class_path: vocos.models.VocosBackbone
  init_args: {input_channels: %d, dim: %d, intermediate_dim: %d, num_layers: %d}
head:
  class_path: vocos.heads.ISTFTHead
  init_args:
    dim: %d
    n_fft: 1024
    hop_length: 256
    padding: %s
""" % (mel_dim, mel_dim, a.vocos_dim, a.vocos_inter, a.vocos_layers, a.vocos_dim, "center" if a.vocos_center else "same"))

    # --- образец: тихая смесь синусоид, 24 кГц ---
    rng = np.random.default_rng(a.seed)
    n = int(1.1 * 24000)
    t = np.arange(n) / 24000.0
    ref = (0.5 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 410 * t * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))) +
           0.1 * rng.normal(size=n)) * (0.5 + 0.5 * np.sin(2 * np.pi * 2.2 * t) ** 2) * 0.06
    write_wav16(os.path.join(a.out, "ref.wav"), ref)
    audio_np, sr = read_wav16(os.path.join(a.out, "ref.wav"))
    ref_text = "Hi there"
    gen_text = "This is a tiny model. It says things; then more, and 42 or 3.5% of them!"
    # нормировка текста образца как в preprocess_ref_audio_text
    rt = ref_text
    if not rt.endswith(". ") and not rt.endswith("。"):
        rt += " " if rt.endswith(".") else ". "

    # --- токенизация ---
    tok_cases = ["Hi there. This is a test, 42 or 3.5%!", "hello world", "世界 ni hao", "A: b'c\"d", "ab  cd", "x;y “q” ‘r’"]
    tok = {}
    for i, s in enumerate(tok_cases):
        syms = convert_char_to_pinyin([s])[0]
        tok[f"tok{i}"] = (s, syms, [vmap.get(c, 0) for c in syms])

    # --- стадии ---
    audio = torch.from_numpy(audio_np)[None]
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < 0.1:
        audio = audio * 0.1 / rms
    with torch.inference_mode():
        mel = model.mel_spec(audio).permute(0, 2, 1)[0]          # [frames, 100]
        N = 60
        g = torch.Generator().manual_seed(11)
        x = torch.randn(1, N, mel_dim, generator=g)
        cond = torch.zeros(1, N, mel_dim)
        cond[:, :30] = torch.randn(1, 30, mel_dim, generator=g)
        text_str = [convert_char_to_pinyin([rt + "ok then"])[0]]
        text = list_str_to_idx(text_str, vmap)
        tt = torch.tensor(0.37)
        pred = transformer(x=x, cond=cond, text=text, time=tt, mask=None, cfg_infer=True, cache=False)
        pc, pu = torch.chunk(pred, 2, dim=0)
        transformer.clear_cache()
        tc = transformer.text_embed(text, seq_len=N, drop_text=False)[0]
        tu = transformer.text_embed(text, seq_len=N, drop_text=True)[0]
        vmel = torch.randn(1, mel_dim, 37, generator=g) * 1.5 - 3.0
        vwave = voc.decode(vmel)[0]
        # полный цикл CFM, шум подставлен
        out = {}
        for tag, nfe, epss in (("a", 7, True), ("b", 5, False)):
            cm = mel[None]
            Cf = cm.shape[1]
            Ns = Cf + 40
            noise = torch.randn(Ns, mel_dim, generator=g)
            orig = torch.randn
            torch.randn = lambda *s_, **k_: noise.clone() if tuple(s_) == (Ns, mel_dim) else orig(*s_, **k_)
            try:
                text_ids = list_str_to_idx([convert_char_to_pinyin([rt + "ok then"])[0]], vmap)
                sampled, _ = model.sample(cond=cm, text=text_ids, duration=Ns, steps=nfe, cfg_strength=2.0, sway_sampling_coef=-1.0,
                                          use_epss=epss)
            finally:
                torch.randn = orig
            out[tag] = (nfe, epss, noise, sampled[0], text_ids[0])

    dump = {"audio": audio[0].numpy(), "mel": mel.numpy(), "x": x[0].numpy(), "cond": cond[0].numpy(), "t": np.array([0.37], np.float32),
            "ids_full": text[0].numpy().astype(np.float32), "temb_c": tc.numpy(), "temb_u": tu.numpy(), "pred_c": pc[0].numpy(),
            "pred_u": pu[0].numpy(), "vmel": vmel[0].T.contiguous().numpy(), "vwave": vwave.numpy()}
    # токенизация: общий тест через метаданные первого случая
    s0, syms0, ids0 = tok["tok0"]
    meta = {"text": s0, "symbols": json.dumps(syms0, ensure_ascii=False), "nfe": str(out["a"][0]), "cfg": "2", "sway": "-1",
            "epss": "1" if out["a"][1] else "0"}
    for i, (k, (s_i, sy_i, id_i)) in enumerate(tok.items()):
        meta[f"tok{i}.text"] = s_i
        meta[f"tok{i}.symbols"] = json.dumps(sy_i, ensure_ascii=False)
    dump.update({"ids": np.array(ids0, np.float32), "noise": out["a"][2].numpy(), "cond_mel": mel.numpy(), "sampled": out["a"][3].numpy(),
                 "sample_ids": out["a"][4].numpy().astype(np.float32)})
    save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in dump.items()}, os.path.join(a.out, "dump.safetensors"), metadata=meta)
    json.dump({k: {"text": s, "symbols": sy, "ids": ids} for k, (s, sy, ids) in tok.items()},
              open(os.path.join(a.out, "tokens.json"), "w", encoding="utf-8"), ensure_ascii=False)

    # --- сквозной прогон через настоящий infer_batch_process ---
    utils_infer.ThreadPoolExecutor = SerialExecutor
    scenarios = {}
    for sc, (text_in, max_chars, speed, nfe, cf) in {
        "single": (gen_text, None, 1.0, 6, 0.15),
        "chunks": ("First sentence is here. Second one follows now! Third? Yes, third; and a fourth, final. Fifth ends it all.", 40, 1.0, 5, 0.15),
        "speed": ("Short.", None, 1.3, 5, 0.0),
        # длина куска из формулы F5 (без max_chars): проверяет и формулу, и нарезку
        "formula": ("Alpha beta gamma delta epsilon zeta eta theta. Iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon. "
                    "Phi chi psi omega again and again, until the limit is reached for sure! More words follow here to fill space; "
                    "enough now? Yes, it is enough, truly.", None, 0.5, 5, 0.15),
    }.items():
        ref_sec = audio_np.shape[0] / sr
        mc = max_chars or int(len(rt.encode("utf-8")) / ref_sec * (22 - ref_sec) * speed)
        batches = utils_infer.chunk_text(text_in, max_chars=mc)
        noises = []
        orig = torch.randn

        def fake(*s_, **k_):
            if len(s_) == 2 and s_[1] == mel_dim:
                r = np.random.default_rng(1000 + len(noises)).normal(size=(s_[0], mel_dim)).astype(np.float32)
                noises.append(r)
                return torch.from_numpy(r)
            return orig(*s_, **k_)
        torch.randn = fake
        try:
            gen = utils_infer.infer_batch_process((torch.from_numpy(audio_np)[None], sr), rt, batches, model, voc, progress=None,
                                                  target_rms=0.1, cross_fade_duration=cf, nfe_step=nfe, cfg_strength=2.0,
                                                  sway_sampling_coef=-1, speed=speed, device="cpu")
            wave_out, _, spec = next(gen)
        finally:
            torch.randn = orig
        if sc == "single":  # то же, что делает infer_batch_process, но с выгрузкой полного mel — для vox verify на длинной последовательности
            rt2 = rt + " " if len(rt[-1].encode("utf-8")) == 1 else rt
            ids_long = list_str_to_idx(convert_char_to_pinyin([rt2 + batches[0]]), vmap)
            Nl = noises[0].shape[0]
            nz = torch.from_numpy(noises[0])
            orig2 = torch.randn
            torch.randn = lambda *s_, **k_: nz.clone() if tuple(s_) == (Nl, mel_dim) else orig2(*s_, **k_)
            try:
                with torch.inference_mode():
                    smp, _ = model.sample(cond=model.mel_spec(torch.from_numpy(audio_np)[None] * (0.1 / rms) if rms < 0.1 else torch.from_numpy(audio_np)[None]).permute(0, 2, 1),
                                          text=ids_long, duration=Nl, steps=nfe, cfg_strength=2.0, sway_sampling_coef=-1.0)
            finally:
                torch.randn = orig2
            save_file({"noise": np.ascontiguousarray(noises[0]), "cond_mel": mel.numpy(), "sample_ids": ids_long[0].numpy().astype(np.float32),
                       "sampled": smp[0].numpy()}, os.path.join(a.out, "dump_long.safetensors"),
                      metadata={"nfe": str(nfe), "cfg": "2", "sway": "-1", "epss": "1"})
        scenarios[sc] = {"gen_text": text_in, "max_chars": mc if max_chars else 0, "speed": speed, "nfe": nfe, "cross_fade": cf,
                         "chunks": len(batches), "batches": batches}
        save_file({f"noise.{i}": np.ascontiguousarray(nz) for i, nz in enumerate(noises)}, os.path.join(a.out, f"noise_{sc}.safetensors"))
        save_file({"wave": np.ascontiguousarray(wave_out, dtype=np.float32), "mel": np.ascontiguousarray(spec.T, dtype=np.float32)}, os.path.join(a.out, f"expected_{sc}.safetensors"))
        print(f"{sc}: {len(batches)} chunk(s), {len(wave_out)} samples", file=sys.stderr)
    json.dump({"ref_text": ref_text, "scenarios": scenarios}, open(os.path.join(a.out, "e2e.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("ok", a.out)


if __name__ == "__main__":
    main()
