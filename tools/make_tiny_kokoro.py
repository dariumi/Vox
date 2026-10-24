#!/usr/bin/env python3
"""Tiny Kokoro (StyleTTS2 + iSTFTNet) with random weights, built from the official `kokoro` classes, plus reference
outputs for tests/test_kokoro_tiny.py.

Writes (--out):
  config.json, kokoro-tiny.pth (same layout as kokoro-v1_0.pth: {bert, bert_encoder, predictor, text_encoder, decoder},
  "module." prefixes, weight_norm as weight_g / weight_v), voices/af_tiny.pt, voices/bm_tiny.pt ([510, 1, 256] packs);
  dump.safetensors — every stage of KModel.forward for one phoneme string (for `vox verify`);
  e2e.json + e2e_<name>.safetensors — whole utterances through the official KPipeline (text → misaki G2P → chunks → audio)
  with the source noise of every chunk recorded, so the engine can be fed the same noise.

The decoder of Kokoro has hard-coded widths (1024 / 512 / 64 channels and a 512-channel generator input), so hidden_dim
and upsample_initial_channel stay 512; the PL-BERT and the prosody predictor depth are small. F0 and durations are biased so
that the source has voiced and unvoiced parts and phonemes get different lengths.
Usage: python tools/make_tiny_kokoro.py --out work/kokoro"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# the real Kokoro-82M vocabulary (config.json of hexgrad/Kokoro-82M)
VOCAB = {";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10, "\"": 11, "(": 12, ")": 13, "“": 14, "”": 15, " ": 16,
         "̃": 17, "ʣ": 18, "ʥ": 19, "ʦ": 20, "ʨ": 21, "ᵝ": 22, "ꭧ": 23, "A": 24, "I": 25, "O": 31, "Q": 33, "S": 35,
         "T": 36, "W": 39, "Y": 41, "ᵊ": 42, "a": 43, "b": 44, "c": 45, "d": 46, "e": 47, "f": 48, "h": 50, "i": 51, "j": 52,
         "k": 53, "l": 54, "m": 55, "n": 56, "o": 57, "p": 58, "q": 59, "r": 60, "s": 61, "t": 62, "u": 63, "v": 64, "w": 65,
         "x": 66, "y": 67, "z": 68, "ɑ": 69, "ɐ": 70, "ɒ": 71, "æ": 72, "β": 75, "ɔ": 76, "ɕ": 77, "ç": 78, "ɖ": 80, "ð": 81,
         "ʤ": 82, "ə": 83, "ɚ": 85, "ɛ": 86, "ɜ": 87, "ɟ": 90, "ɡ": 92, "ɥ": 99, "ɨ": 101, "ɪ": 102, "ʝ": 103, "ɯ": 110,
         "ɰ": 111, "ŋ": 112, "ɳ": 113, "ɲ": 114, "ɴ": 115, "ø": 116, "ɸ": 118, "θ": 119, "œ": 120, "ɹ": 123, "ɾ": 125,
         "ɻ": 126, "ʁ": 128, "ɽ": 129, "ʂ": 130, "ʃ": 131, "ʈ": 132, "ʧ": 133, "ʊ": 135, "ʋ": 136, "ʌ": 138, "ɣ": 139,
         "ɤ": 140, "χ": 142, "ʎ": 143, "ʒ": 147, "ʔ": 148, "ˈ": 156, "ˌ": 157, "ː": 158, "ʰ": 162, "ʲ": 164, "↓": 169,
         "→": 171, "↗": 172, "↘": 173, "ᵻ": 177}


class NoiseTap:
    """Records (and optionally replays) torch.rand / torch.randn_like calls made by the iSTFTNet source module."""

    def __init__(self, torch):
        self.torch = torch
        self.calls = []

    def __enter__(self):
        t = self.torch
        self.orig = (t.rand, t.randn_like)
        tap = self

        def rand(*a, **k):
            r = tap.orig[0](*a, **k)
            tap.calls.append(("rand", r.detach().clone()))
            return r

        def randn_like(x, *a, **k):
            r = tap.orig[1](x, *a, **k)
            tap.calls.append(("randn_like", r.detach().clone()))
            return r
        t.rand, t.randn_like = rand, randn_like
        return self

    def __exit__(self, *e):
        self.torch.rand, self.torch.randn_like = self.orig
        return False

    def tensors(self, prefix):
        """rand_ini [9], sine_noise [L][9], uv_noise [L] in the order SineGen / SourceModuleHnNSF draw them."""
        out = {}
        kinds = [c for c in self.calls]
        assert len(kinds) == 3, [k for k, _ in kinds]
        # randn_like keeps the (transposed) strides of its argument: make the arrays C-contiguous before saving
        out[prefix + "rand_ini"] = np.ascontiguousarray(kinds[0][1].reshape(-1).numpy(), dtype=np.float32)
        out[prefix + "sine_noise"] = np.ascontiguousarray(kinds[1][1][0].contiguous().numpy(), dtype=np.float32)
        out[prefix + "uv_noise"] = np.ascontiguousarray(kinds[2][1].reshape(-1).numpy(), dtype=np.float32)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--bert-hidden", type=int, default=64)
    ap.add_argument("--bert-layers", type=int, default=2)
    ap.add_argument("--gen-ch", type=int, default=512, help="upsample_initial_channel of the generator (the decoder feeds it 512 channels, so 512)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "voices"), exist_ok=True)

    import torch
    from safetensors.numpy import save_file
    torch.manual_seed(a.seed)
    from kokoro.model import KModel

    cfg = {
        "istftnet": {"upsample_kernel_sizes": [20, 12], "upsample_rates": [10, 6], "gen_istft_hop_size": 5, "gen_istft_n_fft": 20,
                     "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]], "resblock_kernel_sizes": [3, 7, 11],
                     "upsample_initial_channel": a.gen_ch},
        "dim_in": 64, "dropout": 0.2, "hidden_dim": 512, "max_conv_dim": 512, "max_dur": 50, "multispeaker": True,
        "n_layer": a.n_layer, "n_mels": 80, "n_token": 178, "style_dim": 128, "text_encoder_kernel_size": 5,
        "plbert": {"hidden_size": a.bert_hidden, "num_attention_heads": 2, "intermediate_size": 2 * a.bert_hidden,
                   "max_position_embeddings": 512, "num_hidden_layers": a.bert_layers, "dropout": 0.1},
        "vocab": VOCAB,
    }
    with open(os.path.join(a.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)

    # random weights: build the modules exactly as KModel does, perturb, save in the official layout
    from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder
    from kokoro.istftnet import Decoder
    from transformers import AlbertConfig
    mods = {
        "bert": CustomAlbert(AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])),
        "predictor": ProsodyPredictor(style_dim=128, d_hid=512, nlayers=a.n_layer, max_dur=50, dropout=0.2),
        "text_encoder": TextEncoder(channels=512, kernel_size=5, depth=a.n_layer, n_symbols=178),
        "decoder": Decoder(dim_in=512, style_dim=128, dim_out=80, **cfg["istftnet"]),
    }
    mods["bert_encoder"] = torch.nn.Linear(a.bert_hidden, 512)
    g = torch.Generator().manual_seed(a.seed + 1)
    with torch.no_grad():
        for name, m in mods.items():
            for k, p in m.named_parameters():
                if k.endswith("weight_g"):
                    p.copy_(p * (0.7 + 0.6 * torch.rand(p.shape, generator=g)))
                elif "alpha" in k:
                    p.copy_(0.5 + torch.rand(p.shape, generator=g))
                elif k.endswith("bias") or k.endswith("beta"):
                    p.copy_(0.05 * torch.randn(p.shape, generator=g))
                elif k.endswith("gamma") or "LayerNorm.weight" in k or "layer_norm.weight" in k:
                    p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
        pr = mods["predictor"]
        pr.duration_proj.linear_layer.bias.fill_(-4.0)                  # ~1-6 frames per phoneme
        pr.duration_proj.linear_layer.weight.mul_(3.0)
        pr.F0_proj.bias.fill_(40.0)                                      # F0 around 40 Hz ± 100: voiced and unvoiced (< 10) parts
        pr.F0_proj.weight.mul_(200.0)
        pr.N_proj.weight.mul_(3.0)
    sd = {k: {("module." + n): v for n, v in m.state_dict().items()} for k, m in mods.items()}
    pth = os.path.join(a.out, "kokoro-tiny.pth")
    torch.save(sd, pth)
    model = KModel(config=os.path.join(a.out, "config.json"), model=pth).eval()
    for name in ("af_tiny", "bm_tiny"):
        pack = 0.25 * torch.randn(510, 1, 256, generator=g)
        torch.save(pack, os.path.join(a.out, "voices", name + ".pt"))
    pack = torch.load(os.path.join(a.out, "voices", "af_tiny.pt"), weights_only=True)

    # ---------------- stage dump for one phoneme string ----------------
    ps = "ðɪs ɪz ɐ tˈɛst, wɪð sˈʌm stɹˈɛs ænd pʌŋktʃuˈeɪʃən!"
    ref_s = pack[len(ps) - 1]
    with torch.no_grad():
        input_ids = [i for i in (model.vocab.get(p) for p in ps) if i is not None]
        ids = torch.LongTensor([[0, *input_ids, 0]])
        L = ids.shape[1]
        lens = torch.LongTensor([L])
        text_mask = torch.zeros(1, L, dtype=torch.bool)
        bert_dur = model.bert(ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = model.predictor.text_encoder(d_en, s, lens, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1)
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        idx = torch.repeat_interleave(torch.arange(L), pred_dur)
        aln = torch.zeros((L, idx.shape[0]))
        aln[idx, torch.arange(idx.shape[0])] = 1
        aln = aln.unsqueeze(0)
        en = d.transpose(-1, -2) @ aln
        F0, N = model.predictor.F0Ntrain(en, s)
        t_en = model.text_encoder(ids, lens, text_mask)
        asr = t_en @ aln
        with NoiseTap(torch) as tap:
            audio = model.decoder(asr, F0, N, ref_s[:, :128]).squeeze()
        nz = tap.tensors("")
        # the source spectrogram the generator saw (same noise replayed): lets `vox verify` check the generator on its own
        gen = model.decoder.generator
        calls = [c[1].clone() for c in tap.calls]
        o_rand, o_rl = torch.rand, torch.randn_like
        torch.rand = lambda *a_, **k_: calls.pop(0).clone()
        torch.randn_like = lambda x_, *a_, **k_: calls.pop(0).clone().reshape(x_.shape)
        try:
            f0u = gen.f0_upsamp(F0[:, None]).transpose(1, 2)
            hs, _, _ = gen.m_source(f0u)
            hs = hs.transpose(1, 2).squeeze(1)
            mag, ph = gen.stft.transform(hs)
            har = torch.cat([mag, ph], dim=1)[0].T.contiguous().numpy()
            har_src = hs[0].contiguous().numpy()
        finally:
            torch.rand, torch.randn_like = o_rand, o_rl
    dump = {"input_ids": np.array(input_ids, np.float32), "bert": np.ascontiguousarray(bert_dur[0].numpy()), "d_en": d_en[0].T.contiguous().numpy(),
            "d": np.ascontiguousarray(d[0].numpy()), "duration": duration[0].numpy(), "pred_dur": pred_dur.numpy().astype(np.float32),
            "F0": F0[0].numpy(), "N": N[0].numpy(), "t_en": t_en[0].T.contiguous().numpy(), "asr": asr[0].T.contiguous().numpy(),
            "ref_s": ref_s[0].numpy(), "audio": audio.numpy().astype(np.float32), "har": har, "har_src": har_src, **nz}
    save_file(dump, os.path.join(a.out, "dump.safetensors"), metadata={"phonemes": ps, "voice": "af_tiny"})
    print(f"stages: {L} tokens, {int(pred_dur.sum())} frames, {audio.shape[0]} samples, "
          f"voiced {float((F0 > 10).float().mean()):.2f}", file=sys.stderr)

    # ---------------- whole utterances through the official pipeline ----------------
    from kokoro import KPipeline
    try:
        from misaki import en
        have_g2p = True
    except Exception:  # noqa: BLE001
        have_g2p = False
    scen = {}
    texts = {
        "phonemes": None,   # raw phoneme string, one chunk
        "short": "Hello world! This is a tiny test.",
        "numbers": "It costs $3.50, or 42 dollars on May 5th, 1999.",
        "long": " ".join(["The quick brown fox jumps over the lazy dog, and then it runs away again."] * 9) +
                "\nA second line after a break.",
    }
    voice_path = os.path.join(a.out, "voices", "af_tiny.pt")
    pipe = KPipeline(lang_code="a", model=model) if have_g2p else None
    for name, text in texts.items():
        if text is None:
            chunks_in = [("", ps)]
            speed = 1.0
        elif pipe is None:
            continue
        else:
            speed = 1.15 if name == "numbers" else 1.0
            chunks_in = None
        out, meta = {}, []
        if chunks_in is not None:
            for i, (gs, p) in enumerate(chunks_in):
                with torch.no_grad(), NoiseTap(torch) as tap:
                    r = model(p, pack[len(p) - 1], speed, return_output=True)
                out.update(tap.tensors(f"{i}."))
                out[f"{i}.audio"] = r.audio.numpy().astype(np.float32)
                meta.append({"graphemes": gs, "phonemes": p})
        else:
            gen = pipe(text, voice=voice_path, speed=speed)
            i = 0
            while True:
                with torch.no_grad(), NoiseTap(torch) as tap:
                    try:
                        r = next(gen)
                    except StopIteration:
                        break
                out.update(tap.tensors(f"{i}."))
                out[f"{i}.audio"] = r.audio.numpy().astype(np.float32)
                meta.append({"graphemes": r.graphemes, "phonemes": r.phonemes})
                i += 1
        save_file(out, os.path.join(a.out, f"e2e_{name}.safetensors"))
        scen[name] = {"text": text if text is not None else ps, "phonemes_in": text is None, "speed": speed, "voice": "af_tiny",
                      "chunks": meta}
        print(f"{name}: {len(meta)} chunk(s), {sum(len(m['phonemes']) for m in meta)} phonemes", file=sys.stderr)
    # misaki G2P without the espeak fallback: what the engine's built-in lexicon G2P is compared with
    g2p_cases = {}
    if have_g2p:
        g = en.G2P(trf=False, british=False, fallback=None, unk="")
        gb = en.G2P(trf=False, british=True, fallback=None, unk="")
        for t in ["Hello world! This is a tiny test.", "It costs $3.50, or 42 dollars on May 5th, 1999.",
                  "The quick brown fox jumps over the lazy dog.", "I can't believe it's already 10:30; we're late!",
                  "Dr. Smith's cats were running, jumped, and played.", "She read the book that he reads every day."]:
            g2p_cases[t] = {"us": g(t)[0], "gb": gb(t)[0]}
    with open(os.path.join(a.out, "e2e.json"), "w", encoding="utf-8") as f:
        json.dump({"scenarios": scen, "g2p": g2p_cases}, f, ensure_ascii=False, indent=1)
    print("ok", a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
