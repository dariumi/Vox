#!/usr/bin/env python3
"""Tiny Chatterbox with random weights, built from the official chatterbox-tts classes, plus reference outputs for
tests/test_chatterbox_tiny.py.

Writes (--out):
  model/  — the layout of ResembleAI/chatterbox: t3_cfg.safetensors (T3 with a 2-layer Llama; the hidden size stays 1024,
            the perceiver is hard-wired to it), s3gen.safetensors (S3Gen with a 2-block encoder, a 64-channel estimator, a
            64-channel HiFT, a 2-layer speech tokenizer; CAM++ at its real size), ve.safetensors (the real VoiceEncoder),
            tokenizer.json (a character BPE trained here, laid out like the release: [STOP] 0, [UNK] 1, [SPACE] 2, [START] 255),
            conds.pt (the built-in voice: ChatterboxTTS.prepare_conditionals of prompt.wav);
  prompt.wav (24 kHz);
  dump.safetensors — every stage of ChatterboxTTS.generate for one utterance with the built-in voice: text tokens, the
            16 kHz prompt and its VoiceEncoder embedding, T3's conditioning rows, the CFG-combined logits of every step and
            the tokens it sampled, the flow's encoder output and starting noise, the mel, HiFT's F0 and source noise, the audio
            (before the Perth watermark, which vox does not add);
  e2e_*.safetensors — whole utterances (several settings), with what the engine needs to repeat them: the speech tokens
            and every random draw of S3Gen.
Usage: python tools/make_tiny_chatterbox.py --out work/chatterbox"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

TRAIN_TEXT = """The quick brown fox jumps over the lazy dog. Hello there, this is a tiny test of the voice.
She sells sea shells by the sea shore! What a wonderful day it is, isn't it? We are going to the park.
Numbers like one, two and three are spelled out. Speech synthesis turns written text into sound.
I can't believe it's already late; we should go. Please call Stella and ask her to bring these things with her."""


def make_tokenizer(path):
    """A character BPE in the shape of Chatterbox's tokenizer.json: Whitespace pre-tokenizer, specials first, [START] = 255."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    alphabet = sorted(set(TRAIN_TEXT.replace(" ", "").replace("\n", "")) | set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?.,;:'\"-()"))
    tr = trainers.BpeTrainer(vocab_size=240, special_tokens=["[STOP]", "[UNK]", "[SPACE]"], initial_alphabet=alphabet, show_progress=False)
    tok.train_from_iterator([TRAIN_TEXT] * 4, tr)
    j = json.loads(tok.to_str())
    vocab = j["model"]["vocab"]
    n = max(vocab.values()) + 1
    assert n <= 255, n
    for s, i in (("[START]", 255), ("[PAD]", 256), ("[SEP]", 257), ("[CLS]", 258), ("[MASK]", 259)):
        vocab[s] = i
        j["added_tokens"].append({"id": i, "content": s, "single_word": False, "lstrip": False, "rstrip": False,
                                  "normalized": False, "special": True})
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(j, ensure_ascii=False))


class Tap:
    """Records torch.rand / torch.randn_like calls (S3Gen: the flow's starting noise randn_like(mu); SineGen's phases
    (Uniform.sample → torch.rand) and its noise randn_like [1, 9, L]; an unused noise of SourceModuleHnNSF)."""

    def __init__(self, torch):
        self.torch, self.calls = torch, []

    def __enter__(self):
        t, tap = self.torch, self
        self.orig = (t.rand, t.randn_like)

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


def randomize(torch, mod, g, scale=1.0, skip=()):
    with torch.no_grad():
        for n, p in mod.named_parameters():
            if any(s in n for s in skip):
                continue
            if p.dim() == 1 and ("norm" in n or n.endswith(".weight")):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif p.dim() == 1:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
            elif "pos_bias" in n:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                fan_in = int(np.prod(p.shape[1:]))
                p.copy_(torch.randn(p.shape, generator=g) * (scale / fan_in ** 0.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--full", action="store_true", help="the real sizes of ResembleAI/chatterbox (random weights, for speed and memory "
                                                         "measurements; the end of speech is rare: limit with max_new_tokens); no reference dumps")
    a = ap.parse_args()

    import torch
    import librosa
    from safetensors.torch import save_file as st_save
    from safetensors.numpy import save_file
    from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

    from chatterbox.tts import ChatterboxTTS, Conditionals, punc_norm
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS, LLAMA_520M_CONFIG_DICT
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.s3gen.flow import CausalMaskedDiffWithXvec
    from chatterbox.models.s3gen.transformer.upsample_encoder import UpsampleConformerEncoder
    from chatterbox.models.s3gen.flow_matching import CausalConditionalCFM
    from chatterbox.models.s3gen.decoder import ConditionalDecoder
    from chatterbox.models.s3gen.configs import CFM_PARAMS
    from chatterbox.models.s3gen.hifigan import HiFTGenerator
    from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
    from chatterbox.models.s3tokenizer import S3Tokenizer
    from chatterbox.models.tokenizers import EnTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder
    from s3tokenizer.model_v2 import S3TokenizerV2, ModelConfig

    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed + 1)
    md = os.path.join(a.out, "model")
    os.makedirs(md, exist_ok=True)
    make_tokenizer(os.path.join(md, "tokenizer.json"))

    # ---- T3: Llama_520M with 2 layers ----
    LLAMA_CONFIGS["vox_tiny"] = dict(LLAMA_520M_CONFIG_DICT, num_hidden_layers=2, intermediate_size=512)
    hp = T3Config.english_only()
    if not a.full:
        hp.llama_config_name = "vox_tiny"
    t3 = T3(hp)
    randomize(torch, t3, g, skip=("text_emb", "speech_emb", "pos_emb", "pre_attention_query"))
    with torch.no_grad():
        # speech tokens: a peaked distribution over a few hundred codes; the end of speech grows likelier with the position
        t3.speech_head.weight.mul_(3.0)
        t3.speech_head.weight[1000:6562].mul_(0.2)
        t3.speech_head.weight[6563:].mul_(0.2)
        t3.speech_pos_emb.emb.weight[:, 7] = 0.0
        t3.speech_pos_emb.emb.weight[:, 7] += torch.arange(t3.speech_pos_emb.emb.weight.shape[0]).float() * 0.08
        t3.speech_head.weight[hp.stop_speech_token].zero_()
        t3.speech_head.weight[hp.stop_speech_token, 7] = 1.2
        t3.cond_enc.perceiver.pre_attention_query.mul_(3.0)
    t3.eval()

    # ---- VoiceEncoder (real size) ----
    ve = VoiceEncoder()
    with torch.no_grad():
        ve.proj.bias.fill_(0.05)
        for n, p in ve.lstm.named_parameters():
            p.mul_(4.0)
    ve.eval()

    # ---- S3Gen: the real class, smaller flow / HiFT / tokenizer swapped in ----
    s3 = S3Gen()
    if a.full:   # the real modules, rescaled so the activations stay in range
        randomize(torch, s3.flow, g)
        randomize(torch, s3.tokenizer, g, scale=1.5, skip=("_mel_filters", "window"))
        with torch.no_grad():
            for n, p in s3.mel2wav.named_parameters():
                if n.endswith("alpha"):
                    p.copy_(0.5 + torch.rand(p.shape, generator=g))
                elif "original0" in n:
                    p.copy_(0.5 + 0.5 * torch.rand(p.shape, generator=g))
                elif p.dim() == 1:
                    p.copy_(0.05 * torch.randn(p.shape, generator=g))
            s3.mel2wav.f0_predictor.classifier.bias.fill_(25.0)
            s3.mel2wav.f0_predictor.classifier.weight.mul_(150.0)
            s3.mel2wav.conv_post.parametrizations.weight.original0.mul_(0.3)
    else:
        enc = UpsampleConformerEncoder(output_size=512, attention_heads=8, linear_units=256, num_blocks=2, dropout_rate=0.1,
                                       positional_dropout_rate=0.1, attention_dropout_rate=0.1, normalize_before=True,
                                       input_layer="linear", pos_enc_layer_type="rel_pos_espnet", selfattention_layer_type="rel_selfattn",
                                       input_size=512, use_cnn_module=False, macaron_style=False)
        est = ConditionalDecoder(in_channels=320, out_channels=80, causal=True, channels=[64], dropout=0.0, attention_head_dim=64,
                                 n_blocks=1, num_mid_blocks=2, num_heads=2, act_fn="gelu", meanflow=False)
        s3.flow = CausalMaskedDiffWithXvec(encoder=enc, decoder=CausalConditionalCFM(spk_emb_dim=80, cfm_params=CFM_PARAMS, estimator=est))
        randomize(torch, s3.flow, g)
        s3.mel2wav = HiFTGenerator(sampling_rate=24000, base_channels=64, upsample_rates=[8, 5, 3], upsample_kernel_sizes=[16, 11, 7],
                                   source_resblock_kernel_sizes=[7, 7, 11], source_resblock_dilation_sizes=[[1, 3, 5]] * 3,
                                   f0_predictor=ConvRNNF0Predictor(cond_channels=64))
        with torch.no_grad():
            for n, p in s3.mel2wav.named_parameters():
                if n.endswith("alpha"):
                    p.copy_(0.5 + torch.rand(p.shape, generator=g))
                elif "original0" in n:
                    p.copy_(0.5 + 0.5 * torch.rand(p.shape, generator=g))
                elif p.dim() == 1:
                    p.copy_(0.05 * torch.randn(p.shape, generator=g))
            s3.mel2wav.f0_predictor.classifier.bias.fill_(25.0)
            s3.mel2wav.f0_predictor.classifier.weight.mul_(150.0)
            s3.mel2wav.conv_post.parametrizations.weight.original0.mul_(0.3)
        tk = S3Tokenizer.__new__(S3Tokenizer)
        S3TokenizerV2.__init__(tk, "speech_tokenizer_v2_25hz",
                               ModelConfig(n_mels=128, n_audio_ctx=1500, n_audio_state=128, n_audio_head=2, n_audio_layer=2, n_codebook_size=3 ** 8))
        tk.n_fft = 400
        tk.register_buffer("_mel_filters", torch.FloatTensor(librosa.filters.mel(sr=16000, n_fft=400, n_mels=128)))
        tk.register_buffer("window", torch.hann_window(400))
        randomize(torch, tk, g, scale=1.5)
        s3.tokenizer = tk
    with torch.no_grad():
        for m in s3.speaker_encoder.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.running_mean.copy_(0.1 * torch.randn(m.running_mean.shape, generator=g))
                m.running_var.copy_(0.5 + torch.rand(m.running_var.shape, generator=g))
                if m.affine:
                    m.weight.copy_(1.0 + 0.1 * torch.randn(m.weight.shape, generator=g))
                    m.bias.copy_(0.1 * torch.randn(m.bias.shape, generator=g))
            elif isinstance(m, torch.nn.PReLU):
                m.weight.copy_(0.25 + 0.05 * torch.randn(m.weight.shape, generator=g))
    s3.eval()

    sd = lambda m: {k: v.detach().contiguous().clone() for k, v in m.state_dict().items()}   # noqa: E731
    st_save(sd(t3), os.path.join(md, "t3_cfg.safetensors"))
    st_save({k: v for k, v in sd(s3).items() if not k.endswith(("_mel_filters", "window", "trim_fade"))},
            os.path.join(md, "s3gen.safetensors"))
    st_save(sd(ve), os.path.join(md, "ve.safetensors"))

    # ---- prompt recording (24 kHz): voiced and unvoiced stretches ----
    sr = 24000
    t = np.arange(int(3.2 * sr)) / sr
    f0 = 120 + 30 * np.sin(2 * np.pi * 1.3 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    env = (np.sin(2 * np.pi * 2.1 * t) > -0.3).astype(np.float64)
    wav = 0.25 * env * (np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)) + 0.01 * np.random.RandomState(1).randn(t.size)
    pcm16 = np.clip(np.round(wav * 32768), -32768, 32767).astype(np.int16)
    import wave
    wpath = os.path.join(a.out, "prompt.wav")
    with wave.open(wpath, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm16.tobytes())

    import types
    import perth
    perth.PerthImplicitWatermarker = lambda: types.SimpleNamespace(apply_watermark=lambda w, sample_rate: w)   # vox does not watermark
    tts = ChatterboxTTS(t3, s3, ve, EnTokenizer(os.path.join(md, "tokenizer.json")), "cpu")
    tts.prepare_conditionals(wpath, exaggeration=0.5)
    tts.conds.save(os.path.join(md, "conds.pt"))
    if a.full:
        print(f"{a.out}: ResembleAI/chatterbox sizes, random weights (no reference dumps)")
        return
    conds = Conditionals.load(os.path.join(md, "conds.pt"))
    # what prepare_conditionals saw, for the per-stage checks
    w24, _ = librosa.load(wpath, sr=24000)
    ref16 = librosa.resample(w24, orig_sr=24000, target_sr=16000)
    # (conds.pt keeps prompt_feat as a transposed view: copy in C order, as safetensors writes the raw buffer)
    cp = lambda x: np.ascontiguousarray(x.detach().cpu().float().numpy(), dtype=np.float32)   # noqa: E731
    ci = lambda x: np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.int32)   # noqa: E731
    clone = {"clone_t3_tokens": ci(conds.t3.cond_prompt_speech_tokens[0]), "clone_ve": cp(conds.t3.speaker_emb[0]),
             "clone_s3_tokens": ci(conds.gen["prompt_token"][0]), "clone_s3_mel": cp(conds.gen["prompt_feat"][0]),
             "clone_xvec": cp(conds.gen["embedding"][0])}

    # ---- whole utterances through ChatterboxTTS.generate, every stage recorded ----
    def run(text, seed, exaggeration=0.5, cfg_weight=0.5, temperature=0.8, repetition_penalty=1.2, min_p=0.05, top_p=1.0):
        rec = {"logits": []}
        orig_rp = RepetitionPenaltyLogitsProcessor.__call__

        def rp(self, ids, scores):
            rec["logits"].append(scores[0].detach().float().clone())
            return orig_rp(self, ids, scores)
        orig_inf, orig_cond = t3.inference, t3.cond_enc.forward

        def inf(**kw):
            r = orig_inf(**kw)
            rec["t3_tokens"] = r[0].detach().clone()
            return r

        def cond_fwd(c):
            r = orig_cond(c)
            rec["t3_cond"] = r[0].detach().clone()
            return r
        dec = s3.flow.decoder
        orig_dec, orig_f0 = dec.forward, s3.mel2wav.f0_predictor.forward

        def dec_fwd(mu, mask, n_timesteps, **kw):
            rec["mu"] = mu[0].detach().clone()
            r = orig_dec(mu, mask, n_timesteps, **kw)
            return r

        def f0_fwd(x):
            r = orig_f0(x)
            rec["f0"] = r[0].detach().clone()
            return r
        orig_flow_inf = s3.flow_inference

        def flow_inf(*aa, **kw):
            r = orig_flow_inf(*aa, **kw)
            rec["mel"] = r[0].detach().clone()
            rec["speech_tokens"] = aa[0].detach().clone() if aa else kw["speech_tokens"].detach().clone()
            return r
        RepetitionPenaltyLogitsProcessor.__call__ = rp
        t3.inference, t3.cond_enc.forward, dec.forward, s3.mel2wav.f0_predictor.forward, s3.flow_inference = inf, cond_fwd, dec_fwd, f0_fwd, flow_inf
        try:
            tts.conds = Conditionals.load(os.path.join(md, "conds.pt"))
            torch.manual_seed(seed)
            with Tap(torch) as tap:
                wav = tts.generate(text, repetition_penalty=repetition_penalty, min_p=min_p, top_p=top_p, exaggeration=exaggeration,
                                   cfg_weight=cfg_weight, temperature=temperature)
        finally:
            RepetitionPenaltyLogitsProcessor.__call__ = orig_rp
            del t3.inference, t3.cond_enc.forward, dec.forward, s3.mel2wav.f0_predictor.forward, s3.flow_inference
        kinds = [k for k, _ in tap.calls]
        assert kinds[:3] == ["randn_like", "rand", "randn_like"], kinds
        z0, ini, sn = tap.calls[0][1][0], tap.calls[1][1].reshape(-1), tap.calls[2][1][0]
        toks = rec["t3_tokens"].numpy().astype(np.int32)
        tt = tts.tokenizer.text_to_tokens(punc_norm(text))[0].numpy()
        tt = np.concatenate([[hp.start_text_token], tt, [hp.stop_text_token]]).astype(np.int32)
        return {
            "text_tokens": tt, "t3_cond": rec["t3_cond"].float().numpy(), "t3_tokens": toks,
            "t3_logits": torch.stack(rec["logits"]).numpy().astype(np.float32),
            "speech_tokens": rec["speech_tokens"].reshape(-1).numpy().astype(np.int32),
            "flow_noise": z0.numpy().astype(np.float32), "flow_mu": rec["mu"].transpose(0, 1).contiguous().numpy().astype(np.float32),
            "flow_mel": rec["mel"].transpose(0, 1).contiguous().numpy().astype(np.float32), "hift_f0": rec["f0"].numpy().astype(np.float32),
            "rand_ini": ini.numpy().astype(np.float32), "sine_noise": sn.contiguous().numpy().astype(np.float32),
            "audio": wav[0].numpy().astype(np.float32),
        }, {"text": text, "exaggeration": str(exaggeration), "cfg_weight": str(cfg_weight), "temperature": str(temperature),
            "repetition_penalty": str(repetition_penalty), "min_p": str(min_p), "top_p": str(top_p)}

    text = "Hello world, this is a tiny test. It speaks two sentences!"
    d, meta = run(text, seed=11)
    d.update({"ref16": ref16.astype(np.float32), "ve": clone["clone_ve"], "t3_prompt_tokens": clone["clone_t3_tokens"],
              "s3_prompt_tokens": clone["clone_s3_tokens"], "s3_prompt_mel": clone["clone_s3_mel"], "xvec": clone["clone_xvec"]})
    # T3 prompt tokens from the official 16 kHz prompt (s3_tokzr.forward([ref16[:6 s]], max_len=150))
    save_file(d, os.path.join(a.out, "dump.safetensors"), metadata=meta)
    print(f"dump: {d['t3_tokens'].size} T3 steps, {d['speech_tokens'].size} speech tokens, {d['audio'].size} samples")
    cases = [
        ("default", "The quick brown fox jumps over the lazy dog.", dict(seed=21)),
        ("expressive", "What a wonderful day it is, isn't it?", dict(seed=22, exaggeration=0.9, cfg_weight=0.3)),
        ("strongcfg", "We are going to the park.", dict(seed=23, cfg_weight=1.0, temperature=1.0)),
        ("topp", "Please call Stella; ask her to bring these things.", dict(seed=24, top_p=0.9, min_p=0.0, repetition_penalty=1.5)),
    ]
    for name, tx, kw in cases:
        e, m = run(tx, **kw)
        save_file(e, os.path.join(a.out, f"e2e_{name}.safetensors"), metadata=m)
        print(f"e2e {name}: {e['t3_tokens'].size} T3 steps, {e['audio'].size} samples")
    save_file(clone, os.path.join(a.out, "clone.safetensors"))
    json.dump({"cases": [c[0] for c in cases]}, open(os.path.join(a.out, "cases.json"), "w"))


if __name__ == "__main__":
    main()
