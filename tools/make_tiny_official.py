#!/usr/bin/env python3
"""Tiny Qwen3-TTS-12Hz model with RANDOM weights, built by the OFFICIAL `qwen_tts` PyTorch classes.

Unlike tools/make_tiny_model.py (which writes tensors by hand), every module here — talker, code predictor,
speaker encoder (ECAPA-TDNN), speech tokenizer encoder (Mimi) and decoder — is instantiated from the official
code with a scaled-down config and saved with `save_pretrained`. The directory therefore has exactly the
layout and tensor names of the real checkpoint, and `Qwen3TTSModel.from_pretrained(dir)` loads it, so vox can be
checked against the reference implementation without downloading gigabytes.

    pip install torch qwen-tts==0.1.1            # reference implementation
    python tools/make_tiny_official.py --vocab qwen2.gguf --out /tmp/tiny-official

Audio is noise, but every stage runs the same math as the 1.7B model. Frame size stays 1920 samples
(12.5 Hz at 24 kHz) like the real model; only channel widths and depths are reduced.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))

TALKER = dict(vocab_size=1088, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, rope_theta=10000.0, num_code_groups=4,
              text_hidden_size=48, text_vocab_size=151936, codec_eos_token_id=1000, codec_think_id=1003,
              codec_nothink_id=1004, codec_think_bos_id=1005, codec_think_eos_id=1006, codec_pad_id=1001,
              codec_bos_id=1002, max_position_embeddings=4096,
              codec_language_id={"english": 1010, "russian": 1011, "chinese": 1012}, spk_id={}, spk_is_dialect={},
              rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "interleaved": True})
PRED = dict(vocab_size=64, hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, rope_theta=10000.0, num_code_groups=4,
            max_position_embeddings=64)
SPK = dict(mel_dim=128, enc_dim=64, enc_channels=[32, 32, 32, 32, 96], enc_kernel_sizes=[5, 3, 3, 3, 1],
           enc_dilations=[1, 2, 3, 4, 1], enc_attention_channels=16, enc_res2net_scale=4, enc_se_channels=16,
           sample_rate=24000)
# Mimi encoder: same strides as the real model (8*6*5*4 = 960, then x2 downsample = 1920), narrow channels.
ENC = dict(sampling_rate=24000, frame_rate=12.5, audio_channels=1, hidden_size=32, num_filters=4, num_residual_layers=1,
           upsampling_ratios=[8, 6, 5, 4], kernel_size=7, last_kernel_size=3, residual_kernel_size=3,
           dilation_growth_rate=2, use_causal_conv=True, pad_mode="constant", compress=2, trim_right_ratio=1.0,
           codebook_size=64, codebook_dim=16, num_quantizers=8, use_conv_shortcut=False,
           vector_quantization_hidden_dimension=16, num_semantic_quantizers=1, upsample_groups=32,
           num_hidden_layers=2, intermediate_size=64, num_attention_heads=4, num_key_value_heads=4, head_dim=8,
           hidden_act="gelu", max_position_embeddings=8000, norm_eps=1e-5, use_cache=False, use_streaming=False,
           rope_theta=10000.0, sliding_window=250, layer_scale_initial_scale=0.01)
DEC = dict(codebook_size=64, hidden_size=32, latent_dim=32, num_attention_heads=4, num_key_value_heads=4, head_dim=8,
           sliding_window=72, intermediate_size=64, layer_scale_initial_scale=0.5, rms_norm_eps=1e-5,
           num_hidden_layers=2, num_quantizers=4, upsample_rates=[8, 5, 4, 3], upsampling_ratios=[2, 2],
           decoder_dim=16, codebook_dim=32, rope_theta=10000.0, max_position_embeddings=8000)


def reinit(module, gen, scale_1d=0.1):
    """Replace the default (near-zero) init with weights that make every stage numerically meaningful."""
    import torch
    with torch.no_grad():
        for name, p in module.named_parameters():
            lname = name.lower()
            if "cluster_usage" in lname:
                p.copy_(torch.rand(p.shape, generator=gen) * 1.5 + 0.5)
            elif "embedding_sum" in lname or "embed_sum" in lname:
                p.copy_(torch.randn(p.shape, generator=gen))
            elif p.dim() >= 2:
                fan_in = int(np.prod(p.shape[1:]))
                p.copy_(torch.randn(p.shape, generator=gen) / np.sqrt(max(1, fan_in)))
            elif "alpha" in lname or "beta" in lname:          # SnakeBeta, log scale
                p.copy_(torch.randn(p.shape, generator=gen) * 0.3)
            elif "scale" in lname:                              # layer scale
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
            elif "norm" in lname and name.endswith("weight"):
                p.copy_(1 + scale_1d * torch.randn(p.shape, generator=gen))
            else:                                               # biases, misc
                p.copy_(scale_1d * torch.randn(p.shape, generator=gen))
        for name, b in module.named_buffers():
            if "cluster_usage" in name:
                b.copy_(torch.rand(b.shape, generator=gen) * 1.5 + 0.5)
            elif "embed_sum" in name or "embedding_sum" in name:
                b.copy_(torch.randn(b.shape, generator=gen))
            elif name.endswith("initialized"):
                b.fill_(1)


def write_text_tokenizer(out, gguf):
    from tok_gguf import read_gguf_kv
    kv = read_gguf_kv(gguf)
    toks, merges = kv["tokenizer.ggml.tokens"], kv["tokenizer.ggml.merges"]
    json.dump({t: i for i, t in enumerate(toks)}, open(os.path.join(out, "vocab.json"), "w"), ensure_ascii=False)
    open(os.path.join(out, "merges.txt"), "w", encoding="utf-8").write("#version: 0.2\n" + "\n".join(merges) + "\n")
    added = {str(i): {"content": toks[i], "lstrip": False, "normalized": False, "rstrip": False, "single_word": False,
                      "special": True} for i in (151643, 151644, 151645)}
    json.dump({"added_tokens_decoder": added, "tokenizer_class": "Qwen2Tokenizer", "model_max_length": 32768,
               "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "bos_token": None, "unk_token": None,
               "clean_up_tokenization_spaces": False, "errors": "replace", "split_special_tokens": False},
              open(os.path.join(out, "tokenizer_config.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({"processor_class": "Qwen3TTSProcessor"}, open(os.path.join(out, "processor_config.json"), "w"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", required=True, help="Qwen2 vocab-only gguf (llama.cpp models/ggml-vocab-qwen2.gguf)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", choices=["base", "custom_voice", "voice_design"], default="base",
                    help="tts_model_type of the tiny model (custom_voice adds built-in speakers incl. dialect ones)")
    a = ap.parse_args()

    import torch
    from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

    gen = torch.Generator().manual_seed(a.seed)
    torch.manual_seed(a.seed)
    os.makedirs(os.path.join(a.out, "speech_tokenizer"), exist_ok=True)

    talker = dict(TALKER)
    if a.variant == "custom_voice":
        talker["codec_language_id"] = dict(TALKER["codec_language_id"], beijing_dialect=1013, sichuan_dialect=1014)
        talker["spk_id"] = {"aiden": 1020, "vivian": 1021, "dylan": 1022, "eric": 1023}
        talker["spk_is_dialect"] = {"aiden": False, "vivian": False, "dylan": "beijing_dialect", "eric": "sichuan_dialect"}
    cfg = Qwen3TTSConfig(talker_config=dict(talker, code_predictor_config=PRED), speaker_encoder_config=SPK,
                         tokenizer_type="qwen3_tts_tokenizer_12hz", tts_model_size="tiny", tts_model_type=a.variant)
    cfg.architectures = ["Qwen3TTSForConditionalGeneration"]
    model = Qwen3TTSForConditionalGeneration(cfg).float().eval()
    reinit(model, gen)
    model.save_pretrained(a.out, safe_serialization=True)
    # Qwen3TTSSpeakerEncoderConfig is not a PretrainedConfig subclass in spirit (its __init__ takes no **kwargs),
    # so keep only its own fields — exactly like the real checkpoint's config.json.
    cp = os.path.join(a.out, "config.json")
    cj = json.load(open(cp))
    cj["speaker_encoder_config"] = SPK
    json.dump(cj, open(cp, "w"), indent=2)

    tcfg = Qwen3TTSTokenizerV2Config(encoder_config=ENC, decoder_config=DEC, encoder_valid_num_quantizers=4,
                                     input_sample_rate=24000, output_sample_rate=24000,
                                     decode_upsample_rate=1920, encode_downsample_rate=1920)
    tcfg.architectures = ["Qwen3TTSTokenizerV2Model"]
    tok = Qwen3TTSTokenizerV2Model(tcfg).float().eval()
    reinit(tok, gen)
    # Random weights saturate the output (clamp to [-1, 1]) and make f32 rounding chaotic; scale the last conv so
    # the pre-clamp signal has std ~0.25 — the comparison then measures the implementation, not the conditioning.
    last = tok.decoder.decoder[-1]
    cap = {}
    h = last.register_forward_hook(lambda m, i, o: cap.__setitem__("y", o.detach()))
    with torch.no_grad():
        codes = torch.randint(0, DEC["codebook_size"], (1, DEC["num_quantizers"], 24), generator=gen)
        tok.decoder(codes)
        k = 0.25 / float(cap["y"].std().clamp(min=1e-6))
        for prm in last.parameters():
            prm.mul_(k)
    h.remove()
    tok.save_pretrained(os.path.join(a.out, "speech_tokenizer"), safe_serialization=True)
    json.dump({"feature_extractor_type": "EncodecFeatureExtractor", "feature_size": 1, "padding_side": "right",
               "padding_value": 0.0, "return_attention_mask": True, "sampling_rate": 24000,
               "chunk_length_s": None, "overlap": None},
              open(os.path.join(a.out, "speech_tokenizer", "preprocessor_config.json"), "w"), indent=1)

    json.dump({"do_sample": True, "temperature": 0.9, "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.05,
               "subtalker_dosample": True, "subtalker_temperature": 0.9, "subtalker_top_k": 50, "subtalker_top_p": 1.0,
               "max_new_tokens": 48}, open(os.path.join(a.out, "generate_config.json"), "w"), indent=1)
    write_text_tokenizer(a.out, a.vocab)
    print("tiny official model written to", a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
