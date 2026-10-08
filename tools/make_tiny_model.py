#!/usr/bin/env python3
"""Крошечная модель Qwen3-TTS со СЛУЧАЙНЫМИ весами в формате настоящего HF-чекпойнта (те же имена тензоров,
те же конфиги, реальный токенизатор Qwen2). Нужна для проверки конвертера и бэкенда без скачивания
гигабайтов: звук — шум, но математика всех стадий (talker, code predictor, декодер кодека) та же.

    python tools/make_tiny_model.py --vocab /tmp/vocab/qwen2.gguf --out /tmp/tiny-qwen3tts
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from vox.formats import save_safetensors  # noqa: E402

T = dict(vocab_size=1100, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
         num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, rope_theta=10000.0, num_code_groups=4,
         text_hidden_size=48, text_vocab_size=151936, codec_eos_token_id=1000, codec_think_id=1003,
         codec_nothink_id=1004, codec_think_bos_id=1005, codec_think_eos_id=1006, codec_pad_id=1001, codec_bos_id=1002)
P = dict(vocab_size=80, hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
         num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, rope_theta=10000.0, num_code_groups=4)
C = dict(codebook_size=80, hidden_size=32, latent_dim=24, num_attention_heads=4, num_key_value_heads=4, head_dim=8,
         sliding_window=5, intermediate_size=64, layer_scale_initial_scale=0.5, rms_norm_eps=1e-5, num_hidden_layers=2,
         num_quantizers=4, upsample_rates=[2, 2], upsampling_ratios=[2, 2], decoder_dim=16, codebook_dim=32,
         rope_theta=10000.0)
LANGS = {"english": 1010, "russian": 1011, "chinese": 1012}


def weights(rng):
    w = {}

    def lin(name, o, i, bias=False, gain=1.0):
        w[name + ".weight"] = (rng.normal(0, gain / np.sqrt(i), (o, i))).astype(np.float32)
        if bias:
            w[name + ".bias"] = rng.normal(0, 0.1, o).astype(np.float32)

    def norm(name, n):
        w[name + ".weight"] = (1 + 0.1 * rng.normal(size=n)).astype(np.float32)

    def xfmr(prefix, H, I, L, nh, nkv, hd, qk, ls=False):
        for i in range(L):
            p = f"{prefix}layers.{i}."
            lin(p + "self_attn.q_proj", nh * hd, H); lin(p + "self_attn.k_proj", nkv * hd, H)
            lin(p + "self_attn.v_proj", nkv * hd, H); lin(p + "self_attn.o_proj", H, nh * hd)
            if qk:
                norm(p + "self_attn.q_norm", hd); norm(p + "self_attn.k_norm", hd)
            lin(p + "mlp.gate_proj", I, H); lin(p + "mlp.up_proj", I, H); lin(p + "mlp.down_proj", H, I)
            norm(p + "input_layernorm", H); norm(p + "post_attention_layernorm", H)
            if ls:
                w[p + "self_attn_layer_scale.scale"] = rng.normal(0, 0.5, H).astype(np.float32)
                w[p + "mlp_layer_scale.scale"] = rng.normal(0, 0.5, H).astype(np.float32)
        norm(prefix + "norm", H)

    D, PH, Q = T["hidden_size"], P["hidden_size"], T["num_code_groups"]
    w["talker.model.codec_embedding.weight"] = rng.normal(0, 1, (T["vocab_size"], D)).astype(np.float32)
    w["talker.model.text_embedding.weight"] = rng.normal(0, 1, (T["text_vocab_size"], T["text_hidden_size"])).astype(np.float32)
    xfmr("talker.model.", D, T["intermediate_size"], T["num_hidden_layers"], T["num_attention_heads"],
         T["num_key_value_heads"], T["head_dim"], True)
    lin("talker.text_projection.linear_fc1", T["text_hidden_size"], T["text_hidden_size"], True)
    lin("talker.text_projection.linear_fc2", D, T["text_hidden_size"], True)
    lin("talker.codec_head", T["vocab_size"], D, gain=2.0)
    xfmr("talker.code_predictor.model.", PH, P["intermediate_size"], P["num_hidden_layers"], P["num_attention_heads"],
         P["num_key_value_heads"], P["head_dim"], True)
    for j in range(Q - 1):
        w[f"talker.code_predictor.model.codec_embedding.{j}.weight"] = rng.normal(0, 1, (P["vocab_size"], D)).astype(np.float32)
        lin(f"talker.code_predictor.lm_head.{j}", P["vocab_size"], PH, gain=2.0)
    lin("talker.code_predictor.small_to_mtp_projection", PH, D, True)
    # ---- кодек ----
    c = {}
    d, cd, lat, hid = C["codebook_dim"] // 2, C["codebook_dim"], C["latent_dim"], C["hidden_size"]

    def clin(name, o, i, bias=False):
        c[name + ".weight"] = rng.normal(0, 1 / np.sqrt(i), (o, i)).astype(np.float32)
        if bias:
            c[name + ".bias"] = rng.normal(0, 0.1, o).astype(np.float32)

    def cconv(name, co, ci, k, gain=1.0):
        c[name + ".weight"] = rng.normal(0, gain / np.sqrt(ci * k), (co, ci, k)).astype(np.float32)
        c[name + ".bias"] = rng.normal(0, 0.1, co).astype(np.float32)

    def snake(name, n):
        c[name + ".alpha"] = rng.normal(0, 0.3, n).astype(np.float32)
        c[name + ".beta"] = rng.normal(0, 0.3, n).astype(np.float32)

    for grp, n in (("rvq_first", 1), ("rvq_rest", Q - 1)):
        c[f"quantizer.{grp}.output_proj.weight"] = rng.normal(0, 1 / np.sqrt(d), (cd, d, 1)).astype(np.float32)
        c[f"quantizer.{grp}.input_proj.weight"] = rng.normal(0, 1, (d, cd, 1)).astype(np.float32)
        for i in range(n):
            c[f"quantizer.{grp}.vq.layers.{i}._codebook.cluster_usage"] = rng.uniform(0.5, 3, C["codebook_size"]).astype(np.float32)
            c[f"quantizer.{grp}.vq.layers.{i}._codebook.embedding_sum"] = rng.normal(0, 2, (C["codebook_size"], d)).astype(np.float32)
    cconv("pre_conv.conv", lat, cd, 3)
    clin("pre_transformer.input_proj", hid, lat, True); clin("pre_transformer.output_proj", lat, hid, True)
    for i in range(C["num_hidden_layers"]):
        p = f"pre_transformer.layers.{i}."
        nh, hd = C["num_attention_heads"], C["head_dim"]
        for n in ("q", "k", "v"):
            clin(p + f"self_attn.{n}_proj", nh * hd, hid)
        clin(p + "self_attn.o_proj", hid, nh * hd)
        clin(p + "mlp.gate_proj", C["intermediate_size"], hid); clin(p + "mlp.up_proj", C["intermediate_size"], hid)
        clin(p + "mlp.down_proj", hid, C["intermediate_size"])
        for n in ("input_layernorm", "post_attention_layernorm"):
            c[p + n + ".weight"] = (1 + 0.1 * rng.normal(size=hid)).astype(np.float32)
        c[p + "self_attn_layer_scale.scale"] = rng.normal(0, 0.5, hid).astype(np.float32)
        c[p + "mlp_layer_scale.scale"] = rng.normal(0, 0.5, hid).astype(np.float32)
    c["pre_transformer.norm.weight"] = (1 + 0.1 * rng.normal(size=hid)).astype(np.float32)
    for s, r in enumerate(C["upsampling_ratios"]):
        c[f"upsample.{s}.0.conv.weight"] = rng.normal(0, 1 / np.sqrt(lat), (lat, lat, r)).astype(np.float32)
        c[f"upsample.{s}.0.conv.bias"] = rng.normal(0, 0.1, lat).astype(np.float32)
        c[f"upsample.{s}.1.dwconv.conv.weight"] = rng.normal(0, 0.4, (lat, 1, 7)).astype(np.float32)
        c[f"upsample.{s}.1.dwconv.conv.bias"] = rng.normal(0, 0.1, lat).astype(np.float32)
        c[f"upsample.{s}.1.norm.weight"] = (1 + 0.1 * rng.normal(size=lat)).astype(np.float32)
        c[f"upsample.{s}.1.norm.bias"] = rng.normal(0, 0.1, lat).astype(np.float32)
        clin(f"upsample.{s}.1.pwconv1", 4 * lat, lat, True); clin(f"upsample.{s}.1.pwconv2", lat, 4 * lat, True)
        c[f"upsample.{s}.1.gamma"] = rng.normal(0, 0.5, lat).astype(np.float32)
    cconv("decoder.0.conv", C["decoder_dim"], lat, 7)
    dim = C["decoder_dim"]
    for b, r in enumerate(C["upsample_rates"]):
        ind, outd = dim >> b, dim >> (b + 1)
        p = f"decoder.{b + 1}.block."
        snake(p + "0", ind)
        c[p + "1.conv.weight"] = rng.normal(0, 1 / np.sqrt(ind), (ind, outd, 2 * r)).astype(np.float32)
        c[p + "1.conv.bias"] = rng.normal(0, 0.1, outd).astype(np.float32)
        for u in range(3):
            q = p + f"{u + 2}."
            snake(q + "act1", outd); cconv(q + "conv1.conv", outd, outd, 7)
            snake(q + "act2", outd); cconv(q + "conv2.conv", outd, outd, 1)
    nb = len(C["upsample_rates"])
    snake(f"decoder.{nb + 1}", dim >> nb)
    cconv(f"decoder.{nb + 2}.conv", 1, dim >> nb, 7, gain=0.02)
    return w, {"decoder." + k: v for k, v in c.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True, help="qwen2 vocab-only gguf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from tok_gguf import read_gguf_kv
    out = a.out
    os.makedirs(os.path.join(out, "speech_tokenizer"), exist_ok=True)
    rng = np.random.default_rng(a.seed)
    main_w, codec_w = weights(rng)
    save_safetensors(os.path.join(out, "model.safetensors"), main_w)
    save_safetensors(os.path.join(out, "speech_tokenizer", "model.safetensors"), codec_w)
    tcfg = dict(T, code_predictor_config=P, codec_language_id=LANGS)
    json.dump({"model_type": "qwen3_tts", "talker_config": tcfg, "tts_model_type": "base", "tts_model_size": "tiny",
               "tokenizer_type": "12hz", "im_start_token_id": 151644, "im_end_token_id": 151645,
               "tts_pad_token_id": 151671, "tts_bos_token_id": 151672, "tts_eos_token_id": 151673},
              open(os.path.join(out, "config.json"), "w"), indent=1)
    json.dump({"decoder_config": C, "output_sample_rate": 24000, "input_sample_rate": 24000},
              open(os.path.join(out, "speech_tokenizer", "config.json"), "w"), indent=1)
    json.dump({"temperature": 0.9, "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.05, "subtalker_temperature": 0.9,
               "subtalker_top_k": 50, "subtalker_top_p": 1.0, "max_new_tokens": 24},
              open(os.path.join(out, "generate_config.json"), "w"))
    kv = read_gguf_kv(a.vocab)
    toks, merges = kv["tokenizer.ggml.tokens"], kv["tokenizer.ggml.merges"]
    json.dump({t: i for i, t in enumerate(toks)}, open(os.path.join(out, "vocab.json"), "w"), ensure_ascii=False)
    open(os.path.join(out, "merges.txt"), "w", encoding="utf-8").write("#version: 0.2\n" + "\n".join(merges) + "\n")
    added = {str(i): {"content": toks[i], "special": True} for i in (151643, 151644, 151645)}
    json.dump({"added_tokens_decoder": added}, open(os.path.join(out, "tokenizer_config.json"), "w"))
    print("tiny model written to", out, file=sys.stderr)


if __name__ == "__main__":
    main()
