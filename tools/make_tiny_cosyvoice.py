#!/usr/bin/env python3
"""Tiny CosyVoice2 with random weights, built from the official CosyVoice classes, plus reference outputs for
tests/test_cosyvoice_tiny.py.

Writes (--out):
  model/  — the layout of FunAudioLLM/CosyVoice2-0.5B: cosyvoice2.yaml, llm.pt, flow.pt, hift.pt (weight_norm as
            weight_g / weight_v), campplus.onnx (exported from the PyTorch CAM++), speech_tokenizer_v2.pt (state dict of
            s3tokenizer.S3TokenizerV2: the real model ships an .onnx that s3tokenizer.onnx2torch turns into this),
            CosyVoice-BlankEN/ (Qwen2 config, tokenizer, weights);
  prompt.wav (24 kHz) and the reference of every stage for one utterance (dump.safetensors, for `vox verify`):
  the text tokens, the frontend (24 kHz mel, Kaldi fbank, CAM++ x-vector, Whisper log-mel, S3 speech tokens), the LLM
  (log-probabilities of every step on the tokens it sampled), the flow (encoder output, mel after 10 Euler steps with the
  fixed noise of CausalConditionalCFM) and HiFT (F0, the source noise it drew, the waveform);
  e2e_*.safetensors — whole utterances (zero-shot, cross-lingual, instruct2) through CosyVoice2Model.tts with the speech
  tokens the official LLM sampled, so the engine can be fed the same tokens and source noise.

Sizes: the LLM is a 2-layer Qwen2 (hidden 64) with the real vocabulary; the flow encoder keeps its hard-wired 512 channels
(2 + 4 blocks, 256 FFN units); the estimator has 64 channels; HiFT has 64 base channels and the real up-sampling (480x).
Usage: VOX_COSYVOICE_SRC=CosyVoice python tools/make_tiny_cosyvoice.py --vocab qwen2.gguf --out work/cosyvoice"""
import argparse
import json
import os
import sys
import warnings
from functools import partial

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))   # tok_gguf

YAML = """# tiny CosyVoice2 (tools/make_tiny_cosyvoice.py): same structure as the released cosyvoice2.yaml
sample_rate: 24000
llm_input_size: {llm_hidden}
llm_output_size: {llm_hidden}
spk_embed_dim: 192
qwen_pretrain_path: ''
token_frame_rate: 25
token_mel_ratio: 2
chunk_size: 25
num_decoding_left_chunks: -1

llm: !new:cosyvoice.llm.llm.Qwen2LM
    llm_input_size: !ref <llm_input_size>
    llm_output_size: !ref <llm_output_size>
    speech_token_size: 6561
    length_normalized_loss: True
    lsm_weight: 0
    mix_ratio: [5, 15]
    llm: !new:cosyvoice.llm.llm.Qwen2Encoder
        pretrain_path: !ref <qwen_pretrain_path>
    sampling: !name:cosyvoice.utils.common.ras_sampling
        top_p: 0.8
        top_k: 25
        win_size: 10
        tau_r: 0.1

flow: !new:cosyvoice.flow.flow.CausalMaskedDiffWithXvec
    input_size: 512
    output_size: 80
    spk_embed_dim: !ref <spk_embed_dim>
    output_type: 'mel'
    vocab_size: 6561
    input_frame_rate: !ref <token_frame_rate>
    only_mask_loss: True
    token_mel_ratio: !ref <token_mel_ratio>
    pre_lookahead_len: 3
    encoder: !new:cosyvoice.transformer.upsample_encoder.UpsampleConformerEncoder
        output_size: 512
        attention_heads: 8
        linear_units: {enc_units}
        num_blocks: {enc_blocks}
        dropout_rate: 0.1
        positional_dropout_rate: 0.1
        attention_dropout_rate: 0.1
        normalize_before: True
        input_layer: 'linear'
        pos_enc_layer_type: 'rel_pos_espnet'
        selfattention_layer_type: 'rel_selfattn'
        input_size: 512
        use_cnn_module: False
        macaron_style: False
        static_chunk_size: !ref <chunk_size>
    decoder: !new:cosyvoice.flow.flow_matching.CausalConditionalCFM
        in_channels: 240
        n_spks: 1
        spk_emb_dim: 80
        cfm_params: !new:omegaconf.DictConfig
            content:
                sigma_min: 1e-06
                solver: 'euler'
                t_scheduler: 'cosine'
                training_cfg_rate: 0.2
                inference_cfg_rate: 0.7
                reg_loss_type: 'l1'
        estimator: !new:cosyvoice.flow.decoder.CausalConditionalDecoder
            in_channels: 320
            out_channels: 80
            channels: [{est_ch}]
            dropout: 0.0
            attention_head_dim: {est_head_dim}
            n_blocks: {est_blocks}
            num_mid_blocks: {est_mid}
            num_heads: {est_heads}
            act_fn: 'gelu'
            static_chunk_size: !ref <chunk_size> * <token_mel_ratio>
            num_decoding_left_chunks: !ref <num_decoding_left_chunks>

hift: !new:cosyvoice.hifigan.generator.HiFTGenerator
    in_channels: 80
    base_channels: {hift_ch}
    nb_harmonics: 8
    sampling_rate: !ref <sample_rate>
    nsf_alpha: 0.1
    nsf_sigma: 0.003
    nsf_voiced_threshold: 10
    upsample_rates: [8, 5, 3]
    upsample_kernel_sizes: [16, 11, 7]
    istft_params:
        n_fft: 16
        hop_len: 4
    resblock_kernel_sizes: [3, 7, 11]
    resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    source_resblock_kernel_sizes: [7, 7, 11]
    source_resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    lrelu_slope: 0.1
    audio_limit: 0.99
    f0_predictor: !new:cosyvoice.hifigan.f0_predictor.ConvRNNF0Predictor
        num_class: 1
        in_channels: 80
        cond_channels: {hift_ch}

feat_extractor: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1920
    num_mels: 80
    sampling_rate: !ref <sample_rate>
    hop_size: 480
    win_size: 1920
    fmin: 0
    fmax: 8000
    center: False

get_tokenizer: !name:cosyvoice.tokenizer.tokenizer.get_qwen_tokenizer
    token_path: !ref <qwen_pretrain_path>
    skip_special_tokens: True
allowed_special: 'all'
"""


class NoiseTap:
    """Records torch.rand / torch.randn_like calls of the HiFT source module (SineGen2 draws rand_ini [1, 9], then
    randn_like [1, L, 9]; SourceModuleHnNSF then draws an unused noise [1, L, 1])."""

    def __init__(self, torch):
        self.torch = torch
        self.calls = []

    def __enter__(self):
        t = self.torch
        self.orig = (t.rand, t.randn_like)
        tap = self

        def rand(*a, **k):
            r = tap.orig[0](*a, **k)
            tap.calls.append(r.detach().clone())
            return r

        def randn_like(x, *a, **k):
            r = tap.orig[1](x, *a, **k)
            tap.calls.append(r.detach().clone())
            return r
        t.rand, t.randn_like = rand, randn_like
        return self

    def __exit__(self, *e):
        self.torch.rand, self.torch.randn_like = self.orig
        return False

    def tensors(self, prefix=""):
        assert len(self.calls) >= 2, len(self.calls)
        return {prefix + "rand_ini": np.ascontiguousarray(self.calls[0].reshape(-1).numpy(), dtype=np.float32),
                prefix + "sine_noise": np.ascontiguousarray(self.calls[1][0].contiguous().numpy(), dtype=np.float32)}


def write_text_tokenizer(out, gguf):
    from tok_gguf import read_gguf_kv
    kv = read_gguf_kv(gguf)
    toks, merges = kv["tokenizer.ggml.tokens"], kv["tokenizer.ggml.merges"]
    # the regular vocabulary is 151643 tokens; <|endoftext|>, <|im_start|>, <|im_end|> are added tokens (so CosyVoice's own
    # specials start at 151646, as in CosyVoice-BlankEN: <|endofprompt|> = 151646)
    json.dump({t: i for i, t in enumerate(toks[:151643])}, open(os.path.join(out, "vocab.json"), "w"), ensure_ascii=False)
    open(os.path.join(out, "merges.txt"), "w", encoding="utf-8").write("#version: 0.2\n" + "\n".join(merges) + "\n")
    added = {str(i): {"content": toks[i], "lstrip": False, "normalized": False, "rstrip": False, "single_word": False,
                      "special": True} for i in (151643, 151644, 151645)}
    json.dump({"added_tokens_decoder": added, "tokenizer_class": "Qwen2Tokenizer", "model_max_length": 32768,
               "eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>", "bos_token": None, "unk_token": None,
               "clean_up_tokenization_spaces": False, "errors": "replace", "split_special_tokens": False},
              open(os.path.join(out, "tokenizer_config.json"), "w"), ensure_ascii=False, indent=1)


def to_weight_g_v(sd):
    """parametrizations.weight.original0/1 → weight_g / weight_v (the layout of the released hift.pt)."""
    out = {}
    for k, v in sd.items():
        if k.endswith(".parametrizations.weight.original0"):
            out[k.replace(".parametrizations.weight.original0", ".weight_g")] = v
        elif k.endswith(".parametrizations.weight.original1"):
            out[k.replace(".parametrizations.weight.original1", ".weight_v")] = v
        else:
            out[k] = v
    return out


def matcha_mel(torch, y):
    """matcha.utils.audio.mel_spectrogram with the parameters of cosyvoice2.yaml (feat_extractor)."""
    from matcha.utils.audio import mel_spectrogram
    return mel_spectrogram(y, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920, fmin=0, fmax=8000,
                           center=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", required=True, help="Qwen2 vocab-only gguf (llama.cpp models/ggml-vocab-qwen2.gguf)")
    ap.add_argument("--src", default="", help="CosyVoice clone (default: $VOX_COSYVOICE_SRC)")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--full", action="store_true", help="the real dimensions of CosyVoice2-0.5B (random weights, for speed and "
                                                         "memory measurements); no reference dumps")
    a = ap.parse_args()
    import cosyvoice_ref
    if not cosyvoice_ref.setup(a.src or None):
        raise SystemExit("set VOX_COSYVOICE_SRC to a clone of https://github.com/FunAudioLLM/CosyVoice (with third_party/Matcha-TTS)")

    import torch
    import torchaudio
    from omegaconf import DictConfig
    from safetensors.numpy import save_file
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from cosyvoice.llm.llm import Qwen2LM, Qwen2Encoder
    from cosyvoice.utils.common import ras_sampling
    from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
    from cosyvoice.transformer.upsample_encoder import UpsampleConformerEncoder
    from cosyvoice.flow.flow_matching import CausalConditionalCFM
    from cosyvoice.flow.decoder import CausalConditionalDecoder
    from cosyvoice.hifigan.generator import HiFTGenerator
    from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor
    from cosyvoice.cli.model import CosyVoice2Model
    from cosyvoice.tokenizer.tokenizer import CosyVoice2Tokenizer
    import torchaudio.compliance.kaldi as kaldi

    torch.manual_seed(a.seed)
    md = os.path.join(a.out, "model")
    blank = os.path.join(md, "CosyVoice-BlankEN")
    os.makedirs(blank, exist_ok=True)
    sz = dict(llm_hidden=64, enc_units=256, enc_blocks=2, est_ch=64, est_head_dim=32, est_blocks=1, est_mid=2, est_heads=2, hift_ch=64)
    qdim = dict(intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    s3dim = dict(n_audio_state=128, n_audio_head=2, n_audio_layer=2)
    if a.full:
        sz = dict(llm_hidden=896, enc_units=2048, enc_blocks=6, est_ch=256, est_head_dim=64, est_blocks=4, est_mid=12, est_heads=8, hift_ch=512)
        qdim = dict(intermediate_size=4864, num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2)
        s3dim = dict(n_audio_state=1280, n_audio_head=20, n_audio_layer=6)
    with open(os.path.join(md, "cosyvoice2.yaml"), "w") as f:
        f.write(YAML.format(**sz))

    # ---- Qwen2 backbone (CosyVoice-BlankEN) + tokenizer ----
    qcfg = Qwen2Config(vocab_size=151936, hidden_size=sz["llm_hidden"], **qdim, max_position_embeddings=32768, rope_theta=1000000.0,
                       rms_norm_eps=1e-6, tie_word_embeddings=True, use_sliding_window=False, torch_dtype="float32")
    qwen = Qwen2ForCausalLM(qcfg)
    with torch.no_grad():
        for n, p in qwen.named_parameters():
            if "norm" in n:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif "bias" in n:
                p.copy_(0.05 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * (0.5 / p.shape[-1] ** 0.5) if p.dim() > 1 else p)
    qwen.save_pretrained(blank, safe_serialization=True)
    write_text_tokenizer(blank, a.vocab)
    tok = CosyVoice2Tokenizer(blank, skip_special_tokens=True)

    llm = Qwen2LM(llm_input_size=sz["llm_hidden"], llm_output_size=sz["llm_hidden"], speech_token_size=6561,
                  llm=Qwen2Encoder(blank), sampling=partial(ras_sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1),
                  length_normalized_loss=True, lsm_weight=0, mix_ratio=[5, 15])
    with torch.no_grad():
        llm.llm_embedding.weight.normal_(0, 0.5)
        llm.speech_embedding.weight.normal_(0, 0.5)
        llm.llm_decoder.weight.normal_(0, 1.0 / sz["llm_hidden"] ** 0.5)
        llm.llm_decoder.bias.zero_()
        llm.llm_decoder.bias[6561] = -2.0   # end of speech: rare, so utterances are a few dozen tokens long
    llm.eval()

    # ---- flow ----
    enc = UpsampleConformerEncoder(input_size=512, output_size=512, attention_heads=8, linear_units=sz["enc_units"],
                                   num_blocks=sz["enc_blocks"], dropout_rate=0.1, positional_dropout_rate=0.1,
                                   attention_dropout_rate=0.1, normalize_before=True, input_layer="linear",
                                   pos_enc_layer_type="rel_pos_espnet", selfattention_layer_type="rel_selfattn",
                                   use_cnn_module=False, macaron_style=False, static_chunk_size=25)
    est = CausalConditionalDecoder(in_channels=320, out_channels=80, channels=[sz["est_ch"]], dropout=0.0,
                                   attention_head_dim=sz["est_head_dim"], n_blocks=sz["est_blocks"], num_mid_blocks=sz["est_mid"],
                                   num_heads=sz["est_heads"], act_fn="gelu", static_chunk_size=50, num_decoding_left_chunks=-1)
    cfm = CausalConditionalCFM(in_channels=240, n_spks=1, spk_emb_dim=80,
                               cfm_params=DictConfig({"sigma_min": 1e-06, "solver": "euler", "t_scheduler": "cosine",
                                                      "training_cfg_rate": 0.2, "inference_cfg_rate": 0.7, "reg_loss_type": "l1"}),
                               estimator=est)
    flow = CausalMaskedDiffWithXvec(input_size=512, output_size=80, spk_embed_dim=192, output_type="mel", vocab_size=6561,
                                    input_frame_rate=25, only_mask_loss=True, token_mel_ratio=2, pre_lookahead_len=3,
                                    encoder=enc, decoder=cfm)
    g = torch.Generator().manual_seed(a.seed + 1)
    with torch.no_grad():   # random but well-scaled: LayerNorm affine ≠ identity, biases ≠ 0
        for n, p in flow.named_parameters():
            if p.dim() == 1 and ("norm" in n or n.endswith(".weight")):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif p.dim() == 1:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
            elif "pos_bias" in n:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                fan_in = int(np.prod(p.shape[1:]))
                p.copy_(torch.randn(p.shape, generator=g) * (1.0 / fan_in ** 0.5))
        flow.input_embedding.weight.mul_(1.0)
    flow.eval()

    # ---- HiFT ----
    hift = HiFTGenerator(in_channels=80, base_channels=sz["hift_ch"], nb_harmonics=8, sampling_rate=24000, nsf_alpha=0.1,
                         nsf_sigma=0.003, nsf_voiced_threshold=10, upsample_rates=[8, 5, 3], upsample_kernel_sizes=[16, 11, 7],
                         istft_params={"n_fft": 16, "hop_len": 4}, resblock_kernel_sizes=[3, 7, 11],
                         resblock_dilation_sizes=[[1, 3, 5]] * 3, source_resblock_kernel_sizes=[7, 7, 11],
                         source_resblock_dilation_sizes=[[1, 3, 5]] * 3, lrelu_slope=0.1, audio_limit=0.99,
                         f0_predictor=ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=sz["hift_ch"]))
    with torch.no_grad():
        for n, p in hift.named_parameters():
            if n.endswith("alpha"):
                p.copy_(0.5 + torch.rand(p.shape, generator=g))
            elif "original0" in n:          # weight_norm magnitudes
                p.copy_(0.5 + 0.5 * torch.rand(p.shape, generator=g))
            elif p.dim() == 1:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
        # F0 around 100 Hz with unvoiced (< 10 Hz) stretches, magnitudes of the spectrum around 1
        hift.f0_predictor.classifier.bias.fill_(25.0)
        hift.f0_predictor.classifier.weight.mul_(150.0)
        hift.conv_post.parametrizations.weight.original0.mul_(0.3)
    hift.eval()

    torch.save(llm.state_dict(), os.path.join(md, "llm.pt"))
    torch.save(flow.state_dict(), os.path.join(md, "flow.pt"))
    torch.save(to_weight_g_v(hift.state_dict()), os.path.join(md, "hift.pt"))

    # ---- speaker encoder (CAM++, exported to ONNX like the released campplus.onnx) ----
    from campplus_ref import CAMPPlus   # 3D-Speaker's CAM++ (copied from chatterbox-tts)
    camp = CAMPPlus(feat_dim=80, embedding_size=192, memory_efficient=False)
    with torch.no_grad():
        for m in camp.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.running_mean.copy_(0.1 * torch.randn(m.running_mean.shape, generator=g))
                m.running_var.copy_(0.5 + torch.rand(m.running_var.shape, generator=g))
                if m.affine:
                    m.weight.copy_(1.0 + 0.1 * torch.randn(m.weight.shape, generator=g))
                    m.bias.copy_(0.1 * torch.randn(m.bias.shape, generator=g))
            elif isinstance(m, torch.nn.PReLU):
                m.weight.copy_(0.25 + 0.05 * torch.randn(m.weight.shape, generator=g))
    camp.eval()
    feats = torch.randn(1, 200, 80)
    torch.onnx.export(camp, (feats,), os.path.join(md, "campplus.onnx"), input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {1: "T"}}, opset_version=14, dynamo=False)
    torch.save(camp.state_dict(), os.path.join(a.out, "campplus_torch.pt"))

    # ---- speech tokenizer (S3TokenizerV2, tiny) ----
    from s3tokenizer.model_v2 import S3TokenizerV2, ModelConfig
    s3 = S3TokenizerV2("speech_tokenizer_v2_25hz", ModelConfig(n_mels=128, n_audio_ctx=1500, **s3dim, n_codebook_size=3 ** 8))
    with torch.no_grad():
        for n, p in s3.named_parameters():
            if p.dim() == 1 and n.endswith(".weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif p.dim() == 1:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
            else:
                fan_in = int(np.prod(p.shape[1:]))
                p.copy_(torch.randn(p.shape, generator=g) * (1.5 / fan_in ** 0.5))
    s3.eval()
    torch.save(s3.state_dict(), os.path.join(md, "speech_tokenizer_v2.pt"))

    # ---- prompt recording: voiced and unvoiced stretches, 24 kHz ----
    sr = 24000
    t = np.arange(int(2.3 * sr)) / sr
    f0 = 120 + 30 * np.sin(2 * np.pi * 1.3 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    env = (np.sin(2 * np.pi * 2.1 * t) > -0.3).astype(np.float64)
    wav = 0.25 * env * (np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)) + 0.01 * np.random.RandomState(1).randn(t.size)
    pcm16 = np.clip(np.round(wav * 32768), -32768, 32767).astype(np.int16)
    import wave
    with wave.open(os.path.join(a.out, "prompt.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm16.tobytes())
    prompt24 = torch.from_numpy(pcm16.astype(np.float32) / 32768.0).unsqueeze(0)
    prompt16 = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)(prompt24)
    prompt_text = "Hello there, this is my voice."

    # ---- frontend (CosyVoiceFrontEnd._extract_*) ----
    feat = matcha_mel(torch, prompt24).squeeze(0).transpose(0, 1)            # [T, 80]
    fb = kaldi.fbank(prompt16, num_mel_bins=80, dither=0, sample_frequency=16000)
    fb = fb - fb.mean(dim=0, keepdim=True)
    import onnxruntime
    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = 1
    sess = onnxruntime.InferenceSession(os.path.join(md, "campplus.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    xvec = sess.run(None, {sess.get_inputs()[0].name: fb.unsqueeze(0).numpy()})[0].flatten()
    from s3tokenizer.utils import log_mel_spectrogram
    wmel = log_mel_spectrogram(prompt16.squeeze(0), n_mels=128)              # [128, T]
    with torch.no_grad():
        codes, _ = s3.quantize(wmel.unsqueeze(0), torch.tensor([wmel.shape[1]]))
    speech_tok = codes[0].numpy().astype(np.int64)
    token_len = min(feat.shape[0] // 2, speech_tok.size)
    feat_c, tok_c = feat[: 2 * token_len], speech_tok[:token_len]

    if a.full:
        print(f"{a.out}: CosyVoice2-0.5B dimensions, random weights (no reference dumps)")
        return
    model = CosyVoice2Model(llm, flow, hift, fp16=False)
    dumps = {}

    def text_tokens(s):
        return np.array(tok.encode(s), dtype=np.int64)

    def run_case(name, text, prompt_text_s, llm_prompt, instruct=False, seed=11):
        """Official LLM sampling + teacher-forced log-probs + flow + HiFT; returns the dump."""
        tt = text_tokens(text)
        pt = text_tokens(prompt_text_s) if prompt_text_s else np.zeros(0, np.int64)
        lp_tok = tok_c if llm_prompt else np.zeros(0, np.int64)
        xv = torch.from_numpy(xvec).float().unsqueeze(0)
        torch.manual_seed(seed)
        out = list(llm.inference(text=torch.from_numpy(tt).unsqueeze(0).int(), text_len=torch.tensor([tt.size], dtype=torch.int32),
                                 prompt_text=torch.from_numpy(pt).unsqueeze(0).int(), prompt_text_len=torch.tensor([pt.size], dtype=torch.int32),
                                 prompt_speech_token=torch.from_numpy(lp_tok).unsqueeze(0).int(),
                                 prompt_speech_token_len=torch.tensor([lp_tok.size], dtype=torch.int32), embedding=xv))
        gen = np.array(out, dtype=np.int64)
        # teacher-forced log-probs of every step (the inputs exactly as Qwen2LM.inference builds them)
        with torch.no_grad():
            emb = llm.llm.model.model.embed_tokens(torch.from_numpy(np.concatenate([pt, tt])).unsqueeze(0))
            sos = llm.llm_embedding.weight[0].reshape(1, 1, -1)
            task = llm.llm_embedding.weight[1].reshape(1, 1, -1)
            sp = llm.speech_embedding(torch.from_numpy(np.concatenate([lp_tok, gen])).unsqueeze(0)) if (lp_tok.size + gen.size) else torch.zeros(1, 0, sz["llm_hidden"])
            x = torch.cat([sos, emb, task, sp], dim=1)
            hs = llm.llm.model(inputs_embeds=x, output_hidden_states=True, return_dict=True).hidden_states[-1]
            first = 1 + emb.shape[1] + 1 + lp_tok.size - 1
            logp = llm.llm_decoder(hs[0, first:]).log_softmax(-1)              # [gen + 1, 6564]
            # flow + HiFT (CosyVoice2Model.token2wav, non-streaming)
            torch.manual_seed(seed + 100)
            h_tok = torch.from_numpy(np.concatenate([tok_c, gen])).unsqueeze(0).int()
            mel_flow, _ = flow.inference(token=torch.from_numpy(gen).unsqueeze(0).int(), token_len=torch.tensor([gen.size], dtype=torch.int32),
                                         prompt_token=torch.from_numpy(tok_c).unsqueeze(0).int(),
                                         prompt_token_len=torch.tensor([tok_c.size], dtype=torch.int32),
                                         prompt_feat=feat_c.unsqueeze(0), prompt_feat_len=torch.tensor([feat_c.shape[0]], dtype=torch.int32),
                                         embedding=xv, streaming=False, finalize=True)
            # the encoder alone (mu of the CFM), for a per-stage check
            e = torch.nn.functional.normalize(xv, dim=1)
            spk = flow.spk_embed_affine_layer(e)
            tk = flow.input_embedding(torch.clamp(h_tok, min=0).long())
            hh, _ = flow.encoder(tk, torch.tensor([h_tok.shape[1]], dtype=torch.int32), streaming=False)
            mu = flow.encoder_proj(hh)
            with NoiseTap(torch) as tap:
                f0 = hift.f0_predictor(mel_flow)
                wav_out, _ = hift.inference(speech_feat=mel_flow, cache_source=torch.zeros(1, 1, 0))
        d = {"text_tokens": tt, "prompt_text_tokens": pt, "llm_prompt_tokens": lp_tok, "gen_tokens": gen,
             "llm_logp": logp.numpy().astype(np.float32), "flow_spk": spk[0].numpy(), "flow_mu": mu[0].numpy().astype(np.float32),
             "flow_mel": mel_flow[0].transpose(0, 1).contiguous().numpy().astype(np.float32),
             "hift_f0": f0[0].numpy().astype(np.float32), "audio": wav_out[0].numpy().astype(np.float32)}
        d.update(tap.tensors())
        dumps[name] = (d, {"text": text, "prompt_text": prompt_text_s, "mode": name, "instruct": "1" if instruct else "0"})
        return d

    run_case("zero_shot", "We are testing a tiny model today.", prompt_text, llm_prompt=True)
    run_case("cross_lingual", "Another sentence, without the prompt text.", "", llm_prompt=False, seed=12)
    run_case("instruct2", "Speak this one happily.", "Speak happily<|endofprompt|>", llm_prompt=False, instruct=True, seed=13)

    common = {"prompt24": prompt24[0].numpy(), "prompt16": prompt16[0].numpy(), "prompt_feat": feat.numpy().astype(np.float32),
              "fbank": fb.numpy().astype(np.float32), "xvec": xvec.astype(np.float32), "whisper_mel": wmel.numpy().astype(np.float32),
              "speech_tokens": speech_tok, "flow_rand_noise_head": cfm.rand_noise[0, :, :64].contiguous().numpy()}
    for name, (d, meta) in dumps.items():
        arrs = {**common, **d}
        arrs = {k: (v.astype(np.int32) if v.dtype.kind in "iu" else np.ascontiguousarray(v, dtype=np.float32)) for k, v in arrs.items()}
        save_file(arrs, os.path.join(a.out, f"dump_{name}.safetensors"), metadata={**meta, "prompt_text_full": prompt_text})
    json.dump({"prompt_text": prompt_text, "cases": list(dumps), "texts": {k: v[1]["text"] for k, v in dumps.items()}},
              open(os.path.join(a.out, "cases.json"), "w"), indent=1)
    print(f"{a.out}: tiny CosyVoice2, prompt {prompt24.shape[1] / 24000:.2f} s, {speech_tok.size} prompt tokens; "
          + ", ".join(f"{k}: {v[0]['gen_tokens'].size} tokens" for k, v in dumps.items()))


if __name__ == "__main__":
    main()
