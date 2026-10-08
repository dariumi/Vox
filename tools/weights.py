"""Общие функции инструментов: поиск модели и отпечаток весов.

weights_sha256 — sha256 по списку "имя_файла:sha256_файла" для всех *.safetensors
модели (рекурсивно, в отсортированном порядке). Тот же алгоритм обязан использовать
`vox build`, иначе профили голосов не узнают свою модель.
"""
import hashlib
import json
import os
from pathlib import Path


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def weights_sha256(model_dir: str) -> str:
    root = Path(model_dir)
    files = sorted(p for p in root.rglob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {model_dir}")
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.relative_to(root).as_posix()}:{_file_sha(p)}\n".encode())
    return h.hexdigest()


def resolve(model: str) -> str:
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model)


def model_info(model: str):
    """→ (локальная папка, family, variant, weights_sha256)."""
    d = resolve(model)
    cfg = json.load(open(os.path.join(d, "config.json")))
    parts = [str(cfg.get(k, "")) for k in ("tokenizer_type", "tts_model_size", "tts_model_type")]
    variant = "-".join(p for p in parts if p).lower() or os.path.basename(os.path.normpath(d))
    return d, "qwen3-tts", variant, weights_sha256(d)
