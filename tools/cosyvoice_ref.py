"""Import helper for the official CosyVoice sources (https://github.com/FunAudioLLM/CosyVoice, with third_party/Matcha-TTS),
used by tools/make_tiny_cosyvoice.py and tests/test_cosyvoice_tiny.py.

CosyVoice is not a pip package: point VOX_COSYVOICE_SRC at a clone (git clone --recursive …). Modules the reference classes
import but inference never uses (lightning for Matcha's training utilities, whisper / tiktoken for the CosyVoice 1 tokenizer)
are replaced by empty stand-ins when they are not installed."""
import os
import sys
import types


def setup(src=None):
    """Adds the CosyVoice clone to sys.path; returns its path or None when it is not there."""
    src = src or os.environ.get("VOX_COSYVOICE_SRC", "")
    if not src or not os.path.isdir(os.path.join(src, "cosyvoice")):
        return None
    for p in (os.path.join(src, "third_party", "Matcha-TTS"), src):
        if p not in sys.path:
            sys.path.insert(0, p)
    import transformers  # noqa: F401  (its optional-package checks must run before the stand-ins exist)
    _stub_missing()
    return src


def _module(name, **attrs):
    import importlib.machinery
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules[name] = m
    return m


def _stub_missing():
    import importlib.util
    if importlib.util.find_spec("lightning") is None:
        class _Any:
            def __init__(self, *a, **k):
                pass
        rank_zero_only = (lambda f: f)
        _module("lightning", Callback=_Any, LightningModule=object, Trainer=_Any, LightningDataModule=object)
        _module("lightning.pytorch", Callback=_Any, LightningModule=object)
        _module("lightning.pytorch.loggers", Logger=_Any)
        _module("lightning.pytorch.utilities", rank_zero_only=rank_zero_only)
        _module("lightning.fabric")
        _module("lightning.fabric.utilities")
        _module("lightning.fabric.utilities.exceptions", MisconfigurationException=Exception)
    for name in ("gdown", "wget"):   # matcha.utils.utils downloads checkpoints with them
        if importlib.util.find_spec(name) is None:
            _module(name)
    if importlib.util.find_spec("whisper") is None:
        _module("whisper", log_mel_spectrogram=None)
        _module("whisper.tokenizer", Tokenizer=object)
    if importlib.util.find_spec("tiktoken") is None:
        _module("tiktoken", Encoding=object)
