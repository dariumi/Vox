"""Imported by the tests: folders made with tempfile.mkdtemp (and TemporaryDirectory) are removed when the test exits,
whether it passed or failed. VOX_KEEP_TMP=1 keeps them for a look afterwards."""
import atexit
import os
import shutil
import tempfile

_made = []
_mkdtemp = tempfile.mkdtemp


def _tracked(*a, **k):
    d = _mkdtemp(*a, **k)
    _made.append(d)
    return d


if os.environ.get("VOX_KEEP_TMP", "") in ("", "0"):
    tempfile.mkdtemp = _tracked
    atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in _made])
