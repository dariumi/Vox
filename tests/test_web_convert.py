"""Загрузка и конвертация моделей внутри vox.js против «Hugging Face» на локальном мок-сервере. Запуск: python tests/test_web_convert.py
Нужны node (>=18), torch + safetensors (чтобы сохранить .pt/.bin) и крошечная модель F5 (VOX_WEB_WORK=<папка> либо соберётся сама, как в test_web.py)."""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "python")]


def main(work=None):
    if shutil.which("node") is None:
        print("SKIP: нужен node")
        return 0
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print("SKIP: нужны torch и safetensors:", e)
        return 0
    from mockhf import MockHF, build_repos
    if work is None:
        work = os.environ.get("VOX_WEB_WORK")
    if work is None:
        try:
            import f5_tts, vocos  # noqa: F401,E401
        except Exception as e:  # noqa: BLE001
            print("SKIP: нужны f5-tts и vocos, чтобы собрать крошечную модель:", e)
            return 0
        work = tempfile.mkdtemp(prefix="voxcv_")
        rc = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_tiny_f5.py"), "--out", work], capture_output=True, text=True)
        assert rc.returncode == 0, rc.stdout[-1500:] + rc.stderr[-1500:]
    assert subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_web.py")], capture_output=True).returncode == 0
    if not os.path.exists(os.path.join(ROOT, "webgpu", "web", "dist", "pinyin.json")):
        subprocess.check_call([sys.executable, os.path.join(ROOT, "tools", "gen_pinyin.py")])
    tmp = tempfile.mkdtemp(prefix="voxcv_srv_")
    repos, refs = build_repos(work, tmp)
    srv = MockHF(repos, private=["mock/private"])
    try:
        rc = subprocess.run(["node", os.path.join(HERE, "web_convert_check.js"), os.path.join(ROOT, "webgpu", "web", "dist", "vox.js"), srv.url, work, refs])
    finally:
        srv.close()
    return rc.returncode


if __name__ == "__main__":
    sys.exit(main())
