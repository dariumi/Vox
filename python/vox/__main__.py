"""python -m vox build|serve ..."""
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    if cmd == "build":
        from .build import main as m
        m(rest)
    elif cmd == "serve":
        from .serve import main as m
        m(rest)
    else:
        print("usage: python -m vox {build|serve} ...\n  build  HF-чекпойнт Qwen3-TTS -> .tmod\n  serve  HTTP-сервер синтеза со стримингом")
        sys.exit(2 if cmd else 0)


if __name__ == "__main__":
    main()
