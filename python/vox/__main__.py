"""python -m vox build|serve|web ..."""
import sys

USAGE = """usage: python -m vox COMMAND ...
  build   Hugging Face Qwen3-TTS checkpoint -> .tmod   (a CosyVoice2 folder too; also: python -m vox.build_f5 / vox.build_kokoro / vox.build_cosyvoice)
  serve   HTTP synthesis server with streaming (one model)
  web     the engine for Vox Studio: models, voice profiles, history; prints a link with a secret"""


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    if cmd == "build":
        from .build import main as m
        m(rest)
    elif cmd == "serve":
        from .serve import main as m
        m(rest)
    elif cmd in ("web", "-web", "--web"):
        from .web import main as m
        m(rest)
    else:
        print(USAGE)
        sys.exit(2 if cmd else 0)


if __name__ == "__main__":
    main()
