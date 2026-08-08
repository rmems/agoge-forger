import sys

from agoge_forger.cli import app

if __name__ == "__main__":
    app(["smoke-vllm"] + sys.argv[1:])
