import sys

from agoge_forger.cli import app

if __name__ == "__main__":
    app(["bench-vllm-frontend"] + sys.argv[1:])
