import sys
from agoge_forger.cli import app
if __name__ == "__main__":
    app(["smoke-eval", "--adapter-path", sys.argv[1] if len(sys.argv) > 1 else ""])
