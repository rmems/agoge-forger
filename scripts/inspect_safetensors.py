import sys
from agoge_forger.cli import app
if __name__ == "__main__":
    app(["inspect-safetensors"] + sys.argv[1:])
