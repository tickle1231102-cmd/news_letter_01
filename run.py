import os, sys
from graph import run

if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        os.environ["DRY_RUN"] = "1"
    out = run()
    for line in out["log"]:
        print(line)
