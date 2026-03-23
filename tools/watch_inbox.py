import time
from pathlib import Path
import shutil
import subprocess

BASE = Path.home() / "Projects/harrington-lmi"
PYTHON = BASE / ".venv/bin/python"

INBOX = BASE / "experiments/inbox"
PROCESSED = BASE / "experiments/processed"

def run(cmd):
    subprocess.run(cmd, check=True)

def process(pkg):
    manifest = pkg / "campaign.yaml"
    if not manifest.exists():
        return

    run([PYTHON, BASE / "tools/generate_campaign.py", manifest, "--out-root", BASE / "campaigns"])
    run([PYTHON, BASE / "tools/run_campaign.py"])

    dest = PROCESSED / pkg.name
    if dest.exists():
        shutil.rmtree(dest)
    pkg.rename(dest)

def main():
    INBOX.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    while True:
        for pkg in INBOX.iterdir():
            if pkg.is_dir() and (pkg / ".complete").exists():
                process(pkg)
        time.sleep(2)

if __name__ == "__main__":
    main()
