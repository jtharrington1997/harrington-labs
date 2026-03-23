from pathlib import Path
import subprocess

BASE = Path.home() / "Projects/harrington-lmi"
PYTHON = BASE / ".venv/bin/python"

CONFIG_DIR = BASE / "campaigns/si_midIR_ablation/configs"
RESULT_DIR = BASE / "campaigns/si_midIR_ablation/results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

for cfg in sorted(CONFIG_DIR.glob("*.json")):
    out = RESULT_DIR / (cfg.stem + ".json")

    subprocess.run([
        PYTHON,
        BASE / "tools/run_digital_twin.py",
        str(cfg),
        str(out)
    ], check=True)

print("[INFO] Campaign complete")
