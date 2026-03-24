from pathlib import Path
import yaml
import subprocess
import uuid

BASE = Path(__file__).resolve().parent.parent.parent.parent.parent
PYTHON = BASE / ".venv/bin/python"

def run_campaign_blocking(spec):
    run_id = f"ui_{uuid.uuid4().hex[:6]}"
    tmp = BASE / "experiments/tmp" / run_id
    tmp.mkdir(parents=True, exist_ok=True)

    yaml_path = tmp / "campaign.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(spec, f)

    subprocess.run([PYTHON, BASE / "tools/generate_campaign.py", yaml_path, "--out-root", BASE / "campaigns"], check=True)
    subprocess.run([PYTHON, BASE / "tools/run_campaign.py"], check=True)

    return run_id
