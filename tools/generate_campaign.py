from pathlib import Path
import yaml
import json
import sys

def main():
    if len(sys.argv) < 4:
        print("Usage: generate_campaign.py <campaign.yaml> --out-root <dir>")
        sys.exit(1)

    manifest = Path(sys.argv[1])
    out_root = Path(sys.argv[3])

    with open(manifest) as f:
        spec = yaml.safe_load(f)

    if spec is None:
        raise ValueError("campaign.yaml is empty")

    samples = spec.get("samples", [])
    beam = spec.get("beam", {})

    base = out_root / "si_midIR_ablation" / "configs"
    base.mkdir(parents=True, exist_ok=True)

    i = 0
    for s in samples:
        for rho in s.get("rho", []):
            i += 1
            cfg = {
                "sample_id": s["id"],
                "rho": rho,
                "type": s["type"],
                "thickness_um": s["thickness_um"],
                "beam": beam,
            }
            with open(base / f"{i:03d}.json", "w") as f:
                json.dump(cfg, f, indent=2)

    print(f"[INFO] Generated {i} configs")

if __name__ == "__main__":
    main()
