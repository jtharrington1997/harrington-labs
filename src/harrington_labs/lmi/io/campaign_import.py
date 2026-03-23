"""
campaign_import.py — Robust importer for tiled Si R/T campaign sheets

Parses Excel sheets with multiple horizontally tiled material blocks.

Each block corresponds to a DIFFERENT MATERIAL, not repeated scans.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# -----------------------------
# LOAD
# -----------------------------
def load_sheet(file_path: str | Path, sheet_name: str):
    return pd.read_excel(file_path, sheet_name=sheet_name, header=None)


# -----------------------------
# LABEL DETECTION
# -----------------------------
def _is_material_label(val) -> bool:
    if not isinstance(val, str):
        return False
    val = val.strip()

    # reject short / generic labels
    if len(val) < 5:
        return False

    # require silicon or blank
    return (
        val.startswith("Si-")
        or val.startswith("Si ")
        or val.startswith("Blank")
    )


# -----------------------------
# BLOCK EXTRACTION
# -----------------------------
def _extract_block(df: pd.DataFrame, r: int, c: int):
    """
    Extract one material block from (r, c)
    """

    data_start = r + 2

    block = df.iloc[data_start:data_start + 10, c:c + 7].copy()

    block.columns = [
        "position_mm",
        "ref_power",
        "ref_std",
        "trans_power",
        "trans_std",
        "norm_ref",
        "norm_trans",
    ]

    block = block.apply(pd.to_numeric, errors="coerce")

    # keep only valid rows
    block = block.dropna(subset=["position_mm"])

    if len(block) < 5:
        return None

    # physics
    block["absorption"] = 1 - (
        block["norm_ref"] + block["norm_trans"]
    )

    # clean
    block = block.dropna(subset=["absorption"])

    # collapse duplicate positions (important)
    block = (
        block.groupby("position_mm", as_index=False)
        .agg({
            "norm_ref": "mean",
            "norm_trans": "mean",
            "absorption": "mean"
        })
    )

    block = block.sort_values("position_mm")

    return block


# -----------------------------
# PARSER
# -----------------------------
def parse_sheet(file_path: str | Path, sheet_name: str):
    df = load_sheet(file_path, sheet_name)

    materials = []

    visited = set()

    nrows, ncols = df.shape

    for r in range(nrows):
        for c in range(ncols):

            val = df.iloc[r, c]

            if not _is_material_label(val):
                continue

            key = (r, c)
            if key in visited:
                continue

            try:
                block = _extract_block(df, r, c)

                if block is None:
                    continue

                materials.append({
                    "name": str(val).strip(),
                    "data": block
                })

                visited.add(key)

            except Exception:
                continue

    # filter out invalid entries
    materials = [
        m for m in materials
        if (
            m.get("data") is not None
            and len(m["data"]) >= 5
            and m["data"]["absorption"].notna().any()
        )
    ]

    return materials


# -----------------------------
# UTIL: SUMMARY
# -----------------------------
def summarize(materials):
    print(f"{len(materials)} materials detected\n")

    for m in materials:
        d = m["data"]

        print(f"--- {m['name']} ---")
        print(f"Points: {len(d)}")
        print(
            "Absorption range:",
            f"{d['absorption'].min():.3f} → {d['absorption'].max():.3f}"
        )
        print()


# -----------------------------
# UTIL: PLOT
# -----------------------------
def plot(materials):
    import matplotlib.pyplot as plt

    plt.figure()

    for m in materials:
        d = m["data"]
        plt.plot(
            d["position_mm"],
            d["absorption"],
            "o-",
            label=m["name"]
        )

    plt.xlabel("Position (mm)")
    plt.ylabel("Absorption")
    plt.legend()
    plt.grid(True)

    plt.savefig("campaign_plot.png", dpi=150)
    print("Saved plot → campaign_plot.png")
