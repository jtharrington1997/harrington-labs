"""Experimental data parsers for model comparison.

Handles messy lab spreadsheets, clean CSV exports, and various
file formats encountered in photonics/laser labs.
No Streamlit imports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ExperimentalDataset:
    """Parsed experimental data ready for comparison."""
    name: str = ""
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    y_uncertainty: Optional[np.ndarray] = None
    x_label: str = ""
    y_label: str = ""
    x_unit: str = ""
    y_unit: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.x)

    def valid(self) -> bool:
        return len(self.x) > 0 and len(self.x) == len(self.y)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _is_numeric(v) -> bool:
    if pd.isna(v):
        return False
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def _is_header_cell(v) -> bool:
    """Check if a cell value looks like a column header (text, not a number)."""
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return bool(re.search(
        r"(ref power|transmitted|norm|std|x/z|blank|position)",
        s,
    ))


# ── Multi-block scanner ────────────────────────────────────────────────────


def _find_all_header_rows(df: pd.DataFrame) -> list[int]:
    """Find all rows that contain column headers (Ref Power, Transmitted, etc.)."""
    header_rows = []
    for i in range(len(df)):
        row_text = " ".join(str(v).strip().lower() for v in df.iloc[i].values if pd.notna(v))
        if any(kw in row_text for kw in ("ref power", "transmitted", "norm ref", "norm trans")):
            header_rows.append(i)
    return header_rows


def _extract_blocks_from_header_row(
    df: pd.DataFrame,
    header_row: int,
) -> list[dict]:
    """Extract all data blocks from a single header row.

    A header row can contain multiple side-by-side blocks separated by
    empty columns or different sample labels.
    """
    ncols = df.shape[1]
    blocks = []

    col = 0
    while col < ncols:
        cell = df.iloc[header_row, col]
        if pd.isna(cell):
            col += 1
            continue

        cell_str = str(cell).strip()
        cell_lower = cell_str.lower()

        # Is this the start of a block? Look for position-like column or sample label
        is_pos = cell_lower in ("x/z", "blank")
        is_sample = bool(re.match(
            r"(?i)(ge|si|intrinsic|p.type|n.type|.*ohm.*)",
            cell_str,
        ))
        is_numeric_pos = _is_numeric(cell)

        if not (is_pos or is_sample or is_numeric_pos):
            col += 1
            continue

        # Map this block's columns
        # Strategy: start at this position, scan right for header cells.
        # Also check the first data row to identify columns with actual data.
        block_start = col
        headers = {}

        # First pass: collect all header cells, stopping at another sample label
        j = col
        while j < ncols:
            h = df.iloc[header_row, j]
            if pd.notna(h):
                h_str = str(h).strip()
                # If we hit another sample/block label after collecting some headers, stop
                if j > col and re.match(r"(?i)(ge|si|intrinsic|blank|x/z)", h_str):
                    if "norm" not in h_str.lower() and "std" not in h_str.lower():
                        break
                headers[j] = h_str
            j += 1
        block_end = j

        # If we only found the position label but no other named headers,
        # check data rows for how wide the numeric data extends
        if len(headers) < 2:
            # Look at the first data row to find block width
            first_data_row = header_row + 1
            if first_data_row < len(df):
                data_j = col
                while data_j < ncols and _is_numeric(df.iloc[first_data_row, data_j]):
                    if data_j not in headers:
                        headers[data_j] = f"col_{data_j}"
                    data_j += 1
                block_end = data_j

        if len(headers) < 2:
            col = block_end
            continue

        # Classify columns
        pos_col = block_start
        ref_col = trans_col = std_trans_col = norm_ref_col = norm_trans_col = None

        for cj, h in headers.items():
            hl = h.lower()
            if cj == block_start:
                continue  # position column
            if "norm trans" in hl or "norm_trans" in hl:
                norm_trans_col = cj
            elif "norm ref" in hl:
                norm_ref_col = cj
            elif "transmitted" in hl:
                trans_col = cj
            elif "ref power" in hl or "ref_power" in hl:
                ref_col = cj
            elif "std" in hl:
                # Assign std to the most recent power column
                if trans_col is not None and std_trans_col is None:
                    std_trans_col = cj

        # Read data rows
        data_rows = []
        for dr in range(header_row + 1, len(df)):
            pos_val = df.iloc[dr, pos_col]
            if not _is_numeric(pos_val):
                # Could be another header row or blank — stop
                break
            row = {"pos": float(pos_val)}
            for cj, h in headers.items():
                if cj != pos_col:
                    v = df.iloc[dr, cj]
                    row[h] = float(v) if _is_numeric(v) else np.nan
            data_rows.append(row)

        if len(data_rows) < 2:
            col = block_end
            continue

        # Find block label — look above header row, at the position column
        block_label = ""
        # First check if the position column header itself is a sample name
        pos_header = headers.get(pos_col, "")
        if re.match(r"(?i)(ge|si|intrinsic|.*ohm.*)", pos_header):
            block_label = pos_header
        else:
            # Scan rows above for a label
            for scan_row in range(max(0, header_row - 4), header_row):
                above = df.iloc[scan_row, pos_col]
                if pd.notna(above) and str(above).strip() and not _is_numeric(above):
                    candidate = str(above).strip()
                    if candidate.lower() not in ("sample", "type", "nan"):
                        block_label = candidate

        blocks.append({
            "label": block_label,
            "header_row": header_row,
            "pos_col": pos_col,
            "ref_col": ref_col,
            "trans_col": trans_col,
            "std_trans_col": std_trans_col,
            "norm_ref_col": norm_ref_col,
            "norm_trans_col": norm_trans_col,
            "headers": headers,
            "data": data_rows,
        })

        col = block_end

    return blocks


def _blocks_to_datasets(
    blocks: list[dict],
    sheet_name: str,
    filepath: Path,
) -> list[ExperimentalDataset]:
    """Convert parsed blocks into ExperimentalDataset objects."""
    datasets = []
    meta_base = {"sheet": sheet_name, "file": filepath.name}

    for block in blocks:
        label = block["label"] or sheet_name
        data = block["data"]
        positions = np.array([r["pos"] for r in data])

        # Normalized transmission
        if block["norm_trans_col"] is not None:
            h = block["headers"][block["norm_trans_col"]]
            values = np.array([r.get(h, np.nan) for r in data])
            valid = ~np.isnan(values)
            if valid.sum() >= 2:
                ds = ExperimentalDataset(
                    name=f"{label} — Norm Trans ({sheet_name})",
                    x=positions[valid],
                    y=values[valid],
                    x_label="Stage Position",
                    y_label="Normalized Transmission",
                    x_unit="mm",
                    metadata={**meta_base, "block_label": label, "type": "norm_trans"},
                )
                datasets.append(ds)

        # Raw transmitted power
        if block["trans_col"] is not None:
            h = block["headers"][block["trans_col"]]
            values = np.array([r.get(h, np.nan) for r in data])
            valid = ~np.isnan(values)
            if valid.sum() >= 2:
                unc = None
                if block["std_trans_col"] is not None:
                    h_std = block["headers"][block["std_trans_col"]]
                    unc_vals = np.array([r.get(h_std, np.nan) for r in data])
                    unc = unc_vals[valid] * 1e-3  # µW → mW

                ds = ExperimentalDataset(
                    name=f"{label} — Raw Trans ({sheet_name})",
                    x=positions[valid],
                    y=values[valid],
                    y_uncertainty=unc,
                    x_label="Stage Position",
                    y_label="Transmitted Power",
                    x_unit="mm",
                    y_unit="mW",
                    metadata={**meta_base, "block_label": label, "type": "raw_trans"},
                )
                datasets.append(ds)

        # Normalized reference
        if block["norm_ref_col"] is not None:
            h = block["headers"][block["norm_ref_col"]]
            values = np.array([r.get(h, np.nan) for r in data])
            valid = ~np.isnan(values)
            if valid.sum() >= 2:
                ds = ExperimentalDataset(
                    name=f"{label} — Norm Ref ({sheet_name})",
                    x=positions[valid],
                    y=values[valid],
                    x_label="Stage Position",
                    y_label="Normalized Reference",
                    x_unit="mm",
                    metadata={**meta_base, "block_label": label, "type": "norm_ref"},
                )
                datasets.append(ds)

    return datasets


def parse_knife_edge_xlsx(
    filepath: str | Path,
    sheet_name: Optional[str] = None,
) -> list[ExperimentalDataset]:
    """Parse knife-edge / z-scan xlsx with messy multi-block layout.

    Handles:
    - Multiple header rows within a single sheet
    - Sample blocks laid out side-by-side across columns
    - Blank reference rows and sample data blocks
    - Normalized and raw transmission data
    - Multiple sheets with different power levels

    Returns a list of ExperimentalDataset, one per data block found.
    """
    filepath = Path(filepath)
    xls = pd.ExcelFile(filepath)
    target_sheets = [sheet_name] if sheet_name else xls.sheet_names
    all_datasets = []

    for sname in target_sheets:
        df = pd.read_excel(xls, sname, header=None)
        if df.empty or df.shape[0] < 3:
            continue

        header_rows = _find_all_header_rows(df)
        all_blocks = []
        for hr in header_rows:
            blocks = _extract_blocks_from_header_row(df, hr)
            all_blocks.extend(blocks)

        datasets = _blocks_to_datasets(all_blocks, sname, filepath)
        all_datasets.extend(datasets)

    return all_datasets


# ── Generic parsers ─────────────────────────────────────────────────────────


def parse_generic_csv(
    filepath: str | Path,
    x_col: int | str = 0,
    y_col: int | str = 1,
    uncertainty_col: Optional[int | str] = None,
    skip_rows: int = 0,
    delimiter: str = ",",
    name: str = "",
) -> ExperimentalDataset:
    """Parse a clean CSV/TSV with numeric columns."""
    df = pd.read_csv(filepath, skiprows=skip_rows, sep=delimiter)
    x_key = df.columns[x_col] if isinstance(x_col, int) else x_col
    y_key = df.columns[y_col] if isinstance(y_col, int) else y_col

    valid = df[[x_key, y_key]].dropna()
    unc = None
    if uncertainty_col is not None:
        u_key = df.columns[uncertainty_col] if isinstance(uncertainty_col, int) else uncertainty_col
        unc = df.loc[valid.index, u_key].values.astype(float)

    return ExperimentalDataset(
        name=name or Path(filepath).stem,
        x=valid[x_key].values.astype(float),
        y=valid[y_key].values.astype(float),
        y_uncertainty=unc,
        x_label=str(x_key),
        y_label=str(y_key),
        metadata={"file": str(filepath), "format": "csv"},
    )


def parse_generic_xlsx(
    filepath: str | Path,
    sheet_name: str | int = 0,
    x_col: int | str = 0,
    y_col: int | str = 1,
    uncertainty_col: Optional[int | str] = None,
    skip_rows: int = 0,
    name: str = "",
) -> ExperimentalDataset:
    """Parse a clean xlsx with numeric columns."""
    df = pd.read_excel(filepath, sheet_name=sheet_name, skiprows=skip_rows)
    x_key = df.columns[x_col] if isinstance(x_col, int) else x_col
    y_key = df.columns[y_col] if isinstance(y_col, int) else y_col

    valid = df[[x_key, y_key]].dropna()
    unc = None
    if uncertainty_col is not None:
        u_key = df.columns[uncertainty_col] if isinstance(uncertainty_col, int) else uncertainty_col
        unc = df.loc[valid.index, u_key].values.astype(float)

    return ExperimentalDataset(
        name=name or Path(filepath).stem,
        x=valid[x_key].values.astype(float),
        y=valid[y_key].values.astype(float),
        y_uncertainty=unc,
        x_label=str(x_key),
        y_label=str(y_key),
        metadata={"file": str(filepath), "format": "xlsx"},
    )


# ── Auto-detect ────────────────────────────────────────────────────────────


def detect_and_parse(filepath: str | Path) -> list[ExperimentalDataset]:
    """Auto-detect file format and parse experimental data.

    Strategy:
    1. xlsx files -> try knife-edge parser first; fall back to generic
    2. csv/tsv -> generic CSV parser
    3. txt -> attempt CSV with whitespace delimiter

    Returns list of datasets found in the file.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in (".xlsx", ".xls", ".xlsm"):
        datasets = parse_knife_edge_xlsx(filepath)
        if datasets:
            return datasets

        xls = pd.ExcelFile(filepath)
        results = []
        for sname in xls.sheet_names:
            try:
                ds = parse_generic_xlsx(filepath, sheet_name=sname, name=f"{filepath.stem} — {sname}")
                if ds.valid():
                    results.append(ds)
            except Exception:
                continue
        return results

    elif suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        try:
            ds = parse_generic_csv(filepath, delimiter=delim, name=filepath.stem)
            return [ds] if ds.valid() else []
        except Exception:
            return []

    elif suffix == ".txt":
        for delim in ["\t", ",", r"\s+"]:
            try:
                ds = parse_generic_csv(filepath, delimiter=delim, name=filepath.stem)
                if ds.valid():
                    return [ds]
            except Exception:
                continue
        return []

    return []
