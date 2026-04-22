from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FILES = [
    "df_input.csv",
    "test_pnl.csv",
    "test_actions.csv",
    "test_inventory.csv",
    "regime_state_a.csv",
    "regime_state_b.csv",
    "signal_confidence_a.csv",
    "signal_confidence_b.csv",
    "run_config_summary.csv",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _normalize_frame(name: str, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    elif out.columns[0].startswith("Unnamed:"):
        out = out.rename(columns={out.columns[0]: "datetime"})
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")

    if name == "run_config_summary.csv":
        return out.sort_values(list(out.columns)).reset_index(drop=True)

    sort_cols = [c for c in ["datetime"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    return out.reset_index(drop=True)


def _compare_numeric(left: pd.Series, right: pd.Series, atol: float) -> pd.Series:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    both_nan = left_num.isna() & right_num.isna()
    both_num = left_num.notna() & right_num.notna()
    equal_num = (left_num - right_num).abs() <= atol
    equal_str = left.astype(str) == right.astype(str)
    return both_nan | (both_num & equal_num) | (~both_num & ~left_num.isna() & ~right_num.isna() & equal_str)


def compare_file(notebook_path: Path, package_path: Path, atol: float) -> str:
    notebook_df = _normalize_frame(notebook_path.name, _read_csv(notebook_path))
    package_df = _normalize_frame(package_path.name, _read_csv(package_path))

    if list(notebook_df.columns) != list(package_df.columns):
        return (
            f"{notebook_path.name}: column mismatch\n"
            f"  notebook: {list(notebook_df.columns)}\n"
            f"  package:  {list(package_df.columns)}"
        )

    if notebook_df.shape != package_df.shape:
        return (
            f"{notebook_path.name}: shape mismatch\n"
            f"  notebook: {notebook_df.shape}\n"
            f"  package:  {package_df.shape}"
        )

    mismatch_mask = pd.Series(False, index=notebook_df.index)
    for col in notebook_df.columns:
        mismatch_mask |= ~_compare_numeric(notebook_df[col], package_df[col], atol)

    if not mismatch_mask.any():
        return f"{notebook_path.name}: OK"

    idx = int(mismatch_mask.idxmax())
    row_note = notebook_df.iloc[idx].to_dict()
    row_pkg = package_df.iloc[idx].to_dict()
    return (
        f"{notebook_path.name}: first mismatch at row {idx}\n"
        f"  notebook: {row_note}\n"
        f"  package:  {row_pkg}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare notebook and package debug export CSVs.")
    parser.add_argument("notebook_dir", help="Directory containing notebook-exported CSVs")
    parser.add_argument("package_dir", help="Directory containing package-exported CSVs")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for numeric comparisons",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=DEFAULT_FILES,
        help="Specific CSV filenames to compare",
    )
    args = parser.parse_args()

    notebook_dir = Path(args.notebook_dir)
    package_dir = Path(args.package_dir)

    for file_name in args.files:
        notebook_path = notebook_dir / file_name
        package_path = package_dir / file_name

        if not notebook_path.exists() or not package_path.exists():
            print(
                f"{file_name}: missing file\n"
                f"  notebook exists: {notebook_path.exists()}\n"
                f"  package exists:  {package_path.exists()}"
            )
            continue

        print(compare_file(notebook_path, package_path, args.atol))


if __name__ == "__main__":
    main()
