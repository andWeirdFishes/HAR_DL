import os
from pathlib import Path
import pandas as pd
from har_dl.definitions import get_project_root


def patch_existing_results_with_end_labels() -> None:
    root = get_project_root()
    dataset_name = "HAR_DL_FEIT_2025"
    results_base_path = root / "data" / "results" / dataset_name

    for file_path in sorted(results_base_path.rglob("*.csv")):
        if not file_path.is_file():
            continue

        df = pd.read_csv(file_path)
        if df.empty or "Label" not in df.columns:
            continue

        # Shift the starting labels up to find the state of the subsequent window frames
        # This naturally finds the end state for almost all overlapping segments
        label_end = df["Label"].shift(-2)

        # Handle the very last trailing segments by letting them hold their final active state
        label_end = label_end.ffill().bfill()

        df["Label_End"] = label_end
        df.to_csv(file_path, index=False)
        print(f"[PATCHED] Successfully synced Label_End tracking on {file_path.relative_to(results_base_path)}")


if __name__ == "__main__":
    patch_existing_results_with_end_labels()