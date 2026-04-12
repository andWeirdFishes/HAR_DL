import os
import pandas as pd
from har_dl.definitions import get_project_root


def assign_folds():
    data_folder = os.path.join(get_project_root(), "data", "preprocessed")

    if not os.path.exists(data_folder):
        print(f"Data folder not found: {data_folder}")
        return

    all_files = []

    for root, dirs, files in os.walk(data_folder):
        for f in files:
            if f.endswith(".csv") and f.startswith("task1"):
                all_files.append(os.path.join(root, f))

    print(f"Found files: {len(all_files)}")
    print(len(all_files))
    for i, file_path in enumerate(all_files):
        fold = ((i - 1) % 6) + 1

        print(f"{file_path} to fold {fold}")

        df = pd.read_csv(file_path)
        df["Fold"] = fold

        df.to_csv(file_path, index=False)

    print("Done!")


if __name__ == "__main__":
    assign_folds()
