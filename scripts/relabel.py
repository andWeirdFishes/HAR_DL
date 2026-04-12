import os
import pandas as pd
from har_dl.definitions import get_project_root


def relabel_activity(label):
    """Map Label to Activity column."""
    mapping = {"sitting": "upright-still", "standing": "upright-still"}
    return mapping.get(label, label)


def process_task1_files():
    data_folder = os.path.join(get_project_root(), "data", "preprocessed")

    print("Project root:", get_project_root())
    print("Data folder:", data_folder)

    if not os.path.exists(data_folder):
        print(f"Data folder not found: {data_folder}")
        return

    all_files = []

    for root, dirs, files in os.walk(data_folder):
        for f in files:
            if f.endswith(".csv") and f.startswith("task1"):
                all_files.append(os.path.join(root, f))

    print("All files:", all_files)

    csv_files = all_files

    print("Filtered CSV files:", csv_files)
    for file in csv_files:
        print(f"Processing file: {file}")
        df = pd.read_csv(file)
        if "Label" not in df.columns:
            print(f"Warning: 'Label' column not found in {file}. Skipping.")
            continue
        df["Activity"] = df["Label"].apply(relabel_activity)
        df.to_csv(file, index=False)
        print(f"Finished processing file: {file}")


if __name__ == "__main__":
    process_task1_files()
