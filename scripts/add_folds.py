import os
import pandas as pd
from har_dl.definitions import get_project_root


def assign_folds() -> None:
    data_folder = os.path.join(get_project_root(), "data", "preprocessed")

    if not os.path.exists(data_folder):
        print(f"Data folder not found: {data_folder}")
        return

    dataset_dirs = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
    
    if not dataset_dirs:
        print("No dataset directories found.")
        return

    dataset_path = os.path.join(data_folder, dataset_dirs[0])
    subject_dirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    print(f"Found {len(subject_dirs)} subjects.")

    for i, subject_id in enumerate(subject_dirs):
        fold = (i % 6) + 1
        subject_full_path = os.path.join(dataset_path, subject_id)
        
        for file in os.listdir(subject_full_path):
            if file.endswith(".csv") and (file.startswith("task1") or file.startswith("task3")):
                file_path = os.path.join(subject_full_path, file)
                print(f"Assigning {file} (Subject: {subject_id}) to fold {fold}")
                
                df = pd.read_csv(file_path)
                df["Fold"] = fold
                df.to_csv(file_path, index=False)

    print("Done!")


if __name__ == "__main__":
    assign_folds()