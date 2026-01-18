import os
import pandas as pd
from pathlib import Path
from typing import List, Optional
from har_dl.config import load_config


class DataLoader:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.raw_path = Path(self.config["raw_path"])
        self.processed_path = Path(self.config["preprocessed_path"])
        self.label_col = self.config["label_col"]
        self.sublabel_col = self.config["sublabel"]
        self.sensor_columns = self.config["sensor_columns_raw"]

    def get_all_files(self, folder: Path) -> list:
        return [Path(root) / file
                for root, _, files in os.walk(folder)
                for file in files if file.endswith(".csv")]

    def load_single_file(self, file_path: Path, subject_name: str) -> pd.DataFrame | None:
        try:
            df = pd.read_csv(file_path)
            expected_cols = self.sensor_columns + [self.label_col] + [self.sublabel_col]
            if not all(col in df.columns for col in expected_cols):
                print(f"Skipping {file_path.name} – missing required columns.")
                return None

            if self.label_col in df.columns:
                if df.empty:
                    print(f"Skipping {file_path.name} – all rows filtered out.")
                    return None

            df["Subject"] = subject_name
            df["File"] = str(file_path.relative_to(self.raw_path))
            return df
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return None

    def load_raw_datasets(self) -> List[pd.DataFrame]:
        all_dfs = []

        for subject_folder in self.raw_path.iterdir():
            if not subject_folder.is_dir():
                continue

            subject_name = subject_folder.name

            for csv_file in subject_folder.glob("*.csv"):
                df = self.load_single_file(csv_file, subject_name)
                if df is not None:
                    all_dfs.append(df)

        return all_dfs

    def load_processed_datasets(self) -> List[pd.DataFrame]:
        all_dfs = []

        for subject_folder in self.processed_path.iterdir():
            if not subject_folder.is_dir():
                continue

            subject_name = subject_folder.name

            for csv_file in subject_folder.rglob("*.csv"):
                try:
                    df = pd.read_csv(csv_file, low_memory=False)
                    df["Subject"] = subject_name
                    df["File"] = str(csv_file.relative_to(self.processed_path))
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Failed to load {csv_file}: {e}")

        return all_dfs

    def merge_all(self, processed: bool = False) -> pd.DataFrame:
        if processed:
            dfs = self.load_processed_datasets()
        else:
            dfs = self.load_raw_datasets()

        if not dfs:
            print("No datasets were loaded.")
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)