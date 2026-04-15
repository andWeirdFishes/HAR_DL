import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
from har_dl.config import load_config
from har_dl.data.loader import DataLoader
from emteqai.utils.processing.signals.filters import lowpass_filter, bandpass_filter, highpass_filter

warnings.filterwarnings("ignore")


class DataPreprocessor:
    def __init__(self,
                 config: Optional[dict] = None,
                 sampling_frequency: Optional[float] = None,
                 sensor_columns: Optional[List[str]] = None):
        self.config = config or load_config()
        self.raw_path = Path(self.config["raw_path"])
        self.processed_path = Path(self.config["preprocessed_path"])
        self.sampling_frequency = sampling_frequency or self.config["sampling_frequency"]
        self.sensor_columns = sensor_columns or self.config["sensor_columns_raw"]
        self.label_col = self.config["label_col"]
        self.sublabel_col = self.config["sublabel"]
        self.f_high_cutoff = self.config.get("f_high_cutoff")
        self.f_low_cutoff = self.config.get("f_low_cutoff")
        self.filter_order = self.config.get("filter_order", 5)
        self.loader = DataLoader(self.config)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def remove_outliers(self, df: pd.DataFrame, method: str = "zscore", factor: float = 1.5) -> pd.DataFrame:
        df_clean = df.copy()
        for col in self.sensor_columns:
            if col not in df_clean.columns:
                continue
            if method == "iqr":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
                df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
            elif method == "zscore":
                z = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                med = df_clean[col].median()
                df_clean.loc[z > factor, col] = med
        return df_clean

    def apply_signal_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filtered = df.copy()

        for col in self.sensor_columns:
            if col not in df_filtered.columns:
                continue

            try:
                if self.f_high_cutoff and self.f_low_cutoff:
                    df_filtered[col] = bandpass_filter(
                        df_filtered[col],
                        order=self.filter_order,
                        fcritical=(self.f_low_cutoff, self.f_high_cutoff),
                        fs=self.sampling_frequency
                    )
                elif self.f_high_cutoff:
                    df_filtered[col] = lowpass_filter(
                        df_filtered[col],
                        order=self.filter_order,
                        fcutoff=self.f_high_cutoff,
                        fs=self.sampling_frequency
                    )
                elif self.f_low_cutoff:
                    df_filtered[col] = highpass_filter(
                        df_filtered[col],
                        order=self.filter_order,
                        fcutoff=self.f_low_cutoff,
                        fs=self.sampling_frequency
                    )
            except Exception as e:
                print(f"Failed to filter {col}: {e}")

        return df_filtered

    def smooth_data(self, df: pd.DataFrame, window_size: int = 3) -> pd.DataFrame:
        df_smooth = df.copy()
        numeric_cols = [col for col in self.sensor_columns if col in df_smooth.columns]

        for col in numeric_cols:
            df_smooth[col] = df_smooth[col].rolling(window=window_size, center=True).mean()

        df_smooth[numeric_cols] = df_smooth[numeric_cols].bfill().ffill()
        return df_smooth

    def add_magnitude_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_mag = df.copy()

        acc_cols = [c for c in self.sensor_columns if "Accelerometer" in c and "Raw" in c]
        gyro_cols = [c for c in self.sensor_columns if "Gyroscope" in c and "Raw" in c]

        if len(acc_cols) == 3:
            df_mag["Accelerometer/Magnitude"] = np.sqrt(
                df_mag[acc_cols[0]] ** 2 + df_mag[acc_cols[1]] ** 2 + df_mag[acc_cols[2]] ** 2
            )

        if len(gyro_cols) == 3:
            df_mag["Gyroscope/Magnitude"] = np.sqrt(
                df_mag[gyro_cols[0]] ** 2 + df_mag[gyro_cols[1]] ** 2 + df_mag[gyro_cols[2]] ** 2
            )

        return df_mag

    def scale_subject_data(self, dfs: List[pd.DataFrame], scaler_type: str = "standard", group_axes: bool = True) -> \
    List[pd.DataFrame]:
        if not dfs or scaler_type == "none":
            return dfs

        all_df = pd.concat(dfs, ignore_index=True)
        sensor_cols = [c for c in self.sensor_columns if c in all_df.columns]

        for c in ["Accelerometer/Magnitude", "Gyroscope/Magnitude"]:
            if c in all_df.columns:
                sensor_cols.append(c)

        if not sensor_cols:
            return dfs

        scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()

        if group_axes:
            acc_cols = [c for c in sensor_cols if "Accelerometer" in c]
            gyro_cols = [c for c in sensor_cols if "Gyroscope" in c]

            scaled_dfs = []
            for df in dfs:
                df_scaled = df.copy()

                if acc_cols:
                    s = StandardScaler() if scaler_type == "standard" else RobustScaler()
                    s.fit(all_df[acc_cols].values.flatten().reshape(-1, 1))
                    for c in acc_cols:
                        df_scaled[c] = s.transform(df_scaled[c].values.reshape(-1, 1)).flatten()

                if gyro_cols:
                    s = StandardScaler() if scaler_type == "standard" else RobustScaler()
                    s.fit(all_df[gyro_cols].values.flatten().reshape(-1, 1))
                    for c in gyro_cols:
                        df_scaled[c] = s.transform(df_scaled[c].values.reshape(-1, 1)).flatten()

                scaled_dfs.append(df_scaled)
        else:
            scaler.fit(all_df[sensor_cols])
            scaled_dfs = []
            for df in dfs:
                df_scaled = df.copy()
                df_scaled[sensor_cols] = scaler.transform(df_scaled[sensor_cols])
                scaled_dfs.append(df_scaled)

        return scaled_dfs

    def add_pressure_features(self, df: pd.DataFrame, window: int = 3000) -> pd.DataFrame:
            from scipy.signal import savgol_filter
            df_p = df.copy()

            if "Pressure/Raw" not in df_p.columns:
                return df_p

            p_raw = df_p["Pressure/Raw"]
            p_med = p_raw.rolling(window=101, center=True).median().bfill().ffill()

            deriv = savgol_filter(p_med, window_length=301, polyorder=2, deriv=1)
            
            df_p["Pressure/Raw.Deriv"] = deriv * self.sampling_frequency

            return df_p

    def preprocess_single_file(self, df: pd.DataFrame,
                               remove_outliers_flag: bool = False,
                               apply_filtering: bool = True,
                               apply_smoothing: bool = True,
                               add_magnitude: bool = True,
                               include_pressure: bool = False) -> pd.DataFrame:
        processed_df = df.copy()

        if remove_outliers_flag:
            processed_df = self.remove_outliers(processed_df)

        if apply_smoothing:
            processed_df = self.smooth_data(processed_df)

        if apply_filtering:
            processed_df = self.apply_signal_filtering(processed_df)

        if add_magnitude:
            processed_df = self.add_magnitude_features(processed_df)

        if include_pressure:
            processed_df = self.add_pressure_features(processed_df)

        return processed_df

    def save_processed_files(self, dfs: List[pd.DataFrame], subject_id: str):
        output_dir = self.processed_path / subject_id
        output_dir.mkdir(parents=True, exist_ok=True)

        sensor_cols_model = self.config.get("sensor_columns_model", self.sensor_columns)
        timestamp_col = self.config.get("timestamp_col", "Timestamp")

        for df in dfs:
            original_file_path = df["File"].iloc[0]
            original_path = Path(original_file_path)
            filename = original_path.name.replace(".csv", "_PROCESSED.csv")

            keep_cols = []

            if timestamp_col in df.columns:
                keep_cols.append(timestamp_col)

            for c in sensor_cols_model:
                if c in df.columns:
                    keep_cols.append(c)

            if self.label_col in df.columns:
                keep_cols.append(self.label_col)

            if self.sublabel_col in df.columns:
                keep_cols.append(self.sublabel_col)

            df_to_save = df[keep_cols]
            save_path = output_dir / filename
            df_to_save.to_csv(save_path, index=False)
            print(f"      Saved: {save_path}")

    def process_all_data(self,
                         remove_outliers_flag: bool = False,
                         apply_filtering: bool = True,
                         apply_smoothing: bool = False,
                         add_magnitude: bool = True,
                         apply_scaling: bool = False,
                         scaler_type: str = "standard",
                         group_axes: bool = True,
                         include_pressure: bool = False,
                         save_files: bool = True) -> Dict[str, List[pd.DataFrame]]:
        print("=== STARTING DATA PREPROCESSING ===")

        dfs = self.loader.load_raw_datasets()
        processed_data = {}

        grouped = {}
        for df in dfs:
            subject_id = df["Subject"].iloc[0]
            grouped.setdefault(subject_id, []).append(df)

        for subject_id, subject_dfs in grouped.items():
            print(f"  Processing subject: {subject_id} ({len(subject_dfs)} files)")

            processed_dfs = []
            for df in subject_dfs:
                processed_df = self.preprocess_single_file(
                    df,
                    remove_outliers_flag=remove_outliers_flag,
                    apply_filtering=apply_filtering,
                    apply_smoothing=apply_smoothing,
                    add_magnitude=add_magnitude,
                    include_pressure=include_pressure
                )
                processed_dfs.append(processed_df)

            if apply_scaling:
                processed_dfs = self.scale_subject_data(processed_dfs, scaler_type=scaler_type, group_axes=group_axes)

            if save_files:
                self.save_processed_files(processed_dfs, subject_id)

            processed_data[subject_id] = processed_dfs
            print(f"    Successfully processed {len(processed_dfs)} files")

        print("\n=== PREPROCESSING COMPLETED ===")
        return processed_data