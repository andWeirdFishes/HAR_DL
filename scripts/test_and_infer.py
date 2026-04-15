import math
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from har_dl.config import load_config
from har_dl.data.preprocessor import DataPreprocessor
from har_dl.definitions import get_project_root
from har_dl.architectures.cnn_lstm import CNNLSTM
from har_dl.architectures.cnn_lstm_absmax import CNNLSTMAbsMax
from emteqai.utils.processing.data.segmentation import find_label_segments


def _load_label_map(artifacts_path: Path) -> tuple[dict, dict]:
    lm = pd.read_csv(artifacts_path / "label_map.csv")
    label2idx = dict(zip(lm["label"], lm["idx"]))
    idx2label = dict(zip(lm["idx"], lm["label"]))
    return label2idx, idx2label


def _preprocess_dataset(dataset_path: Path, config: dict) -> dict[str, list[pd.DataFrame]]:
    cfg = config.copy()
    cfg["raw_path"] = str(dataset_path.parent)
    cfg["f_low_cutoff"] = None
    preprocessor = DataPreprocessor(config=cfg)

    result: dict[str, list[pd.DataFrame]] = {}

    for subject_folder in sorted(dataset_path.iterdir()):
        if not subject_folder.is_dir():
            continue
        subject_id = subject_folder.name
        dfs = []

        for csv_file in sorted(subject_folder.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file, low_memory=False)
            except Exception as exc:
                print(f"  [skip] {csv_file.name}: {exc}")
                continue

            missing = [c for c in cfg["sensor_columns_raw"] if c not in df.columns]
            if missing:
                print(f"  [skip] {csv_file.name}: missing sensor cols {missing}")
                continue

            if df.empty:
                print(f"  [skip] {csv_file.name}: empty file")
                continue

            df["Subject"] = subject_id
            df["File"] = str(csv_file.relative_to(dataset_path))

            df = preprocessor.preprocess_single_file(
                df,
                remove_outliers_flag=False,
                apply_filtering=True,
                apply_smoothing=False,
                add_magnitude=True,
            )
            dfs.append(df)
            print(f"  preprocessed: {csv_file.name}")

        if dfs:
            result[subject_id] = dfs

    return result


def _make_windows(
    data: np.ndarray, timestamps: np.ndarray, window: int, step: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(data)
    if n < window:
        return np.empty((0, data.shape[1], window), dtype=np.float32), np.empty(0)
    win_count = math.floor((n - window + step) / step)
    windows = np.stack([data[i * step: i * step + window] for i in range(win_count)])
    midpoints = np.array(
        [np.median(timestamps[i * step: i * step + window]) for i in range(win_count)]
    )
    return windows.transpose(0, 2, 1).astype(np.float32), midpoints

def _load_model(
    model_path: Path, in_channels: int, num_classes: int, device: torch.device
) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_type" in checkpoint:
        model_type = checkpoint["model_type"]
    else:
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
        keys = list(state_dict.keys())

        if any("absmax" in k.lower() for k in keys):
            model_type = "cnn_lstm_absmax"
        else:
            model_type = "cnn_lstm"

    if model_type == "cnn_lstm_absmax":
        model = CNNLSTMAbsMax(in_channels=in_channels, num_classes=num_classes)
    else:
        model = CNNLSTM(in_channels=in_channels, num_classes=num_classes)

    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def _predict(
    model: torch.nn.Module, windows: np.ndarray, device: torch.device, batch_size: int = 64
) -> np.ndarray:
    all_preds = []
    tensor = torch.tensor(windows, dtype=torch.float32)
    for i in range(0, len(tensor), batch_size):
        preds = model(tensor[i: i + batch_size].to(device)).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
    return np.concatenate(all_preds)


def _normalize(label: str) -> str:
    return label.replace("-", "_")


def _get_results_dir(dataset_path: Path, model_path: Path, project_root: Path) -> Path:
    base = project_root / "data" / "results"
    base.mkdir(parents=True, exist_ok=True)
    run_idx = 0
    name = f"{dataset_path.name}_{model_path.stem}"
    while (base / f"{name}_{run_idx}").exists():
        run_idx += 1
    results_dir = base / f"{name}_{run_idx}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def test(
    model_path: str | Path,
    dataset_path: str | Path,
    artifacts_path: str | Path,
    classes_to_keep: list[str],
    rename_classes: dict[str, str],
) -> dict[str, pd.DataFrame]:
    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    artifacts_path = Path(artifacts_path)
    project_root = Path(get_project_root())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config()
    label2idx, idx2label = _load_label_map(artifacts_path)
    sensor_cols = config["sensor_columns_model"]
    timestamp_col = config["timestamp_col"]
    label_col = config["label_col"]
    window_size = int(config["window_size_sec"] * config["sampling_frequency"])
    window_step = int(config["window_slide_sec"] * config["sampling_frequency"])

    model = _load_model(model_path, len(sensor_cols), len(label2idx), device)
    processed = _preprocess_dataset(dataset_path, config)
    results_dir = _get_results_dir(dataset_path, model_path, project_root)

    keep_set = {_normalize(c) for c in classes_to_keep}
    rename_map = {_normalize(k): _normalize(v) for k, v in rename_classes.items()}

    all_results: dict[str, pd.DataFrame] = {}

    for subject_id, dfs in processed.items():
        for df in dfs:
            file_name = Path(df["File"].iloc[0]).stem if "File" in df.columns else subject_id

            if label_col not in df.columns:
                print(f"  [skip] {file_name}: no label column for test()")
                continue

            df = df.copy()
            df[label_col] = df[label_col].astype(str).apply(_normalize)
            df = df[df[label_col].isin(keep_set)].reset_index(drop=True)

            if df.empty:
                print(f"  [skip] {file_name}: no rows left after filtering to classes_to_keep")
                continue

            missing = [c for c in sensor_cols if c not in df.columns]
            if missing:
                print(f"  [skip] {file_name}: missing sensor cols {missing}")
                continue

            segments = find_label_segments(df, label_col)
            rows = []

            for _, seg in segments.iterrows():
                raw_label = seg["label"]
                model_label = rename_map.get(raw_label, raw_label)
                if model_label not in label2idx:
                    continue

                start, end = int(seg["start"]), int(seg["end"])
                seg_data = df.iloc[start:end][sensor_cols].values
                seg_ts = (
                    df.iloc[start:end][timestamp_col].values
                    if timestamp_col in df.columns
                    else np.arange(end - start, dtype=float)
                )

                windows, midpoints = _make_windows(seg_data, seg_ts, window_size, window_step)
                if len(windows) == 0:
                    continue

                preds = _predict(model, windows, device)

                for ts, pred_idx in zip(midpoints, preds):
                    rows.append({
                        "Time": ts,
                        "TrueLabel": raw_label,
                        "Prediction": idx2label[pred_idx],
                    })

            if not rows:
                print(f"  [skip] {file_name}: no windows produced")
                continue

            result_df = pd.DataFrame(rows)
            all_results[file_name] = result_df

            subject_dir = results_dir / subject_id
            subject_dir.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(subject_dir / f"{file_name}_test.csv", index=False)
            print(f"  saved: {subject_dir / f'{file_name}_test.csv'}")

    return all_results


def infer(
    model_path: str | Path,
    dataset_path: str | Path,
    artifacts_path: str | Path,
) -> dict[str, pd.DataFrame]:
    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    artifacts_path = Path(artifacts_path)
    project_root = Path(get_project_root())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config()
    label2idx, idx2label = _load_label_map(artifacts_path)
    sensor_cols = config["sensor_columns_model"]
    timestamp_col = config["timestamp_col"]
    window_size = int(config["window_size_sec"] * config["sampling_frequency"])
    window_step = int(config["window_slide_sec"] * config["sampling_frequency"])

    model = _load_model(model_path, len(sensor_cols), len(label2idx), device)
    processed = _preprocess_dataset(dataset_path, config)
    results_dir = _get_results_dir(dataset_path, model_path, project_root)

    all_results: dict[str, pd.DataFrame] = {}

    for subject_id, dfs in processed.items():
        for df in dfs:
            file_name = Path(df["File"].iloc[0]).stem if "File" in df.columns else subject_id

            missing = [c for c in sensor_cols if c not in df.columns]
            if missing:
                print(f"  [skip] {file_name}: missing sensor cols {missing}")
                continue

            data = df[sensor_cols].values
            ts = (
                df[timestamp_col].values
                if timestamp_col in df.columns
                else np.arange(len(df), dtype=float)
            )

            windows, midpoints = _make_windows(data, ts, window_size, window_step)
            if len(windows) == 0:
                print(f"  [skip] {file_name}: no windows produced")
                continue

            preds = _predict(model, windows, device)

            result_df = pd.DataFrame({
                "Time": midpoints,
                "Prediction": [idx2label[p] for p in preds],
            })

            all_results[file_name] = result_df

            subject_dir = results_dir / subject_id
            subject_dir.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(subject_dir / f"{file_name}_infer.csv", index=False)
            print(f"  saved: {subject_dir / f'{file_name}_infer.csv'}")

    return all_results


if __name__ == "__main__":
    results = test(
        model_path="artifacts/results_absmax/final_model/checkpoints/best_model.pt",
        dataset_path="data/raw/HAR_jumping",
        artifacts_path="artifacts/results_absmax",
        classes_to_keep=["walking", "sitting", "standing", "jogging"],
        rename_classes={
            "walking": "walking",
            "jogging": "running",
            "sitting": "upright_still",
            "standing": "upright_still",
        },
    )