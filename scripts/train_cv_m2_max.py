import math
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from har_dl.config import load_config
from har_dl.definitions import get_project_root
from har_dl.architectures.cnn_lstm import CNNLSTM
from har_dl.architectures.trainer import Trainer, HARDataset

EPOCHS: int = 200
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 64

def _make_sliding_windows(
    df: pd.DataFrame, 
    sensor_cols: list[str], 
    label_col: str, 
    window: int, 
    step: int, 
    label2idx: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    data: np.ndarray = df[sensor_cols].values.astype(np.float32)
    labels: np.ndarray = df[label_col].values
    n: int = len(data)
    if n < window:
        return np.empty((0, len(sensor_cols), window)), np.empty((0,))

    win_count: int = math.floor((n - window + step) / step)
    window_list: list[np.ndarray] = []
    label_list: list[int] = []
    
    for i in range(win_count):
        start: int = i * step
        end: int = start + window
        win_data: np.ndarray = data[start:end].T 
        win_label_str: str = str(labels[end - 1]).replace("-", "_")
        
        if win_label_str in label2idx:
            window_list.append(win_data)
            label_list.append(label2idx[win_label_str])
            
    return np.array(window_list), np.array(label_list)

def build_fold_data(
    preprocessed_path: Path,
    valid_labels: list[str],
    sensor_cols: list[str],
    label_col: str,
    window_size: int,
    window_step: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, int]]:
    label2idx: dict[str, int] = {str(l).replace("-", "_"): i for i, l in enumerate(sorted(set(valid_labels)))}
    fold_windows: dict[int, list[np.ndarray]] = defaultdict(list)
    fold_labels: dict[int, list[np.ndarray]] = defaultdict(list)

    for subject_folder in sorted(preprocessed_path.iterdir()):
        if not subject_folder.is_dir():
            continue
        for csv_file in sorted(subject_folder.rglob("*.csv")):
            df: pd.DataFrame = pd.read_csv(csv_file, low_memory=False)
            if "Fold" not in df.columns or label_col not in df.columns:
                continue
            fold_id: int = int(df["Fold"].iloc[0])
            wins, labs = _make_sliding_windows(df, sensor_cols, label_col, window_size, window_step, label2idx)
            if len(wins) > 0:
                fold_windows[fold_id].append(wins)
                fold_labels[fold_id].append(labs)

    fold_arrays: dict[int, np.ndarray] = {f: np.concatenate(fold_windows[f], axis=0) for f in fold_windows}
    label_arrays: dict[int, np.ndarray] = {f: np.concatenate(fold_labels[f], axis=0) for f in fold_labels}
    return fold_arrays, label_arrays, label2idx

def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _save_loss_plot(history: list[dict], path: Path, title: str) -> None:
    epochs: list[int] = [h["epoch"] for h in history]
    losses: list[float] = [h["train_loss"] for h in history]
    accs: list[float] = [h["train_acc"] for h in history]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(epochs, losses, color="tab:red", label="Train Loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(epochs, accs, color="tab:blue", label="Train Acc")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    plt.title(title)
    fig.tight_layout()
    plt.savefig(path)
    plt.close()

def _run_single_fold(args: tuple) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    fid, all_fids, fold_arrays, label_arrays, in_ch, n_classes, class_names, art_dir = args
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train: np.ndarray = np.concatenate([fold_arrays[f] for f in all_fids if f != fid])
    y_train: np.ndarray = np.concatenate([label_arrays[f] for f in all_fids if f != fid])
    X_test, y_test = fold_arrays[fid], label_arrays[fid]
    
    model: CNNLSTM = CNNLSTM(in_channels=in_ch, num_classes=n_classes).to(device)
    trainer: Trainer = Trainer(model, device, EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY)
    fold_path: Path = art_dir / f"fold_{fid}"
    history: list[dict] = trainer.fit(DataLoader(HARDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True), fold_path)
    _save_loss_plot(history, fold_path / "loss_plot.png", f"Fold {fid} Training Progress")
    _, _, preds, targets = trainer.evaluate(DataLoader(HARDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False))
    _save_confusion_matrix(targets, preds, class_names, fold_path / "cm.png", f"Fold {fid}")
    return preds, targets, history

def main() -> None:
    mp.set_start_method("spawn", force=True)
    config: dict = load_config()
    project_root: Path = get_project_root()
    art_path: Path = project_root / "artifacts" / "results_m2_max"
    art_path.mkdir(parents=True, exist_ok=True)

    fold_arrays, label_arrays, label2idx = build_fold_data(
        project_root / "data" / "preprocessed" / "HAR_DL_FEIT_2025",
        config["second_model_labels"], config["sensor_columns_model"], "Label",
        int(config["window_size_sec"] * config["sampling_frequency"]),
        int(config["window_slide_sec"] * config["sampling_frequency"])
    )
    all_fids: list[int] = sorted(fold_arrays.keys())
    class_names: list[str] = [k for k, _ in sorted(label2idx.items(), key=lambda x: x[1])]
    
    all_p, all_t = [], []
    chunks: list[list[int]] = [all_fids[i:i+3] for i in range(0, len(all_fids), 3)]
    for chunk in chunks:
        args = [(f, all_fids, fold_arrays, label_arrays, len(config["sensor_columns_model"]), len(class_names), class_names, art_path) for f in chunk]
        with ProcessPoolExecutor(max_workers=len(chunk)) as ex:
            for p, t, _ in list(ex.map(_run_single_fold, args)):
                all_p.append(p); all_t.append(t)

    report: dict = classification_report(np.concatenate(all_t), np.concatenate(all_p), target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(art_path / "total_report.csv")
    
    print("Training final model on all data...")
    all_X, all_y = np.concatenate(list(fold_arrays.values())), np.concatenate(list(label_arrays.values()))
    dev: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_m: CNNLSTM = CNNLSTM(in_channels=len(config["sensor_columns_model"]), num_classes=len(class_names)).to(dev)
    Trainer(final_m, dev, EPOCHS, lr=LR).fit(DataLoader(HARDataset(all_X, all_y), batch_size=BATCH_SIZE, shuffle=True), art_path / "final_model")

if __name__ == "__main__":
    main()