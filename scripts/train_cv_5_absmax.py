import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from har_dl.config import load_config
from har_dl.definitions import get_project_root
# Updated import to use the AbsMax variant
from har_dl.architectures.cnn_lstm_absmax import CNNLSTMAbsMax
from har_dl.architectures.trainer import Trainer, HARDataset
from emteqai.utils.processing.data.segmentation import find_label_segments


EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64


def _make_windows(data: np.ndarray, window: int, step: int) -> np.ndarray:
    n = len(data)
    if n < window:
        return np.empty((0, data.shape[1], window), dtype=np.float32)
    win_count = math.floor((n - window + step) / step)
    slices = np.stack(
        [data[i * step: i * step + window] for i in range(win_count)]
    )
    return slices.transpose(0, 2, 1).astype(np.float32)


def _normalize_label(label: str) -> str:
    return label.replace("-", "_")


def build_fold_data(
    preprocessed_path: Path,
    valid_labels: list[str],
    sensor_cols: list[str],
    activity_col: str,
    window_size: int,
    window_step: int,
) -> tuple[dict, dict, dict]:
    valid_set = set(_normalize_label(l) for l in valid_labels)
    label2idx = {l: i for i, l in enumerate(sorted(valid_set))}

    fold_windows: dict[int, list] = defaultdict(list)
    fold_labels: dict[int, list] = defaultdict(list)

    for subject_folder in sorted(preprocessed_path.iterdir()):
        if not subject_folder.is_dir():
            continue
        for csv_file in sorted(subject_folder.rglob("*.csv")):
            try:
                df = pd.read_csv(csv_file, low_memory=False)
            except Exception as exc:
                print(f"  [skip] {csv_file.name}: {exc}")
                continue

            if "Fold" not in df.columns or activity_col not in df.columns:
                continue

            fold_id = int(df["Fold"].iloc[0])
            df[activity_col] = df[activity_col].astype(str).apply(_normalize_label)
            df = df[df[activity_col].isin(valid_set)].reset_index(drop=True)

            if df.empty:
                continue

            missing = [c for c in sensor_cols if c not in df.columns]
            if missing:
                print(f"  [skip] {csv_file.name}: missing sensor cols {missing}")
                continue

            segments = find_label_segments(df, activity_col)

            for _, seg in segments.iterrows():
                label = seg["label"]
                if label not in label2idx:
                    continue
                start, end = int(seg["start"]), int(seg["end"])
                seg_data = df.iloc[start:end][sensor_cols].values.astype(np.float32)
                windows = _make_windows(seg_data, window_size, window_step)
                if len(windows) == 0:
                    continue
                fold_windows[fold_id].extend(windows)
                fold_labels[fold_id].extend([label2idx[label]] * len(windows))

    fold_arrays = {
        fid: np.stack(fold_windows[fid]) for fid in fold_windows if fold_windows[fid]
    }
    label_arrays = {
        fid: np.array(fold_labels[fid]) for fid in fold_labels if fold_labels[fid]
    }
    return fold_arrays, label_arrays, label2idx


def _save_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path, title: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title or save_path.parent.name.replace("_", " ").title(), fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _make_loader(windows: np.ndarray, labels: np.ndarray, shuffle: bool) -> DataLoader:
    return DataLoader(
        HARDataset(windows, labels),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
    )


def _run_fold(
    test_fold: int,
    all_fold_ids: list[int],
    fold_arrays: dict,
    label_arrays: dict,
    in_channels: int,
    num_classes: int,
    class_names: list[str],
    device: torch.device,
    artifact_dir: Path,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    train_folds = [f for f in all_fold_ids if f != test_fold]

    train_windows = np.concatenate([fold_arrays[f] for f in train_folds], axis=0)
    train_labels = np.concatenate([label_arrays[f] for f in train_folds], axis=0)
    test_windows = fold_arrays[test_fold]
    test_labels = label_arrays[test_fold]

    print(f"  train={len(train_labels):,}  test={len(test_labels):,}")

    train_loader = _make_loader(train_windows, train_labels, shuffle=True)
    test_loader = _make_loader(test_windows, test_labels, shuffle=False)

    # Instantiate CNNLSTMAbsMax instead of CNNLSTM
    model = CNNLSTMAbsMax(in_channels=in_channels, num_classes=num_classes).to(device)
    trainer = Trainer(model, device, EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY)

    fold_dir = artifact_dir / f"fold_{test_fold}"
    history = trainer.fit(train_loader, save_dir=fold_dir / "checkpoints")

    _, test_acc, preds, targets = trainer.evaluate(test_loader)
    print(f"  test_acc={test_acc:.4f}")

    cm = confusion_matrix(targets, preds)
    _save_confusion_matrix(cm, class_names, fold_dir / "confusion_matrix.png")

    report = classification_report(
        targets, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    metrics_rows = [
        {"fold": test_fold, "class": cls, **report[cls]}
        for cls in class_names
    ]
    metrics_rows.append(
        {
            "fold": test_fold,
            "class": "macro_avg",
            **report["macro avg"],
        }
    )
    metrics_rows.append(
        {
            "fold": test_fold,
            "class": "overall_accuracy",
            "precision": report["accuracy"],
        }
    )

    pd.DataFrame(metrics_rows).to_csv(fold_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)

    return metrics_rows, cm, preds, targets


def main():
    config = load_config()
    project_root = get_project_root()

    preprocessed_path = (
        Path(project_root) / "data" / "preprocessed" / "HAR_DL_FEIT_2025"
    )
    # Target folder changed to results_absmax
    artifacts_path = Path(project_root) / "artifacts" / "results_absmax"
    artifacts_path.mkdir(parents=True, exist_ok=True)

    valid_labels = config["first_model_labels"]
    sensor_cols = config["sensor_columns_model"]
    activity_col = "Activity"
    sampling_frequency = config["sampling_frequency"]
    window_size = int(config["window_size_sec"] * sampling_frequency)
    window_step = int(config["window_slide_sec"] * sampling_frequency)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"window_size={window_size} samples  window_step={window_step} samples")

    print("\nbuilding fold data...")
    fold_arrays, label_arrays, label2idx = build_fold_data(
        preprocessed_path,
        valid_labels,
        sensor_cols,
        activity_col,
        window_size,
        window_step,
    )

    idx2label = {v: k for k, v in label2idx.items()}
    class_names = [idx2label[i] for i in range(len(label2idx))]
    num_classes = len(label2idx)
    in_channels = len(sensor_cols)
    all_fold_ids = sorted(fold_arrays.keys())

    print(f"classes: {class_names}")
    print(f"folds: {all_fold_ids}")
    for fid in all_fold_ids:
        print(f"  fold {fid}: {len(label_arrays[fid]):,} windows")

    pd.DataFrame(
        [{"label": l, "idx": i} for l, i in label2idx.items()]
    ).to_csv(artifacts_path / "label_map.csv", index=False)

    all_metrics = []
    all_cms = []
    all_preds = []
    all_targets = []
    for test_fold in all_fold_ids:
        print(f"\n{'='*55}")
        print(f"fold {test_fold} → TEST")
        fold_metrics, cm, preds, targets = _run_fold(
            test_fold,
            all_fold_ids,
            fold_arrays,
            label_arrays,
            in_channels,
            num_classes,
            class_names,
            device,
            artifacts_path,
        )
        all_metrics.extend(fold_metrics)
        all_cms.append(cm)
        all_preds.append(preds)
        all_targets.append(targets)

    pd.DataFrame(all_metrics).to_csv(
        artifacts_path / "all_folds_metrics.csv", index=False
    )

    total_cm = np.sum(all_cms, axis=0)
    _save_confusion_matrix(
        cm=total_cm,
        class_names=class_names,
        save_path=artifacts_path / "confusion_matrix_total.png",
        title="All Folds — Aggregated Confusion Matrix (AbsMax)",
    )

    total_preds = np.concatenate(all_preds)
    total_targets = np.concatenate(all_targets)
    total_report = classification_report(
        total_targets, total_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    total_rows = [{"class": cls, **total_report[cls]} for cls in class_names]
    total_rows.append({"class": "macro_avg", **total_report["macro avg"]})
    total_rows.append({"class": "weighted_avg", **total_report["weighted avg"]})
    total_rows.append({"class": "overall_accuracy", "precision": total_report["accuracy"]})
    pd.DataFrame(total_rows).to_csv(artifacts_path / "total_report.csv", index=False)

    print(f"\n{'='*55}")
    print("training final model on ALL folds...")

    all_windows = np.concatenate(list(fold_arrays.values()), axis=0)
    all_labels = np.concatenate(list(label_arrays.values()), axis=0)
    full_loader = _make_loader(all_windows, all_labels, shuffle=True)

    # Use AbsMax for the final model
    final_model = CNNLSTMAbsMax(in_channels=in_channels, num_classes=num_classes).to(device)
    final_trainer = Trainer(
        final_model, device, EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY
    )
    final_dir = artifacts_path / "final_model"
    final_history = final_trainer.fit(full_loader, save_dir=final_dir / "checkpoints")

    pd.DataFrame(final_history).to_csv(
        final_dir / "training_history.csv", index=False
    )

    print(f"\ndone. artifacts at: {artifacts_path}")


if __name__ == "__main__":
    main()