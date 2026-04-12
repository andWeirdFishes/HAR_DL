import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from pathlib import Path
from har_dl.data.loader import DataLoader as HARDataLoader
from har_dl.architectures.cnn_lstm import CNNLSTM
from har_dl.definitions import get_project_root
from har_dl.config import load_config

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 4
IN_CHANNELS = 8

config = load_config()
WINDOW_SIZE = int(
    config["window_size_sec"] * config["sampling_frequency"]
)  # 2.5 * 50 = 125
STEP_SIZE = WINDOW_SIZE

ACTIVITY_COL = "Activity"
FOLD_COL = "Fold"
VALID_CLASSES = {"walking", "running", "upright-still", "laying"}


class WindowedDataset(Dataset):
    def __init__(self, df, feature_cols, label_col, window_size=125, step=50):
        self.windows = []
        self.labels = []

        for (subject, file), group in df.groupby(["Subject", "File"]):
            group = group.reset_index(drop=True)
            X = group[feature_cols].values
            y = group[label_col].values

            for start in range(0, len(X) - window_size + 1, step):
                window = X[start : start + window_size]
                label = pd.Series(y[start : start + window_size]).mode()[0]
                self.windows.append(window)
                self.labels.append(label)

        self.windows = torch.tensor(np.array(self.windows), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx].permute(1, 0), self.labels[idx]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = correct / total
    return acc, all_preds, all_labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    har_loader = HARDataLoader()
    df = har_loader.merge_all(processed=True)

    all_preds_dfs = []

    if df.empty:
        print("No data loaded.")
        return

    for col in [ACTIVITY_COL, FOLD_COL]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Make sure it was added to preprocessed files."
            )

    before = len(df)
    df = df[df[ACTIVITY_COL].notna() & (df[ACTIVITY_COL].str.strip() != "")]
    print(f"Dropped {before - len(df)} unlabeled rows, {len(df)} remaining.")

    before = len(df)
    df = df[df[ACTIVITY_COL].isin(VALID_CLASSES)]
    print(f"Dropped {before - len(df)} out-of-scope rows, {len(df)} remaining.")

    classes = sorted(df[ACTIVITY_COL].unique())
    print(f"Classes: {classes}")
    assert (
        len(classes) == NUM_CLASSES
    ), f"Expected {NUM_CLASSES} classes, got {len(classes)}: {classes}"
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label_encoded"] = df[ACTIVITY_COL].map(class_to_idx)

    NON_FEATURE = {
        ACTIVITY_COL,
        FOLD_COL,
        "Subject",
        "File",
        "label_encoded",
        "Label",
        "Sublabel",
        "SoftwareTimestamp",
    }
    feature_cols = [col for col in df.columns if col not in NON_FEATURE]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    assert (
        len(feature_cols) == IN_CHANNELS
    ), f"Expected {IN_CHANNELS} feature columns, got {len(feature_cols)}. Adjust IN_CHANNELS."

    results = []

    for fold in range(1, 7):
        print(f"\n===== Fold {fold} =====")

        train_df = df[df[FOLD_COL] != fold].copy()
        test_df = df[df[FOLD_COL] == fold].copy()

        print(f"  Building windows for train set...")
        train_dataset = WindowedDataset(
            train_df, feature_cols, "label_encoded", WINDOW_SIZE, STEP_SIZE
        )
        print(f"  Building windows for test set...")
        test_dataset = WindowedDataset(
            test_df, feature_cols, "label_encoded", WINDOW_SIZE, STEP_SIZE
        )

        print(
            f"  Train windows: {len(train_dataset)}, Test windows: {len(test_dataset)}"
        )

        train_loader = TorchDataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        test_loader = TorchDataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        model = CNNLSTM(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            acc, _, _ = evaluate(model, test_loader, device)
            print(f"  Epoch {epoch+1:02d}: Loss={loss:.4f}, Val-Acc={acc:.4f}")

        final_acc, preds, labels = evaluate(model, test_loader, device)
        print(f"Fold {fold} Final Accuracy: {final_acc:.4f}")
        preds_df = pd.DataFrame(
            {
                "fold": fold,
                "pred": preds,
                "true": labels,
            }
        )
        all_preds_dfs.append(preds_df)
        results.append({"fold": fold, "accuracy": final_acc})

    results_df = pd.DataFrame(results)
    results_path = Path(get_project_root()) / "data" / "results" / "6fcv_no_overlap.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    preds_path = (
        Path(get_project_root()) / "data" / "results" / "6fcv_no_overlap_preds.csv"
    )
    pd.concat(all_preds_dfs, ignore_index=True).to_csv(preds_path, index=False)
    print(f"Predictions saved to: {preds_path}")

    print(
        f"\nMean accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}"
    )
    print(f"Results saved to: {results_path}")
    print(results_df)


if __name__ == "__main__":
    main()
