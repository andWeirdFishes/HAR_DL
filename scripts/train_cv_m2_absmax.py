import copy
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict, Counter
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from har_dl.config import load_config
from har_dl.definitions import get_project_root
from har_dl.architectures.cnn_lstm_absmax import CNNLSTMAbsMax
from har_dl.architectures.trainer import Trainer, HARDataset

EPOCHS: int = 200
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 64


def _make_sliding_windows(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int,
    step: int,
    label2idx: dict,
    label_col: str,
    sanity_label: str,
    sublabel_col: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    data = df[sensor_cols].values.astype(np.float32)
    win_labels = df[label_col].values
    orig_labels = df[sanity_label].values
    sublabels = df[sublabel_col].values

    n = len(data)
    if n < window:
        return np.empty((0, len(sensor_cols), window)), np.empty((0,)), []

    win_count = math.floor((n - window + step) / step)
    w_list, l_list, meta_list = [], [], []

    for i in range(win_count):
        s = i * step
        e = s + window
        w_slice = win_labels[s:e]

        sequence = []
        if len(w_slice) > 0:
            sequence.append(str(w_slice[0]).replace("-", "_"))
            for j in range(1, len(w_slice)):
                lbl = str(w_slice[j]).replace("-", "_")
                if lbl != sequence[-1]:
                    sequence.append(lbl)

        assigned_label: str | None = None

        if len(sequence) == 1:
            assigned_label = sequence[0]
        elif len(sequence) == 3:
            if sequence[0] == "upright_still" and sequence[2] == "upright_still":
                trans_type = sequence[1]
                start_orig = str(orig_labels[s]).replace("-", "_")
                end_orig = str(orig_labels[e - 1]).replace("-", "_")
                if trans_type == "transition_up" and start_orig == "sitting" and end_orig == "standing":
                    assigned_label = "transition_up"
                elif trans_type == "transition_down" and start_orig == "standing" and end_orig == "sitting":
                    assigned_label = "transition_down"

        if assigned_label and assigned_label in label2idx:
            w_list.append(data[s:e].T)
            l_list.append(label2idx[assigned_label])
            meta_list.append({
                "WindowLabel": assigned_label,
                "Label": str(orig_labels[s]).replace("-", "_"),
                "Sublabel": str(sublabels[s]).replace("-", "_"),
            })

    return np.array(w_list), np.array(l_list), meta_list


def build_fold_data(
    pre_path: Path,
    valid_l: list[str],
    s_cols: list[str],
    l_col: str,
    s_col: str,
    sub_col: str,
    w_size: int,
    w_step: int,
) -> tuple[dict, dict, dict, dict]:
    l2idx = {str(l).replace("-", "_"): i for i, l in enumerate(sorted(set(valid_l)))}
    f_w: dict = defaultdict(list)
    f_l: dict = defaultdict(list)
    file_info: dict = {}

    for sub in sorted(pre_path.iterdir()):
        if not sub.is_dir():
            continue
        subject_id = sub.name
        for csv in sorted(sub.rglob("*.csv")):
            df = pd.read_csv(csv)
            fid = int(df["Fold"].iloc[0])
            task_name = "task1" if csv.name.startswith("task1") else "task3_1"
            file_key = f"{subject_id}_{task_name}"

            wins, labs, meta = _make_sliding_windows(df, s_cols, w_size, w_step, l2idx, l_col, s_col, sub_col)
            if len(wins) == 0:
                continue

            f_w[fid].append(wins)
            f_l[fid].append(labs)

            if file_key not in file_info:
                file_info[file_key] = {
                    "subject_id": subject_id,
                    "task": task_name,
                    "windows": [],
                    "labels": [],
                    "metadata": [],
                }

            file_info[file_key]["windows"].append(wins)
            file_info[file_key]["labels"].append(labs)
            file_info[file_key]["metadata"].extend(meta)

    for key in file_info:
        file_info[key]["windows"] = np.concatenate(file_info[key]["windows"])
        file_info[key]["labels"] = np.concatenate(file_info[key]["labels"])

    return (
        {f: np.concatenate(f_w[f]) for f in f_w},
        {f: np.concatenate(f_l[f]) for f in f_l},
        l2idx,
        file_info,
    )


def sanity_check_transitions(l_arr: dict, l2i: dict) -> None:
    all_labels = np.concatenate(list(l_arr.values()))
    counts = Counter(all_labels.tolist())
    print("\n--- Transition Window Sanity Check ---")
    for name in ["transition_up", "transition_down"]:
        if name not in l2i:
            print(f"[WARNING] {name.upper()} NOT IN LABEL MAP — CHECK CONFIG")
            continue
        idx = l2i[name]
        count = counts.get(idx, 0)
        if count == 0:
            print(f"[WARNING] NO {name.upper()} WINDOWS FOUND — CHECK WINDOWING LOGIC OR RAW DATA")
        else:
            print(f"[OK] {name}: {count} windows across all folds")
    print("--------------------------------------\n")


def _save_fold_loss_plot(hist: list[dict], path: Path, title: str) -> None:
    epochs = [h["epoch"] for h in hist]
    losses = [h["train_loss"] for h in hist]
    accs = [h["train_acc"] for h in hist]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(epochs, losses, color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(epochs, accs, color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    plt.title(title)
    fig.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_train_val_loss_plot(hist: list[dict], path: Path, title: str) -> None:
    epochs = [h["epoch"] for h in hist]
    train_losses = [h["train_loss"] for h in hist]
    val_losses = [h["val_loss"] for h in hist]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(epochs, train_losses, color="tab:red", label="Train Loss")
    ax.plot(epochs, val_losses, color="tab:blue", label="Val Loss")
    ax.legend()
    plt.title(title)
    fig.tight_layout()
    plt.savefig(path)
    plt.close()


def _train_with_val(
    model: nn.Module,
    dev: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    save_dir: Path,
) -> list[dict]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_correct += (logits.argmax(dim=1) == y).sum().item()
            tr_total += x.size(0)
        scheduler.step()
        train_loss = tr_loss / tr_total
        train_acc = tr_correct / tr_total

        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_sum += loss.item() * x.size(0)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += x.size(0)
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": scheduler.get_last_lr()[0],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"    epoch {epoch:>3}/{epochs}"
                f"  train_loss={train_loss:.4f}"
                f"  val_loss={val_loss:.4f}"
                f"  val_acc={val_acc:.4f}"
            )

    torch.save(best_state, save_dir / "best_model.pt")
    torch.save(model.state_dict(), save_dir / "last_model.pt")
    model.load_state_dict(best_state)
    return history


def _run_single_fold(args: tuple) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    fid, all_fids, f_arr, l_arr, in_ch, n_cls, art_dir = args
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = np.concatenate([f_arr[f] for f in all_fids if f != fid])
    y_tr = np.concatenate([l_arr[f] for f in all_fids if f != fid])

    model = CNNLSTMAbsMax(in_channels=in_ch, num_classes=n_cls).to(dev)
    trainer = Trainer(model, dev, EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY)

    f_path = art_dir / f"fold_{fid}"
    hist = trainer.fit(DataLoader(HARDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True), f_path)
    _save_fold_loss_plot(hist, f_path / "loss_plot.png", f"Fold {fid} Training Progress")

    _, _, p, t = trainer.evaluate(DataLoader(HARDataset(f_arr[fid], l_arr[fid]), batch_size=BATCH_SIZE))
    return p, t, hist


def main() -> None:
    mp.set_start_method("spawn", force=True)
    cfg = load_config()
    root = get_project_root()
    art = root / "artifacts" / "results_m2_absmax"
    art.mkdir(parents=True, exist_ok=True)

    dataset_name = "HAR_DL_FEIT_2025"

    f_a, l_a, l2i, file_info = build_fold_data(
        root / "data" / "preprocessed" / dataset_name,
        cfg["second_model_labels"],
        cfg["sensor_columns_model"],
        "WindowLabel",
        "Label",
        "Sublabel",
        int(cfg["window_size_sec"] * cfg["sampling_frequency"]),
        int(cfg["window_slide_sec"] * cfg["sampling_frequency"]),
    )

    sanity_check_transitions(l_a, l2i)

    ids = sorted(f_a.keys())
    names = [k for k, _ in sorted(l2i.items(), key=lambda x: x[1])]
    in_ch = len(cfg["sensor_columns_model"])
    n_cls = len(names)

    all_p, all_t = [], []
    for chunk in [ids[i:i + 3] for i in range(0, len(ids), 3)]:
        args = [(f, ids, f_a, l_a, in_ch, n_cls, art) for f in chunk]
        with ProcessPoolExecutor(max_workers=len(chunk)) as ex:
            for p, t, _ in list(ex.map(_run_single_fold, args)):
                all_p.append(p)
                all_t.append(t)

    pd.DataFrame(
        classification_report(
            np.concatenate(all_t),
            np.concatenate(all_p),
            target_names=names,
            labels=list(range(n_cls)),
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(art / "total_report.csv")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_fold = ids[-1]
    train_folds = [f for f in ids if f != val_fold]

    print(f"\nFinal model (with val): training on folds {train_folds}, validating on fold {val_fold}")

    X_val_tr = np.concatenate([f_a[f] for f in train_folds])
    y_val_tr = np.concatenate([l_a[f] for f in train_folds])
    X_val_vl = f_a[val_fold]
    y_val_vl = l_a[val_fold]

    final_with_val = CNNLSTMAbsMax(in_channels=in_ch, num_classes=n_cls).to(dev)
    val_hist = _train_with_val(
        final_with_val, dev,
        DataLoader(HARDataset(X_val_tr, y_val_tr), batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(HARDataset(X_val_vl, y_val_vl), batch_size=BATCH_SIZE),
        EPOCHS, LR, WEIGHT_DECAY,
        art / "final_model_with_val",
    )
    _save_train_val_loss_plot(val_hist, art / "final_model_with_val" / "train_val_loss.png", "Final Model — Train vs Val Loss")

    print("\nFull model: training on all folds")
    X_all = np.concatenate(list(f_a.values()))
    y_all = np.concatenate(list(l_a.values()))
    full_model = CNNLSTMAbsMax(in_channels=in_ch, num_classes=n_cls).to(dev)
    trainer_full = Trainer(full_model, dev, EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY)
    trainer_full.fit(DataLoader(HARDataset(X_all, y_all), batch_size=BATCH_SIZE, shuffle=True), art / "final_model_full")

    idx2name = {v: k for k, v in l2i.items()}

    for file_key, info in file_info.items():
        X = info["windows"]
        y_true = info["labels"]
        metadata = info["metadata"]
        subject_id = info["subject_id"]
        task = info["task"]

        _, _, y_pred, _ = trainer_full.evaluate(DataLoader(HARDataset(X, y_true), batch_size=BATCH_SIZE))

        pred_df = pd.DataFrame({
            "SubjectID": [subject_id] * len(y_pred),
            "WindowLabel": [m["WindowLabel"] for m in metadata],
            "Prediction": [idx2name[p] for p in y_pred],
            "Label": [m["Label"] for m in metadata],
            "Sublabel": [m["Sublabel"] for m in metadata],
        })

        out_dir = root / "data" / "results" / dataset_name / subject_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"predictions_{task}.csv"
        pred_df.to_csv(out_path, index=False)
        print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
