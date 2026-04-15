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
from har_dl.architectures.cnn_lstm_absmax import CNNLSTMAbsMax
from har_dl.architectures.trainer import Trainer, HARDataset

EPOCHS: int = 200
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 64

def _make_sliding_windows(df, sensor_cols, label_col, window, step, label2idx) -> tuple[np.ndarray, np.ndarray]:
    data, labels = df[sensor_cols].values.astype(np.float32), df[label_col].values
    n = len(data)
    if n < window: return np.empty((0, len(sensor_cols), window)), np.empty((0,))
    win_count = math.floor((n - window + step) / step)
    w_list, l_list = [], []
    for i in range(win_count):
        e = i * step + window
        s_label = str(labels[e - 1]).replace("-", "_")
        if s_label in label2idx:
            w_list.append(data[i * step:e].T)
            l_list.append(label2idx[s_label])
    return np.array(w_list), np.array(l_list)

def build_fold_data(pre_path, valid_l, s_cols, l_col, w_size, w_step) -> tuple[dict, dict, dict]:
    l2idx = {str(l).replace("-", "_"): i for i, l in enumerate(sorted(set(valid_l)))}
    f_w, f_l = defaultdict(list), defaultdict(list)
    for sub in sorted(pre_path.iterdir()):
        if not sub.is_dir(): continue
        for csv in sorted(sub.rglob("*.csv")):
            df = pd.read_csv(csv)
            fid = int(df["Fold"].iloc[0])
            wins, labs = _make_sliding_windows(df, s_cols, l_col, w_size, w_step, l2idx)
            if len(wins) > 0: f_w[fid].append(wins); f_l[fid].append(labs)
    return {f: np.concatenate(f_w[f]) for f in f_w}, {f: np.concatenate(f_l[f]) for f in f_l}, l2idx

def _save_loss_plot(hist, path, title) -> None:
    e, l, a = [h["epoch"] for h in hist], [h["train_loss"] for h in hist], [h["train_acc"] for h in hist]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(e, l, color="tab:red"); ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx(); ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(e, a, color="tab:blue"); ax2.tick_params(axis="y", labelcolor="tab:blue")
    plt.title(title); fig.tight_layout(); plt.savefig(path); plt.close()

def _run_single_fold(args) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    fid, all_fids, f_arr, l_arr, in_ch, n_cls, names, art_dir = args
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr = np.concatenate([f_arr[f] for f in all_fids if f != fid])
    y_tr = np.concatenate([l_arr[f] for f in all_fids if f != fid])
    model = CNNLSTMAbsMax(in_channels=in_ch, num_classes=n_cls).to(dev)
    trainer = Trainer(model, dev, EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY)
    f_path = art_dir / f"fold_{fid}"
    hist = trainer.fit(DataLoader(HARDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True), f_path)
    _save_loss_plot(hist, f_path / "loss_plot.png", f"Fold {fid} Progress")
    _, _, p, t = trainer.evaluate(DataLoader(HARDataset(f_arr[fid], l_arr[fid]), batch_size=BATCH_SIZE))
    return p, t, hist

def main() -> None:
    mp.set_start_method("spawn", force=True)
    cfg = load_config(); root = get_project_root()
    art = root / "artifacts" / "results_m2_absmax"
    art.mkdir(parents=True, exist_ok=True)
    f_a, l_a, l2i = build_fold_data(root / "data" / "preprocessed" / "HAR_DL_FEIT_2025", cfg["second_model_labels"], cfg["sensor_columns_model"], "Label", int(cfg["window_size_sec"] * cfg["sampling_frequency"]), int(cfg["window_slide_sec"] * cfg["sampling_frequency"]))
    ids = sorted(f_a.keys()); names = [k for k, _ in sorted(l2i.items(), key=lambda x: x[1])]
    all_p, all_t = [], []
    for chunk in [ids[i:i+3] for i in range(0, len(ids), 3)]:
        args = [(f, ids, f_a, l_a, len(cfg["sensor_columns_model"]), len(names), names, art) for f in chunk]
        with ProcessPoolExecutor(max_workers=len(chunk)) as ex:
            for p, t, _ in list(ex.map(_run_single_fold, args)):
                all_p.append(p); all_t.append(t)
    pd.DataFrame(classification_report(np.concatenate(all_t), np.concatenate(all_p), target_names=names, output_dict=True)).transpose().to_csv(art / "total_report.csv")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final = CNNLSTMAbsMax(in_channels=len(cfg["sensor_columns_model"]), num_classes=len(names)).to(dev)
    Trainer(final, dev, EPOCHS, lr=LR).fit(DataLoader(HARDataset(np.concatenate(list(f_a.values())), np.concatenate(list(l_a.values()))), batch_size=BATCH_SIZE, shuffle=True), art / "final_model")

if __name__ == "__main__":
    main()