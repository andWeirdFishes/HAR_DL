import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from har_dl.definitions import get_project_root


def run_metrics(path_to_results: str, final_dir_name: str) -> None:
    root = get_project_root()

    entire_path = Path(os.path.join(root, "data", "results", path_to_results))
    artifacts_path = Path(os.path.join(root, "artifacts", final_dir_name))
    artifacts_path.mkdir(parents=True, exist_ok=True)

    dfs = []
    for file in entire_path.rglob("*.csv"):
        df = pd.read_csv(file)
        dfs.append(df)

    df_all = pd.concat(dfs, axis=0).reset_index(drop=True)

    y_true = df_all["WindowLabel"]
    y_pred = df_all["Prediction"]

    labels = sorted(list(set(y_true.dropna()) | set(y_pred.dropna())))

    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(artifacts_path / "classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm_norm, cmap="Blues")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            norm = cm_norm[i, j]
            text = f"{val}\n{norm:.2f}"
            cl = "white" if i==j else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=cl)

    ax.set_xlabel("Prediction")
    ax.set_ylabel("WindowLabel")

    fig.tight_layout()
    plt.savefig(artifacts_path / "confusion_matrix.png")
    plt.close(fig)

if __name__=='__main__':
    run_metrics(path_to_results="HAR_DL_FEIT_2025", final_dir_name="results_m2_attempt2_absmax")