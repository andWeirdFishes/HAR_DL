import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from har_dl.definitions import get_project_root

TRANSITION_CLASSES = {"transition_up", "transition_down"}
DYNAMIC_CLASSES = {"walking", "running", "stairs_up", "stairs_down"}
STATIC_CLASSES = {"sitting", "standing", "laying", "running", "walking", "stairs_up", "stairs_down"}


def post_process_session(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    preds = df["Prediction"].tolist()

    first_label = df["Label"].iloc[0]
    current_static_state = first_label if first_label in {"sitting", "standing"} else "sitting"

    pp_predictions = []

    for i, pred in enumerate(preds):
        if pred == "transition_up":
            current_static_state = "standing"
            pp_predictions.append("transition_up")

        elif pred == "transition_down":
            current_static_state = "sitting"
            pp_predictions.append("transition_down")

        elif pred in DYNAMIC_CLASSES:
            current_static_state = "standing"
            pp_predictions.append(pred)

        elif pred == "upright_still":
            j = i - 1
            while j >= 0 and preds[j] in DYNAMIC_CLASSES:
                j -= 1
            if j < i - 1 and j >= 0 and preds[j] not in TRANSITION_CLASSES:
                current_static_state = "standing"

            pp_predictions.append(current_static_state)

        else:
            pp_predictions.append(pred)

    df["PP_Prediction"] = pp_predictions
    return df


def run_post_processing_metrics(path_to_results: str, final_dir_name: str) -> None:
    root = get_project_root()
    entire_path = Path(os.path.join(root, "data", "results", path_to_results))
    artifacts_path = Path(os.path.join(root, "artifacts", final_dir_name))
    artifacts_path.mkdir(parents=True, exist_ok=True)

    processed_dfs = []
    for file in sorted(entire_path.rglob("*.csv")):
        df = pd.read_csv(file)
        if df.empty:
            continue
        df_processed = post_process_session(df)
        processed_dfs.append(df_processed)
        df_processed.to_csv(file, index=False)

    df_all = pd.concat(processed_dfs, axis=0).reset_index(drop=True)

    y_true = df_all["Label"]
    y_pred = df_all["PP_Prediction"]

    evaluation_mask = (
        y_true.isin(STATIC_CLASSES) &
        y_pred.isin(STATIC_CLASSES)
    )
    y_true_filtered = y_true[evaluation_mask]
    y_pred_filtered = y_pred[evaluation_mask]

    labels = sorted(list(set(y_true_filtered) | set(y_pred_filtered)))

    report_dict = classification_report(
        y_true_filtered, y_pred_filtered,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report_dict).transpose().to_csv(artifacts_path / "pp_classification_report.csv")

    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float), row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm_norm, cmap="Blues")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i==j:
                cl = "white"
            else:
                cl = "black"
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=9, color=cl)

    ax.set_xlabel("PP_Prediction")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    plt.savefig(artifacts_path / "pp_confusion_matrix.png")
    plt.close(fig)


if __name__ == "__main__":
    run_post_processing_metrics(
        path_to_results="HAR_DL_FEIT_2025",
        final_dir_name="results_m2_attempt2_PP_anchors_absmax",
    )