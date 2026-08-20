import logging
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


import logging

from pathlib import Path
script_dir = Path(__file__).resolve().parent
log_file_path = script_dir / "evaluate.log"
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
        logging.StreamHandler()
    ]
)


LABELS = ["positive", "negative", "neutral"]


def print_evaluation(df: pd.DataFrame):
    eval_df = df[df["predicted_sentiment"] != "unknown"].copy()


    report_dict = classification_report(
        eval_df["true_sentiment"],
        eval_df["predicted_sentiment"],
        labels=["positive", "negative", "neutral"],
        zero_division=0,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    logger.info("Classification Report:\n" + str(report_df))

    cm = confusion_matrix(eval_df["true_sentiment"], eval_df["predicted_sentiment"], labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in LABELS],
        columns=[f"pred_{l}" for l in LABELS],
    )
    logger.info("Confusion Matrix:\n" + str(cm_df))

    return report_df,cm_df


def plot_confusion_matrix(cm_df: pd.DataFrame, output_path="confusion_matrix.png"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(output_path)
    logger.info(f"Saved confusion matrix plot to {output_path}")
    plt.show()
    plt.close()