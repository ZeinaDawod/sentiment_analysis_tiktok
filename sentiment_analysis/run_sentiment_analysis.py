import logging
import sys
from pathlib import Path
import pandas as pd
from sentiment_analysis.qwen_7b.inference import generate_sentiment
from sentiment_analysis.evaluate import print_evaluation, plot_confusion_matrix
import logging

from pathlib import Path
script_dir = Path(__file__).resolve().parent
log_file_path = script_dir / "run_sentiment_analysis_model.log"
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompt"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CLASS_TO_SENTIMENT = {"pos": "positive", "neg": "negative", "neutral": "neutral"}


def run_sentiment_analysis_model(df: pd.DataFrame, prompt_name: str = "prompt_fewshot.txt"):
    prompt_template = (PROMPTS_DIR / prompt_name).read_text(encoding="utf-8")
    df = df[df["transcript"].notna()].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows")

    predicted_sentiments, raw_outputs = [], []

    for idx, row in df.iterrows():
        result = generate_sentiment(
            row["transcript"], row["description"], row["facial_expression"], prompt_template
        )
        predicted_sentiments.append(result["predicted_sentiment"])
        raw_outputs.append(result["raw_output"])
        logger.debug(f"[{idx}] Predicted: {result['predicted_sentiment']}")

    df["true_sentiment"] = df["class"].map(CLASS_TO_SENTIMENT)
    df["predicted_sentiment"] = predicted_sentiments
    df["raw_model_output"] = raw_outputs
    df["is_correct"] = df["true_sentiment"] == df["predicted_sentiment"]

    accuracy = df["is_correct"].mean()
    logger.info(f"Accuracy: {accuracy:.2%}")

    output_path = RESULTS_DIR / "predictions_df.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")

    return df

