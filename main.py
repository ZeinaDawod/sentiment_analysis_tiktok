from process_data.fix_or_remove_mislabeled_neutral import fix_or_remove_mislabeled_neutral
from process_data.dedup import remove_duplicates
from data_collection.tiktok_scraper import main as scrape_main
from pathlib import Path
import asyncio
import time
from datetime import datetime
import json
import logging
import traceback
import pandas as pd
from extract import (
    get_audio_from_url,
    download_tiktok_video,
    is_music_only,
    is_arabic,
    transcribe,
    predict_facial_expression,
    correct_spelling
)
from process_data.fix_transcript import fix_transcript
from sentiment_analysis.run_sentiment_analysis import run_sentiment_analysis_model, RESULTS_DIR
from sentiment_analysis.evaluate import print_evaluation,plot_confusion_matrix
BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "data" / "tiktok_results_all.csv"
FINAL_CSV = BASE_DIR / "data" / "final_dataset.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_VIDEOS_REQUIRED = 50
logger = logging.getLogger("main_pipeline")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(console_handler)


    log_file = OUTPUT_DIR / "main_pipeline.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

REQUIRED_COLUMNS = ["true_sentiment", "predicted_sentiment", "raw_model_output", "is_correct"]


def should_skip_execution(predictions_csv_path: Path) -> bool:
    if not predictions_csv_path.exists():
        return False

    try:
        df = pd.read_csv(predictions_csv_path)
        has_required_cols = all(col in df.columns for col in REQUIRED_COLUMNS)
        if not has_required_cols:
            return False
        no_nulls = df[REQUIRED_COLUMNS].notna().all().all()
        return bool(no_nulls)
    except Exception as e:
        logger.warning(f"Error checking existing predictions file: {e}")
        return False
def load_existing_class_overrides() -> dict:
    if not FINAL_CSV.exists():
        return {}
    existing_df = pd.read_csv(FINAL_CSV)
    if "class" not in existing_df.columns:
        return {}
    return dict(zip(existing_df["video_url"], existing_df["class"]))

def should_skip_scraping() -> bool:

    if not RAW_CSV.exists():
        logger.info("RAW_CSV not found -> scraping is required")
        return False

    try:
        df = pd.read_csv(RAW_CSV)
    except Exception:
        logger.warning("RAW_CSV exists but couldn't be read -> scraping is required")
        return False

    if len(df) < MIN_VIDEOS_REQUIRED:
        logger.info(f"RAW_CSV has only {len(df)} rows (< {MIN_VIDEOS_REQUIRED}) -> scraping is required")
        return False

    logger.info(f"RAW_CSV has {len(df)} rows (>= {MIN_VIDEOS_REQUIRED}) -> skipping scraping step")
    return True

def load_existing_results() -> dict:
    if not FINAL_CSV.exists():
        return {}
    existing_df = pd.read_csv(FINAL_CSV)
    cache = {}
    for _, row in existing_df.iterrows():
        if pd.notna(row.get("transcript")) and pd.notna(row.get("facial_expression")):
            cache[row["video_url"]] = row.to_dict()
    return cache
async def run():
    run_started_at = datetime.now()
    run_id = run_started_at.strftime("%Y%m%d_%H%M%S")
    report = {
        "run_id": run_id,
        "started_at": run_started_at.isoformat(),
        "steps": {},
    }

    logger.info("=" * 60)
    logger.info("STEP 1: Scraping")
    logger.info("=" * 60)

    step1_start = time.time()
    if should_skip_scraping():
        report["steps"]["scraping"] = {"status": "skipped", "reason": "sufficient videos already exist"}
    else:
        try:
            await scrape_main()
            report["steps"]["scraping"] = {"status": "executed",
                                           "duration_seconds": round(time.time() - step1_start, 2)}
        except Exception as e:
            logger.error(f"Scraping step failed: {e}")
            logger.error(traceback.format_exc())
            report["steps"]["scraping"] = {"status": "failed", "error": str(e)}
    
    logger.info("=" * 60)
    logger.info("STEP 2: Cleaning raw data (dedup + mislabeled)")
    logger.info("=" * 60)

    df = pd.read_csv(RAW_CSV)
    logger.info(f"Loaded {len(df)} raw rows")
    df = remove_duplicates(df)
    df = fix_or_remove_mislabeled_neutral(df)

    class_overrides = load_existing_class_overrides()
    if class_overrides:
        overridden_count = df["video_url"].isin(class_overrides.keys()).sum()
        logger.info(f"Preserving {overridden_count} manually-verified class labels from existing final dataset")
        df["class"] = df.apply(
            lambda row: class_overrides.get(row["video_url"], row["class"]),
            axis=1
        )

    logger.info(f"Class distribution after cleaning:\n{df['class'].value_counts()}")

    for col in ["audio_path", "video_path", "transcript", "facial_expression"]:
        if col not in df.columns:
            df[col] = None

    report["steps"]["cleaning"] = {
        "status": "executed",
        "rows_after_cleaning": len(df),
        "manually_verified_labels_preserved": len(class_overrides),
    }




    logger.info("=" * 60)
    logger.info("STEP 3: Processing videos (download + filter + models)")
    logger.info("=" * 60)

    cache = load_existing_results()
    logger.info(f"Found {len(cache)} already fully-processed videos - will skip them entirely")

    urls_to_process = [url for url in df["video_url"] if url not in cache]
    if not urls_to_process:
        logger.info("All videos already fully processed -> skipping Step 3 entirely")
        report["steps"]["processing"] = {"status": "skipped", "reason": "all videos already processed"}
        for idx, row in df.iterrows():
            url = row["video_url"]
            df.at[idx, "audio_path"] = cache[url]["audio_path"]
            df.at[idx, "video_path"] = cache[url]["video_path"]
            df.at[idx, "transcript"] = cache[url]["transcript"]
            df.at[idx, "facial_expression"] = cache[url]["facial_expression"]
    else:
        logger.info(f"{len(urls_to_process)} new videos need processing")

        transcribe_times, facial_expression_times = [], []
        excluded_indices = []
        excluded_reasons = []
        new_videos = []
        failed_videos = []

        for idx, row in df.iterrows():
            url = row["video_url"]

            if url in cache:
                df.at[idx, "audio_path"] = cache[url]["audio_path"]
                df.at[idx, "video_path"] = cache[url]["video_path"]
                df.at[idx, "transcript"] = cache[url]["transcript"]
                df.at[idx, "facial_expression"] = cache[url]["facial_expression"]
                logger.debug(f"[{idx}] Skipped entirely (already processed): {url[:50]}")
                continue

            try:
                audio_path = get_audio_from_url(video_url=url, cleanup_video=False)
                video_path = download_tiktok_video(video_url=url)

                if is_music_only(audio_path):
                    logger.info(f"[{idx}] Excluded (music only): {url[:50]}")
                    excluded_indices.append(idx)
                    excluded_reasons.append("music_only")
                    continue

                if is_arabic(audio_path):
                    logger.info(f"[{idx}] Excluded (Arabic detected): {url[:50]}")
                    excluded_indices.append(idx)
                    excluded_reasons.append("arabic_detected")
                    continue

                df.at[idx, "audio_path"] = audio_path
                df.at[idx, "video_path"] = video_path

                start = time.time()
                transcript = transcribe(audio_path)
                transcribe_times.append(time.time() - start)
                transcript = fix_transcript(transcript)
                transcript = correct_spelling(transcript)
                logger.debug(f"[{idx}] Transcript: {transcript}")

                start = time.time()
                facial_expression = predict_facial_expression(video_path)
                facial_expression_times.append(time.time() - start)
                logger.debug(f"[{idx}] Facial_Expression: {facial_expression}")

                df.at[idx, "transcript"] = transcript
                df.at[idx, "facial_expression"] = facial_expression
                new_videos.append(url)

            except Exception as e:
                logger.error(f"[{idx}] Failed (couldn't extract audio/video or run models): {url[:50]} - {e}")
                logger.debug(traceback.format_exc())
                excluded_indices.append(idx)
                excluded_reasons.append("processing_error")
                failed_videos.append(url)

            df.to_csv(FINAL_CSV, index=False)

        if excluded_indices:
            logger.info(f"Removing {len(excluded_indices)} excluded rows (music/arabic/failed)")

            excluded_df = df.loc[excluded_indices].copy()
            excluded_df["exclusion_reason"] = excluded_reasons
            excluded_df.to_csv(BASE_DIR / "data" / "excluded_for_review.csv", index=False)

            df = df.drop(index=excluded_indices).reset_index(drop=True)
            df.to_csv(FINAL_CSV, index=False)

        report["steps"]["processing"] = {
            "status": "executed",
            "new_videos_count": len(new_videos),
            "new_videos": new_videos,
            "excluded_count": len(excluded_indices),
            "failed_count": len(failed_videos),
            "failed_videos": failed_videos,
            "transcribe_total_seconds": round(sum(transcribe_times), 2) if transcribe_times else 0,
            "facial_expression_total_seconds": round(sum(facial_expression_times), 2) if facial_expression_times else 0,
        }

        logger.info(f"New videos processed: {len(new_videos)}")
        for url in new_videos:
            logger.info(f"  - {url}")


    logger.info("=" * 60)
    logger.info("STEP 4: Sentiment Analysis")
    logger.info("=" * 60)

    predictions_file = RESULTS_DIR / "predictions_df.csv"

    if should_skip_execution(predictions_file):
        logger.info(f"Skipping sentiment analysis: '{predictions_file.name}' already exists with complete data.")
        report["steps"]["sentiment_analysis"] = {
            "status": "skipped",
            "reason": "predictions file already exists with complete data",
        }
    else:
        step4_start = time.time()
        try:
            logger.info("Starting sentiment analysis model pipeline...")
            sentiment_input_df = pd.read_csv(FINAL_CSV)

            results_df = run_sentiment_analysis_model(sentiment_input_df, prompt_name="prompt_fewshot.txt")
            results_df.to_csv(predictions_file, index=False)
            logger.info(f"Saved predictions to {predictions_file}")

            report_df, cm_df = print_evaluation(results_df)
            report_df.to_csv(RESULTS_DIR / "classification_report.csv")
            cm_df.to_csv(RESULTS_DIR / "confusion_matrix.csv")

            cm_plot_path = RESULTS_DIR / "confusion_matrix.png"
            plot_confusion_matrix(cm_df, output_path=str(cm_plot_path))

            accuracy = results_df["is_correct"].mean()
            logger.info(f"Sentiment analysis accuracy: {accuracy:.2%}")

            report["steps"]["sentiment_analysis"] = {
                "status": "executed",
                "duration_seconds": round(time.time() - step4_start, 2),
                "rows_evaluated": len(results_df),
                "accuracy": round(accuracy, 4),
                "predictions_file": str(predictions_file),
                "classification_report_file": str(RESULTS_DIR / "classification_report.csv"),
                "confusion_matrix_file": str(RESULTS_DIR / "confusion_matrix.csv"),
            }

        except Exception as e:
            logger.error(f"Sentiment analysis step failed: {e}")
            logger.error(traceback.format_exc())
            report["steps"]["sentiment_analysis"] = {"status": "failed", "error": str(e)}


    report["finished_at"] = datetime.now().isoformat()
    report["final_dataset_rows"] = len(df)

    report_path = OUTPUT_DIR / f"run_report_{run_id}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"DONE. Final dataset ({len(df)} rows) saved to {FINAL_CSV}")
    logger.info(f"Run report saved to {report_path}")
    logger.info("=" * 60)
if __name__ == "__main__":
    asyncio.run(run())

