from process_data.remove_mislabeled_neutral import remove_mislabeled_neutral
from process_data.dedup import remove_duplicates
from data_collection.tiktok_scraper import main as scrape_main
from pathlib import Path
import asyncio
import time
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
BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "data" / "tiktok_results_all.csv"
FINAL_CSV = BASE_DIR / "data" / "final_dataset.csv"


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

    print("=" * 60)
    print("STEP 1: Scraping new videos")
    print("=" * 60)
    await scrape_main()

    print("\n" + "=" * 60)
    print("STEP 2: Cleaning raw data (dedup + mislabeled)")
    print("=" * 60)
    df = pd.read_csv(RAW_CSV)
    print(f"[*] Loaded {len(df)} raw rows")

    df = remove_mislabeled_neutral(df)
    df = remove_duplicates(df)
    print(df["class"].value_counts())

    for col in ["audio_path", "video_path", "transcript", "facial_expression"]:
        if col not in df.columns:
            df[col] = None
    cache = load_existing_results()
    print(f"[*] Found {len(cache)} already fully-processed videos - will skip them entirely")

    print("\n" + "=" * 60)
    print("STEP 3: Processing videos (download + filter + models)")
    print("=" * 60)

    transcribe_times, facial_expression_times = [], []

    excluded_indices = []
    new_videos = []

    for idx, row in df.iterrows():
        url = row["video_url"]
        if url in cache:
            df.at[idx, "audio_path"] = cache[url]["audio_path"]
            df.at[idx, "video_path"] = cache[url]["video_path"]
            df.at[idx, "transcript"] = cache[url]["transcript"]
            df.at[idx, "facial_expression"] = cache[url]["facial_expression"]
            print(f"[{idx}] Skipped entirely (already processed): {url[:50]}")
            continue

        try:
            audio_path = get_audio_from_url(video_url=url, cleanup_video=False)
            video_path = download_tiktok_video(video_url=url)

            if is_music_only(audio_path):
                print(f"[{idx}] Excluded (music only): {url[:50]}")
                excluded_indices.append(idx)
                continue

            if is_arabic(audio_path):
                print(f"[{idx}] Excluded (Arabic detected): {url[:50]}")
                excluded_indices.append(idx)
                continue

            df.at[idx, "audio_path"] = audio_path
            df.at[idx, "video_path"] = video_path

            start = time.time()
            transcript = transcribe(audio_path)
            transcribe_times.append(time.time() - start)
            transcript = fix_transcript(transcript)
            transcript = correct_spelling(transcript)
            print(f"[{idx}] Transcript: {transcript}")

            start = time.time()
            facial_expression = predict_facial_expression(video_path)
            facial_expression_times.append(time.time() - start)
            print(f"[{idx}] Facial_Expression: {facial_expression}")

            df.at[idx, "transcript"] = transcript
            df.at[idx, "facial_expression"] = facial_expression
            new_videos.append(url)

        except Exception:
            print(f"[{idx}] Failed (couldn't extract audio/video or run models): {url[:50]}")
            traceback.print_exc()
            excluded_indices.append(idx)


        df.to_csv(FINAL_CSV, index=False)


    if excluded_indices:
        print(f"\n[*] Removing {len(excluded_indices)} excluded rows (music/arabic)")
        df = df.drop(index=excluded_indices).reset_index(drop=True)
        df.to_csv(FINAL_CSV, index=False)
    print(f"\n{'=' * 60}")
    print(f"num of videos  {len(new_videos)}")
    print(f"{'=' * 60}")
    for url in new_videos:
        print(url)

    if transcribe_times:
        print(f"\n[*] speech_to_text -> total: {sum(transcribe_times):.2f}s")
    if facial_expression_times:
        print(f"[*] facial_expression -> total: {sum(facial_expression_times):.2f}s")

    print(f"\n[*] DONE. Final dataset ({len(df)} rows) saved to {FINAL_CSV}")


if __name__ == "__main__":
    asyncio.run(run())

