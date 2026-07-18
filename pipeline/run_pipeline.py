
from pathlib import Path
import pandas as pd
import time
import traceback
from extract import transcribe, predict_facial_expression
from pipeline.fix_transcript import fix_transcript



BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "process_data" / "tiktok_results_processed.csv"
OUTPUT_CSV = BASE_DIR / "pipeline" / "final_dataset.csv"


def load_existing_results() -> dict:

    if not OUTPUT_CSV.exists():
        return {}

    existing_df = pd.read_csv(OUTPUT_CSV)
    cache = {}
    for _, row in existing_df.iterrows():
        if pd.notna(row.get("transcript")) and pd.notna(row.get("facial_expression")):
            cache[row["video_url"]] = {
                "transcript": row["transcript"],
                "facial_expression": row["facial_expression"],
            }
    return cache
def run():
    df = pd.read_csv(INPUT_CSV)
    print(f"[*] Loaded {len(df)} rows with audio ready")

    # منجهز الأعمدة مسبقاً بقيم فاضية، بعدين منعبيها صف صف
    if "transcript" not in df.columns:
        df["transcript"] = None
    if "facial_expression" not in df.columns:
        df["facial_expression"] = None

    cache = load_existing_results()
    print(f"[*] Found {len(cache)} already-processed videos in existing output, will skip them")

    transcribe_times, facial_expression_times = [], []

    for idx, row in df.iterrows():
        url = row["video_url"]
        audio_path = row["audio_path"]
        video_path = row["video_path"]


        if url in cache:
            df.at[idx, "transcript"] = cache[url]["transcript"]
            df.at[idx, "facial_expression"] = cache[url]["facial_expression"]
            print(f"[{idx}] Skipped (already processed): {url[:50]}")
            continue

        if audio_path is None or (isinstance(audio_path, float) and pd.isna(audio_path)):
            df.at[idx, "transcript"] = None
            df.at[idx, "facial_expression"] = None
            df.to_csv(OUTPUT_CSV, index=False)
            continue

        try:
            print("Before transcribe")
            start = time.time()
            transcript = transcribe(audio_path)
            transcribe_times.append(time.time() - start)
            transcript = fix_transcript(transcript)
            print(f"Transcript: {transcript}")
            print(f"the time for this transcript {transcribe_times[-1]}")
            print("After transcribe")

            print("Before Facial_Expression")
            start = time.time()
            facial_expression = predict_facial_expression(video_path)
            facial_expression_times.append(time.time() - start)
            print(f"Facial_Expression: {facial_expression}")
            print(f"the time for this Facial_Expression {facial_expression_times[-1]}")
            print("After Facial_Expression")

            df.at[idx, "transcript"] = transcript
            df.at[idx, "facial_expression"] = facial_expression

        except Exception:
            traceback.print_exc()
            df.at[idx, "transcript"] = None
            df.at[idx, "facial_expression"] = None


        df.to_csv(OUTPUT_CSV, index=False)
        print(f"[{idx}] Saved progress to {OUTPUT_CSV}")

    if transcribe_times:
        print(f"\n[*] speech_to_text -> total: {sum(transcribe_times):.2f}s ")
    if facial_expression_times:
        print(f"[*] facial_expression_times -> total: {sum(facial_expression_times):.2f}s ")

    print(f"[*] Final dataset ({len(df)} rows) saved to {OUTPUT_CSV}")




if __name__ == "__main__":
    run()