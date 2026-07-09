
from pathlib import Path
import pandas as pd
from clean_data.dedup import remove_duplicates
from clean_data.remove_mislabeled_neutral import remove_mislabeled_neutral
import traceback

from extract import transcribe, predict_tone
from extract import get_audio_from_url
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "data_collection" / "tiktok_results_all.csv"
OUTPUT_CSV = BASE_DIR / "pipeline" / "final_dataset.csv"
SUMMARY_CSV = BASE_DIR / "pipeline" / "class_transcript_expression_tone.csv"


def run():


    NEW_CSV_NEG="data_collection/tiktok_results_neg_new.csv"
    NEW_CSV_POS = "data_collection/tiktok_results_positive_new.csv"
    NEW_CSV_NEUTRAL="data_collection/tiktok_results_neutral_new.csv"
    df_main = pd.read_csv(INPUT_CSV)
    df_new_pos = pd.read_csv(NEW_CSV_POS)
    df_new_neg = pd.read_csv(NEW_CSV_NEG)
    df_new_neu = pd.read_csv(NEW_CSV_NEUTRAL)
    df_combined = pd.concat([df_main, df_new_pos,df_new_neg,df_new_neu], ignore_index=True)


    df_combined = remove_duplicates(df_combined)
    df_combined = remove_mislabeled_neutral(df_combined)

    df_combined.to_csv(INPUT_CSV, index=False)
    print(f"Combined: {len(df_combined)} total rows")
    print(df_combined["class"].value_counts())





    audio_paths = []

    for idx, row in df_combined.iterrows():
        url = row["video_url"]

        try:
            audio_path = get_audio_from_url(
                video_url=url,
                cleanup_video=True
            )

            audio_paths.append(audio_path)
            print(f"[{idx}] Audio extracted")

        except Exception as e:
            print(f"[{idx}] Error: {e}")
            audio_paths.append(None)
    df_combined["audio_path"] = audio_paths

    transcripts = []
    tones = []


    for idx, row in df_combined.iterrows():

        audio_path = row["audio_path"]

        if audio_path is None:
            transcripts.append(None)
            tones.append(None)

            continue

        try:
            print("Before transcribe")
            transcript = transcribe(audio_path)
            print("After transcribe")

            print("Before tone")
            tone = predict_tone(audio_path)
            print("After tone")


            transcripts.append(transcript)
            tones.append(tone)


        except Exception :
            traceback.print_exc()
            transcripts.append(None)
            tones.append(None)


    df_combined["transcript"] = transcripts
    df_combined["tone"] = tones

    df_combined.to_csv(OUTPUT_CSV, index=False)
    print(f"[*] Saved final dataset ({len(df_combined)} rows) to {OUTPUT_CSV}")


    summary_df = df_combined[["video_url","class", "transcript",  "tone"]]
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"[*] Saved summary file ({len(summary_df)} rows) to {SUMMARY_CSV}")


if __name__ == "__main__":
    run()