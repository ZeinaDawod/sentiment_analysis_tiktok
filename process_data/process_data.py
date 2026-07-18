

from pathlib import Path
import pandas as pd

from process_data.dedup import remove_duplicates
from process_data.remove_mislabeled_neutral import remove_mislabeled_neutral
from extract.bais import get_audio_from_url
from extract.bais import download_tiktok_video

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = BASE_DIR / "data_collection" / "tiktok_results_all.csv"
PROCESSED_CSV = BASE_DIR / "process_data" / "tiktok_results_processed.csv"


def run():

    df = pd.read_csv(RAW_CSV)
    print(f"[*] Loaded {len(df)} raw rows")

    df = remove_mislabeled_neutral(df)
    df = remove_duplicates(df)
    print(df["class"].value_counts())

    audio_paths = []
    video_paths = []
    for idx, row in df.iterrows():
        url = row["video_url"]
        try:
            audio_path = get_audio_from_url(video_url=url, cleanup_video=False)
            video_path = download_tiktok_video(video_url=url)
            audio_paths.append(audio_path)
            video_paths.append(video_path)
            print(f"[{idx}] Audio + Video ready: {url[:50]}")
        except Exception as e:
            print(f"[{idx}] Error: {e}")
            audio_paths.append(None)
            video_paths.append(None)

    df["audio_path"] = audio_paths
    df["video_path"] = video_paths

    before = len(df)
    df = df[df["audio_path"].notna() & df["video_path"].notna()].reset_index(drop=True)
    after = len(df)
    print(f"[*] Removed {before - after} rows with failed audio ({before} -> {after})")
    print(df["class"].value_counts())


    df.to_csv(PROCESSED_CSV, index=False)
    print(f"[*] Saved {len(df)} processed rows to {PROCESSED_CSV}")


if __name__ == "__main__":
    run()