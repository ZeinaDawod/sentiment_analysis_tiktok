

import pandas as pd


def remove_duplicates(df: pd.DataFrame, subset: str = "video_url") -> pd.DataFrame:

    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    after = len(df)
    print(f"[dedup] Removed {before - after} duplicate rows ({before} -> {after})")
    return df


if __name__ == "__main__":

    INPUT_PATH = "../data_collection/tiktok_results_all.csv"
    OUTPUT_PATH = "tiktok_results_deduped.csv"

    df = pd.read_csv(INPUT_PATH)
    df = remove_duplicates(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[dedup] Saved cleaned file to {OUTPUT_PATH}")