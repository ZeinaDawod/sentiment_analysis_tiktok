

import pandas as pd


def remove_duplicates(df: pd.DataFrame, subset: str = "video_url") -> pd.DataFrame:

    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    after = len(df)
    print(f"[dedup] Removed {before - after} duplicate rows ({before} -> {after})")
    return df


