from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "data_collection" / "tiktok_results_all.csv"
NEW_CSV_NEG = BASE_DIR / "data_collection" / "tiktok_results_neg_new.csv"
NEW_CSV_POS = BASE_DIR / "data_collection" / "tiktok_results_positive_new.csv"
NEW_CSV_NEUTRAL = BASE_DIR / "data_collection" / "tiktok_results_neutral_new.csv"


def run():
    df_main = pd.read_csv(INPUT_CSV)
    df_new_pos = pd.read_csv(NEW_CSV_POS)
    df_new_neg = pd.read_csv(NEW_CSV_NEG)
    df_new_neu = pd.read_csv(NEW_CSV_NEUTRAL)

    df_combined = pd.concat([df_main, df_new_pos, df_new_neg, df_new_neu], ignore_index=True)

    df_combined.to_csv(INPUT_CSV, index=False)
    print(f"Combined: {len(df_combined)} total rows")
    print(df_combined["class"].value_counts())


if __name__ == "__main__":
    run()