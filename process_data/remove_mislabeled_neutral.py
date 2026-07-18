

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = BASE_DIR / "data_collection" / "tiktok_results_all.csv"
CLEANED_CSV = BASE_DIR / "process_data" / "tiktok_results_no_mislabeled.csv"


POSITIVE_SIGNAL_WORDS = [
    "love it", "love this", "obsessed","thank you", "amazing", "holy grail",
    "best product", "best skincare", "life changing", "game changer",
    "highly recommend", "must have", "worth it", "in love with",
    "favorite", "favourite", "10/10", "perfect for", "works wonders",
    "changed my skin", "cleared my skin", "glowing", "so good",
    "saved my skin", "the best", "ever had", "could cry",
    "great results", "so soft", "worth to buy", "worth buying",
    "lifesaver", "life saver", "i love", "excited",
    "🥳", "😍", "🤍", "❤️", "🥰","💙",

]
NEGATIVE_SIGNAL_WORDS = [
    "worst", "hate it", "hate this", "waste of money", "don't buy",
    "do not buy", "disappointed", "disappointing", "broke me out",
    "broke out", "made it worse", "caused breakouts", "irritated my skin",
    "allergic reaction", "rash", "burning sensation", "regret",
    "never again", "not worth", "not worth it", "terrible", "awful",
    "horrible", "ruined my skin", "damaged my skin", "returning this",
    "returned it", "refund", "would not recommend", "wouldn't recommend",
    "avoid this", "avoid at all costs", "scam",
    "😡", "😭😭", "🤮", "👎",
]



def is_actually_positive(description: str) -> bool:

    if not isinstance(description, str):
        return False
    desc_lower = description.lower()
    return any(phrase in desc_lower for phrase in POSITIVE_SIGNAL_WORDS)
def is_actually_negative(description: str) -> bool:
    if not isinstance(description, str):
        return False
    desc_lower = description.lower()
    return any(phrase in desc_lower for phrase in NEGATIVE_SIGNAL_WORDS)


def remove_mislabeled_neutral(df: pd.DataFrame) -> pd.DataFrame:
    is_neutral = df["class"].str.lower() == "neutral"
    is_flagged_positive = df["description"].apply(is_actually_positive)
    is_flagged_negative = df["description"].apply(is_actually_negative)

    mislabeled_mask = is_neutral & (is_flagged_positive | is_flagged_negative)
    removed_count = mislabeled_mask.sum()

    df_cleaned = df[~mislabeled_mask].reset_index(drop=True)

    print(f"[remove_mislabeled_neutral] Removed {removed_count} mislabeled rows "
          f"({len(df)} -> {len(df_cleaned)})")

    return df_cleaned


