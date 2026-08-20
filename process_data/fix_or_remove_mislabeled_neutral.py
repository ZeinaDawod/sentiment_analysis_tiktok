import logging
import pandas as pd
from pathlib import Path
script_dir = Path(__file__).resolve().parent
log_file_path = script_dir / "fix_or_remove_mislabeled_neutral.log"
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
        logging.StreamHandler()
    ]
)




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


def fix_or_remove_mislabeled_neutral(df: pd.DataFrame) -> pd.DataFrame:
    unique_videos_count = len(df)


    is_neutral = df["class"].str.lower() == "neutral"
    is_pos = df["description"].apply(is_actually_positive)
    is_neg = df["description"].apply(is_actually_negative)


    if unique_videos_count > 50:

        mislabeled_mask = is_neutral & (is_pos | is_neg)
        removed_count = mislabeled_mask.sum()

        df_cleaned = df[~mislabeled_mask].reset_index(drop=True)

        logger.info(f"[DELETE MODE] Unique video count ({unique_videos_count}) is > 50.")
        logger.info(f"Removed {removed_count} mislabeled rows. Dataset size: {len(df)} -> {len(df_cleaned)}")
        return df_cleaned

    else:

        df_cleaned = df.copy()

        pos_target = is_neutral & is_pos & (~is_neg)
        neg_target = is_neutral & is_neg & (~is_pos)

        df_cleaned.loc[pos_target, "class"] = "positive"
        df_cleaned.loc[neg_target, "class"] = "negative"

        fixed_pos_count = pos_target.sum()
        fixed_neg_count = neg_target.sum()
        total_fixed = fixed_pos_count + fixed_neg_count

        logger.info(f"[RELABEL MODE] Unique video count ({unique_videos_count}) is <= 50.")
        logger.info(f"Relabeled {total_fixed} rows ({fixed_pos_count} -> positive, {fixed_neg_count} -> negative).")
        return df_cleaned