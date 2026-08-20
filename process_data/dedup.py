

import pandas as pd
import logging
from pathlib import Path
script_dir = Path(__file__).resolve().parent
log_file_path = script_dir / "dedup.log"
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
        logging.StreamHandler()
    ]
)


def remove_duplicates(df: pd.DataFrame, subset: str = "video_url") -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    after = len(df)
    logger.info(f"[dedup] Removed {before - after} duplicate rows ({before} -> {after})")
    return df


