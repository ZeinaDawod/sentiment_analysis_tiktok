from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DOWNLOAD_DIR = Path(
    os.getenv(
        "BASE_DOWNLOAD_DIR",
        str(Path.home() / "Desktop" / "tiktok_downloads")
    )
)

VIDEO_DIR = BASE_DOWNLOAD_DIR / "videos"
AUDIO_DIR = BASE_DOWNLOAD_DIR / "audio"
FRAME_DIR = BASE_DOWNLOAD_DIR / "frames"

WHISPER_MODEL = os.getenv("WHISPER_MODEL")
FACE_MODEL = os.getenv("FACE_MODEL")
NUM_FRAMES = os.getenv("NUM_FRAMES")

