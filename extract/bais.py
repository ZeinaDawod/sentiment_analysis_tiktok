import os
import subprocess
import yt_dlp
import imageio_ffmpeg
import pandas as pd
from pathlib import Path
from config import VIDEO_DIR, AUDIO_DIR, FRAME_DIR
import logging

from pathlib import Path
script_dir = Path(__file__).resolve().parent
log_file_path = script_dir / "bais.log"
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
        logging.StreamHandler()
    ]
)



VIDEO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)


def get_video_id_from_url(video_url: str) -> str:
    clean_url = video_url.split("?")[0].rstrip("/")
    return clean_url.split("/")[-1]
def find_existing_file(directory: Path, video_id: str) -> str:
    matches = list(Path(directory).glob(f"{video_id}.*"))
    return str(matches[0]) if matches else None


def download_tiktok_video(video_url: str, output_dir: str = None) -> str:

    if output_dir is None:
        output_dir = str(VIDEO_DIR)

    os.makedirs(output_dir, exist_ok=True)

    video_id = get_video_id_from_url(video_url)
    existing_video = find_existing_file(Path(output_dir), video_id)
    if existing_video:
        logger.debug(f"[skip] Video already downloaded: {existing_video}")
        return existing_video

    ydl_opts = {
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "format": "bv*+ba/b",
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        filename = ydl.prepare_filename(info)
        logger.info(f"Video successfully downloaded: {filename}")

    return filename


def extract_audio(video_path: str, output_path: str = None) -> str:
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = str(AUDIO_DIR / f"{video_name}.wav")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    if os.path.exists(output_path):
        logger.debug(f"[skip] Audio already extracted: {output_path}")
        return output_path

    command = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        output_path,
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg Error:\n{e.stderr}")
        raise

    return output_path


def extract_frame(video_path: str, output_path: str = None, timestamp: str = "00:00:01") -> str:
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = str(FRAME_DIR / f"{video_name}.jpg")

    if os.path.exists(output_path):
        logger.debug(f"[skip] Frame already extracted: {output_path}")
        return output_path

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-ss", timestamp,
        "-vframes", "1",
        output_path,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg Error (frame extraction):\n{e.stderr}")
        raise

    return output_path


def get_video_duration(video_path: str) -> float:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg_exe, "-i", video_path],
        capture_output=True, text=True
    )
    for line in result.stderr.splitlines():
        if "Duration" in line:
            duration_str = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = duration_str.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def extract_frames(video_path: str, num_frames: int = 3) -> list:
    duration = get_video_duration(video_path)
    if duration <= 0:
        return [extract_frame(video_path)]

    video_name = Path(video_path).stem
    frame_paths = []


    for i in range(num_frames):
        fraction = (i + 1) / (num_frames + 1)
        timestamp_seconds = duration * fraction
        timestamp_str = f"{int(timestamp_seconds // 3600):02}:{int((timestamp_seconds % 3600) // 60):02}:{timestamp_seconds % 60:05.2f}"

        output_path = str(FRAME_DIR / f"{video_name}_frame{i}.jpg")
        frame_path = extract_frame(video_path, output_path=output_path, timestamp=timestamp_str)
        frame_paths.append(frame_path)

    return frame_paths


def get_audio_from_url(video_url: str,output_dir: str = None,cleanup_video: bool = False) -> str:

    video_path = download_tiktok_video(video_url, output_dir)

    audio_path = extract_audio(video_path)

    if cleanup_video and os.path.exists(video_path):
        os.remove(video_path)

    return audio_path


