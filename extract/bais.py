

import os
import subprocess
import tempfile
import yt_dlp
import imageio_ffmpeg
from pathlib import Path

BASE_DOWNLOAD_DIR = Path.home() / "Desktop" / "tiktok_downloads"
VIDEO_DIR = BASE_DOWNLOAD_DIR / "videos"
AUDIO_DIR = BASE_DOWNLOAD_DIR / "audio"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def download_video(video_url: str, output_dir: str = None) -> str:

    if output_dir is None:
        output_dir = str(VIDEO_DIR)

    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "outtmpl": output_template,
        "format": "mp4/best",
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        video_path = ydl.prepare_filename(info)

    return video_path


def extract_audio(video_path: str, output_path: str = None) -> str:

    if output_path is None:

        video_name = Path(video_path).stem
        output_path = str(AUDIO_DIR / f"{video_name}.wav")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        output_path,
    ]
    subprocess.run(command, check=True, capture_output=True)

    return output_path


def get_audio_from_url(video_url: str, output_dir: str = None, cleanup_video: bool = False) -> str:

    video_path = download_video(video_url, output_dir)
    audio_path = extract_audio(video_path)

    if cleanup_video and os.path.exists(video_path):
        os.remove(video_path)

    return audio_path