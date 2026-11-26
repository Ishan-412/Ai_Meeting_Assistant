from dotenv import load_dotenv
import yt_dlp
from pydub import AudioSegment
import os
from pathlib import Path
load_dotenv()
DOWNLOAD_DIR = 'downloads'

def download_youtube_audio(url: str, work_dir: str | None = None) -> str:
    destination = work_dir or DOWNLOAD_DIR
    os.makedirs(destination, exist_ok=True)
    output_path = os.path.join(destination, "%(id)s.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'ffmpeg_location': os.getenv("FFMPEG_PATH"),
        'postprocessors': [
            {
                'key': "FFmpegExtractAudio",
                'preferredcodec': "wav",
                "preferredquality": "192",
            }
        ],
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            wav_filename = os.path.splitext(filename)[0] + ".wav"
    except yt_dlp.utils.DownloadError as error:
        if "403" in str(error):
            raise RuntimeError(
                "YouTube refused the audio download (HTTP 403). The video may be "
                "restricted, or YouTube may be blocking this hosting server. "
                "Try uploading the audio or video file directly instead."
            ) from error
        raise

    return wav_filename



def convert_to_wav(input_file: str, work_dir: str | None = None) -> str:
    """Converts any audio/video file to WAV format using pydub."""
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    output_path = os.path.join(work_dir or os.path.dirname(input_file), f"{Path(input_file).stem}_converted.wav")
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path,format="wav")
    return output_path




def chunk_audio(wav_path: str, chunk_minutes: int = 10, work_dir: str | None = None) -> list:
    if chunk_minutes < 1:
        raise ValueError("chunk_minutes must be at least 1")
    audio = AudioSegment.from_wav(wav_path)
    if len(audio) == 0:
        raise ValueError("The input audio file is empty.")
    chunk_ms = chunk_minutes * 60 * 1000  # Convert minutes to milliseconds
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = os.path.join(work_dir or os.path.dirname(wav_path), f"chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks


def process_input(source: str, work_dir: str | None = None) -> list:
    if not source or not source.strip():
        raise ValueError("Provide a YouTube URL or a local audio/video file.")
    source = source.strip()
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected Youtube URL. Downloading audio...")
        wav_path = download_youtube_audio(source, work_dir)
    else:
        print("Detected local file. Converting to WAV...")
        if work_dir:
            os.makedirs(work_dir, exist_ok=True)
        wav_path = convert_to_wav(source, work_dir)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path, work_dir=work_dir)
    print(f"Audio processing complete. Generated {len(chunks)} chunks.")
    return chunks
