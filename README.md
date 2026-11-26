# AI Video Assistant

A Streamlit application that turns a YouTube video or local audio/video file into a transcript, meeting summary, action items, decisions, and questions. It also lets you ask questions grounded in the transcript.

## Requirements

- Python 3.11 (recommended)
- FFmpeg available on your system path
- A Google Gemini API key for summaries, extraction, and transcript chat
- A Sarvam API key for Hinglish transcription; English transcription uses local Whisper

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install FFmpeg if it is not already available. On Debian/Ubuntu use `sudo apt install ffmpeg`; on macOS with Homebrew use `brew install ffmpeg`; on Windows install FFmpeg and add its `bin` folder to `PATH`.

Copy `.env.example` to `.env` and add the API keys you need. Keep `.env` private and do not commit it. Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/app/apikey). The default model is `gemini-3.8-flash`; you can change it with `GEMINI_MODEL`. For Streamlit Community Cloud, add `GOOGLE_API_KEY`, optional `GEMINI_MODEL`, and `SARVAM_API_KEY` under the app's Secrets settings instead of uploading `.env`. Gemini free-tier requests have usage limits and may be used to improve Google's products; avoid sending confidential meeting transcripts through the free tier.

## Run

```bash
streamlit run app.py
```

Open the local address printed by Streamlit (normally `http://localhost:8501`). Choose a YouTube URL or upload an audio/video file, select English or Hinglish, and press **Analyse**. Whisper downloads its selected model the first time English transcription runs. Longer recordings take longer to process and may be limited by Gemini and Sarvam quotas.

For a terminal workflow, run `python main.py` and enter a YouTube URL or local file path when prompted.

## Project layout

```text
app.py                  Streamlit user interface
main.py                 Terminal entry point
core/transcriber.py     Whisper and Sarvam transcription
core/summarizer.py      Title and summary generation
core/extractor.py       Action item, decision, and question extraction
core/rag_engine.py      Transcript question answering
core/vector_store.py    Transcript embeddings and retrieval
utils/audio_processor.py Audio download, conversion, and chunking
```

Generated audio files and the downloaded Whisper model are local runtime data. They are not source files and should not be submitted as project code. Include any separate dataset or media that your instructor requires and that you have permission to share.

## Troubleshooting

- **FFmpeg not found:** install FFmpeg and restart the terminal/IDE so its `bin` folder is on `PATH`. Set `FFMPEG_PATH` in `.env` to the FFmpeg executable or installation directory if automatic discovery fails.
- **Gemini/Sarvam authentication:** check the matching key in `.env`; Gemini is needed for analysis/chat, while Sarvam is only needed for Hinglish transcription.
- **Whisper is slow:** the first run downloads the model. Set `WHISPER_MODEL=tiny` in `.env` for faster, less accurate local transcription, or `small` for the default.
- **YouTube download fails:** confirm the URL is accessible and update the yt-dlp package with `pip install --upgrade yt-dlp`.
