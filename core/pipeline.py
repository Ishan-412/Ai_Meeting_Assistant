"""UI-independent orchestration for analysing a video or audio recording."""

from collections.abc import Callable
from tempfile import TemporaryDirectory
from typing import Any

from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain
from core.summarizer import generate_title, summarize
from core.transcriber import transcribe_all
from utils.audio_processor import process_input

ProgressCallback = Callable[[str, str, str | None, dict[str, Any]], None]


def run_pipeline(
    source: str,
    language: str = "english",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run analysis, reporting each completed stage and retaining partial results."""
    result: dict[str, Any] = {
        "title": "", "transcript": "", "summary": "", "action_items": "",
        "key_decisions": "", "open_questions": "", "rag_chain": None,
    }

    def report(stage: str, status: str, detail: str | None = None) -> None:
        if progress_callback:
            progress_callback(stage, status, detail, result.copy())

    with TemporaryDirectory(prefix="video-assistant-") as work_dir:
        report("audio", "active", "Preparing audio from the selected source…")
        chunks = process_input(source, work_dir=work_dir)
        report("audio", "done")

        report("transcript", "active", f"Transcribing {len(chunks)} audio segment(s)…")

        def transcription_progress(message: str) -> None:
            report("transcript", "active", message)

        result["transcript"] = transcribe_all(chunks, language, progress_callback=transcription_progress)
        report("transcript", "done")

        for stage, operation, key, detail in [
            ("title", generate_title, "title", "Generating a title…"),
            ("summary", summarize, "summary", "Writing the summary…"),
            ("action_items", extract_action_items, "action_items", "Extracting action items…"),
            ("key_decisions", extract_key_decisions, "key_decisions", "Extracting decisions…"),
            ("open_questions", extract_questions, "open_questions", "Extracting questions…"),
        ]:
            report(stage, "active", detail)
            result[key] = operation(result["transcript"])
            report(stage, "done")

        report("rag", "active", "Building transcript search…")
        result["rag_chain"] = build_rag_chain(result["transcript"])
        report("rag", "done")

    return result
