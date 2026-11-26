"""Streamlit interface for the AI Video Assistant."""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.llm import get_llm_api_key, get_llm_provider
from core.pipeline import run_pipeline
from core.rag_engine import ask_question
from core.settings import get_setting

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

STAGES = {
    "audio": "Preparing audio",
    "transcript": "Transcribing",
    "title": "Naming the meeting",
    "summary": "Writing the summary",
    "action_items": "Finding action items",
    "key_decisions": "Finding decisions",
    "open_questions": "Finding open questions",
    "rag": "Building transcript search",
}
STAGE_PROGRESS = {
    "audio": 0.08,
    "transcript": 0.34,
    "title": 0.47,
    "summary": 0.60,
    "action_items": 0.70,
    "key_decisions": 0.77,
    "open_questions": 0.84,
    "rag": 0.94,
}

for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "pipeline_steps": {},
    "pipeline_error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def llm_provider_label() -> str:
    try:
        return "OpenRouter" if get_llm_provider() == "openrouter" else "Gemini"
    except RuntimeError:
        return "LLM provider not configured"


def render_stage_status() -> None:
    steps = st.session_state.pipeline_steps
    if not steps:
        return
    st.markdown("#### Analysis progress")
    for stage, label in STAGES.items():
        state = steps.get(stage)
        if state == "done":
            st.caption(f"✓  {label}")
        elif state == "active":
            st.caption(f"◌  {label}")
        elif state == "failed":
            st.caption(f"!  {label}")


with st.sidebar:
    st.title("🎬 Video assistant")
    st.caption("Turn recordings into useful meeting notes.")
    st.divider()

    with st.form("analysis_form"):
        input_method = st.radio(
            "Source",
            ["YouTube link", "Upload a file"],
            horizontal=True,
        )

        source = ""
        uploaded_file = None
        if input_method == "YouTube link":
            source = st.text_input(
                "YouTube URL",
                placeholder="https://youtube.com/watch?v=…",
            ).strip()
        else:
            uploaded_file = st.file_uploader(
                "Audio or video file",
                type=["mp4", "mkv", "mp3", "wav", "m4a", "webm", "aac"],
                help="Supported formats: MP4, MKV, MP3, WAV, M4A, WebM, AAC.",
            )

        language_label = st.radio(
            "Spoken language",
            ["English", "Hinglish"],
            horizontal=True,
        )
        analyze_clicked = st.form_submit_button(
            "Analyze meeting",
            type="primary",
            width="stretch",
        )

    st.caption(f"Language model: **{llm_provider_label()}**")
    render_stage_status()

st.title("AI Video Assistant")
st.caption("TRANSCRIBE · SUMMARISE · FIND ACTIONS · ASK QUESTIONS")

if analyze_clicked:
    try:
        llm_provider = get_llm_provider()
        provider_key = get_llm_api_key(llm_provider)
        provider_error = None
    except RuntimeError as error:
        llm_provider = None
        provider_key = None
        provider_error = str(error)

    if input_method == "YouTube link" and not source:
        st.error("Add a YouTube link to start the analysis.")
    elif input_method == "Upload a file" and uploaded_file is None:
        st.error("Choose an audio or video file to start the analysis.")
    elif provider_error:
        st.error(provider_error)
    elif not provider_key:
        key_name = "OPENROUTER_API_KEY" if llm_provider == "openrouter" else "GOOGLE_API_KEY"
        st.error(f"Add {key_name} to .env or Streamlit secrets before analyzing.")
    elif language_label == "Hinglish" and not get_setting("SARVAM_API_KEY"):
        st.error("Add SARVAM_API_KEY to .env or Streamlit secrets to use Hinglish transcription.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}
        st.session_state.pipeline_error = None

        progress_bar = st.progress(0, text="Starting analysis…")
        progress_detail = st.empty()
        upload_path = None
        active_stage = {"name": "starting"}

        try:
            if uploaded_file is not None:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(
                    prefix="video-upload-",
                    suffix=suffix,
                    delete=False,
                ) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    upload_path = temp_file.name
                pipeline_source = upload_path
            else:
                pipeline_source = source

            def report_progress(stage, status, detail, partial_result):
                active_stage["name"] = stage
                st.session_state.result = partial_result
                st.session_state.pipeline_steps[stage] = status
                label = STAGES.get(stage, "Analyzing")
                value = STAGE_PROGRESS.get(stage, 0)
                if status == "done":
                    value = min(value + 0.07, 0.99)
                progress_bar.progress(value, text=f"{label}…")
                if detail:
                    progress_detail.caption(detail)

            result = run_pipeline(
                pipeline_source,
                language_label.lower(),
                progress_callback=report_progress,
            )
            st.session_state.result = result
            st.session_state.pipeline_done = True
            progress_bar.progress(1.0, text="Analysis complete")
            progress_detail.empty()
            st.success("Analysis complete. Your meeting notes are ready below.")
        except Exception as error:
            stage = active_stage["name"]
            if stage in STAGES:
                st.session_state.pipeline_steps[stage] = "failed"
            st.session_state.pipeline_error = f"Analysis stopped during {STAGES.get(stage, stage)}: {error}"
            progress_bar.empty()
            progress_detail.empty()
        finally:
            if upload_path and os.path.exists(upload_path):
                os.remove(upload_path)

if st.session_state.pipeline_error:
    st.error(st.session_state.pipeline_error)

result = st.session_state.result
has_results = bool(
    result
    and any(
        result.get(key)
        for key in ("transcript", "title", "summary", "action_items", "key_decisions", "open_questions")
    )
)

if has_results:
    st.divider()
    st.header(result.get("title") or "Meeting analysis")
    if st.session_state.pipeline_done:
        st.caption("Analysis complete · Review the summary, follow-ups, and transcript below.")
    else:
        st.caption("Partial results are available. The failed step is shown above.")

    overview_tab, transcript_tab, chat_tab = st.tabs(
        ["Overview", "Transcript", "Ask the transcript"]
    )

    with overview_tab:
        st.subheader("Meeting summary")
        with st.container(border=True):
            st.markdown(result.get("summary") or "Summary was not generated.")

        st.subheader("Follow-ups and discussion")
        action_col, decision_col, question_col = st.columns(3)
        with action_col:
            with st.container(border=True):
                st.markdown("#### ✅ Action items")
                st.markdown(result.get("action_items") or "No action items were generated.")
        with decision_col:
            with st.container(border=True):
                st.markdown("#### 🔑 Key decisions")
                st.markdown(result.get("key_decisions") or "No decisions were generated.")
        with question_col:
            with st.container(border=True):
                st.markdown("#### ❓ Open questions")
                st.markdown(result.get("open_questions") or "No open questions were generated.")

    with transcript_tab:
        st.subheader("Full transcript")
        transcript = result.get("transcript", "")
        if transcript:
            st.text_area(
                "Transcript text",
                value=transcript,
                height=480,
                disabled=True,
                label_visibility="collapsed",
            )
            st.download_button(
                "Download transcript",
                data=transcript,
                file_name="meeting-transcript.txt",
                mime="text/plain",
            )
        else:
            st.info("The transcript was not generated.")

    with chat_tab:
        st.subheader("Ask about this meeting")
        if result.get("rag_chain") is None:
            st.info("Transcript search wasn’t built, so questions aren’t available for this analysis.")
        else:
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            else:
                st.caption("Ask about decisions, owners, dates, or details mentioned in the transcript.")

            with st.form("meeting_chat_form", clear_on_submit=True):
                question = st.text_input(
                    "Your question",
                    placeholder="What follow-up was assigned to John’s case?",
                )
                send_question = st.form_submit_button("Send question", type="primary")

            if send_question and question.strip():
                try:
                    with st.spinner("Searching the transcript…"):
                        answer = ask_question(result["rag_chain"], question.strip())
                    st.session_state.chat_history.extend(
                        [
                            {"role": "user", "content": question.strip()},
                            {"role": "assistant", "content": answer},
                        ]
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"Couldn’t answer that question. {error}")

            if st.session_state.chat_history and st.button("Clear conversation"):
                st.session_state.chat_history = []
                st.rerun()
else:
    st.divider()
    with st.container(border=True):
        st.subheader("Turn a recording into clear next steps")
        st.write(
            "Add a YouTube link or upload a recording. The assistant will transcribe it, "
            "summarize the discussion, identify follow-ups, and let you ask questions."
        )
    feature_cols = st.columns(3)
    for column, heading, description in zip(
        feature_cols,
        ["Transcript", "Meeting notes", "Grounded chat"],
        [
            "Get a readable transcript from audio or video.",
            "Pull out the summary, decisions, and action items.",
            "Ask questions and get answers from the transcript.",
        ],
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"#### {heading}")
                st.caption(description)
