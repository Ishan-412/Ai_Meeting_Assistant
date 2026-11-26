"""Shared Gemini chat model configuration."""

from langchain_google_genai import ChatGoogleGenerativeAI
from core.settings import get_setting


def get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    api_key = get_setting("GOOGLE_API_KEY") or get_setting("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Google Gemini API key is missing. Set GOOGLE_API_KEY in .env "
            "or in Streamlit secrets."
        )

    return ChatGoogleGenerativeAI(
        model=get_setting("GEMINI_MODEL", "gemini-3.8-flash"),
        api_key=api_key,
        temperature=temperature,
        max_retries=2,
    )
