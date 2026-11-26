"""Shared chat model configuration for Gemini and OpenRouter."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from core.settings import get_setting


def get_llm_provider() -> str:
    provider = (get_setting("LLM_PROVIDER", "gemini") or "gemini").lower()
    if provider not in {"gemini", "openrouter"}:
        raise RuntimeError("LLM_PROVIDER must be either 'gemini' or 'openrouter'.")
    return provider


def get_llm_api_key(provider: str | None = None) -> str | None:
    """Return only the key for the selected provider; never silently switch providers."""
    provider = provider or get_llm_provider()
    if provider == "openrouter":
        return get_setting("OPENROUTER_API_KEY")
    if provider == "gemini":
        return get_setting("GOOGLE_API_KEY") or get_setting("GEMINI_API_KEY")
    raise RuntimeError("LLM provider must be either 'gemini' or 'openrouter'.")


def get_llm(temperature: float = 0.3) -> BaseChatModel:
    provider = get_llm_provider()
    api_key = get_llm_api_key(provider)
    if not api_key:
        key_name = "OPENROUTER_API_KEY" if provider == "openrouter" else "GOOGLE_API_KEY"
        raise RuntimeError(f"{key_name} is missing. Set it in .env or Streamlit secrets.")

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=get_setting("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_retries=2,
        )

    return ChatGoogleGenerativeAI(
        model=get_setting("GEMINI_MODEL", "gemini-3.8-flash"),
        api_key=api_key,
        temperature=temperature,
        max_retries=2,
    )
