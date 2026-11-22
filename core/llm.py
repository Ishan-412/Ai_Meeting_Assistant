"""Shared Gemini chat model configuration."""

import os

from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it to your .env file.")

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.7-flash"),
        api_key=api_key,
        temperature=temperature,
    )
