"""Read application settings from environment variables or Streamlit secrets."""

import os
from dotenv import load_dotenv


# Load project-local settings consistently for Streamlit, CLI, and direct use.
load_dotenv()


def get_setting(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()

    # Streamlit Cloud exposes configured credentials through st.secrets.
    # In CLI/local runs without a secrets file, this falls back to the default.
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except FileNotFoundError:
        return default

    if value is None or not str(value).strip():
        return default
    return str(value).strip()
