"""
LLM factory with automatic Groq -> local Qwen fallback.

Reads GROQ_API_KEY from the .env file (via python-dotenv, already loaded
at app startup). If it's set, uses Groq's cloud API for fast inference
(ChatGroq, real streaming). If it's not set, falls back to the local
Qwen model -- no crash, no prompt, the user doesn't have to choose.

Note on privacy: using Groq sends the retrieved InBody context and the
question to Groq's servers. The local Qwen path keeps everything on the
user's machine. This module only decides WHICH backend to use; it does
not change that trade-off -- see main_app.py / README for how it's
surfaced to the user.
"""

import os
from typing import Any, Optional

import streamlit as st

def get_groq_llm(streaming: bool = True, temperature: float = 0.3) -> Optional[Any]:
    """Returns a LangChain-compatible ChatGroq instance if a Groq API key
    is configured, otherwise None. Never raises -- callers should fall
    back to the local model when this returns None."""
    api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass  # no secrets.toml configured -- that's fine, fall through to .env

    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from langchain_groq import ChatGroq
    except ImportError:
        st.warning(
            "GROQ_API_KEY is set but the `langchain-groq` package isn't installed. "
            "Run `pip install -r requirements.txt`. Falling back to the local model."
        )
        return None

    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    try:
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            streaming=streaming,
            temperature=temperature,
        )
    except Exception as exc:
        st.warning(f"Groq is configured but couldn't be reached ({exc}). Falling back to the local model.")
        return None


def using_groq() -> bool:
    """Quick check for the UI (e.g. to show a badge: 'Fast mode (Groq)' vs
    'Local mode (private)') without constructing a client."""
    try:
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
            return True
    except Exception:
        pass
    return bool(os.getenv("GROQ_API_KEY"))
