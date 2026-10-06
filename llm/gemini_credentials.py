"""Gemini key access through the OS credential store (environment fallback)."""
from __future__ import annotations

import os

SERVICE = "VideoHighlighter/Gemini"
ACCOUNT = "api_key"


def get_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        import keyring
        return (keyring.get_password(SERVICE, ACCOUNT) or "").strip()
    except Exception:
        return ""


def save_key(value: str) -> None:
    value = value.strip()
    if not value:
        raise ValueError("Escribe una API key antes de guardarla.")
    try:
        import keyring
        keyring.set_password(SERVICE, ACCOUNT, value)
    except Exception:
        raise RuntimeError("No se pudo guardar la clave en el administrador de credenciales de Windows.") from None


def delete_key() -> None:
    try:
        import keyring
        keyring.delete_password(SERVICE, ACCOUNT)
    except Exception:
        pass
