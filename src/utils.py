import os
import google.generativeai as genai
import streamlit as st


def initialize_gemini_client() -> bool:
    """Configures the Gemini API client using environment variables or Streamlit secrets."""
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception:
            return False
    return False
