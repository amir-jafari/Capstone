META = dict(
    name="Google Gemini API",
    domain="AI & Machine Learning",
    docs="https://ai.google.dev/gemini-api/docs",
    auth="API key (header `x-goog-api-key`)",
    env="GEMINI_API_KEY",
    signup="Get a key at https://aistudio.google.com/apikey (Google account). A free tier with lower rate limits is "
           "available for some models; see https://ai.google.dev/gemini-api/docs/rate-limits.",
    provides=["Text and multimodal generation (gemini-3.8-flash; gemini-3.5-flash-lite for high volume)",
              "Image, audio and video understanding; image generation and text-to-speech models", "Embeddings, Live API (real-time audio)"],
    ideas="Multimodal capstones (video/audio analysis), low-cost bulk labeling with Flash-Lite.",
    endpoint="POST https://generativelanguage.googleapis.com/v1beta/interactions",
    params=[
        ("model", "gemini-3.8-flash | gemini-3.5-flash-lite | ... (GET /v1beta/models lists them)", "gemini-3.8-flash"),
        ("input", "A string, or a list of content objects ({type: text | image, ...})", "Suggest three capstone ideas."),
        ("system_instruction", "System prompt", "You are a concise research assistant."),
        ("generation_config.temperature", "Sampling temperature", "1.0"),
        ("generation_config.thinking_level", "How much the model thinks", "low"),
        ("store / previous_interaction_id", "store=false for stateless calls; chain turns server-side by ID", "false"),
        ("x-goog-api-key (header)", "Your key", "<GEMINI_API_KEY>"),
    ],
    sample="""{
  "model": "gemini-3.8-flash",
  "text": "1. Predict wildfire smoke exposure by joining NASA EONET events with OpenAQ readings...\\n2. ...\\n3. ..."
}""",
)
import os

import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, key, show


def main():
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    r = requests.post("https://generativelanguage.googleapis.com/v1beta/interactions", timeout=120, headers={
        "x-goog-api-key": key("GEMINI_API_KEY"), "User-Agent": USER_AGENT}, json={
        "model": model, "store": False,
        "system_instruction": "You are a concise research assistant for data-science students.",
        "input": "Suggest three capstone project ideas that combine public APIs with an LLM. One line each.",
        "generation_config": {"thinking_level": "low"},
    })
    r.raise_for_status()
    show({"model": model, "text": r.json()["steps"][-1]["content"][0]["text"]}, max_str=1000)


if __name__ == "__main__":
    main()
