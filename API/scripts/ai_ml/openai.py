META = dict(
    name="OpenAI API",
    domain="AI & Machine Learning",
    docs="https://developers.openai.com/api/docs",
    auth="API key (header `Authorization: Bearer`)",
    env="OPENAI_API_KEY",
    signup="Create an account at https://platform.openai.com -> API keys -> Create new secret key, and add billing "
           "credits (pay-as-you-go). Small model example: gpt-6-luna at $0.10 in / $0.50 out per million tokens.",
    provides=["Text generation via the Responses API (gpt-6-astra flagship, gpt-6.1-sol balanced, gpt-6-luna small)",
              "Embeddings, image generation, speech-to-text and text-to-speech, moderation", "Batch API, fine-tuning, file search"],
    ideas="Embedding-based clustering of text datasets, zero-shot classification baselines, transcription of audio data.",
    endpoint="POST https://api.openai.com/v1/responses",
    params=[
        ("model", "gpt-6-astra | gpt-6.1-sol | gpt-6-luna (GET /v1/models lists what your key can use)", "gpt-6-luna"),
        ("input", "A string, or a list of {role, content} messages", "Suggest three capstone ideas."),
        ("instructions", "System-style instructions", "You are a concise research assistant."),
        ("max_output_tokens", "Cap on generated tokens", "500"),
        ("stream", "true -> server-sent events", "true"),
        ("/v1/embeddings", "Sibling endpoint: vector embeddings (model, input)", "—"),
        ("Authorization (header)", "Bearer token", "Bearer <OPENAI_API_KEY>"),
    ],
    sample="""{
  "model": "gpt-6-luna",
  "text": "1. Track how news sentiment (GDELT) leads stock volatility...\\n2. ...\\n3. ...",
  "usage": {"input_tokens": 32, "output_tokens": 88}
}""",
)
import os

import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, key, show


def main():
    r = requests.post("https://api.openai.com/v1/responses", timeout=120, headers={
        "Authorization": f"Bearer {key('OPENAI_API_KEY')}", "User-Agent": USER_AGENT}, json={
        "model": os.getenv("OPENAI_MODEL", "gpt-6-luna"),
        "instructions": "You are a concise research assistant for data-science students.",
        "input": "Suggest three capstone project ideas that combine public APIs with an LLM. One line each.",
        "max_output_tokens": 500,
    })
    r.raise_for_status()
    d = r.json()
    # Text lives in output items of type "message" -> content items of type "output_text"
    text = "".join(c["text"] for item in d["output"] if item["type"] == "message"
                   for c in item["content"] if c["type"] == "output_text")
    show({"model": d["model"], "text": text, "usage": {k: d["usage"][k] for k in ("input_tokens", "output_tokens")}},
         max_str=1000)


if __name__ == "__main__":
    main()
