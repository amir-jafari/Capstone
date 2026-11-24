META = dict(
    name="Ollama (local LLMs)",
    domain="AI & Machine Learning",
    docs="https://docs.ollama.com/api/introduction",
    auth="None (runs on your own machine)",
    env="OLLAMA_MODEL (optional, default llama3.2)",
    signup="Install from https://ollama.com/download, then `ollama pull llama3.2` (or any model from "
           "https://ollama.com/library). Free; speed depends on your hardware. Data never leaves your machine.",
    provides=["Chat / generation with open-weight models running locally (Llama, Qwen, Gemma, Mistral, DeepSeek...)",
              "Embeddings (/api/embed), JSON-schema structured output, tool calling", "OpenAI-compatible endpoint at /v1/chat/completions"],
    ideas="Private-data projects (no data sent to a vendor), reproducible experiments with pinned local models.",
    endpoint="POST http://localhost:11434/api/chat",
    params=[
        ("model", "Any pulled model (GET /api/tags lists them)", "llama3.2"),
        ("messages", "List of {role, content}", "[{\"role\": \"user\", \"content\": \"...\"}]"),
        ("stream", "false -> one JSON response (default streams JSON lines)", "false"),
        ("format", "\"json\" or a JSON schema for structured output", "json"),
        ("options", "Sampling/runtime options: temperature, num_ctx, seed, ...", "{\"temperature\": 0, \"seed\": 42}"),
        ("/api/embed", "Sibling endpoint: embeddings (model, input)", "model=nomic-embed-text"),
    ],
    sample="""{
  "model": "llama3.2",
  "reply": "The EPA Air Quality System (AQS) and OpenAQ are two widely used public air-quality datasets.",
  "eval_count": 24
}""",
)
import os
import sys

import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import show


def main():
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    try:
        r = requests.post("http://localhost:11434/api/chat", timeout=300, json={
            "model": model, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "user", "content": "Name one public dataset for studying air quality."}]})
    except requests.ConnectionError:
        print("[skip] Ollama is not running on localhost:11434 (install it and run `ollama pull llama3.2`).")
        sys.exit(2)
    r.raise_for_status()
    d = r.json()
    show({"model": d["model"], "reply": d["message"]["content"], "eval_count": d.get("eval_count")})


if __name__ == "__main__":
    main()
