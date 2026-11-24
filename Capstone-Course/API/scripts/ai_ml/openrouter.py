META = dict(
    name="OpenRouter",
    domain="AI & Machine Learning",
    docs="https://openrouter.ai/docs/api_reference/overview",
    auth="None to list models; API key (Bearer) to run them",
    env="OPENROUTER_API_KEY (needed for chat completions)",
    signup="Sign in at https://openrouter.ai -> Keys -> Create Key. Pay-as-you-go credits; some models are free "
           "(IDs ending in ':free', with low rate limits).",
    provides=["One OpenAI-compatible API for hundreds of models from many labs (OpenAI, Anthropic, Google, Meta, Qwen, ...)",
              "Public model catalog: context length, modalities, per-token prices, max output",
              "Automatic provider fallback and routing by price or latency"],
    ideas="Model-comparison studies at fixed cost, tracking LLM price trends over time from the public catalog.",
    endpoint="GET https://openrouter.ai/api/v1/models  ·  POST https://openrouter.ai/api/v1/chat/completions",
    params=[
        ("model", "Model ID from /models (add :free for free variants where offered)", "qwen/qwen3.8-27b"),
        ("messages", "OpenAI-style list of {role, content}", "[{\"role\": \"user\", \"content\": \"...\"}]"),
        ("max_tokens / temperature", "Output cap and sampling", "300 / 0.7"),
        ("models", "Fallback list tried in order if the first model fails", "[\"model-a\", \"model-b\"]"),
        ("provider.sort", "Route by price | throughput | latency", "price"),
        ("stream", "true -> server-sent events", "true"),
        ("Authorization (header)", "Bearer key", "Bearer <OPENROUTER_API_KEY>"),
    ],
)
import os

import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, get, show


def main():
    models = get("https://openrouter.ai/api/v1/models")["data"]
    free = [m["id"] for m in models if m["pricing"].get("prompt") == "0" and m["pricing"].get("completion") == "0"]
    out = {"models_available": len(models), "free_models": len(free), "newest": [
        {"id": m["id"], "context_length": m["context_length"],
         "usd_per_M_input": round(float(m["pricing"]["prompt"]) * 1e6, 2),
         "usd_per_M_output": round(float(m["pricing"]["completion"]) * 1e6, 2)}
        for m in sorted(models, key=lambda m: -m["created"])[:5]]}
    if os.getenv("OPENROUTER_API_KEY") and free:  # running a model needs a key; listing them does not
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=120, headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "User-Agent": USER_AGENT}, json={
            "model": free[0], "max_tokens": 300,
            "messages": [{"role": "user", "content": "Name one public dataset for studying air quality."}]})
        r.raise_for_status()
        out["chat_reply"] = r.json()["choices"][0]["message"]["content"]
    show(out)


if __name__ == "__main__":
    main()
