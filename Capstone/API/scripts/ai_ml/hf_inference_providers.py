META = dict(
    name="Hugging Face Inference Providers",
    domain="AI & Machine Learning",
    docs="https://huggingface.co/docs/inference-providers",
    auth="None to list models; Hugging Face token (Bearer) to run them",
    env="HF_TOKEN (needed for chat completions)",
    signup="Create a token at https://huggingface.co/settings/tokens (enable 'Make calls to Inference Providers'). "
           "Free accounts get a small monthly credit; PRO accounts more. One token reaches many providers.",
    provides=["One OpenAI-compatible endpoint for open-weight models (Qwen, Llama, DeepSeek, Mistral, ...) served by many providers",
              "Live model list with per-provider price, context length, latency, throughput, tool-use support",
              "Also text-to-image, speech, embeddings via provider-specific routes"],
    ideas="Compare open-weight models on your task, cost/latency benchmarking across providers.",
    endpoint="GET https://router.huggingface.co/v1/models  ·  POST https://router.huggingface.co/v1/chat/completions",
    params=[
        ("model", "Hub model ID; append :provider or :fastest / :cheapest to pick a provider", "Qwen/Qwen3.8-27B:cheapest"),
        ("messages", "OpenAI-style list of {role, content}", "[{\"role\": \"user\", \"content\": \"...\"}]"),
        ("max_tokens", "Cap on generated tokens", "300"),
        ("temperature / top_p", "Sampling controls", "0.7"),
        ("stream", "true -> server-sent events", "true"),
        ("tools / response_format", "Function calling / JSON output (where the provider supports it)", "{\"type\": \"json_object\"}"),
        ("Authorization (header)", "Bearer token", "Bearer <HF_TOKEN>"),
    ],
)
import os

import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, get, show


def main():
    models = get("https://router.huggingface.co/v1/models")["data"]
    out = {"models_available": len(models), "sample_models": [
        {"id": m["id"], "providers": len(m["providers"]),
         "cheapest_input_usd_per_M": min((p["pricing"]["input"] for p in m["providers"] if p.get("pricing")), default=None)}
        for m in models[:5]]}
    if os.getenv("HF_TOKEN"):  # running a model needs a token; listing them does not
        r = requests.post("https://router.huggingface.co/v1/chat/completions", timeout=120, headers={
            "Authorization": f"Bearer {os.environ['HF_TOKEN']}", "User-Agent": USER_AGENT}, json={
            "model": f'{models[0]["id"]}:cheapest', "max_tokens": 300,
            "messages": [{"role": "user", "content": "Name one public dataset for studying air quality."}]})
        r.raise_for_status()
        out["chat_reply"] = r.json()["choices"][0]["message"]["content"]
    show(out)


if __name__ == "__main__":
    main()
