META = dict(
    name="Anthropic Claude API",
    domain="AI & Machine Learning",
    docs="https://platform.claude.com/docs/en/api/overview · models & pricing: https://platform.claude.com/docs/en/about-claude/pricing",
    auth="API key (header `x-api-key`; the Python SDK reads `ANTHROPIC_API_KEY`)",
    env="ANTHROPIC_API_KEY",
    signup="Create an account at https://platform.claude.com -> Settings -> API Keys -> Create Key, then add credits under "
           "Billing (pay-as-you-go). Example prices per million tokens: Claude Opus 5.5 $4 in / $20 out, Claude Haiku 5.5 "
           "$0.10 / $0.50. Install the SDK with `pip install anthropic`.",
    provides=["Text generation, summarization, classification, extraction, and Q&A over long documents (1M-token context)",
              "Vision (images, PDFs), tool use / function calling, structured JSON outputs",
              "Batch API (50% cheaper, asynchronous), prompt caching, token counting, Models API"],
    ideas="LLM-assisted labeling of text data, summarizing documents pulled from the other APIs here, LLM-as-judge evaluation.",
    endpoint="POST https://api.anthropic.com/v1/messages  (Python SDK: client.messages.create / client.beta.messages.create)",
    params=[
        ("x-api-key / anthropic-version (headers)", "Your key; API version header for raw HTTP (the SDK sets both)", "anthropic-version: 2023-06-01"),
        ("model", "claude-opus-5-5 (default, most capable Opus), claude-sonnet-5-5, claude-haiku-5-5 (fastest, cheapest)", "claude-opus-5-5"),
        ("max_tokens", "Required cap on output tokens; ~16,000 for non-streaming calls (stream for larger)", "16000"),
        ("messages", "List of {role: user | assistant, content}; the first must be user", "[{\"role\": \"user\", \"content\": \"...\"}]"),
        ("system", "System prompt", "You are a concise research assistant."),
        ("output_config.effort", "low | medium | high | xhigh | max - how much the model thinks (default medium on Opus 5.5)", "low"),
        ("output_config.format", "Structured outputs: force JSON that matches a schema", "{\"type\": \"json_schema\", \"schema\": {...}}"),
        ("stream", "true -> server-sent events (SDK: client.messages.stream)", "true"),
        ("tools", "Tool / function definitions the model can call", "[{\"name\": ..., \"input_schema\": {...}}]"),
        ("fallbacks + anthropic-beta header", "\"default\" retries a safety-classifier refusal on a recommended fallback model", "server-side-fallback-2026-07-01"),
        ("temperature / top_p / top_k", "Not accepted on current models (returns 400) - steer with the prompt and effort instead", "—"),
    ],
    sample="""{
  "model": "claude-opus-5-5",
  "stop_reason": "end_turn",
  "text": "1. Policy-text classifier: label Federal Register rules by topic with an LLM, then train a cheap model on the labels.\\n2. News-to-markets: summarize GDELT coverage per company and test whether sentiment predicts SEC-filing events.\\n3. Education Q&A: answer questions over College Scorecard data with tool calls that query the API live.",
  "usage": {"input_tokens": 41, "output_tokens": 96}
}""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import key, show


def main():
    key("ANTHROPIC_API_KEY")  # the SDK reads the key from the environment (.env is loaded by _common)
    import anthropic  # only this example needs the SDK: pip install anthropic

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # if a safety classifier declines, retry on Anthropic's recommended fallback model
        output_config={"effort": "low"},  # a short, simple task; raise effort for harder reasoning
        system="You are a concise research assistant for data-science students.",
        messages=[{"role": "user", "content": "Suggest three capstone project ideas that combine public APIs "
                                              "with an LLM. One line each."}],
    )
    if response.stop_reason == "refusal":  # check before reading content
        show({"stop_reason": "refusal", "category": response.stop_details and response.stop_details.category})
        return
    show({"model": response.model, "stop_reason": response.stop_reason,
          "text": "".join(b.text for b in response.content if b.type == "text"),
          "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}},
         max_str=1000)


if __name__ == "__main__":
    main()
