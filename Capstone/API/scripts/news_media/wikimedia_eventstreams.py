META = dict(
    name="Wikimedia EventStreams (real-time edits)",
    domain="News & Media",
    docs="https://wikitech.wikimedia.org/wiki/Event_Platform/EventStreams_HTTP_Service",
    auth="None",
    signup="No registration. A never-ending Server-Sent Events stream - read with stream=True, process events as they "
           "arrive, and stop after N events or seconds. Use since=<timestamp> to resume.",
    provides=["Every edit, page creation, log action and categorization across all Wikimedia wikis, live (~30+ events/second)",
              "Wiki, page title, user (or IP), bot flag, size change, comment, timestamp"],
    ideas="Breaking-news detection from editing bursts, bot vs. human activity, edit-war monitoring, streaming-data exercises.",
    endpoint="GET https://stream.wikimedia.org/v2/stream/{stream}",
    params=[
        ("stream", "recentchange | page-create | revision-create | mediawiki.page-delete ... (comma list allowed)", "recentchange"),
        ("since", "Start from a timestamp (ISO 8601) or offset to resume", "2026-10-12T00:00:00Z"),
        ("event fields (recentchange)", "wiki, type (edit, new, log, categorize), title, user, bot, length {old,new}, timestamp", "—"),
        ("format", "SSE lines: 'data: {json}'", "—"),
    ],
)
import json
from collections import Counter

import requests

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import USER_AGENT, show


def main():
    events = []
    with requests.get("https://stream.wikimedia.org/v2/stream/recentchange", stream=True, timeout=60,
                      headers={"User-Agent": USER_AGENT}) as r:
        for line in r.iter_lines():
            if line.startswith(b"data: "):
                events.append(json.loads(line[6:]))
                if len(events) >= 300:  # the stream never ends - stop after a sample
                    break
    span = (max(e["timestamp"] for e in events) - min(e["timestamp"] for e in events)) or 1
    show({"events_sampled": len(events), "approx_events_per_second": round(len(events) / span, 1),
          "by_wiki": Counter(e["wiki"] for e in events).most_common(5),
          "by_type": Counter(e["type"] for e in events).most_common(),
          "bot_share": round(sum(bool(e.get("bot")) for e in events) / len(events), 2)})


if __name__ == "__main__":
    main()
