META = dict(
    name="GDELT Television 2.0 API (TV news captions)",
    domain="News & Media",
    docs="https://blog.gdeltproject.org/gdelt-2-0-television-api-debuts/",
    auth="None",
    signup="No registration. Searches closed captions of U.S. and international TV news (Internet Archive TV News "
           "Archive, 2009 onward). Note: queries for periods after mid-2024 currently return no data (the archive stopped "
           "updating after the Internet Archive's 2024 outage), so treat it as a historical source.",
    provides=["Share of airtime mentioning a phrase, per station and day (CNN, Fox News, MSNBC, BBC, network evening news...)",
              "Station comparisons, top clips with links, word clouds of co-occurring words",
              "15-second resolution caption matches"],
    ideas="Media attention and partisan agenda-setting (CNN vs. Fox vs. MSNBC), coverage of elections, climate or wars.",
    endpoint="GET https://api.gdeltproject.org/api/v2/tv/tv",
    params=[
        ("query", "Keywords plus at least one station:NAME (OR several in parentheses)", "climate (station:CNN OR station:FOXNEWS OR station:MSNBC)"),
        ("mode", "timelinevol | stationchart | clipgallery | wordcloud | timelinevolnorm", "stationchart"),
        ("startdatetime / enddatetime", "YYYYMMDDHHMMSS (use dates before mid-2024)", "20240101000000 / 20240701000000"),
        ("format", "json | csv | html", "json"),
        ("maxrecords", "For clipgallery (max 3,000)", "10"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    d = get("https://api.gdeltproject.org/api/v2/tv/tv", params={
        "query": '"climate change" (station:CNN OR station:FOXNEWS OR station:MSNBC)', "mode": "stationchart",
        "format": "json", "startdatetime": "20240101000000", "enddatetime": "20240701000000"})
    show({"query": '"climate change" on cable news, Jan-Jun 2024',
          "pct_of_15s_airtime_blocks": {s["station"]: s["count"] for s in d["stationchart"]}})


if __name__ == "__main__":
    main()
