# Public API Catalog for Data Science Capstones

> A curated, runnable list of free APIs across news, social media, weather, finance, census,
> political science, health, science, engineering, transportation, and culture.
> Every entry has: how to register / get a token, what data you can get, a short Python
> example, and a **sample output**.

## Quick Start

```bash
cd API
pip install -r requirements.txt
cp .env.example .env            # paste in any keys you register for
python scripts/weather_environment/open_meteo.py   # run any single example
python build_readme.py          # re-run everything & regenerate this README
```

- Scripts share helpers in [`scripts/_common.py`](scripts/_common.py): `get()` (User-Agent + retry on 429),
  `key()` (reads `.env`), and `show()` (truncated pretty-print).
- Sample outputs marked **live run** were produced by actually running the script (saved in [`samples/`](samples/)).
  Outputs marked **⚠️ illustrative** could not be run live (key required or rate-limited); they show the response shape from the official docs.
- **One key, many APIs:** a free [api.data.gov key](https://api.data.gov/signup/) works for Congress.gov,
  OpenFEC, Regulations.gov, NASA, and NLR (formerly NREL). Set it as `DATA_GOV_API_KEY`.
- Always check each API's terms of use (non-commercial clauses, attribution, rate limits) before
  publishing a dataset built from it, and **never commit `.env`** (it is git-ignored).

## Catalog at a Glance

| # | Domain | API | Auth | Key / env var | Script |
|--:|:--|:--|:--|:--|:--|
| 1 | 📰 News & Media | [GDELT DOC 2.0](#gdelt-doc-20) | 🟢 no key | — | [`gdelt.py`](scripts/news_media/gdelt.py) |
| 2 | 📰 News & Media | [The Guardian Open Platform](#the-guardian-open-platform) | 🔑 key | `GUARDIAN_API_KEY` | [`guardian.py`](scripts/news_media/guardian.py) |
| 3 | 📰 News & Media | [Hacker News (Firebase API)](#hacker-news-firebase-api) | 🟢 no key | — | [`hacker_news.py`](scripts/news_media/hacker_news.py) |
| 4 | 📰 News & Media | [NewsAPI.org](#newsapiorg) | 🔑 key | `NEWSAPI_KEY` | [`newsapi.py`](scripts/news_media/newsapi.py) |
| 5 | 📰 News & Media | [New York Times APIs](#new-york-times-apis) | 🔑 key | `NYT_API_KEY` | [`nyt.py`](scripts/news_media/nyt.py) |
| 6 | 📰 News & Media | [Wikimedia REST API (Pageviews)](#wikimedia-rest-api-pageviews) | 🟢 no key | — | [`wikipedia_pageviews.py`](scripts/news_media/wikipedia_pageviews.py) |
| 7 | 💬 Social Media | [Bluesky (AT Protocol) Public AppView](#bluesky-at-protocol-public-appview) | 🟢 no key | — | [`bluesky.py`](scripts/social_media/bluesky.py) |
| 8 | 💬 Social Media | [Mastodon](#mastodon) | 🟢 no key | — | [`mastodon.py`](scripts/social_media/mastodon.py) |
| 9 | 💬 Social Media | [Reddit Data API](#reddit-data-api) | 🔑 key | `REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET` | [`reddit.py`](scripts/social_media/reddit.py) |
| 10 | 💬 Social Media | [Stack Exchange API](#stack-exchange-api) | 🟢 no key | — | [`stackexchange.py`](scripts/social_media/stackexchange.py) |
| 11 | 💬 Social Media | [YouTube Data API v3](#youtube-data-api-v3) | 🔑 key | `YOUTUBE_API_KEY` | [`youtube.py`](scripts/social_media/youtube.py) |
| 12 | 🌦️ Weather, Climate & Environment | [NOAA Climate Data Online (CDO)](#noaa-climate-data-online-cdo) | 🔑 key | `NOAA_TOKEN` | [`noaa_cdo.py`](scripts/weather_environment/noaa_cdo.py) |
| 13 | 🌦️ Weather, Climate & Environment | [US National Weather Service (api.weather.gov)](#us-national-weather-service-apiweathergov) | 🟢 no key | — | [`nws.py`](scripts/weather_environment/nws.py) |
| 14 | 🌦️ Weather, Climate & Environment | [Open-Meteo](#open-meteo) | 🟢 no key | — | [`open_meteo.py`](scripts/weather_environment/open_meteo.py) |
| 15 | 🌦️ Weather, Climate & Environment | [OpenAQ v3 (Air Quality)](#openaq-v3-air-quality) | 🔑 key | `OPENAQ_API_KEY` | [`openaq.py`](scripts/weather_environment/openaq.py) |
| 16 | 🌦️ Weather, Climate & Environment | [OpenWeatherMap](#openweathermap) | 🔑 key | `OPENWEATHER_API_KEY` | [`openweathermap.py`](scripts/weather_environment/openweathermap.py) |
| 17 | 🌦️ Weather, Climate & Environment | [USGS Earthquake Catalog](#usgs-earthquake-catalog) | 🟢 no key | — | [`usgs_earthquakes.py`](scripts/weather_environment/usgs_earthquakes.py) |
| 18 | 💵 Finance & Economics | [Alpha Vantage](#alpha-vantage) | 🟢 no key | `ALPHAVANTAGE_API_KEY` | [`alpha_vantage.py`](scripts/finance_economics/alpha_vantage.py) |
| 19 | 💵 Finance & Economics | [US Bureau of Labor Statistics (BLS)](#us-bureau-of-labor-statistics-bls) | 🟢 no key | `BLS_API_KEY (optional)` | [`bls.py`](scripts/finance_economics/bls.py) |
| 20 | 💵 Finance & Economics | [CoinGecko](#coingecko) | 🟢 no key | — | [`coingecko.py`](scripts/finance_economics/coingecko.py) |
| 21 | 💵 Finance & Economics | [Frankfurter (ECB exchange rates)](#frankfurter-ecb-exchange-rates) | 🟢 no key | — | [`frankfurter.py`](scripts/finance_economics/frankfurter.py) |
| 22 | 💵 Finance & Economics | [FRED (Federal Reserve Economic Data)](#fred-federal-reserve-economic-data) | 🔑 key | `FRED_API_KEY` | [`fred.py`](scripts/finance_economics/fred.py) |
| 23 | 💵 Finance & Economics | [SEC EDGAR (data.sec.gov)](#sec-edgar-datasecgov) | 🟢 no key | — | [`sec_edgar.py`](scripts/finance_economics/sec_edgar.py) |
| 24 | 💵 Finance & Economics | [US Treasury Fiscal Data](#us-treasury-fiscal-data) | 🟢 no key | — | [`treasury_fiscal.py`](scripts/finance_economics/treasury_fiscal.py) |
| 25 | 💵 Finance & Economics | [World Bank Indicators API](#world-bank-indicators-api) | 🟢 no key | — | [`world_bank.py`](scripts/finance_economics/world_bank.py) |
| 26 | 👥 Census & Demographics | [IPUMS API (microdata)](#ipums-api-microdata) | 🔑 key | `IPUMS_API_KEY` | [`ipums.py`](scripts/census_demographics/ipums.py) |
| 27 | 👥 Census & Demographics | [UN Data Portal — World Population Prospects](#un-data-portal--world-population-prospects) | 🔑 key | `UN_POP_TOKEN` | [`un_population.py`](scripts/census_demographics/un_population.py) |
| 28 | 👥 Census & Demographics | [US Census Bureau Data API](#us-census-bureau-data-api) | 🔑 key | `CENSUS_API_KEY` | [`us_census.py`](scripts/census_demographics/us_census.py) |
| 29 | 🏛️ Political Science & Government | [ACLED (Armed Conflict Location & Event Data)](#acled-armed-conflict-location--event-data) | 🔑 key | `ACLED_EMAIL, ACLED_PASSWORD` | [`acled.py`](scripts/politics_government/acled.py) |
| 30 | 🏛️ Political Science & Government | [Congress.gov API](#congressgov-api) | 🟢 no key | `DATA_GOV_API_KEY` | [`congress_gov.py`](scripts/politics_government/congress_gov.py) |
| 31 | 🏛️ Political Science & Government | [Federal Register API](#federal-register-api) | 🟢 no key | — | [`federal_register.py`](scripts/politics_government/federal_register.py) |
| 32 | 🏛️ Political Science & Government | [Open States / Plural (US state legislatures)](#open-states--plural-us-state-legislatures) | 🔑 key | `OPENSTATES_API_KEY` | [`open_states.py`](scripts/politics_government/open_states.py) |
| 33 | 🏛️ Political Science & Government | [OpenFEC (Federal Election Commission)](#openfec-federal-election-commission) | 🟢 no key | `DATA_GOV_API_KEY` | [`openfec.py`](scripts/politics_government/openfec.py) |
| 34 | 🏛️ Political Science & Government | [Regulations.gov API v4](#regulationsgov-api-v4) | 🟢 no key | `DATA_GOV_API_KEY` | [`regulations_gov.py`](scripts/politics_government/regulations_gov.py) |
| 35 | 🏛️ Political Science & Government | [UK Parliament APIs](#uk-parliament-apis) | 🟢 no key | — | [`uk_parliament.py`](scripts/politics_government/uk_parliament.py) |
| 36 | 🏛️ Political Science & Government | [USAspending.gov](#usaspendinggov) | 🟢 no key | — | [`usaspending.py`](scripts/politics_government/usaspending.py) |
| 37 | 🩺 Health & Social Science | [CDC Open Data (Socrata SODA)](#cdc-open-data-socrata-soda) | 🟢 no key | `SOCRATA_APP_TOKEN (optional)` | [`cdc_socrata.py`](scripts/health/cdc_socrata.py) |
| 38 | 🩺 Health & Social Science | [ClinicalTrials.gov API v2](#clinicaltrialsgov-api-v2) | 🟢 no key | — | [`clinicaltrials.py`](scripts/health/clinicaltrials.py) |
| 39 | 🩺 Health & Social Science | [openFDA](#openfda) | 🟢 no key | `OPENFDA_API_KEY (optional)` | [`openfda.py`](scripts/health/openfda.py) |
| 40 | 🩺 Health & Social Science | [Our World in Data (Grapher API)](#our-world-in-data-grapher-api) | 🟢 no key | — | [`our_world_in_data.py`](scripts/health/our_world_in_data.py) |
| 41 | 🩺 Health & Social Science | [WHO Global Health Observatory (GHO OData)](#who-global-health-observatory-gho-odata) | 🟢 no key | — | [`who_gho.py`](scripts/health/who_gho.py) |
| 42 | 🔬 Science & Research | [arXiv API](#arxiv-api) | 🟢 no key | — | [`arxiv.py`](scripts/science_research/arxiv.py) |
| 43 | 🔬 Science & Research | [Crossref REST API](#crossref-rest-api) | 🟢 no key | — | [`crossref.py`](scripts/science_research/crossref.py) |
| 44 | 🔬 Science & Research | [NASA Open APIs (APOD, NeoWs, DONKI, EONET...)](#nasa-open-apis-apod-neows-donki-eonet) | 🟢 no key | `DATA_GOV_API_KEY` | [`nasa.py`](scripts/science_research/nasa.py) |
| 45 | 🔬 Science & Research | [OpenAlex (scholarly works)](#openalex-scholarly-works) | 🟢 no key | — | [`openalex.py`](scripts/science_research/openalex.py) |
| 46 | 🔬 Science & Research | [PubChem PUG-REST](#pubchem-pug-rest) | 🟢 no key | — | [`pubchem.py`](scripts/science_research/pubchem.py) |
| 47 | 🔬 Science & Research | [Semantic Scholar Academic Graph](#semantic-scholar-academic-graph) | 🟢 no key | `S2_API_KEY (optional)` | [`semantic_scholar.py`](scripts/science_research/semantic_scholar.py) |
| 48 | ⚙️ Engineering, Energy & Tech | [US Energy Information Administration (EIA) v2](#us-energy-information-administration-eia-v2) | 🔑 key | `EIA_API_KEY` | [`eia.py`](scripts/engineering_energy/eia.py) |
| 49 | ⚙️ Engineering, Energy & Tech | [GitHub REST API](#github-rest-api) | 🟢 no key | `GITHUB_TOKEN (optional)` | [`github.py`](scripts/engineering_energy/github.py) |
| 50 | ⚙️ Engineering, Energy & Tech | [Hugging Face Hub API](#hugging-face-hub-api) | 🟢 no key | `HF_TOKEN (optional)` | [`huggingface.py`](scripts/engineering_energy/huggingface.py) |
| 51 | ⚙️ Engineering, Energy & Tech | [NLR (formerly NREL) Developer Network — solar, EV charging](#nlr-formerly-nrel-developer-network--solar-ev-charging) | 🟢 no key | `DATA_GOV_API_KEY` | [`nrel.py`](scripts/engineering_energy/nrel.py) |
| 52 | ⚙️ Engineering, Energy & Tech | [NIST National Vulnerability Database (CVE API 2.0)](#nist-national-vulnerability-database-cve-api-20) | 🟢 no key | `NVD_API_KEY (optional)` | [`nvd_cve.py`](scripts/engineering_energy/nvd_cve.py) |
| 53 | ⚙️ Engineering, Energy & Tech | [USGS Water Services (streamflow)](#usgs-water-services-streamflow) | 🟢 no key | — | [`usgs_water.py`](scripts/engineering_energy/usgs_water.py) |
| 54 | 🗺️ Transportation & Geospatial | [OpenStreetMap Nominatim (geocoding)](#openstreetmap-nominatim-geocoding) | 🟢 no key | — | [`nominatim.py`](scripts/transportation_geo/nominatim.py) |
| 55 | 🗺️ Transportation & Geospatial | [OpenSky Network (live flights)](#opensky-network-live-flights) | 🟢 no key | — | [`opensky.py`](scripts/transportation_geo/opensky.py) |
| 56 | 🗺️ Transportation & Geospatial | [OpenStreetMap Overpass API](#openstreetmap-overpass-api) | 🟢 no key | — | [`overpass.py`](scripts/transportation_geo/overpass.py) |
| 57 | 🗺️ Transportation & Geospatial | [WMATA (DC Metro) API](#wmata-dc-metro-api) | 🔑 key | `WMATA_API_KEY` | [`wmata.py`](scripts/transportation_geo/wmata.py) |
| 58 | 🎬 Culture, Books & Entertainment | [The Met Museum Collection API](#the-met-museum-collection-api) | 🟢 no key | — | [`met_museum.py`](scripts/culture_misc/met_museum.py) |
| 59 | 🎬 Culture, Books & Entertainment | [Open Library (books)](#open-library-books) | 🟢 no key | — | [`open_library.py`](scripts/culture_misc/open_library.py) |
| 60 | 🎬 Culture, Books & Entertainment | [TMDB (The Movie Database)](#tmdb-the-movie-database) | 🔑 key | `TMDB_TOKEN` | [`tmdb.py`](scripts/culture_misc/tmdb.py) |

🟢 = runs out of the box (no key, or a public demo key)  ·  🔑 = free registration required

---

## 📰 News & Media

### GDELT DOC 2.0

| | |
|:--|:--|
| **Docs** | https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/ |
| **Auth** | None |
| **Register / token** | No registration. Be polite: ~1 request every 5 seconds. |
| **Script** | [`scripts/news_media/gdelt.py`](scripts/news_media/gdelt.py) |

**What you can get:**

- Global news articles (65+ languages) from the last ~3 months matching a query
- Article URL, title, domain, source country, language, timestamp
- Timeline modes: volume / tone (sentiment) of coverage over time

**Capstone ideas:** Media framing & sentiment around elections, conflicts, policy events; event detection.

```python
from _common import get, show


def main():
    data = get("https://api.gdeltproject.org/api/v2/doc/doc", params={
        "query": '"artificial intelligence" sourcecountry:US',
        "mode": "artlist", "maxrecords": 5, "format": "json", "sort": "datedesc",
    })
    show([{k: a[k] for k in ("title", "domain", "seendate", "language", "sourcecountry")}
          for a in data.get("articles", [])])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:14</summary>

```json
[
  {
    "title": "Meta Platforms ( NASDAQ : META ) Shares Unloaded by Rep . Maria Elvira Salazar",
    "domain": "themarketsdaily.com",
    "seendate": "20261010T064500Z",
    "language": "English",
    "sourcecountry": "United States"
  },
  {
    "title": "We Must Embrace , Not Pause , AI and Data Centers",
    "domain": "townhall.com",
    "seendate": "20261010T064500Z",
    "language": "English",
    "sourcecountry": "United States"
  },
  {
    "title": "Op - ed : West Virginia AI future depends on 7 , 000 bridges",
    "domain": "newsandsentinel.com",
    "seendate": "20261010T064500Z",
    "language": "English",
    "sourcecountry": "United States"
  },
  {
    "title": "How AI is changing lawyers , billable hours and legal careers",
    "domain": "cnbc.com",
    "seendate": "20261010T064500Z",
    "language": "English",
    "sourcecountry": "United States"
  },
  {
    "title": "Mobileye Global ( NASDAQ : MBLY ) Stock Gets Price Target Cut from UBS Group",
    "domain": "themarketsdaily.com",
    "seendate": "20261010T064500Z",
    "language": "English",
    "sourcecountry": "United States"
  }
]
```
</details>

---

### The Guardian Open Platform

| | |
|:--|:--|
| **Docs** | https://open-platform.theguardian.com/documentation/ |
| **Auth** | API key (query `api-key`) |
| **Register / token** | Register for a free 'Developer key' at https://open-platform.theguardian.com/access/ (emailed instantly). 1 call/sec, 500 calls/day, non-commercial. |
| **Env var** | `GUARDIAN_API_KEY` |
| **Script** | [`scripts/news_media/guardian.py`](scripts/news_media/guardian.py) |

**What you can get:**

- Search 2M+ articles back to 1999 with full body text (`show-fields=bodyText`)
- Section, tags (keywords, contributors), publication date
- Great for longitudinal corpora — the archive is free and complete

**Capstone ideas:** Long-run framing of immigration/climate in the press, NLP corpora with full text.

```python
from _common import get, key, show


def main():
    data = get("https://content.guardianapis.com/search", params={
        "q": "election", "page-size": 5, "order-by": "newest",
        "show-fields": "wordcount", "api-key": key("GUARDIAN_API_KEY"),
    })["response"]
    show({"total": data["total"], "results": [
        {"webTitle": r["webTitle"], "sectionName": r["sectionName"],
         "webPublicationDate": r["webPublicationDate"], "wordcount": r["fields"]["wordcount"], "webUrl": r["webUrl"]}
        for r in data["results"]]})
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "total": 18432,
  "results": [
    {
      "webTitle": "What the polls say three weeks before the midterms",
      "sectionName": "US news",
      "webPublicationDate": "2026-10-10T11:00:12Z",
      "wordcount": "1243",
      "webUrl": "https://www.theguardian.com/us-news/2026/oct/10/..."
    }
  ]
}
```
</details>

---

### Hacker News (Firebase API)

| | |
|:--|:--|
| **Docs** | https://github.com/HackerNews/API |
| **Auth** | None |
| **Register / token** | No registration, no rate limit published. |
| **Script** | [`scripts/news_media/hacker_news.py`](scripts/news_media/hacker_news.py) |

**What you can get:**

- Top / new / best / Ask / Show story IDs
- Story title, URL, score, author, comment count, time
- Full comment trees (item `kids`)
- User profiles (karma, submissions)

**Capstone ideas:** Tech-trend detection, comment toxicity / sentiment, predicting story popularity.

```python
from _common import get, show

BASE = "https://hacker-news.firebaseio.com/v0"


def main():
    ids = get(f"{BASE}/topstories.json")[:5]
    stories = [get(f"{BASE}/item/{i}.json") for i in ids]
    show([{k: s.get(k) for k in ("title", "url", "score", "by", "descendants")} for s in stories])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "title": "Total Annihilation, rebuilt for modern computers",
    "url": "https://nanolathe.gg",
    "score": 291,
    "by": "jttnr",
    "descendants": 122
  },
  {
    "title": "Cartographers of North Korea",
    "url": "https://cartographers-nk.wonyoung.so/",
    "score": 130,
    "by": "edent",
    "descendants": 20
  },
  {
    "title": "The gap in shared understanding of LLM capability is widening",
    "url": "https://twitter.com/karpathy/status/2109361546505966046",
    "score": 131,
    "by": "albertzeyer",
    "descendants": 173
  },
  {
    "title": "So, You’d Like To Sell Your Print Issues Of The Onion?",
    "url": "https://theonion.com/archivehunt/",
    "score": 213,
    "by": "mooreds",
    "descendants": 25
  },
  {
    "title": "Google Data Centers around the world",
    "url": "https://datacenters.google/locations/",
    "score": 4,
    "by": "andsoitis",
    "descendants": 0
  }
]
```
</details>

---

### NewsAPI.org

| | |
|:--|:--|
| **Docs** | https://newsapi.org/docs |
| **Auth** | API key (header `X-Api-Key`) |
| **Register / token** | Register at https://newsapi.org/register -> key shown on dashboard. Free 'Developer' plan: 100 requests/day, articles up to 1 month old, development use only. |
| **Env var** | `NEWSAPI_KEY` |
| **Script** | [`scripts/news_media/newsapi.py`](scripts/news_media/newsapi.py) |

**What you can get:**

- Headlines by country / category / source (~150k sources)
- Full-text search across articles (`/v2/everything`)
- Title, description, source, author, URL, image, publishedAt, content snippet

**Capstone ideas:** News topic modeling, headline sentiment vs. stock moves, source bias comparison.

```python
from _common import get, key, show


def main():
    data = get("https://newsapi.org/v2/top-headlines",
               params={"country": "us", "category": "business", "pageSize": 5},
               headers={"X-Api-Key": key("NEWSAPI_KEY")})
    show([{"source": a["source"]["name"], "title": a["title"], "publishedAt": a["publishedAt"], "url": a["url"]}
          for a in data["articles"]])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {
    "source": "Reuters",
    "title": "Fed holds rates steady, signals patience on cuts",
    "publishedAt": "2026-10-10T18:42:00Z",
    "url": "https://www.reuters.com/markets/..."
  },
  {
    "source": "The Verge",
    "title": "...",
    "publishedAt": "2026-10-10T17:05:11Z",
    "url": "https://www.theverge.com/..."
  }
]
```
</details>

---

### New York Times APIs

| | |
|:--|:--|
| **Docs** | https://developer.nytimes.com/apis |
| **Auth** | API key (query `api-key`) |
| **Register / token** | Create an account at https://developer.nytimes.com -> Apps -> New App -> enable 'Article Search API' (and Archive/Top Stories as needed). 10 req/min, 4,000 req/day. |
| **Env var** | `NYT_API_KEY` |
| **Script** | [`scripts/news_media/nyt.py`](scripts/news_media/nyt.py) |

**What you can get:**

- Article Search: headline, abstract, keywords, section, byline, pub_date (back to 1851)
- Archive API: every article metadata for a given month
- Top Stories, Most Popular, Books best-seller lists

**Capstone ideas:** 150+ years of headline data for historical discourse analysis; best-seller trends.

```python
from _common import get, key, show


def main():
    docs = get("https://api.nytimes.com/svc/search/v2/articlesearch.json",
               params={"q": "supreme court", "sort": "newest", "api-key": key("NYT_API_KEY")})["response"]["docs"]
    show([{"headline": d["headline"]["main"], "pub_date": d["pub_date"], "section_name": d.get("section_name"),
           "keywords": [k["value"] for k in d.get("keywords", [])][:3]} for d in docs])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {
    "headline": "Supreme Court Opens New Term With Major Cases on Agency Power",
    "pub_date": "2026-10-06T09:00:05+0000",
    "section_name": "U.S.",
    "keywords": ["Supreme Court (US)", "Law and Legislation", "Regulation and Deregulation of Industry"]
  }
]
```
</details>

---

### Wikimedia REST API (Pageviews)

| | |
|:--|:--|
| **Docs** | https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/ |
| **Auth** | None (descriptive User-Agent required) |
| **Register / token** | No registration. Keep under ~100 req/s and send a User-Agent with contact info. |
| **Script** | [`scripts/news_media/wikipedia_pageviews.py`](scripts/news_media/wikipedia_pageviews.py) |

**What you can get:**

- Daily/monthly pageviews per article, per project, per country
- Most-viewed articles per day
- Article summaries, revisions, edit counts

**Capstone ideas:** Public attention as a proxy signal: predicting elections, disease outbreaks, stock volatility.

```python
from _common import get, show


def main():
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
           "en.wikipedia/all-access/user/Artificial_intelligence/daily/20260901/20260907")
    show([{"date": i["timestamp"][:8], "views": i["views"]} for i in get(url)["items"]], max_items=7)
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "date": "20260901",
    "views": 12087
  },
  {
    "date": "20260902",
    "views": 13608
  },
  {
    "date": "20260903",
    "views": 13897
  },
  {
    "date": "20260904",
    "views": 12755
  },
  {
    "date": "20260905",
    "views": 9826
  },
  {
    "date": "20260906",
    "views": 10354
  },
  {
    "date": "20260907",
    "views": 13622
  }
]
```
</details>

---

## 💬 Social Media

### Bluesky (AT Protocol) Public AppView

| | |
|:--|:--|
| **Docs** | https://docs.bsky.app/docs/category/http-reference |
| **Auth** | None for public read endpoints (search needs an app password) |
| **Register / token** | Reading public profiles/feeds needs nothing. For search, create an account and an app password at Settings -> Privacy and security -> App passwords, then call `com.atproto.server.createSession`. The full firehose is open too (Jetstream: wss://jetstream2.us-east.bsky.network). |
| **Script** | [`scripts/social_media/bluesky.py`](scripts/social_media/bluesky.py) |

**What you can get:**

- Profiles: followers, following, post counts, bio
- Author feeds: post text, likes, reposts, replies
- Follower graphs
- Real-time firehose of every public post

**Capstone ideas:** The most open large social network for research now that X/Twitter API is paid; network & discourse analysis.

```python
from _common import get, show

BASE = "https://public.api.bsky.app/xrpc"


def main():
    prof = get(f"{BASE}/app.bsky.actor.getProfile", params={"actor": "bsky.app"})
    feed = get(f"{BASE}/app.bsky.feed.getAuthorFeed", params={"actor": prof["did"], "limit": 3})["feed"]
    show({
        "profile": {k: prof.get(k) for k in ("handle", "displayName", "followersCount", "postsCount")},
        "recent_posts": [{"text": f["post"]["record"].get("text"), "likes": f["post"].get("likeCount"),
                          "reposts": f["post"].get("repostCount"), "createdAt": f["post"]["record"].get("createdAt")}
                         for f in feed],
    })
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
{
  "profile": {
    "handle": "bsky.app",
    "displayName": "Bluesky",
    "followersCount": 35225915,
    "postsCount": 867
  },
  "recent_posts": [
    {
      "text": "Students made their first minicomics in my intro making course - mostly standard format but this student, a first time maker(!), did something so off the charts...",
      "likes": 3957,
      "reposts": 867,
      "createdAt": "2026-10-10T19:19:54.753Z"
    },
    {
      "text": " Australopithecus 3 million years ago:",
      "likes": 7773,
      "reposts": 1504,
      "createdAt": "2026-10-11T06:28:42.288Z"
    },
    {
      "text": "i got so many people calling this tweet fakeposting back when i made it but since this is bluesky and things have less reach on here im just going to post the p...",
      "likes": 19785,
      "reposts": 5040,
      "createdAt": "2026-10-10T19:49:31.289Z"
    }
  ]
}
```
</details>

---

### Mastodon

| | |
|:--|:--|
| **Docs** | https://docs.joinmastodon.org/methods/ |
| **Auth** | None for public timelines/trends (per-instance) |
| **Register / token** | No key for public endpoints. For more, register an app on any instance (Preferences -> Development). |
| **Script** | [`scripts/social_media/mastodon.py`](scripts/social_media/mastodon.py) |

**What you can get:**

- Trending hashtags with daily usage history
- Public / local timelines, hashtag timelines
- Account info, statuses (HTML content, boosts, favourites)

**Capstone ideas:** Decentralized-social-network studies, hashtag diffusion across instances.

```python
from _common import get, show


def main():
    tags = get("https://mastodon.social/api/v1/trends/tags", params={"limit": 5})
    show([{"tag": t["name"], "uses_today": t["history"][0]["uses"], "accounts_today": t["history"][0]["accounts"]}
          for t in tags])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "tag": "SilentSunday",
    "uses_today": "894",
    "accounts_today": "776"
  },
  {
    "tag": "FragrantPlaces",
    "uses_today": "272",
    "accounts_today": "86"
  },
  {
    "tag": "nationalcomingoutday",
    "uses_today": "36",
    "accounts_today": "24"
  },
  {
    "tag": "豊臣兄弟",
    "uses_today": "241",
    "accounts_today": "23"
  },
  {
    "tag": "glucksmann",
    "uses_today": "34",
    "accounts_today": "23"
  }
]
```
</details>

---

### Reddit Data API

| | |
|:--|:--|
| **Docs** | https://www.reddit.com/dev/api/ |
| **Auth** | OAuth2 (client id + secret, 'script' app) |
| **Register / token** | 1) Log in, go to https://www.reddit.com/prefs/apps -> 'create another app' -> type **script**, redirect uri `http://localhost:8080`. 2) The string under the app name is the client id; 'secret' is the secret. 3) Since 2024 you must also accept the Data API terms (https://support.reddithelp.com/hc/en-us/articles/14945211791892). Free tier: 100 QPM per client. Unauthenticated `.json` URLs now return 403 for most scripts. |
| **Env var** | `REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET` |
| **Script** | [`scripts/social_media/reddit.py`](scripts/social_media/reddit.py) |

**What you can get:**

- Posts per subreddit (hot/new/top), title, selftext, score, upvote ratio, num_comments
- Full comment trees
- Subreddit metadata, user public history
- Search across Reddit

**Capstone ideas:** Opinion mining, community detection, mental-health discourse, meme-stock sentiment (r/wallstreetbets).

```python
import requests

from _common import USER_AGENT, get, key, show


def main():
    # Application-only OAuth (client_credentials) is enough for reading public data.
    tok = requests.post("https://www.reddit.com/api/v1/access_token",
                        auth=(key("REDDIT_CLIENT_ID"), key("REDDIT_CLIENT_SECRET")),
                        data={"grant_type": "client_credentials"},
                        headers={"User-Agent": USER_AGENT}, timeout=30).json()["access_token"]
    posts = get("https://oauth.reddit.com/r/datascience/top", params={"t": "week", "limit": 5},
                headers={"Authorization": f"bearer {tok}"})["data"]["children"]
    show([{k: p["data"][k] for k in ("title", "score", "num_comments", "upvote_ratio", "created_utc")}
          for p in posts])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {
    "title": "What's one data-science skill that paid off the most in your career?",
    "score": 812,
    "num_comments": 241,
    "upvote_ratio": 0.97,
    "created_utc": 1760000000.0
  },
  {
    "title": "Weekly Entering & Transitioning Thread",
    "score": 15,
    "num_comments": 98,
    "upvote_ratio": 0.9,
    "created_utc": 1759800000.0
  }
]
```
</details>

---

### Stack Exchange API

| | |
|:--|:--|
| **Docs** | https://api.stackexchange.com/docs |
| **Auth** | None (optional key raises quota) |
| **Register / token** | Works without a key (300 req/day per IP). Register at https://stackapps.com/apps/oauth/register for a key -> 10,000 req/day. |
| **Script** | [`scripts/social_media/stackexchange.py`](scripts/social_media/stackexchange.py) |

**What you can get:**

- Questions, answers, comments across 170+ sites (Stack Overflow, Cross Validated, ...)
- Score, views, tags, accepted answer, timestamps
- Tag popularity, user reputation

**Capstone ideas:** Technology adoption trends, answer-quality prediction, developer-community analysis.

```python
from _common import get, show


def main():
    items = get("https://api.stackexchange.com/2.3/questions", params={
        "order": "desc", "sort": "votes", "tagged": "pandas", "site": "stackoverflow", "pagesize": 5,
    })["items"]
    show([{"title": q["title"], "score": q["score"], "views": q["view_count"], "tags": q["tags"]} for q in items])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "title": "How can I iterate over rows in a Pandas DataFrame?",
    "score": 4253,
    "views": 8112609,
    "tags": [
      "python",
      "pandas",
      "dataframe",
      "loops"
    ]
  },
  {
    "title": "How do I select rows from a DataFrame based on column values?",
    "score": 3635,
    "views": 6898626,
    "tags": [
      "python",
      "pandas",
      "dataframe",
      "indexing",
      "filter"
    ]
  },
  {
    "title": "How do I change the size of figures drawn with Matplotlib?",
    "score": 3266,
    "views": 6435668,
    "tags": [
      "python",
      "pandas",
      "matplotlib",
      "seaborn",
      "figsize"
    ]
  },
  {
    "title": "Renaming column names in Pandas",
    "score": 3074,
    "views": 7042039,
    "tags": [
      "python",
      "pandas",
      "replace",
      "dataframe",
      "rename"
    ]
  },
  {
    "title": "Delete a column from a Pandas DataFrame",
    "score": 2342,
    "views": 4677217,
    "tags": [
      "python",
      "pandas",
      "dataframe",
      "del"
    ]
  }
]
```
</details>

---

### YouTube Data API v3

| | |
|:--|:--|
| **Docs** | https://developers.google.com/youtube/v3/docs |
| **Auth** | API key (Google Cloud) |
| **Register / token** | 1) https://console.cloud.google.com -> create project. 2) APIs & Services -> Library -> enable 'YouTube Data API v3'. 3) Credentials -> Create credentials -> API key. Free quota: 10,000 units/day (a search costs 100, a videos.list costs 1). |
| **Env var** | `YOUTUBE_API_KEY` |
| **Script** | [`scripts/social_media/youtube.py`](scripts/social_media/youtube.py) |

**What you can get:**

- Video search; title, description, channel, publish date
- Statistics: views, likes, comment counts
- Comment threads (text, likes, replies)
- Channel subscribers & uploads

**Capstone ideas:** Comment sentiment, misinformation spread, creator-economy analytics.

```python
from _common import get, key, show


def main():
    k = key("YOUTUBE_API_KEY")
    ids = [i["id"]["videoId"] for i in get("https://www.googleapis.com/youtube/v3/search", params={
        "part": "snippet", "q": "neural network", "type": "video", "maxResults": 3, "key": k})["items"]]
    vids = get("https://www.googleapis.com/youtube/v3/videos",
               params={"part": "snippet,statistics", "id": ",".join(ids), "key": k})["items"]
    show([{"title": v["snippet"]["title"], "channel": v["snippet"]["channelTitle"],
           "views": v["statistics"].get("viewCount"), "likes": v["statistics"].get("likeCount"),
           "comments": v["statistics"].get("commentCount")} for v in vids])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {
    "title": "But what is a neural network? | Deep learning chapter 1",
    "channel": "3Blue1Brown",
    "views": "21534876",
    "likes": "498211",
    "comments": "14232"
  }
]
```
</details>

---

## 🌦️ Weather, Climate & Environment

### NOAA Climate Data Online (CDO)

| | |
|:--|:--|
| **Docs** | https://www.ncdc.noaa.gov/cdo-web/webservices/v2 |
| **Auth** | Token (header `token`) |
| **Register / token** | Request a token at https://www.ncdc.noaa.gov/cdo-web/token (enter email; token is emailed). 5 req/sec, 10,000 req/day. |
| **Env var** | `NOAA_TOKEN` |
| **Script** | [`scripts/weather_environment/noaa_cdo.py`](scripts/weather_environment/noaa_cdo.py) |

**What you can get:**

- Daily/monthly station observations (GHCND): TMAX, TMIN, PRCP, SNOW
- Station metadata, datasets back to the 1800s

**Capstone ideas:** Long-term local climate trends, heat-wave studies, weather/health linkage.

```python
from _common import get, key, show


def main():
    d = get("https://www.ncei.noaa.gov/cdo-web/api/v2/data", headers={"token": key("NOAA_TOKEN")}, params={
        "datasetid": "GHCND", "stationid": "GHCND:USW00013743",  # Reagan National Airport
        "datatypeid": "TMAX,TMIN", "startdate": "2025-07-01", "enddate": "2025-07-07", "units": "metric",
    })
    show([{k: r[k] for k in ("date", "datatype", "station", "value")} for r in d["results"]])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"date": "2025-07-01T00:00:00", "datatype": "TMAX", "station": "GHCND:USW00013743", "value": 33.9},
  {"date": "2025-07-01T00:00:00", "datatype": "TMIN", "station": "GHCND:USW00013743", "value": 24.4},
  {"date": "2025-07-02T00:00:00", "datatype": "TMAX", "station": "GHCND:USW00013743", "value": 35.0}
]
```
</details>

---

### US National Weather Service (api.weather.gov)

| | |
|:--|:--|
| **Docs** | https://www.weather.gov/documentation/services-web-api |
| **Auth** | None (User-Agent with contact required) |
| **Register / token** | No registration. Just send a User-Agent such as `(myapp, me@gwu.edu)`. |
| **Script** | [`scripts/weather_environment/nws.py`](scripts/weather_environment/nws.py) |

**What you can get:**

- Official NWS forecasts (12-hour periods & hourly) for US locations
- Active weather alerts (watches/warnings) by state or point
- Station observations

**Capstone ideas:** Severe-weather alert analysis, linking alerts to outages/traffic/311 calls.

```python
from _common import get, show


def main():
    point = get("https://api.weather.gov/points/38.8977,-77.0365")["properties"]
    periods = get(point["forecast"])["properties"]["periods"][:3]
    alerts = get("https://api.weather.gov/alerts/active", params={"area": "VA"})["features"]
    show({
        "office": point["gridId"],
        "forecast": [{k: p[k] for k in ("name", "temperature", "temperatureUnit", "shortForecast")} for p in periods],
        "active_alerts_VA": [a["properties"]["headline"] for a in alerts[:3]],
    })
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
{
  "office": "LWX",
  "forecast": [
    {
      "name": "Tonight",
      "temperature": 62,
      "temperatureUnit": "F",
      "shortForecast": "Showers And Thunderstorms"
    },
    {
      "name": "Columbus Day",
      "temperature": 73,
      "temperatureUnit": "F",
      "shortForecast": "Rain Showers Likely then Mostly Cloudy"
    },
    {
      "name": "Monday Night",
      "temperature": 58,
      "temperatureUnit": "F",
      "shortForecast": "Partly Cloudy"
    }
  ],
  "active_alerts_VA": [
    "Coastal Flood Advisory issued October 11 at 6:58PM EDT until October 11 at 11:00PM EDT by NWS Baltimore MD/Washington DC",
    "Flood Watch issued October 11 at 3:01PM EDT until October 12 at 2:00AM EDT by NWS Baltimore MD/Washington DC",
    "Beach Hazards Statement issued October 11 at 1:38PM EDT until October 11 at 8:00PM EDT by NWS Wakefield VA"
  ]
}
```
</details>

---

### Open-Meteo

| | |
|:--|:--|
| **Docs** | https://open-meteo.com/en/docs |
| **Auth** | None |
| **Register / token** | No registration. Free for non-commercial use (<10,000 calls/day). |
| **Script** | [`scripts/weather_environment/open_meteo.py`](scripts/weather_environment/open_meteo.py) |

**What you can get:**

- Hourly/daily forecasts up to 16 days for any lat/lon
- Historical weather back to 1940 (ERA5 reanalysis) via archive-api.open-meteo.com
- Air quality, marine, flood, climate-change (CMIP6) projections, geocoding

**Capstone ideas:** Weather as a covariate for sales/energy/mobility models; climate trend analysis.

```python
from _common import get, show


def main():
    d = get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": 38.90, "longitude": -77.04,  # Washington, DC
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "America/New_York", "forecast_days": 3,
    })
    show({"current": d["current"], "daily": d["daily"]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
{
  "current": {
    "time": "2026-10-11T19:00",
    "interval": 900,
    "temperature_2m": 17.8,
    "relative_humidity_2m": 97,
    "wind_speed_10m": 6.8
  },
  "daily": {
    "time": [
      "2026-10-11",
      "2026-10-12",
      "2026-10-13"
    ],
    "temperature_2m_max": [
      18.4,
      20.8,
      27.7
    ],
    "temperature_2m_min": [
      16.1,
      15.1,
      12.6
    ],
    "precipitation_sum": [
      53.8,
      0.0,
      0.0
    ]
  }
}
```
</details>

---

### OpenAQ v3 (Air Quality)

| | |
|:--|:--|
| **Docs** | https://docs.openaq.org/ |
| **Auth** | API key (header `X-API-Key`) |
| **Register / token** | Create a free account at https://explore.openaq.org/register -> Settings -> API Key. 60 req/min, 2,000 req/hour. |
| **Env var** | `OPENAQ_API_KEY` |
| **Script** | [`scripts/weather_environment/openaq.py`](scripts/weather_environment/openaq.py) |

**What you can get:**

- Ground-station air-quality measurements worldwide: PM2.5, PM10, O3, NO2, SO2, CO
- Location metadata (provider, sensors, coordinates)
- Hourly / daily aggregates

**Capstone ideas:** Environmental justice studies, pollution vs. health outcomes, wildfire smoke impact.

```python
from _common import get, key, show


def main():
    d = get("https://api.openaq.org/v3/locations", headers={"X-API-Key": key("OPENAQ_API_KEY")},
            params={"coordinates": "38.9,-77.04", "radius": 20000, "limit": 5})
    show([{"location": l["name"], "country": l["country"]["code"],
           "sensors": [f'{s["parameter"]["name"]} {s["parameter"]["units"]}' for s in l["sensors"]]}
          for l in d["results"]])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"location": "Washington DC - River Terrace", "country": "US", "sensors": ["pm25 µg/m³", "o3 ppm", "no2 ppm"]},
  {"location": "McMillan Reservoir", "country": "US", "sensors": ["pm25 µg/m³", "pm10 µg/m³"]}
]
```
</details>

---

### OpenWeatherMap

| | |
|:--|:--|
| **Docs** | https://openweathermap.org/api |
| **Auth** | API key (query `appid`) |
| **Register / token** | Sign up at https://home.openweathermap.org/users/sign_up -> 'API keys' tab. Free plan: 60 calls/min, 1M calls/month (current weather, 5-day forecast, air pollution). New keys take ~10-60 min to activate. |
| **Env var** | `OPENWEATHER_API_KEY` |
| **Script** | [`scripts/weather_environment/openweathermap.py`](scripts/weather_environment/openweathermap.py) |

**What you can get:**

- Current weather by city / lat-lon / zip
- 5-day / 3-hour forecast
- Air pollution (AQI, PM2.5, O3...)
- Geocoding

**Capstone ideas:** Quick weather features for any global city; air-quality dashboards.

```python
from _common import get, key, show


def main():
    d = get("https://api.openweathermap.org/data/2.5/weather",
            params={"q": "Washington,DC,US", "units": "metric", "appid": key("OPENWEATHER_API_KEY")})
    show({"city": d["name"], "temp_C": d["main"]["temp"], "humidity": d["main"]["humidity"],
          "description": d["weather"][0]["description"], "wind_mps": d["wind"]["speed"]})
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "city": "Washington",
  "temp_C": 18.4,
  "humidity": 62,
  "description": "scattered clouds",
  "wind_mps": 3.6
}
```
</details>

---

### USGS Earthquake Catalog

| | |
|:--|:--|
| **Docs** | https://earthquake.usgs.gov/fdsnws/event/1/ |
| **Auth** | None |
| **Register / token** | No registration. |
| **Script** | [`scripts/weather_environment/usgs_earthquakes.py`](scripts/weather_environment/usgs_earthquakes.py) |

**What you can get:**

- Global earthquakes with magnitude, depth, location, time, tsunami flag, felt reports
- GeoJSON output; real-time feeds and full historical catalog

**Capstone ideas:** Spatio-temporal clustering, aftershock forecasting, geo-visualization.

```python
from _common import get, show


def main():
    d = get("https://earthquake.usgs.gov/fdsnws/event/1/query", params={
        "format": "geojson", "minmagnitude": 5, "orderby": "time", "limit": 5})
    show([{"mag": f["properties"]["mag"], "place": f["properties"]["place"],
           "time_ms": f["properties"]["time"], "depth_km": f["geometry"]["coordinates"][2],
           "tsunami": f["properties"]["tsunami"]} for f in d["features"]])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "mag": 5.3,
    "place": "150 km W of Gorontalo, Indonesia",
    "time_ms": 1791754493822,
    "depth_km": 110.467,
    "tsunami": 0
  },
  {
    "mag": 5.2,
    "place": "101 km SSW of Severo-Kuril’sk, Russia",
    "time_ms": 1791738029277,
    "depth_km": 83.386,
    "tsunami": 0
  },
  {
    "mag": 6.2,
    "place": "79 km SSE of Panguna, Papua New Guinea",
    "time_ms": 1791727643188,
    "depth_km": 53.473,
    "tsunami": 0
  },
  {
    "mag": 5.2,
    "place": "18 km SSE of Camaná, Peru",
    "time_ms": 1791726179788,
    "depth_km": 44.92,
    "tsunami": 0
  },
  {
    "mag": 5.2,
    "place": "180 km S of Pelabuhanratu, Indonesia",
    "time_ms": 1791689417430,
    "depth_km": 10,
    "tsunami": 0
  }
]
```
</details>

---

## 💵 Finance & Economics

### Alpha Vantage

| | |
|:--|:--|
| **Docs** | https://www.alphavantage.co/documentation/ |
| **Auth** | API key (query `apikey`); `demo` works for documented examples (IBM) |
| **Register / token** | Claim a free key instantly at https://www.alphavantage.co/support/#api-key. Free: 25 req/day. |
| **Env var** | `ALPHAVANTAGE_API_KEY` |
| **Script** | [`scripts/finance_economics/alpha_vantage.py`](scripts/finance_economics/alpha_vantage.py) |

**What you can get:**

- Daily/intraday OHLCV for stocks, ETFs, FX, crypto
- Fundamentals: income statement, balance sheet, earnings
- 50+ technical indicators
- News & sentiment scores per ticker
- Economic indicators (CPI, GDP)

**Capstone ideas:** Stock forecasting, news-sentiment trading signals, portfolio analysis.

```python
from _common import get, show


def main():
    import os
    d = get("https://www.alphavantage.co/query", params={
        "function": "TIME_SERIES_DAILY", "symbol": "IBM", "apikey": os.getenv("ALPHAVANTAGE_API_KEY", "demo")})
    ts = d["Time Series (Daily)"]
    show({"symbol": d["Meta Data"]["2. Symbol"],
          "last_5_days": [{"date": day, **{k[3:]: v for k, v in ts[day].items()}} for day in list(ts)[:5]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
{
  "symbol": "IBM",
  "last_5_days": [
    {
      "date": "2026-10-09",
      "open": "225.8700",
      "high": "228.7600",
      "low": "224.0501",
      "close": "227.1300",
      "volume": "3352852"
    },
    {
      "date": "2026-10-08",
      "open": "220.1500",
      "high": "227.3050",
      "low": "217.7950",
      "close": "226.6100",
      "volume": "8929853"
    },
    {
      "date": "2026-10-07",
      "open": "221.9000",
      "high": "222.4100",
      "low": "218.7900",
      "close": "220.5100",
      "volume": "3895497"
    },
    {
      "date": "2026-10-06",
      "open": "222.7100",
      "high": "224.5100",
      "low": "221.0000",
      "close": "221.2900",
      "volume": "3938659"
    },
    {
      "date": "2026-10-05",
      "open": "222.3000",
      "high": "222.8800",
      "low": "220.3000",
      "close": "221.5800",
      "volume": "6318079"
    }
  ]
}
```
</details>

---

### US Bureau of Labor Statistics (BLS)

| | |
|:--|:--|
| **Docs** | https://www.bls.gov/developers/ |
| **Auth** | None for v1; free key for v2 |
| **Register / token** | v1 works keyless (25 queries/day, 10 years). Register at https://data.bls.gov/registrationEngine/ for v2: 500 queries/day, 20 years, 50 series per request, net/percent changes. |
| **Env var** | `BLS_API_KEY (optional)` |
| **Script** | [`scripts/finance_economics/bls.py`](scripts/finance_economics/bls.py) |

**What you can get:**

- CPI (inflation), unemployment, employment by industry (CES), wages, productivity
- Local area unemployment (LAUS) for states/counties/metros
- Producer prices, JOLTS

**Capstone ideas:** Labor-market analysis, inflation forecasting, regional employment shocks.

```python
from _common import get, show


def main():
    d = get("https://api.bls.gov/publicAPI/v1/timeseries/data/CUUR0000SA0")  # CPI-U all items
    s = d["Results"]["series"][0]
    show({"series": s["seriesID"] + " (CPI-U, all items, 1982-84=100)",
          "data": [{"period": f'{x["year"]}-{x["periodName"]}', "value": x["value"]} for x in s["data"][:6]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
{
  "series": "CUUR0000SA0 (CPI-U, all items, 1982-84=100)",
  "data": [
    {
      "period": "2026-August",
      "value": "334.980"
    },
    {
      "period": "2026-July",
      "value": "333.918"
    },
    {
      "period": "2026-June",
      "value": "333.952"
    },
    {
      "period": "2026-May",
      "value": "335.123"
    },
    {
      "period": "2026-April",
      "value": "333.020"
    },
    "... (1 more)"
  ]
}
```
</details>

---

### CoinGecko

| | |
|:--|:--|
| **Docs** | https://docs.coingecko.com/v3.0.1/reference/introduction |
| **Auth** | None for public endpoints (free 'Demo' key recommended) |
| **Register / token** | Works keyless at low volume. For a stable 30 calls/min, 10k/month register a free Demo key at https://www.coingecko.com/en/developers/dashboard and pass header `x-cg-demo-api-key`. |
| **Script** | [`scripts/finance_economics/coingecko.py`](scripts/finance_economics/coingecko.py) |

**What you can get:**

- Prices, market cap, volume for 15k+ crypto assets
- Historical OHLC & market charts
- Exchanges, trending coins, categories

**Capstone ideas:** Crypto volatility modeling, cross-asset correlation, social-hype vs price.

```python
from _common import get, show


def main():
    d = get("https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 5})
    show([{k: c[k] for k in ("name", "current_price", "market_cap", "price_change_percentage_24h")} for c in d])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "name": "Bitcoin",
    "current_price": 83647,
    "market_cap": 1681037143329,
    "price_change_percentage_24h": 0.77903
  },
  {
    "name": "Ethereum",
    "current_price": 2534.12,
    "market_cap": 309481162747,
    "price_change_percentage_24h": 1.06014
  },
  {
    "name": "Tether",
    "current_price": 0.999133,
    "market_cap": 184090665514,
    "price_change_percentage_24h": -0.00873
  },
  {
    "name": "BNB",
    "current_price": 753.42,
    "market_cap": 100323096959,
    "price_change_percentage_24h": 0.3609
  },
  {
    "name": "XRP",
    "current_price": 1.41,
    "market_cap": 88842983516,
    "price_change_percentage_24h": 0.288
  }
]
```
</details>

---

### Frankfurter (ECB exchange rates)

| | |
|:--|:--|
| **Docs** | https://frankfurter.dev/ |
| **Auth** | None |
| **Register / token** | No registration, no limits. |
| **Script** | [`scripts/finance_economics/frankfurter.py`](scripts/finance_economics/frankfurter.py) |

**What you can get:**

- Daily reference FX rates published by the European Central Bank (~30 currencies)
- Historical rates back to 1999 and time-series ranges

**Capstone ideas:** Exchange-rate forecasting, currency-crisis analysis, converting multi-country price data.

```python
from _common import get, show


def main():
    latest = get("https://api.frankfurter.dev/v1/latest", params={"base": "USD", "symbols": "EUR,GBP,JPY,CNY,INR"})
    series = get("https://api.frankfurter.dev/v1/2026-09-01..2026-09-05", params={"base": "USD", "symbols": "EUR"})
    show({"latest": latest, "usd_eur_series": series["rates"]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
{
  "latest": {
    "amount": 1.0,
    "base": "USD",
    "date": "2026-10-09",
    "rates": {
      "CNY": 6.6921,
      "EUR": 0.89238,
      "GBP": 0.75641,
      "INR": 96.73,
      "JPY": 158.25
    }
  },
  "usd_eur_series": {
    "2026-09-01": {
      "EUR": 0.86281
    },
    "2026-09-02": {
      "EUR": 0.86371
    },
    "2026-09-03": {
      "EUR": 0.86096
    },
    "2026-09-04": {
      "EUR": 0.86044
    }
  }
}
```
</details>

---

### FRED (Federal Reserve Economic Data)

| | |
|:--|:--|
| **Docs** | https://fred.stlouisfed.org/docs/api/fred/ |
| **Auth** | API key (query `api_key`) |
| **Register / token** | Create a free account at https://fredaccount.stlouisfed.org -> 'API Keys' -> Request API Key (describe use, e.g. 'GWU capstone'). 120 req/min. |
| **Env var** | `FRED_API_KEY` |
| **Script** | [`scripts/finance_economics/fred.py`](scripts/finance_economics/fred.py) |

**What you can get:**

- 800,000+ US & international time series: GDP, CPI, unemployment, rates, housing, M2...
- Series search, release calendars, vintage (real-time) data via ALFRED

**Capstone ideas:** Macro forecasting, recession prediction, nowcasting, economic regime detection.

```python
from _common import get, key, show


def main():
    obs = get("https://api.stlouisfed.org/fred/series/observations", params={
        "series_id": "UNRATE", "api_key": key("FRED_API_KEY"), "file_type": "json",
        "sort_order": "desc", "limit": 5})["observations"]
    show({"series": "UNRATE (Unemployment Rate, %)",
          "observations": [{"date": o["date"], "value": o["value"]} for o in reversed(obs)]})
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "series": "UNRATE (Unemployment Rate, %)",
  "observations": [
    {"date": "2026-04-01", "value": "4.2"},
    {"date": "2026-05-01", "value": "4.2"},
    {"date": "2026-06-01", "value": "4.1"},
    {"date": "2026-07-01", "value": "4.2"},
    {"date": "2026-08-01", "value": "4.3"}
  ]
}
```
</details>

---

### SEC EDGAR (data.sec.gov)

| | |
|:--|:--|
| **Docs** | https://www.sec.gov/search-filings/edgar-application-programming-interfaces |
| **Auth** | None (User-Agent with email required) |
| **Register / token** | No registration. Must send `User-Agent: Name email@domain`; max 10 req/sec. |
| **Script** | [`scripts/finance_economics/sec_edgar.py`](scripts/finance_economics/sec_edgar.py) |

**What you can get:**

- Every company filing (10-K, 10-Q, 8-K, S-1...) with dates & links
- XBRL 'company facts': every reported financial number (revenue, net income, EPS...) over time
- Full-text search (efts.sec.gov) across filings since 2001

**Capstone ideas:** NLP on 10-K risk factors, fraud detection, fundamentals-based models, ESG text mining.

```python
from _common import get, show

HDR = {"User-Agent": "GWU Capstone research your_email@gwu.edu"}


def main():
    sub = get("https://data.sec.gov/submissions/CIK0000320193.json", headers=HDR)  # Apple
    recent = sub["filings"]["recent"]
    facts = get("https://data.sec.gov/api/xbrl/companyconcept/CIK0000320193/us-gaap/NetIncomeLoss.json",
                headers=HDR)["units"]["USD"]
    annual = [f for f in facts if f.get("form") == "10-K" and f.get("fp") == "FY"][-3:]
    show({
        "company": sub["name"], "sic": sub["sicDescription"],
        "recent_filings": [{"form": recent["form"][i], "date": recent["filingDate"][i]} for i in range(5)],
        "net_income_10K": [{"fy": f["fy"], "end": f["end"], "USD": f["val"]} for f in annual],
    })
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
{
  "company": "Apple Inc.",
  "sic": "Electronic Computers",
  "recent_filings": [
    {
      "form": "4",
      "date": "2026-10-08"
    },
    {
      "form": "4",
      "date": "2026-10-05"
    },
    {
      "form": "4",
      "date": "2026-10-05"
    },
    {
      "form": "4",
      "date": "2026-10-05"
    },
    {
      "form": "4",
      "date": "2026-10-05"
    }
  ],
  "net_income_10K": [
    {
      "fy": 2024,
      "end": "2024-09-28",
      "USD": 93736000000
    },
    {
      "fy": 2025,
      "end": "2024-09-28",
      "USD": 93736000000
    },
    {
      "fy": 2025,
      "end": "2025-09-27",
      "USD": 112010000000
    }
  ]
}
```
</details>

---

### US Treasury Fiscal Data

| | |
|:--|:--|
| **Docs** | https://fiscaldata.treasury.gov/api-documentation/ |
| **Auth** | None |
| **Register / token** | No registration. |
| **Script** | [`scripts/finance_economics/treasury_fiscal.py`](scripts/finance_economics/treasury_fiscal.py) |

**What you can get:**

- Debt to the Penny (daily national debt)
- Average interest rates on Treasury securities
- Monthly Treasury Statement: receipts, outlays, deficit
- Exchange rates used by Treasury

**Capstone ideas:** Fiscal-policy analysis, debt dynamics, government spending composition.

```python
from _common import get, show


def main():
    d = get("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/debt_to_penny",
            params={"sort": "-record_date", "page[size]": 5,
                    "fields": "record_date,tot_pub_debt_out_amt,debt_held_public_amt,intragov_hold_amt"})
    show(d["data"])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "record_date": "2026-10-08",
    "tot_pub_debt_out_amt": "40305316210829.72",
    "debt_held_public_amt": "32454086117711.32",
    "intragov_hold_amt": "7851230093118.40"
  },
  {
    "record_date": "2026-10-07",
    "tot_pub_debt_out_amt": "40284036147367.19",
    "debt_held_public_amt": "32439260138550.17",
    "intragov_hold_amt": "7844776008817.02"
  },
  {
    "record_date": "2026-10-06",
    "tot_pub_debt_out_amt": "40273024579219.17",
    "debt_held_public_amt": "32440264284905.73",
    "intragov_hold_amt": "7832760294313.44"
  },
  {
    "record_date": "2026-10-05",
    "tot_pub_debt_out_amt": "40249104431078.48",
    "debt_held_public_amt": "32426124989617.49",
    "intragov_hold_amt": "7822979441460.99"
  },
  {
    "record_date": "2026-10-02",
    "tot_pub_debt_out_amt": "40242446619209.33",
    "debt_held_public_amt": "32428913787327.44",
    "intragov_hold_amt": "7813532831881.89"
  }
]
```
</details>

---

### World Bank Indicators API

| | |
|:--|:--|
| **Docs** | https://datahelpdesk.worldbank.org/knowledgebase/articles/889392 |
| **Auth** | None |
| **Register / token** | No registration. |
| **Script** | [`scripts/finance_economics/world_bank.py`](scripts/finance_economics/world_bank.py) |

**What you can get:**

- 16,000+ development indicators for 200+ countries since 1960
- GDP, poverty, education, health, CO2, internet use, governance (WGI)...

**Capstone ideas:** Cross-country development economics, panel regressions, global inequality studies.

```python
from _common import get, show


def main():
    meta, rows = get("https://api.worldbank.org/v2/country/USA;CHN;IND;BRA;NGA/indicator/NY.GDP.PCAP.CD",
                     params={"format": "json", "date": "2023", "per_page": 50})
    show([{"country": r["country"]["value"], "year": r["date"], "gdp_per_capita_usd": round(r["value"] or 0)}
          for r in rows])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "country": "Brazil",
    "year": "2023",
    "gdp_per_capita_usd": 10378
  },
  {
    "country": "China",
    "year": "2023",
    "gdp_per_capita_usd": 12951
  },
  {
    "country": "India",
    "year": "2023",
    "gdp_per_capita_usd": 2434
  },
  {
    "country": "Nigeria",
    "year": "2023",
    "gdp_per_capita_usd": 2139
  },
  {
    "country": "United States",
    "year": "2023",
    "gdp_per_capita_usd": 82587
  }
]
```
</details>

---

## 👥 Census & Demographics

### IPUMS API (microdata)

| | |
|:--|:--|
| **Docs** | https://developer.ipums.org/docs/v2/ |
| **Auth** | API key (header `Authorization`) |
| **Register / token** | Register for the collection you need (e.g. IPUMS USA, CPS, NHGIS) at https://uma.pop.umn.edu/, then create a key at https://account.ipums.org/api_keys. Python client: `pip install ipumspy`. |
| **Env var** | `IPUMS_API_KEY` |
| **Script** | [`scripts/census_demographics/ipums.py`](scripts/census_demographics/ipums.py) |

**What you can get:**

- Harmonized person-level census & survey microdata (US 1850-today, CPS, international censuses)
- NHGIS: aggregate census tables + GIS boundary files for every census since 1790

**Capstone ideas:** Intergenerational mobility, historical migration, labor economics with individual records.

```python
from _common import get, key, show


def main():
    d = get("https://api.ipums.org/metadata/samples", params={"collection": "usa", "version": 2, "pageSize": 5},
            headers={"Authorization": key("IPUMS_API_KEY")})
    show(d["data"])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"name": "us2022a", "description": "2022 ACS"},
  {"name": "us2021a", "description": "2021 ACS"},
  {"name": "us2020a", "description": "2020 ACS"}
]
```
</details>

---

### UN Data Portal — World Population Prospects

| | |
|:--|:--|
| **Docs** | https://population.un.org/dataportalapi/index.html |
| **Auth** | Bearer token (free) |
| **Register / token** | Email population@un.org (subject 'Data Portal API token request') or follow the link on the docs page; a bearer token is returned. Pass as `Authorization: Bearer <token>`. |
| **Env var** | `UN_POP_TOKEN` |
| **Script** | [`scripts/census_demographics/un_population.py`](scripts/census_demographics/un_population.py) |

**What you can get:**

- Population by age/sex for every country 1950-2100 (projections)
- Fertility, mortality, life expectancy, migration, dependency ratios

**Capstone ideas:** Aging & dependency forecasting, demographic dividend studies.

```python
from _common import get, key, show


def main():
    # Indicator 49 = total population; location 566 = Nigeria
    d = get("https://population.un.org/dataportalapi/api/v1/data/indicators/49/locations/566/start/2025/end/2050",
            headers={"Authorization": f"Bearer {key('UN_POP_TOKEN')}"})
    show([{"location": r["location"], "indicator": r["indicator"], "year": r["timeLabel"], "value": r["value"]}
          for r in d["data"] if r["sex"] == "Both sexes" and r["timeLabel"] in ("2025", "2050")])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"location": "Nigeria", "indicator": "Total population by sex (thousands)", "year": 2025, "value": 237527.8},
  {"location": "Nigeria", "indicator": "Total population by sex (thousands)", "year": 2050, "value": 359427.6}
]
```
</details>

---

### US Census Bureau Data API

| | |
|:--|:--|
| **Docs** | https://www.census.gov/data/developers/guidance/api-user-guide.html |
| **Auth** | API key (query `key`) — now required |
| **Register / token** | Request a key at https://api.census.gov/data/key_signup.html (emailed; click the activation link). Free; 500 queries/day per IP without key is no longer honored for most datasets. Variable lists: https://api.census.gov/data/2023/acs/acs5/variables.html |
| **Env var** | `CENSUS_API_KEY` |
| **Script** | [`scripts/census_demographics/us_census.py`](scripts/census_demographics/us_census.py) |

**What you can get:**

- American Community Survey (ACS 1-yr/5-yr): income, education, race, housing, commuting, language...
- Decennial Census counts
- Geographies: nation, state, county, tract, block group, ZIP (ZCTA)
- Population estimates, County Business Patterns, international trade

**Capstone ideas:** Demographic covariates for any US spatial study, gentrification, segregation indices, equity analysis.

```python
from _common import get, key, show


def main():
    rows = get("https://api.census.gov/data/2023/acs/acs5", params={
        "get": "NAME,B01003_001E,B19013_001E",  # total population, median household income
        "for": "state:11,24,51", "key": key("CENSUS_API_KEY")})
    header, *data = rows
    show([{"NAME": r[0], "population": r[1], "median_household_income": r[2]} for r in data])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"NAME": "District of Columbia", "population": "672079", "median_household_income": "106287"},
  {"NAME": "Maryland", "population": "6161707", "median_household_income": "98461"},
  {"NAME": "Virginia", "population": "8624511", "median_household_income": "87249"}
]
```
</details>

---

## 🏛️ Political Science & Government

### ACLED (Armed Conflict Location & Event Data)

| | |
|:--|:--|
| **Docs** | https://acleddata.com/acled-api-documentation |
| **Auth** | OAuth token (account email + password) |
| **Register / token** | Register at https://acleddata.com/register/ with your university email (free academic access). Then POST your email/password to https://acleddata.com/oauth/token to get a 24-hour bearer token. |
| **Env var** | `ACLED_EMAIL, ACLED_PASSWORD` |
| **Script** | [`scripts/politics_government/acled.py`](scripts/politics_government/acled.py) |

**What you can get:**

- Geocoded political violence & protest events worldwide (battles, riots, protests, violence vs civilians)
- Date, actors, location (lat/lon, admin areas), fatalities, notes text

**Capstone ideas:** Conflict forecasting, protest diffusion, election violence, spatial econometrics.

```python
import requests

from _common import get, key, show


def main():
    tok = requests.post("https://acleddata.com/oauth/token", data={
        "username": key("ACLED_EMAIL"), "password": key("ACLED_PASSWORD"),
        "grant_type": "password", "client_id": "acled"}, timeout=30).json()["access_token"]
    d = get("https://acleddata.com/api/acled/read", headers={"Authorization": f"Bearer {tok}"},
            params={"country": "Kenya", "limit": 5, "_format": "json"})
    show([{k: e[k] for k in ("event_date", "event_type", "sub_event_type", "country", "location", "fatalities", "notes")}
          for e in d["data"]])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"event_date": "2026-09-28", "event_type": "Protests", "sub_event_type": "Peaceful protest",
   "country": "Kenya", "location": "Nairobi", "fatalities": 0,
   "notes": "On 28 September 2026, students demonstrated outside parliament against tuition increases..."}
]
```
</details>

---

### Congress.gov API

| | |
|:--|:--|
| **Docs** | https://github.com/LibraryOfCongress/api.congress.gov |
| **Auth** | api.data.gov key (`DEMO_KEY` works at low volume) |
| **Register / token** | Sign up at https://api.congress.gov/sign-up/ (an api.data.gov key — the same key also works for OpenFEC, Regulations.gov, NREL, NASA). 5,000 req/hour; DEMO_KEY: 30/hour. |
| **Env var** | `DATA_GOV_API_KEY` |
| **Script** | [`scripts/politics_government/congress_gov.py`](scripts/politics_government/congress_gov.py) |

**What you can get:**

- Bills & resolutions: title, sponsor, cosponsors, actions, status, subjects, text versions
- Members of Congress (party, state, terms)
- Roll-call votes (House), committees, hearings, nominations, treaties

**Capstone ideas:** Legislative effectiveness, bipartisanship networks from cosponsorship, policy-agenda topic modeling.

```python
import os

from _common import get, show


def main():
    k = os.getenv("DATA_GOV_API_KEY", "DEMO_KEY")
    bills = get("https://api.congress.gov/v3/bill/119", params={"api_key": k, "limit": 5, "format": "json",
                                                              "sort": "updateDate+desc"})["bills"]
    show([{"bill": f'{b["type"]} {b["number"]}', "title": b["title"],
           "latestAction": b["latestAction"]["text"], "date": b["latestAction"]["actionDate"]} for b in bills])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "bill": "S 3018",
    "title": "A bill to permit visiting dignitaries and service members from Taiwan to display the flag of the Republic of China.",
    "latestAction": "Placed on Senate Legislative Calendar under General Orders. Calendar No. 499.",
    "date": "2026-07-27"
  },
  {
    "bill": "HR 5744",
    "title": "Targeting Online Sales of Fentanyl Act",
    "latestAction": "Referred to the Committee on Energy and Commerce, and in addition to the Committee on the Judiciary, for a period to be subsequently determined by the Speaker, ...",
    "date": "2025-10-10"
  },
  {
    "bill": "HCONRES 29",
    "title": "Expressing support for the withdrawal of the United States from the Paris Agreement.",
    "latestAction": "Referred to the House Committee on Foreign Affairs.",
    "date": "2025-04-21"
  },
  {
    "bill": "S 2943",
    "title": "ACE Veterans Act",
    "latestAction": "Read twice and referred to the Committee on Veterans' Affairs.",
    "date": "2025-09-30"
  },
  {
    "bill": "SRES 542",
    "title": "A resolution commemorating the 50th anniversary of Southeast Asian refugee resettlement and the many contributions and sacrifices of Southeast Asian Americans t...",
    "latestAction": "Referred to the Committee on the Judiciary. (text: CR S8676)",
    "date": "2025-12-11"
  }
]
```
</details>

---

### Federal Register API

| | |
|:--|:--|
| **Docs** | https://www.federalregister.gov/developers/documentation/api/v1 |
| **Auth** | None |
| **Register / token** | No registration. |
| **Script** | [`scripts/politics_government/federal_register.py`](scripts/politics_government/federal_register.py) |

**What you can get:**

- Every rule, proposed rule, notice, and presidential document (executive orders) since 1994
- Agency, type, abstract, publication date, full-text/PDF links, comment deadlines

**Capstone ideas:** Regulatory-activity indices by administration, executive-order text analysis.

```python
from _common import get, show


def main():
    d = get("https://www.federalregister.gov/api/v1/documents.json", params={
        "conditions[presidential_document_type]": "executive_order", "order": "newest", "per_page": 5,
        "fields[]": ["title", "executive_order_number", "signing_date", "publication_date"]})
    show({"count": d["count"], "results": d["results"]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
{
  "count": 1571,
  "results": [
    {
      "title": "Emergency Tax Relief on Diesel Fuel",
      "executive_order_number": "14435",
      "signing_date": "2026-10-05",
      "publication_date": "2026-10-09"
    },
    {
      "title": "Inaugurating the Era of Super Intelligence",
      "executive_order_number": "14434",
      "signing_date": "2026-09-29",
      "publication_date": "2026-10-02"
    },
    {
      "title": "Eliminating Disease-Carrying Pests and Restoring Enjoyment of the Great Outdoors",
      "executive_order_number": "14433",
      "signing_date": "2026-09-29",
      "publication_date": "2026-10-02"
    },
    {
      "title": "Streamlining Access to Government Services Through America.gov",
      "executive_order_number": "14432",
      "signing_date": "2026-09-29",
      "publication_date": "2026-10-02"
    },
    {
      "title": "Enhancing Program Integrity and Interagency Coordination in the Administration of the H-1B Nonimmigrant Visa Program",
      "executive_order_number": "14431",
      "signing_date": "2026-09-18",
      "publication_date": "2026-09-23"
    }
  ]
}
```
</details>

---

### Open States / Plural (US state legislatures)

| | |
|:--|:--|
| **Docs** | https://docs.openstates.org/api-v3/ |
| **Auth** | API key (header `X-API-KEY`) |
| **Register / token** | Register at https://open.pluralpolicy.com/accounts/profile/ -> request API key. Free tier: 500 req/day. |
| **Env var** | `OPENSTATES_API_KEY` |
| **Script** | [`scripts/politics_government/open_states.py`](scripts/politics_government/open_states.py) |

**What you can get:**

- Bills, votes, sponsors for all 50 state legislatures + DC/PR
- State legislators (party, district, contact)
- Committees, events

**Capstone ideas:** Policy diffusion across states (e.g., model legislation), state-level polarization.

```python
from _common import get, key, show


def main():
    d = get("https://v3.openstates.org/bills", headers={"X-API-KEY": key("OPENSTATES_API_KEY")},
            params={"jurisdiction": "Virginia", "q": "artificial intelligence", "sort": "updated_desc", "per_page": 5})
    show([{"identifier": b["identifier"], "title": b["title"], "jurisdiction": b["jurisdiction"]["name"],
           "latest_action": b["latest_action_description"], "date": b["latest_action_date"]} for b in d["results"]])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"identifier": "HB 1234", "title": "Relating to artificial intelligence systems used in employment decisions",
   "jurisdiction": "Virginia", "latest_action": "Referred to Committee on Labor and Commerce", "date": "2026-01-14"}
]
```
</details>

---

### OpenFEC (Federal Election Commission)

| | |
|:--|:--|
| **Docs** | https://api.open.fec.gov/developers/ |
| **Auth** | api.data.gov key (`DEMO_KEY` works at low volume) |
| **Register / token** | Use the same api.data.gov key (https://api.data.gov/signup/). 1,000 req/hour. |
| **Env var** | `DATA_GOV_API_KEY` |
| **Script** | [`scripts/politics_government/openfec.py`](scripts/politics_government/openfec.py) |

**What you can get:**

- Candidates & committees, total receipts / disbursements / cash on hand
- Itemized individual contributions (Schedule A): donor name, employer, occupation, ZIP, amount
- Independent expenditures, PAC spending, filings

**Capstone ideas:** Campaign-finance networks, donor geography, money vs. electoral outcomes.

```python
import os

from _common import get, show


def main():
    d = get("https://api.open.fec.gov/v1/candidates/totals/", params={
        "api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY"), "election_year": 2024, "office": "P",
        "sort": "-receipts", "per_page": 5, "is_active_candidate": True})
    show([{"name": c["name"], "party": c["party"], "receipts": c["receipts"],
           "disbursements": c["disbursements"]} for c in d["results"]])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "name": "BIDEN, JOSEPH R JR",
    "party": "DEM",
    "receipts": 1187297098.58,
    "disbursements": 1185795517.32
  },
  {
    "name": "HARRIS, KAMALA",
    "party": "DEM",
    "receipts": 1175903792.49,
    "disbursements": 1176074620.47
  },
  {
    "name": "KENNEDY, ROBERT, F. JR., SHANAHAN, NICOLE",
    "party": "IND",
    "receipts": 67278247.27,
    "disbursements": 66539069.36
  },
  {
    "name": "RAMASWAMY, VIVEK",
    "party": "REP",
    "receipts": 66197196.43,
    "disbursements": 66197196.43
  },
  {
    "name": "HALEY, NIKKI",
    "party": "REP",
    "receipts": 58378347.5,
    "disbursements": 56690680.61
  }
]
```
</details>

---

### Regulations.gov API v4

| | |
|:--|:--|
| **Docs** | https://open.gsa.gov/api/regulationsgov/ |
| **Auth** | api.data.gov key (`DEMO_KEY` works at low volume) |
| **Register / token** | Use an api.data.gov key (https://api.data.gov/signup/). 1,000 req/hour. |
| **Env var** | `DATA_GOV_API_KEY` |
| **Script** | [`scripts/politics_government/regulations_gov.py`](scripts/politics_government/regulations_gov.py) |

**What you can get:**

- Federal rulemaking dockets & documents
- Public comments submitted on proposed rules (text + attachments)
- Agency, docket ID, comment period dates

**Capstone ideas:** Public-comment analysis, detecting mass/astroturf comment campaigns, interest-group influence.

```python
import os

from _common import get, show


def main():
    d = get("https://api.regulations.gov/v4/documents", params={
        "api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY"), "filter[documentType]": "Proposed Rule",
        "sort": "-postedDate", "page[size]": 5})
    show([{k: x["attributes"][k] for k in ("title", "agencyId", "postedDate", "commentEndDate")} | {"id": x["id"]}
          for x in d["data"]])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "title": "Hearings, Meetings, Proceedings, etc.: Application of the Personal Responsibility and Work Opportunity Reconciliation Act to the Refunded Portion of Certain Fed...",
    "agencyId": "IRS",
    "postedDate": "2026-10-09T04:00:00Z",
    "commentEndDate": null,
    "id": "IRS-2026-1057-14754"
  },
  {
    "title": "Extensions of Credit to Insiders",
    "agencyId": "FDIC",
    "postedDate": "2026-10-09T04:00:00Z",
    "commentEndDate": "2026-11-05T04:59:59Z",
    "id": "FDIC-2026-1981-0001"
  },
  {
    "title": "Requirements for Additional Traceability Records for Certain Foods: Enforcement Policy for Certain Retail Food Establishments and Restaurants; Draft Guidance fo...",
    "agencyId": "FDA",
    "postedDate": "2026-10-09T04:00:00Z",
    "commentEndDate": "2026-11-24T04:59:59Z",
    "id": "FDA-2025-D-6952-0001"
  },
  {
    "title": "Marketing Order: Almonds Grown in California",
    "agencyId": "AMS",
    "postedDate": "2026-10-08T04:00:00Z",
    "commentEndDate": null,
    "id": "AMS_FRDOC_0001-2764"
  },
  {
    "title": "Air Quality State Implementation Plans; Approvals and Promulgations: Arizona; Maricopa County Air Quality Department; New Source Review; Emission Reduction Cred...",
    "agencyId": "EPA",
    "postedDate": "2026-10-08T04:00:00Z",
    "commentEndDate": "2026-11-10T04:59:59Z",
    "id": "EPA-R09-OAR-2026-4654-0001"
  }
]
```
</details>

---

### UK Parliament APIs

| | |
|:--|:--|
| **Docs** | https://developer.parliament.uk/ |
| **Auth** | None |
| **Register / token** | No registration. |
| **Script** | [`scripts/politics_government/uk_parliament.py`](scripts/politics_government/uk_parliament.py) |

**What you can get:**

- Members (MPs/Lords), party, constituency
- Commons & Lords divisions (votes) with every member's vote
- Written questions & answers, bills, committees

**Capstone ideas:** Comparative legislative behavior, party-cohesion / rebellion analysis.

```python
from _common import get, show


def main():
    divs = get("https://commonsvotes-api.parliament.uk/data/divisions.json/search", params={"queryParameters.take": 5})
    show([{"title": d["Title"], "date": d["Date"][:10], "ayes": d["AyeCount"], "noes": d["NoCount"]} for d in divs])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "title": "Terminally Ill Adults (End of Life) Bill: Second Reading",
    "date": "2026-09-11",
    "ayes": 270,
    "noes": 286
  },
  {
    "title": "Closure motion",
    "date": "2026-09-11",
    "ayes": 293,
    "noes": 218
  },
  {
    "title": "Social Housing Bill [Lords]: Reasoned Amendment to Second Reading",
    "date": "2026-09-10",
    "ayes": 77,
    "noes": 292
  },
  {
    "title": "Draft Plant Health, Seeds, Seed Potatoes and Plant Propagating Material  (Amendment) (Northern Ireland) Regulations 2026",
    "date": "2026-09-09",
    "ayes": 371,
    "noes": 100
  },
  {
    "title": "Health Bill: Report Stage: Amendment 1",
    "date": "2026-09-08",
    "ayes": 162,
    "noes": 297
  }
]
```
</details>

---

### USAspending.gov

| | |
|:--|:--|
| **Docs** | https://api.usaspending.gov/docs/endpoints |
| **Auth** | None |
| **Register / token** | No registration. |
| **Script** | [`scripts/politics_government/usaspending.py`](scripts/politics_government/usaspending.py) |

**What you can get:**

- Every federal contract, grant, loan award since FY2001: recipient, amount, agency, location, NAICS
- Spending by agency, state, congressional district, budget function

**Capstone ideas:** Pork-barrel politics, federal spending vs. district partisanship, contractor concentration.

```python
import requests

from _common import USER_AGENT, show


def main():
    r = requests.post("https://api.usaspending.gov/api/v2/search/spending_by_category/awarding_agency/", json={
        "filters": {"time_period": [{"start_date": "2024-10-01", "end_date": "2025-09-30"}]}, "limit": 5,
    }, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    show([{"agency": x["name"], "obligated_usd": round(x["amount"])} for x in r.json()["results"]])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "agency": "Department of Health and Human Services",
    "obligated_usd": 2024158651519
  },
  {
    "agency": "Social Security Administration",
    "obligated_usd": 1634789118457
  },
  {
    "agency": "Department of Defense",
    "obligated_usd": 501689539224
  },
  {
    "agency": "Department of Veterans Affairs",
    "obligated_usd": 287885676666
  },
  {
    "agency": "Department of Agriculture",
    "obligated_usd": 186422112686
  }
]
```
</details>

---

## 🩺 Health & Social Science

### CDC Open Data (Socrata SODA)

| | |
|:--|:--|
| **Docs** | https://dev.socrata.com/foundry/data.cdc.gov/swc5-untb |
| **Auth** | None (optional app token raises throttle) |
| **Register / token** | Works keyless. For heavy use create an app token at https://data.cdc.gov/profile/edit/developer_settings. The same SODA query language works on data.cityofchicago.org, data.cityofnewyork.us, opendata.dc.gov, etc. |
| **Env var** | `SOCRATA_APP_TOKEN (optional)` |
| **Script** | [`scripts/health/cdc_socrata.py`](scripts/health/cdc_socrata.py) |

**What you can get:**

- CDC PLACES: county/tract estimates of obesity, diabetes, depression, smoking, insurance...
- Overdose deaths, COVID, vaccination, BRFSS, NCHS mortality
- SQL-like filtering ($where, $select, $group)

**Capstone ideas:** Social determinants of health, small-area health disparities, joining with Census ACS.

```python
from _common import get, show


def main():
    # CDC PLACES county-level data: depression prevalence in DC/MD/VA counties
    d = get("https://data.cdc.gov/resource/swc5-untb.json", params={
        "$select": "stateabbr,locationname,measure,data_value,year", "measureid": "DEPRESSION",
        "datavaluetypeid": "CrdPrv", "$where": "stateabbr in ('DC','MD','VA')",
        "$order": "data_value DESC", "$limit": 5})
    show(d)
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "stateabbr": "VA",
    "locationname": "Radford",
    "measure": "Depression among adults",
    "data_value": "30.0",
    "year": "2023"
  },
  {
    "stateabbr": "VA",
    "locationname": "Norton",
    "measure": "Depression among adults",
    "data_value": "27.1",
    "year": "2023"
  },
  {
    "stateabbr": "VA",
    "locationname": "Tazewell",
    "measure": "Depression among adults",
    "data_value": "26.9",
    "year": "2023"
  },
  {
    "stateabbr": "VA",
    "locationname": "Harrisonburg",
    "measure": "Depression among adults",
    "data_value": "26.9",
    "year": "2023"
  },
  {
    "stateabbr": "VA",
    "locationname": "Buena Vista",
    "measure": "Depression among adults",
    "data_value": "26.9",
    "year": "2023"
  }
]
```
</details>

---

### ClinicalTrials.gov API v2

| | |
|:--|:--|
| **Docs** | https://clinicaltrials.gov/data-api/api |
| **Auth** | None |
| **Register / token** | No registration (~50 req/min). |
| **Script** | [`scripts/health/clinicaltrials.py`](scripts/health/clinicaltrials.py) |

**What you can get:**

- 500k+ registered studies: conditions, interventions, phase, status, sponsors, enrollment, locations
- Results & adverse events for completed trials

**Capstone ideas:** Trial-success prediction, geographic access to trials, sponsor landscape analysis.

```python
from _common import get, show


def main():
    d = get("https://clinicaltrials.gov/api/v2/studies", params={
        "query.cond": "Alzheimer Disease", "filter.overallStatus": "RECRUITING", "pageSize": 5,
        "fields": "NCTId,BriefTitle,Phase,LeadSponsorName,EnrollmentCount", "countTotal": "true"})
    show({"total_recruiting": d.get("totalCount"), "studies": [
        {"nct": s["protocolSection"]["identificationModule"]["nctId"],
         "title": s["protocolSection"]["identificationModule"]["briefTitle"],
         "phase": s["protocolSection"].get("designModule", {}).get("phases"),
         "sponsor": s["protocolSection"]["sponsorCollaboratorsModule"]["leadSponsor"]["name"]}
        for s in d["studies"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
{
  "total_recruiting": 634,
  "studies": [
    {
      "nct": "NCT05820919",
      "title": "Enhancing Sleep Quality for Nursing Home Residents With Dementia - R33 Phase",
      "phase": [
        "NA"
      ],
      "sponsor": "University of Alabama, Tuscaloosa"
    },
    {
      "nct": "NCT07721467",
      "title": "Neuroplasticity Enhancement From Cognitive Training Reinforced by Psilocybin",
      "phase": [
        "PHASE2"
      ],
      "sponsor": "National Institute on Aging (NIA)"
    },
    {
      "nct": "NCT06611878",
      "title": "Memory for Music: Individual Intensive Musical Training in Alzheimer's Disease",
      "phase": [
        "NA"
      ],
      "sponsor": "NORCE Norwegian Research Centre AS"
    },
    {
      "nct": "NCT06869135",
      "title": "Saliva and Extracellular Vesicles for Neurodegenerative Diseases",
      "phase": null,
      "sponsor": "Fondazione Don Carlo Gnocchi ETS"
    },
    {
      "nct": "NCT07766174",
      "title": "Transforming Maternal Mental Health Care",
      "phase": [
        "NA"
      ],
      "sponsor": "University of Utah"
    }
  ]
}
```
</details>

---

### openFDA

| | |
|:--|:--|
| **Docs** | https://open.fda.gov/apis/ |
| **Auth** | None (optional key) |
| **Register / token** | Keyless: 1,000 req/day per IP. Free key at https://open.fda.gov/apis/authentication/ -> 120k/day. |
| **Env var** | `OPENFDA_API_KEY (optional)` |
| **Script** | [`scripts/health/openfda.py`](scripts/health/openfda.py) |

**What you can get:**

- Adverse event reports for drugs, devices, food (FAERS)
- Drug labels, recalls/enforcement, NDC directory
- `count=` aggregations directly in the API

**Capstone ideas:** Pharmacovigilance / adverse-event signal detection, recall analysis.

```python
from _common import get, show


def main():
    d = get("https://api.fda.gov/drug/event.json", params={
        "search": 'patient.drug.openfda.generic_name:"semaglutide"',
        "count": "patient.reaction.reactionmeddrapt.exact", "limit": 6})
    show({"drug": "semaglutide", "top_reported_reactions": d["results"]}, max_items=6)
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
{
  "drug": "semaglutide",
  "top_reported_reactions": [
    {
      "term": "NAUSEA",
      "count": 10674
    },
    {
      "term": "OFF LABEL USE",
      "count": 7325
    },
    {
      "term": "VOMITING",
      "count": 6964
    },
    {
      "term": "DIARRHOEA",
      "count": 6462
    },
    {
      "term": "DECREASED APPETITE",
      "count": 4905
    },
    {
      "term": "CONSTIPATION",
      "count": 4724
    }
  ]
}
```
</details>

---

### Our World in Data (Grapher API)

| | |
|:--|:--|
| **Docs** | https://docs.owid.io/projects/etl/api/chart-api/ |
| **Auth** | None |
| **Register / token** | No registration. Append `.csv` / `.metadata.json` to any chart URL on ourworldindata.org. |
| **Script** | [`scripts/health/our_world_in_data.py`](scripts/health/our_world_in_data.py) |

**What you can get:**

- Thousands of curated social-science series: poverty, democracy indices, education, CO2, health, happiness
- Clean country-year CSVs with sources documented

**Capstone ideas:** Fast panel data for social science (e.g., democracy vs. growth, happiness vs. income).

```python
import csv
import io

from _common import get, show


def main():
    text = get("https://ourworldindata.org/grapher/happiness-cantril-ladder.csv",
               params={"useColumnShortNames": "true"}, as_json=False)
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r["code"] in ("USA", "FIN", "IND", "BRA", "DEU")]
    latest = max(r["year"] for r in rows)
    show([r for r in rows if r["year"] == latest])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:14</summary>

```json
[
  {
    "entity": "Brazil",
    "code": "BRA",
    "year": "2025",
    "cantril_ladder_score": "6.634"
  },
  {
    "entity": "Finland",
    "code": "FIN",
    "year": "2025",
    "cantril_ladder_score": "7.764"
  },
  {
    "entity": "Germany",
    "code": "DEU",
    "year": "2025",
    "cantril_ladder_score": "6.882"
  },
  {
    "entity": "India",
    "code": "IND",
    "year": "2025",
    "cantril_ladder_score": "4.536"
  },
  {
    "entity": "United States",
    "code": "USA",
    "year": "2025",
    "cantril_ladder_score": "6.816"
  }
]
```
</details>

---

### WHO Global Health Observatory (GHO OData)

| | |
|:--|:--|
| **Docs** | https://www.who.int/data/gho/info/gho-odata-api |
| **Auth** | None |
| **Register / token** | No registration. |
| **Script** | [`scripts/health/who_gho.py`](scripts/health/who_gho.py) |

**What you can get:**

- 2,000+ health indicators for 194 countries: life expectancy, mortality, immunization, NCDs, health spending
- Breakdowns by sex, age, year

**Capstone ideas:** Global health inequality, health-system efficiency, cross-country panel models.

```python
from _common import get, show


def main():
    d = get("https://ghoapi.azureedge.net/api/WHOSIS_000001", params={  # life expectancy at birth
        "$filter": "SpatialDim in ('USA','JPN','NGA','IND','BRA') and Dim1 eq 'SEX_BTSX' and TimeDim eq 2021"})
    show([{"country": r["SpatialDim"], "year": r["TimeDim"], "life_expectancy": round(r["NumericValue"], 1)}
          for r in d["value"]])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "country": "JPN",
    "year": 2021,
    "life_expectancy": 84.3
  },
  {
    "country": "USA",
    "year": 2021,
    "life_expectancy": 76.4
  },
  {
    "country": "BRA",
    "year": 2021,
    "life_expectancy": 72.4
  },
  {
    "country": "IND",
    "year": 2021,
    "life_expectancy": 68.1
  },
  {
    "country": "NGA",
    "year": 2021,
    "life_expectancy": 61.6
  }
]
```
</details>

---

## 🔬 Science & Research

### arXiv API

| | |
|:--|:--|
| **Docs** | https://info.arxiv.org/help/api/user-manual.html |
| **Auth** | None |
| **Register / token** | No registration. Wait ~3 seconds between calls. Returns Atom XML. |
| **Script** | [`scripts/science_research/arxiv.py`](scripts/science_research/arxiv.py) |

**What you can get:**

- Preprints in CS, stats, physics, math, econ, q-bio: title, abstract, authors, categories, dates, PDF link

**Capstone ideas:** Abstract topic modeling, paper recommendation, tracking ML research trends.

```python
import xml.etree.ElementTree as ET

from _common import get, show

NS = {"a": "http://www.w3.org/2005/Atom"}


def main():
    xml = get("http://export.arxiv.org/api/query", params={
        "search_query": "cat:cs.CL AND abs:retrieval", "sortBy": "submittedDate", "max_results": 3}, as_json=False)
    root = ET.fromstring(xml)
    show([{"title": " ".join(e.find("a:title", NS).text.split()),
           "published": e.find("a:published", NS).text,
           "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)][:3],
           "abstract": " ".join(e.find("a:summary", NS).text.split())}
          for e in root.findall("a:entry", NS)], max_str=200)
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "title": "ViSkill: Reinforcing VLM Agents with Evolving Visual-Native Skills",
    "published": "2026-10-08T17:44:19Z",
    "authors": [
      "Hongxing Li",
      "Dingming Li",
      "Yixin Li"
    ],
    "abstract": "Skill-augmented agents improve sample efficiency by distilling successful trajectories into reusable strategies. Yet most existing approaches remain text-centric, linearizing spatial layouts and actio..."
  },
  {
    "title": "Which Skill to Distill? SGUID: Selecting a Compact Skill Bank for Model-Skill Co-Evolution",
    "published": "2026-10-08T17:27:53Z",
    "authors": [
      "Yuhan Liu",
      "Xiyao Ma",
      "Zhongkai Sun"
    ],
    "abstract": "Skills, reusable procedural guidance added at inference, can substantially improve LLM downstream performance (Li et al., 2026). Prior work retrieves skills from a bank by semantic relevance, then use..."
  },
  {
    "title": "Accurate but Not Humble: Evaluating Epistemic Humility in LLM Agents under Knowledge Conflict",
    "published": "2026-10-08T17:25:06Z",
    "authors": [
      "Kaiser Sun",
      "Bernal Jimenez Gutierrez",
      "Hongjun Liu"
    ],
    "abstract": "When retrieved evidence contradicts an agent's prior beliefs, does it revise its answer, acknowledge uncertainty, or persist with an incorrect conclusion? Existing evaluations of agentic systems focus..."
  }
]
```
</details>

---

### Crossref REST API

| | |
|:--|:--|
| **Docs** | https://api.crossref.org/swagger-ui/index.html |
| **Auth** | None (add `mailto=` for polite pool) |
| **Register / token** | No registration. |
| **Script** | [`scripts/science_research/crossref.py`](scripts/science_research/crossref.py) |

**What you can get:**

- Metadata for 150M+ DOIs: title, authors, journal, publisher, dates, references, funders, licenses

**Capstone ideas:** Bibliometrics, funder-impact analysis, DOI metadata enrichment.

```python
from _common import get, show


def main():
    d = get("https://api.crossref.org/works", params={
        "query.bibliographic": "data science education", "rows": 3, "sort": "is-referenced-by-count",
        "order": "desc", "mailto": "your_email@gwu.edu"})["message"]
    show([{"DOI": w["DOI"], "title": w["title"][0], "journal": (w.get("container-title") or [None])[0],
           "year": w["issued"]["date-parts"][0][0], "cited_by": w["is-referenced-by-count"]} for w in d["items"]])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "DOI": "10.1006/meth.2001.1262",
    "title": "Analysis of Relative Gene Expression Data Using Real-Time Quantitative PCR and the 2−ΔΔCT Method",
    "journal": "Methods",
    "year": 2001,
    "cited_by": 166791
  },
  {
    "DOI": "10.1186/s13059-014-0550-8",
    "title": "Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2",
    "journal": "Genome Biology",
    "year": 2014,
    "cited_by": 92838
  },
  {
    "DOI": "10.1016/0927-0256(96)00008-0",
    "title": "Efficiency of ab-initio total energy calculations for metals and semiconductors using a plane-wave basis set",
    "journal": "Computational Materials Science",
    "year": 1996,
    "cited_by": 74879
  }
]
```
</details>

---

### NASA Open APIs (APOD, NeoWs, DONKI, EONET...)

| | |
|:--|:--|
| **Docs** | https://api.nasa.gov/ |
| **Auth** | api.data.gov key (`DEMO_KEY` works at low volume) |
| **Register / token** | Get a key instantly at https://api.nasa.gov/ (same api.data.gov key). 1,000 req/hour; DEMO_KEY 30/hour. |
| **Env var** | `DATA_GOV_API_KEY` |
| **Script** | [`scripts/science_research/nasa.py`](scripts/science_research/nasa.py) |

**What you can get:**

- Near-Earth asteroids: size, velocity, miss distance, hazard flag (NeoWs)
- Astronomy Picture of the Day, Mars rover photos, space weather (DONKI)
- Natural events: wildfires, storms, volcanoes (EONET, keyless)

**Capstone ideas:** Asteroid hazard classification, wildfire event tracking, image datasets.

```python
import os

from _common import get, show


def main():
    d = get("https://api.nasa.gov/neo/rest/v1/feed", params={
        "start_date": "2026-10-01", "end_date": "2026-10-01", "api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY")})
    neos = d["near_earth_objects"]["2026-10-01"]
    show({"count": d["element_count"], "asteroids": [
        {"name": n["name"], "diameter_m_max": round(n["estimated_diameter"]["meters"]["estimated_diameter_max"]),
         "hazardous": n["is_potentially_hazardous_asteroid"],
         "miss_km": round(float(n["close_approach_data"][0]["miss_distance"]["kilometers"])),
         "velocity_kph": round(float(n["close_approach_data"][0]["relative_velocity"]["kilometers_per_hour"]))}
        for n in neos]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
{
  "count": 6,
  "asteroids": [
    {
      "name": "499496 (2010 MR87)",
      "diameter_m_max": 755,
      "hazardous": false,
      "miss_km": 28345948,
      "velocity_kph": 67816
    },
    {
      "name": "(2008 TX3)",
      "diameter_m_max": 62,
      "hazardous": false,
      "miss_km": 65905706,
      "velocity_kph": 60405
    },
    {
      "name": "(2014 RX22)",
      "diameter_m_max": 38,
      "hazardous": false,
      "miss_km": 37754644,
      "velocity_kph": 46816
    },
    {
      "name": "(2016 FB)",
      "diameter_m_max": 110,
      "hazardous": false,
      "miss_km": 64412687,
      "velocity_kph": 39905
    },
    {
      "name": "(2018 LC1)",
      "diameter_m_max": 62,
      "hazardous": false,
      "miss_km": 71893233,
      "velocity_kph": 30362
    },
    "... (1 more)"
  ]
}
```
</details>

---

### OpenAlex (scholarly works)

| | |
|:--|:--|
| **Docs** | https://docs.openalex.org/ |
| **Auth** | None (add `mailto=` for the faster 'polite pool') |
| **Register / token** | No registration. 100k req/day. Add `mailto=you@gwu.edu` to requests. |
| **Script** | [`scripts/science_research/openalex.py`](scripts/science_research/openalex.py) |

**What you can get:**

- 250M+ papers: title, authors, institutions, venues, concepts/topics, citations, open-access links
- Author / institution / funder entities
- Abstracts (inverted index), citation graph, group-by aggregations

**Capstone ideas:** Science-of-science, citation networks, research-trend forecasting, institution rankings.

```python
from _common import get, show


def main():
    d = get("https://api.openalex.org/works", params={
        "search": "large language models", "filter": "from_publication_date:2024-01-01",
        "sort": "cited_by_count:desc", "per-page": 5, "mailto": "your_email@gwu.edu"})
    show({"total": d["meta"]["count"], "top_cited": [
        {"title": w["title"], "year": w["publication_year"], "cited_by": w["cited_by_count"],
         "venue": (w.get("primary_location") or {}).get("source", {}) and w["primary_location"]["source"].get("display_name")}
        for w in d["results"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
{
  "total": 1610170,
  "top_cited": [
    {
      "title": "Detecting Functionality-Specific Vulnerabilities via Retrieving Individual Functionality-Equivalent APIs in Open-Source Repositories",
      "year": 2025,
      "cited_by": 15857,
      "venue": "Leibniz international proceedings in informatics"
    },
    {
      "title": "Accurate structure prediction of biomolecular interactions with AlphaFold 3",
      "year": 2024,
      "cited_by": 15554,
      "venue": "Nature"
    },
    {
      "title": "The Difference between Emergency Remote Teaching and Online Learning",
      "year": 2024,
      "cited_by": 6265,
      "venue": "BRILL eBooks"
    },
    {
      "title": "Global incidence, prevalence, years lived with disability (YLDs), disability-adjusted life-years (DALYs), and healthy life expectancy (HALE) for 371 diseases an...",
      "year": 2024,
      "cited_by": 5035,
      "venue": "The Lancet"
    },
    {
      "title": "Kaldi Speech Recognition Toolkit",
      "year": 2024,
      "cited_by": 4823,
      "venue": "Infoscience (Ecole Polytechnique Fédérale de Lausanne)"
    }
  ]
}
```
</details>

---

### PubChem PUG-REST

| | |
|:--|:--|
| **Docs** | https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest |
| **Auth** | None |
| **Register / token** | No registration. Max 5 req/sec. |
| **Script** | [`scripts/science_research/pubchem.py`](scripts/science_research/pubchem.py) |

**What you can get:**

- 100M+ chemical compounds: formula, weight, SMILES, InChI, XLogP, synonyms
- Bioassay results, similarity/substructure search

**Capstone ideas:** Molecular property prediction (cheminformatics + ML), drug-likeness models.

```python
from _common import get, show


def main():
    # CIDs: 2519 caffeine, 2244 aspirin, 3672 ibuprofen (look up a CID via /compound/name/<name>/cids/JSON)
    props = "MolecularFormula,MolecularWeight,ConnectivitySMILES,XLogP,HBondDonorCount"
    d = get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2519,2244,3672/property/{props}/JSON")
    show(d["PropertyTable"]["Properties"])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:12</summary>

```json
[
  {
    "CID": 2519,
    "MolecularFormula": "C8H10N4O2",
    "MolecularWeight": "194.19",
    "ConnectivitySMILES": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "XLogP": -0.1,
    "HBondDonorCount": 0
  },
  {
    "CID": 2244,
    "MolecularFormula": "C9H8O4",
    "MolecularWeight": "180.16",
    "ConnectivitySMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "XLogP": 1.2,
    "HBondDonorCount": 1
  },
  {
    "CID": 3672,
    "MolecularFormula": "C13H18O2",
    "MolecularWeight": "206.28",
    "ConnectivitySMILES": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "XLogP": 3.5,
    "HBondDonorCount": 1
  }
]
```
</details>

---

### Semantic Scholar Academic Graph

| | |
|:--|:--|
| **Docs** | https://api.semanticscholar.org/api-docs/ |
| **Auth** | None (shared pool) / free key (header `x-api-key`) |
| **Register / token** | Works keyless in principle, but the shared pool is often rate-limited (HTTP 429). Request a free key at https://www.semanticscholar.org/product/api#api-key-form -> 1 req/sec dedicated. |
| **Env var** | `S2_API_KEY (optional)` |
| **Script** | [`scripts/science_research/semantic_scholar.py`](scripts/science_research/semantic_scholar.py) |

**What you can get:**

- Papers with TLDR summaries, abstracts, citations/references, influential-citation counts
- SPECTER2 paper embeddings
- Author h-index & paper lists
- Bulk dataset downloads

**Capstone ideas:** Citation-intent analysis, embedding-based paper clustering, literature-review automation.

```python
import os

from _common import get, show


def main():
    hdr = {"x-api-key": os.environ["S2_API_KEY"]} if os.getenv("S2_API_KEY") else {}
    d = get("https://api.semanticscholar.org/graph/v1/paper/search", headers=hdr, params={
        "query": "attention is all you need", "limit": 3, "fields": "title,year,citationCount,tldr"})
    show([{"title": p["title"], "year": p["year"], "citationCount": p["citationCount"],
           "tldr": (p.get("tldr") or {}).get("text")} for p in d["data"]])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"title": "Attention is All you Need", "year": 2017, "citationCount": 160000,
   "tldr": "A new simple network architecture, the Transformer, based solely on attention mechanisms..."}
]
```
</details>

---

## ⚙️ Engineering, Energy & Tech

### US Energy Information Administration (EIA) v2

| | |
|:--|:--|
| **Docs** | https://www.eia.gov/opendata/documentation.php |
| **Auth** | API key (query `api_key`) |
| **Register / token** | Register at https://www.eia.gov/opendata/register.php (key emailed). Free, ~5,000 req/hour. |
| **Env var** | `EIA_API_KEY` |
| **Script** | [`scripts/engineering_energy/eia.py`](scripts/engineering_energy/eia.py) |

**What you can get:**

- Hourly electricity demand & generation by fuel for every US balancing authority
- Electricity prices by state/sector, petroleum, natural gas, coal, renewables, CO2 emissions
- Short-Term & Annual Energy Outlook forecasts

**Capstone ideas:** Load forecasting, renewable-share analysis, grid-carbon-intensity modeling.

```python
from _common import get, key, show


def main():
    d = get("https://api.eia.gov/v2/electricity/rto/region-data/data/", params={
        "api_key": key("EIA_API_KEY"), "frequency": "hourly", "data[0]": "value",
        "facets[respondent][]": "PJM", "facets[type][]": "D",
        "sort[0][column]": "period", "sort[0][direction]": "desc", "length": 5})
    show([{k: r[k] for k in ("period", "respondent", "type-name", "value", "value-units")} for r in d["response"]["data"]])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"period": "2026-10-10T23", "respondent": "PJM", "type-name": "Demand", "value": 92841, "value-units": "megawatthours"},
  {"period": "2026-10-10T22", "respondent": "PJM", "type-name": "Demand", "value": 95310, "value-units": "megawatthours"}
]
```
</details>

---

### GitHub REST API

| | |
|:--|:--|
| **Docs** | https://docs.github.com/en/rest |
| **Auth** | None (60 req/hour) / personal access token (5,000 req/hour) |
| **Register / token** | Optional: GitHub -> Settings -> Developer settings -> Personal access tokens -> Fine-grained -> public-repo read-only. Pass as `Authorization: Bearer <token>`. |
| **Env var** | `GITHUB_TOKEN (optional)` |
| **Script** | [`scripts/engineering_energy/github.py`](scripts/engineering_energy/github.py) |

**What you can get:**

- Repository search & stats: stars, forks, language, topics, license, issues
- Commits, contributors, pull requests, issues, releases
- User / org profiles

**Capstone ideas:** Open-source ecosystem health, developer-collaboration networks, technology adoption.

```python
import os

from _common import get, show


def main():
    hdr = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"} if os.getenv("GITHUB_TOKEN") else {}
    d = get("https://api.github.com/search/repositories", headers=hdr, params={
        "q": "topic:machine-learning language:python", "sort": "stars", "per_page": 5})
    show({"total": d["total_count"], "repos": [
        {"repo": r["full_name"], "stars": r["stargazers_count"], "forks": r["forks_count"],
         "open_issues": r["open_issues_count"], "license": (r.get("license") or {}).get("spdx_id")} for r in d["items"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
{
  "total": 88218,
  "repos": [
    {
      "repo": "huggingface/transformers",
      "stars": 167522,
      "forks": 34839,
      "open_issues": 2335,
      "license": "Apache-2.0"
    },
    {
      "repo": "pytorch/pytorch",
      "stars": 104458,
      "forks": 32100,
      "open_issues": 17703,
      "license": "NOASSERTION"
    },
    {
      "repo": "d2l-ai/d2l-zh",
      "stars": 81567,
      "forks": 12421,
      "open_issues": 127,
      "license": "Apache-2.0"
    },
    {
      "repo": "openbq-org/OpenBB",
      "stars": 74118,
      "forks": 7659,
      "open_issues": 102,
      "license": "NOASSERTION"
    },
    {
      "repo": "scikit-learn/scikit-learn",
      "stars": 67553,
      "forks": 27504,
      "open_issues": 2166,
      "license": "BSD-3-Clause"
    }
  ]
}
```
</details>

---

### Hugging Face Hub API

| | |
|:--|:--|
| **Docs** | https://huggingface.co/docs/hub/api |
| **Auth** | None for public metadata (token for gated models / inference) |
| **Register / token** | Optional: https://huggingface.co/settings/tokens -> New token (read). `pip install huggingface_hub`. |
| **Env var** | `HF_TOKEN (optional)` |
| **Script** | [`scripts/engineering_energy/huggingface.py`](scripts/engineering_energy/huggingface.py) |

**What you can get:**

- Model / dataset / Space search: downloads, likes, tags, pipeline task, license
- Model cards & dataset cards
- Datasets Server: row previews and Parquet files for any public dataset

**Capstone ideas:** ML-ecosystem analytics, finding pretrained models & datasets for your capstone.

```python
from _common import get, show


def main():
    d = get("https://huggingface.co/api/models", params={"pipeline_tag": "text-classification",
                                                         "sort": "downloads", "direction": -1, "limit": 5})
    show([{k: m.get(k) for k in ("id", "downloads", "likes", "library_name")} for m in d])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "id": "BAAI/bge-reranker-v2-m3",
    "downloads": 16162257,
    "likes": 1229,
    "library_name": "sentence-transformers"
  },
  {
    "id": "ProsusAI/finbert",
    "downloads": 4377369,
    "likes": 1264,
    "library_name": "transformers"
  },
  {
    "id": "BAAI/bge-reranker-base",
    "downloads": 3877205,
    "likes": 248,
    "library_name": "sentence-transformers"
  },
  {
    "id": "SamLowe/roberta-base-go_emotions",
    "downloads": 3745871,
    "likes": 696,
    "library_name": "transformers"
  },
  {
    "id": "meta-llama/Prompt-Guard-86M",
    "downloads": 2805342,
    "likes": 422,
    "library_name": "transformers"
  }
]
```
</details>

---

### NLR (formerly NREL) Developer Network — solar, EV charging

| | |
|:--|:--|
| **Docs** | https://developer.nlr.gov/docs/ |
| **Auth** | api.data.gov key (`DEMO_KEY` works at low volume) |
| **Register / token** | Sign up at https://developer.nlr.gov/signup/ (api.data.gov key). 1,000 req/hour. |
| **Env var** | `DATA_GOV_API_KEY` |
| **Script** | [`scripts/engineering_energy/nrel.py`](scripts/engineering_energy/nrel.py) |

**What you can get:**

- PVWatts: estimated solar PV output for any location/system
- Solar resource (GHI/DNI) data
- Alternative-fuel / EV charging stations (location, connectors, network)
- Utility rates

**Capstone ideas:** EV-charger access equity, rooftop-solar potential mapping, techno-economic analysis.

```python
import os

from _common import get, show


def main():
    d = get("https://developer.nlr.gov/api/alt-fuel-stations/v1/nearest.json", params={
        "api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY"), "latitude": 38.9, "longitude": -77.04,
        "fuel_type": "ELEC", "limit": 5})
    show({"station_counts_near_DC": d["total_results"], "stations": [
        {k: s[k] for k in ("station_name", "ev_network", "ev_level2_evse_num", "ev_dc_fast_num")} |
        {"distance_mi": round(s["distance"], 2)} for s in d["fuel_stations"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:12</summary>

```json
{
  "station_counts_near_DC": 630,
  "stations": [
    {
      "station_name": "1750 H St. NW. Garage",
      "ev_network": "FLASH",
      "ev_level2_evse_num": 1,
      "ev_dc_fast_num": null,
      "distance_mi": 0.03
    },
    {
      "station_name": "1875 Pennsylvania Ave NW",
      "ev_network": "Blink Network",
      "ev_level2_evse_num": 1,
      "ev_dc_fast_num": null,
      "distance_mi": 0.14
    },
    {
      "station_name": "World Bank",
      "ev_network": "EV Connect",
      "ev_level2_evse_num": 2,
      "ev_dc_fast_num": null,
      "distance_mi": 0.17
    },
    {
      "station_name": "The Hay-Adams - Tesla Destination",
      "ev_network": "Tesla Destination",
      "ev_level2_evse_num": 1,
      "ev_dc_fast_num": null,
      "distance_mi": 0.17
    },
    {
      "station_name": "World Bank",
      "ev_network": "EV Connect",
      "ev_level2_evse_num": 4,
      "ev_dc_fast_num": null,
      "distance_mi": 0.18
    }
  ]
}
```
</details>

---

### NIST National Vulnerability Database (CVE API 2.0)

| | |
|:--|:--|
| **Docs** | https://nvd.nist.gov/developers/vulnerabilities |
| **Auth** | None (optional key) |
| **Register / token** | Keyless: 5 req / 30 s. Request a key at https://nvd.nist.gov/developers/request-an-api-key -> 50 req / 30 s. |
| **Env var** | `NVD_API_KEY (optional)` |
| **Script** | [`scripts/engineering_energy/nvd_cve.py`](scripts/engineering_energy/nvd_cve.py) |

**What you can get:**

- Every CVE: description, CVSS scores/severity, CWE weakness type, affected products (CPE), references

**Capstone ideas:** Vulnerability-severity prediction from text, exploit-likelihood modeling, cyber-risk dashboards.

```python
from _common import get, show


def main():
    d = get("https://services.nvd.nist.gov/rest/json/cves/2.0", params={
        "keywordSearch": "pytorch", "resultsPerPage": 5})
    out = []
    for v in d["vulnerabilities"]:
        c = v["cve"]
        m = (c.get("metrics", {}).get("cvssMetricV31") or [{}])[0].get("cvssData", {})
        out.append({"id": c["id"], "published": c["published"][:10], "severity": m.get("baseSeverity"),
                    "score": m.get("baseScore"), "description": c["descriptions"][0]["value"]})
    show({"total": d["totalResults"], "cves": out}, max_str=140)
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
{
  "total": 77,
  "cves": [
    {
      "id": "CVE-2021-43811",
      "published": "2021-12-08",
      "severity": "HIGH",
      "score": 7.8,
      "description": "Sockeye is an open-source sequence-to-sequence framework for Neural Machine Translation built on PyTorch. Sockeye uses YAML to store model a..."
    },
    {
      "id": "CVE-2021-4118",
      "published": "2021-12-23",
      "severity": "HIGH",
      "score": 7.8,
      "description": "pytorch-lightning is vulnerable to Deserialization of Untrusted Data"
    },
    {
      "id": "CVE-2022-0845",
      "published": "2022-03-05",
      "severity": "CRITICAL",
      "score": 9.8,
      "description": "Code Injection in GitHub repository pytorchlightning/pytorch-lightning prior to 1.6.0."
    },
    {
      "id": "CVE-2022-45907",
      "published": "2022-11-26",
      "severity": "CRITICAL",
      "score": 9.8,
      "description": "In PyTorch before trunk/89695, torch.jit.annotations.parse_type_line can cause arbitrary code execution because eval is used unsafely."
    },
    {
      "id": "CVE-2023-43654",
      "published": "2023-09-28",
      "severity": "CRITICAL",
      "score": 10.0,
      "description": "TorchServe is a tool for serving and scaling PyTorch models in production. TorchServe default configuration lacks proper input validation, e..."
    }
  ]
}
```
</details>

---

### USGS Water Services (streamflow)

| | |
|:--|:--|
| **Docs** | https://waterservices.usgs.gov/docs/instantaneous-values/ |
| **Auth** | None |
| **Register / token** | No registration. (USGS is migrating to https://api.waterdata.usgs.gov — same data, OGC format.) |
| **Script** | [`scripts/engineering_energy/usgs_water.py`](scripts/engineering_energy/usgs_water.py) |

**What you can get:**

- Real-time & historical streamflow (discharge), gage height, water temperature, water quality
- 10,000+ monitoring sites with 15-minute readings

**Capstone ideas:** Flood forecasting, hydrologic time-series ML, drought monitoring.

```python
from _common import get, show


def main():
    d = get("https://waterservices.usgs.gov/nwis/iv/", params={
        "format": "json", "sites": "01646500", "parameterCd": "00060,00065", "period": "PT2H"})  # Potomac @ Little Falls
    show([{"site": ts["sourceInfo"]["siteName"], "variable": ts["variable"]["variableName"],
           "latest": ts["values"][0]["value"][-1]} for ts in d["value"]["timeSeries"]])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "site": "POTOMAC RIVER NEAR WASH, DC LITTLE FALLS PUMP STA",
    "variable": "Streamflow, ft&#179;/s",
    "latest": {
      "value": "5780",
      "qualifiers": [
        "P"
      ],
      "dateTime": "2026-10-11T18:50:00.000-04:00"
    }
  },
  {
    "site": "POTOMAC RIVER NEAR WASH, DC LITTLE FALLS PUMP STA",
    "variable": "Gage height, ft",
    "latest": {
      "value": "3.47",
      "qualifiers": [
        "P"
      ],
      "dateTime": "2026-10-11T18:50:00.000-04:00"
    }
  }
]
```
</details>

---

## 🗺️ Transportation & Geospatial

### OpenStreetMap Nominatim (geocoding)

| | |
|:--|:--|
| **Docs** | https://nominatim.org/release-docs/latest/api/Overview/ |
| **Auth** | None (User-Agent required) |
| **Register / token** | No registration. Max 1 req/sec; no bulk geocoding on the public server (self-host for bulk). |
| **Script** | [`scripts/transportation_geo/nominatim.py`](scripts/transportation_geo/nominatim.py) |

**What you can get:**

- Address -> lat/lon (forward geocoding) and lat/lon -> address (reverse)
- OSM class/type, bounding box, structured address parts

**Capstone ideas:** Geocoding addresses in any dataset before spatial joins with Census tracts.

```python
from _common import get, show


def main():
    d = get("https://nominatim.openstreetmap.org/search",
            params={"q": "George Washington University, Washington DC", "format": "jsonv2", "limit": 2})
    show([{k: r[k] for k in ("display_name", "lat", "lon", "category", "type")} for r in d])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
[
  {
    "display_name": "George Washington University, Ward 2, Washington, District of Columbia, 20037, United States",
    "lat": "38.8998650",
    "lon": "-77.0474485",
    "category": "place",
    "type": "neighbourhood"
  }
]
```
</details>

---

### OpenSky Network (live flights)

| | |
|:--|:--|
| **Docs** | https://openskynetwork.github.io/opensky-api/rest.html |
| **Auth** | None (anonymous: 400 credits/day) / OAuth client for more |
| **Register / token** | Anonymous works for live state vectors. Register at https://opensky-network.org for 4,000 credits/day and historical flights (create an API client under Account -> API Client). |
| **Script** | [`scripts/transportation_geo/opensky.py`](scripts/transportation_geo/opensky.py) |

**What you can get:**

- Live aircraft state vectors: callsign, origin country, lat/lon, altitude, velocity, heading
- Arrivals/departures by airport, flight tracks (authenticated)

**Capstone ideas:** Air-traffic pattern mining, emissions estimation, delay propagation.

```python
from _common import get, show


def main():
    d = get("https://opensky-network.org/api/states/all",  # bounding box around Washington DC
            params={"lamin": 38.6, "lomin": -77.6, "lamax": 39.2, "lomax": -76.6})
    states = d.get("states") or []
    show({"aircraft_in_box": len(states), "sample": [
        {"callsign": s[1].strip(), "origin_country": s[2], "lat": s[6], "lon": s[5], "baro_alt_m": s[7],
         "velocity_mps": s[9]} for s in states]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:09</summary>

```json
{
  "aircraft_in_box": 58,
  "sample": [
    {
      "callsign": "EJA738",
      "origin_country": "United States",
      "lat": 38.9503,
      "lon": -77.4432,
      "baro_alt_m": null,
      "velocity_mps": 6.69
    },
    {
      "callsign": "SWA4523",
      "origin_country": "United States",
      "lat": 38.8439,
      "lon": -77.0411,
      "baro_alt_m": null,
      "velocity_mps": 1.03
    },
    {
      "callsign": "ENY4169",
      "origin_country": "United States",
      "lat": 38.8479,
      "lon": -77.0388,
      "baro_alt_m": null,
      "velocity_mps": 1.03
    },
    {
      "callsign": "AAL544",
      "origin_country": "United States",
      "lat": 38.8427,
      "lon": -77.0383,
      "baro_alt_m": null,
      "velocity_mps": 0
    },
    {
      "callsign": "SWA2160",
      "origin_country": "United States",
      "lat": 39.0548,
      "lon": -76.7812,
      "baro_alt_m": 1242.06,
      "velocity_mps": 106.76
    },
    "... (53 more)"
  ]
}
```
</details>

---

### OpenStreetMap Overpass API

| | |
|:--|:--|
| **Docs** | https://wiki.openstreetmap.org/wiki/Overpass_API |
| **Auth** | None |
| **Register / token** | No registration. Keep queries small; heavy users should self-host. |
| **Script** | [`scripts/transportation_geo/overpass.py`](scripts/transportation_geo/overpass.py) |

**What you can get:**

- Query any OSM feature in an area: roads, bike lanes, bus stops, restaurants, schools, hospitals, buildings
- Tags (name, cuisine, opening_hours, amenity...) and coordinates/geometry

**Capstone ideas:** Food deserts, walkability scores, amenity access, urban-form features for ML.

```python
from _common import USER_AGENT, show
import requests


def main():
    q = """[out:json][timeout:25];
    node["amenity"="cafe"](around:600,38.8997,-77.0486);
    out 5;"""
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q},
                      headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    show([{"name": e["tags"].get("name"), "lat": e["lat"], "lon": e["lon"],
           "opening_hours": e["tags"].get("opening_hours")} for e in r.json()["elements"]])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:10</summary>

```json
[
  {
    "name": "Starbucks",
    "lat": 38.8961368,
    "lon": -77.0446667,
    "opening_hours": "Mo-Fr 06:00-19:00; Sa-Su 07:00-17:30"
  },
  {
    "name": "Starbucks",
    "lat": 38.9010118,
    "lon": -77.0507502,
    "opening_hours": null
  },
  {
    "name": "Starbucks",
    "lat": 38.902294,
    "lon": -77.043818,
    "opening_hours": null
  },
  {
    "name": "SecreTea",
    "lat": 38.8961397,
    "lon": -77.044562,
    "opening_hours": null
  },
  {
    "name": "Cafe Aria",
    "lat": 38.897517,
    "lon": -77.043879,
    "opening_hours": null
  }
]
```
</details>

---

### WMATA (DC Metro) API

| | |
|:--|:--|
| **Docs** | https://developer.wmata.com/apis |
| **Auth** | API key (header `api_key`) |
| **Register / token** | Sign up at https://developer.wmata.com -> Products -> subscribe to 'Default Tier' -> Profile shows primary key. 10 calls/sec, 50,000/day. GTFS & GTFS-RT feeds included. |
| **Env var** | `WMATA_API_KEY` |
| **Script** | [`scripts/transportation_geo/wmata.py`](scripts/transportation_geo/wmata.py) |

**What you can get:**

- Real-time rail predictions (next trains) per station
- Bus positions & predictions
- Rail incidents, elevator/escalator outages
- Station & route metadata, GTFS

**Capstone ideas:** Transit reliability analysis, ridership/delay prediction, accessibility (elevator outage) studies.

```python
from _common import get, key, show


def main():
    d = get("https://api.wmata.com/StationPrediction.svc/json/GetPrediction/A01",  # Metro Center
            headers={"api_key": key("WMATA_API_KEY")})
    show([{k: t[k] for k in ("Line", "Destination", "Min", "Car")} for t in d["Trains"]])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"Line": "RD", "Destination": "Glenmont", "Min": "3", "Car": "8"},
  {"Line": "RD", "Destination": "Shady Grove", "Min": "5", "Car": "6"},
  {"Line": "RD", "Destination": "Glenmont", "Min": "11", "Car": "8"}
]
```
</details>

---

## 🎬 Culture, Books & Entertainment

### The Met Museum Collection API

| | |
|:--|:--|
| **Docs** | https://metmuseum.github.io/ |
| **Auth** | None |
| **Register / token** | No registration. Max 80 req/sec. Get every object ID via `/public/collection/v1/objects`. |
| **Script** | [`scripts/culture_misc/met_museum.py`](scripts/culture_misc/met_museum.py) |

**What you can get:**

- 470k+ artworks: title, artist, date, culture, medium, department, dimensions, public-domain image URLs

**Capstone ideas:** Computer-vision on art images, cultural-history analytics.

```python
from _common import get, show


def main():
    # The /search endpoint was retired (HTTP 410); /objects (all IDs) and /objects/<id> still work.
    ids = [436524, 437984, 45734]  # Met object IDs (436524 = Van Gogh, 'Sunflowers')
    objs = [get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{i}") for i in ids]
    show([{k: o.get(k) for k in ("title", "artistDisplayName", "objectDate", "medium", "department", "primaryImageSmall")}
          for o in objs])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:12</summary>

```json
[
  {
    "title": "Sunflowers",
    "artistDisplayName": "Vincent van Gogh",
    "objectDate": "1887",
    "medium": "Oil on canvas",
    "department": "European Paintings",
    "primaryImageSmall": "https://images.metmuseum.org/CRDImages/ep/web-large/DP-41223-001.jpg"
  },
  {
    "title": "La Berceuse (Woman Rocking a Cradle; Augustine-Alix Pellicot Roulin, 1851–1930)",
    "artistDisplayName": "Vincent van Gogh",
    "objectDate": "1889",
    "medium": "Oil on canvas",
    "department": "European Paintings",
    "primaryImageSmall": "https://images.metmuseum.org/CRDImages/ep/web-large/DP-19279-001.jpg"
  },
  {
    "title": "Quail and Millet",
    "artistDisplayName": "Kiyohara Yukinobu",
    "objectDate": "late 17th century",
    "medium": "Hanging scroll; ink and color on silk",
    "department": "Asian Art",
    "primaryImageSmall": "https://images.metmuseum.org/CRDImages/as/web-large/DP251139.jpg"
  }
]
```
</details>

---

### Open Library (books)

| | |
|:--|:--|
| **Docs** | https://openlibrary.org/developers/api |
| **Auth** | None |
| **Register / token** | No registration. |
| **Script** | [`scripts/culture_misc/open_library.py`](scripts/culture_misc/open_library.py) |

**What you can get:**

- Book search: title, authors, first publish year, subjects, editions, ISBNs
- Author & cover data

**Capstone ideas:** Cultural analytics, book-recommendation systems, publishing trends.

```python
from _common import get, show


def main():
    d = get("https://openlibrary.org/search.json", params={"q": "data science", "limit": 5,
                                                          "fields": "title,author_name,first_publish_year,edition_count"})
    show({"numFound": d["numFound"], "docs": d["docs"]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 19:10</summary>

```json
{
  "numFound": 32288,
  "docs": [
    {
      "author_name": [
        "Joel Grus"
      ],
      "edition_count": 4,
      "first_publish_year": 2015,
      "title": "Data science from scratch"
    },
    {
      "author_name": [
        "Tiffany-Anne Timbers",
        "Trevor Campbell",
        "Melissa Lee"
      ],
      "edition_count": 6,
      "first_publish_year": 2022,
      "title": "Data Science"
    },
    {
      "author_name": [
        "Foster Provost",
        "Tom Fawcett"
      ],
      "edition_count": 5,
      "first_publish_year": 2013,
      "title": "Data Science for Business"
    },
    {
      "author_name": [
        "Lillian Pierson"
      ],
      "edition_count": 10,
      "first_publish_year": 2015,
      "title": "Data science for dummies"
    },
    {
      "author_name": [
        "Nick Singh",
        "Kevin Huo"
      ],
      "edition_count": 1,
      "first_publish_year": 2021,
      "title": "Ace the Data Science Interview"
    }
  ]
}
```
</details>

---

### TMDB (The Movie Database)

| | |
|:--|:--|
| **Docs** | https://developer.themoviedb.org/docs |
| **Auth** | API read-access token (Bearer) |
| **Register / token** | Create an account at https://www.themoviedb.org/signup -> Settings -> API -> request a Developer key (short form). Use the 'API Read Access Token' as a Bearer token. ~50 req/sec. |
| **Env var** | `TMDB_TOKEN` |
| **Script** | [`scripts/culture_misc/tmdb.py`](scripts/culture_misc/tmdb.py) |

**What you can get:**

- Movies & TV: title, release date, genres, budget, revenue, runtime, popularity, ratings
- Cast & crew credits, keywords, reviews, streaming providers by country

**Capstone ideas:** Box-office prediction, recommender systems, gender representation in film crews.

```python
from _common import get, key, show


def main():
    d = get("https://api.themoviedb.org/3/trending/movie/week",
            headers={"Authorization": f"Bearer {key('TMDB_TOKEN')}"})
    show([{k: m[k] for k in ("title", "release_date", "vote_average", "vote_count", "popularity")}
          for m in d["results"]])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"title": "Example Blockbuster", "release_date": "2026-10-03", "vote_average": 7.6, "vote_count": 812, "popularity": 1543.2}
]
```
</details>

---
