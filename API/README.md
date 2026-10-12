# Public API Catalog for Data Science Capstones

> A curated, runnable list of free APIs across news, social media, weather, finance, census, housing,
> political science, law, health, education, science, AI & machine learning, engineering, transportation,
> sports, and culture.
> Every entry has: how to register / get a token, what data you can get, the endpoint and its
> **parameters**, a short Python example, and a **sample output**.

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
  OpenFEC, Regulations.gov, College Scorecard, NASA, and NLR (formerly NREL). Set it as `DATA_GOV_API_KEY`.
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
| 29 | 🏠 Housing & Real Estate | [CFPB Consumer Complaint Database](#cfpb-consumer-complaint-database) | 🟢 no key | — | [`cfpb_complaints.py`](scripts/housing/cfpb_complaints.py) |
| 30 | 🏠 Housing & Real Estate | [FHFA House Price Index](#fhfa-house-price-index) | 🟢 no key | — | [`fhfa_hpi.py`](scripts/housing/fhfa_hpi.py) |
| 31 | 🏠 Housing & Real Estate | [HMDA Data Browser (mortgage applications)](#hmda-data-browser-mortgage-applications) | 🟢 no key | — | [`hmda.py`](scripts/housing/hmda.py) |
| 32 | 🏠 Housing & Real Estate | [HUD User API (Fair Market Rents, Income Limits, ZIP crosswalk)](#hud-user-api-fair-market-rents-income-limits-zip-crosswalk) | 🔑 key | `HUD_API_TOKEN` | [`hud_user.py`](scripts/housing/hud_user.py) |
| 33 | 🏠 Housing & Real Estate | [OpenFEMA — Flood Insurance Claims & Disaster Declarations](#openfema--flood-insurance-claims--disaster-declarations) | 🟢 no key | — | [`openfema_nfip.py`](scripts/housing/openfema_nfip.py) |
| 34 | 🏠 Housing & Real Estate | [Redfin Data Center (housing market tracker)](#redfin-data-center-housing-market-tracker) | 🟢 no key | — | [`redfin_data_center.py`](scripts/housing/redfin_data_center.py) |
| 35 | 🏠 Housing & Real Estate | [RentCast (property records, rent & value estimates)](#rentcast-property-records-rent--value-estimates) | 🔑 key | `RENTCAST_API_KEY` | [`rentcast.py`](scripts/housing/rentcast.py) |
| 36 | 🏠 Housing & Real Estate | [Zillow Research Data (ZHVI home values, ZORI rents)](#zillow-research-data-zhvi-home-values-zori-rents) | 🟢 no key | — | [`zillow_research.py`](scripts/housing/zillow_research.py) |
| 37 | 🏛️ Political Science & Government | [ACLED (Armed Conflict Location & Event Data)](#acled-armed-conflict-location--event-data) | 🔑 key | `ACLED_EMAIL, ACLED_PASSWORD` | [`acled.py`](scripts/politics_government/acled.py) |
| 38 | 🏛️ Political Science & Government | [Congress.gov API](#congressgov-api) | 🟢 no key | `DATA_GOV_API_KEY` | [`congress_gov.py`](scripts/politics_government/congress_gov.py) |
| 39 | 🏛️ Political Science & Government | [European Parliament Open Data API](#european-parliament-open-data-api) | 🟢 no key | — | [`european_parliament.py`](scripts/politics_government/european_parliament.py) |
| 40 | 🏛️ Political Science & Government | [Federal Register API](#federal-register-api) | 🟢 no key | — | [`federal_register.py`](scripts/politics_government/federal_register.py) |
| 41 | 🏛️ Political Science & Government | [GovInfo API (U.S. Government Publishing Office)](#govinfo-api-us-government-publishing-office) | 🟢 no key | `DATA_GOV_API_KEY` | [`govinfo.py`](scripts/politics_government/govinfo.py) |
| 42 | 🏛️ Political Science & Government | [Grants.gov Search API](#grantsgov-search-api) | 🟢 no key | — | [`grants_gov.py`](scripts/politics_government/grants_gov.py) |
| 43 | 🏛️ Political Science & Government | [Harvard Dataverse (MIT Election Lab & replication data)](#harvard-dataverse-mit-election-lab--replication-data) | 🟢 no key | `DATAVERSE_API_TOKEN (optional)` | [`harvard_dataverse.py`](scripts/politics_government/harvard_dataverse.py) |
| 44 | 🏛️ Political Science & Government | [Open States / Plural (US state legislatures)](#open-states--plural-us-state-legislatures) | 🔑 key | `OPENSTATES_API_KEY` | [`open_states.py`](scripts/politics_government/open_states.py) |
| 45 | 🏛️ Political Science & Government | [OpenFEC (Federal Election Commission)](#openfec-federal-election-commission) | 🟢 no key | `DATA_GOV_API_KEY` | [`openfec.py`](scripts/politics_government/openfec.py) |
| 46 | 🏛️ Political Science & Government | [Regulations.gov API v4](#regulationsgov-api-v4) | 🟢 no key | `DATA_GOV_API_KEY` | [`regulations_gov.py`](scripts/politics_government/regulations_gov.py) |
| 47 | 🏛️ Political Science & Government | [Lobbying Disclosure Act (LDA) API](#lobbying-disclosure-act-lda-api) | 🟢 no key | `LDA_API_KEY (optional)` | [`senate_lda.py`](scripts/politics_government/senate_lda.py) |
| 48 | 🏛️ Political Science & Government | [UK Parliament APIs](#uk-parliament-apis) | 🟢 no key | — | [`uk_parliament.py`](scripts/politics_government/uk_parliament.py) |
| 49 | 🏛️ Political Science & Government | [USAJOBS Search API (federal jobs)](#usajobs-search-api-federal-jobs) | 🔑 key | `USAJOBS_API_KEY, USAJOBS_EMAIL` | [`usajobs.py`](scripts/politics_government/usajobs.py) |
| 50 | 🏛️ Political Science & Government | [USAspending.gov](#usaspendinggov) | 🟢 no key | — | [`usaspending.py`](scripts/politics_government/usaspending.py) |
| 51 | ⚖️ Law & Justice | [CourtListener (Free Law Project)](#courtlistener-free-law-project) | 🟢 no key | `COURTLISTENER_TOKEN (optional)` | [`courtlistener.py`](scripts/law/courtlistener.py) |
| 52 | ⚖️ Law & Justice | [U.S. Department of Justice News API](#us-department-of-justice-news-api) | 🟢 no key | — | [`doj_press_releases.py`](scripts/law/doj_press_releases.py) |
| 53 | ⚖️ Law & Justice | [eCFR API (Code of Federal Regulations)](#ecfr-api-code-of-federal-regulations) | 🟢 no key | — | [`ecfr.py`](scripts/law/ecfr.py) |
| 54 | ⚖️ Law & Justice | [FBI Crime Data Explorer API](#fbi-crime-data-explorer-api) | 🟢 no key | `DATA_GOV_API_KEY` | [`fbi_crime_data.py`](scripts/law/fbi_crime_data.py) |
| 55 | ⚖️ Law & Justice | [Oyez (U.S. Supreme Court cases)](#oyez-us-supreme-court-cases) | 🟢 no key | — | [`oyez.py`](scripts/law/oyez.py) |
| 56 | ⚖️ Law & Justice | [Supreme Court Database (SCDB)](#supreme-court-database-scdb) | 🟢 no key | — | [`supreme_court_database.py`](scripts/law/supreme_court_database.py) |
| 57 | 🩺 Health & Social Science | [CDC Open Data (Socrata SODA)](#cdc-open-data-socrata-soda) | 🟢 no key | `SOCRATA_APP_TOKEN (optional)` | [`cdc_socrata.py`](scripts/health/cdc_socrata.py) |
| 58 | 🩺 Health & Social Science | [ClinicalTrials.gov API v2](#clinicaltrialsgov-api-v2) | 🟢 no key | — | [`clinicaltrials.py`](scripts/health/clinicaltrials.py) |
| 59 | 🩺 Health & Social Science | [openFDA](#openfda) | 🟢 no key | `OPENFDA_API_KEY (optional)` | [`openfda.py`](scripts/health/openfda.py) |
| 60 | 🩺 Health & Social Science | [Our World in Data (Grapher API)](#our-world-in-data-grapher-api) | 🟢 no key | — | [`our_world_in_data.py`](scripts/health/our_world_in_data.py) |
| 61 | 🩺 Health & Social Science | [WHO Global Health Observatory (GHO OData)](#who-global-health-observatory-gho-odata) | 🟢 no key | — | [`who_gho.py`](scripts/health/who_gho.py) |
| 62 | 🎓 Education | [College Scorecard (US Dept. of Education)](#college-scorecard-us-dept-of-education) | 🟢 no key | `DATA_GOV_API_KEY` | [`college_scorecard.py`](scripts/education/college_scorecard.py) |
| 63 | 🎓 Education | [ERIC (Education Resources Information Center)](#eric-education-resources-information-center) | 🟢 no key | — | [`eric.py`](scripts/education/eric.py) |
| 64 | 🎓 Education | [Hipolabs Universities List](#hipolabs-universities-list) | 🟢 no key | — | [`hipolabs_universities.py`](scripts/education/hipolabs_universities.py) |
| 65 | 🎓 Education | [NAEP — The Nation's Report Card](#naep--the-nations-report-card) | 🟢 no key | — | [`naep.py`](scripts/education/naep.py) |
| 66 | 🎓 Education | [UNESCO Institute for Statistics (UIS) Data API](#unesco-institute-for-statistics-uis-data-api) | 🟢 no key | — | [`unesco_uis.py`](scripts/education/unesco_uis.py) |
| 67 | 🎓 Education | [Urban Institute Education Data Portal](#urban-institute-education-data-portal) | 🟢 no key | — | [`urban_education_data.py`](scripts/education/urban_education_data.py) |
| 68 | 🔬 Science & Research | [arXiv API](#arxiv-api) | 🟢 no key | — | [`arxiv.py`](scripts/science_research/arxiv.py) |
| 69 | 🔬 Science & Research | [Crossref REST API](#crossref-rest-api) | 🟢 no key | — | [`crossref.py`](scripts/science_research/crossref.py) |
| 70 | 🔬 Science & Research | [NASA Open APIs (APOD, NeoWs, DONKI, EONET...)](#nasa-open-apis-apod-neows-donki-eonet) | 🟢 no key | `DATA_GOV_API_KEY` | [`nasa.py`](scripts/science_research/nasa.py) |
| 71 | 🔬 Science & Research | [OpenAlex (scholarly works)](#openalex-scholarly-works) | 🟢 no key | — | [`openalex.py`](scripts/science_research/openalex.py) |
| 72 | 🔬 Science & Research | [PubChem PUG-REST](#pubchem-pug-rest) | 🟢 no key | — | [`pubchem.py`](scripts/science_research/pubchem.py) |
| 73 | 🔬 Science & Research | [Semantic Scholar Academic Graph](#semantic-scholar-academic-graph) | 🟢 no key | `S2_API_KEY (optional)` | [`semantic_scholar.py`](scripts/science_research/semantic_scholar.py) |
| 74 | 🤖 AI & Machine Learning | [Anthropic Claude API](#anthropic-claude-api) | 🔑 key | `ANTHROPIC_API_KEY` | [`anthropic_claude.py`](scripts/ai_ml/anthropic_claude.py) |
| 75 | 🤖 AI & Machine Learning | [Epoch AI — Data on AI Models](#epoch-ai--data-on-ai-models) | 🟢 no key | — | [`epoch_ai.py`](scripts/ai_ml/epoch_ai.py) |
| 76 | 🤖 AI & Machine Learning | [Google Gemini API](#google-gemini-api) | 🔑 key | `GEMINI_API_KEY` | [`google_gemini.py`](scripts/ai_ml/google_gemini.py) |
| 77 | 🤖 AI & Machine Learning | [Hugging Face Daily Papers](#hugging-face-daily-papers) | 🟢 no key | — | [`hf_daily_papers.py`](scripts/ai_ml/hf_daily_papers.py) |
| 78 | 🤖 AI & Machine Learning | [Hugging Face Inference Providers](#hugging-face-inference-providers) | 🟢 no key | `HF_TOKEN (needed for chat completions)` | [`hf_inference_providers.py`](scripts/ai_ml/hf_inference_providers.py) |
| 79 | 🤖 AI & Machine Learning | [Kaggle API](#kaggle-api) | 🔑 key | `KAGGLE_USERNAME, KAGGLE_KEY` | [`kaggle.py`](scripts/ai_ml/kaggle.py) |
| 80 | 🤖 AI & Machine Learning | [Ollama (local LLMs)](#ollama-local-llms) | 🟢 no key | `OLLAMA_MODEL (optional, default llama3.2)` | [`ollama.py`](scripts/ai_ml/ollama.py) |
| 81 | 🤖 AI & Machine Learning | [OpenAI API](#openai-api) | 🔑 key | `OPENAI_API_KEY` | [`openai.py`](scripts/ai_ml/openai.py) |
| 82 | 🤖 AI & Machine Learning | [OpenML](#openml) | 🟢 no key | `OPENML_API_KEY (optional)` | [`openml.py`](scripts/ai_ml/openml.py) |
| 83 | 🤖 AI & Machine Learning | [OpenRouter](#openrouter) | 🟢 no key | `OPENROUTER_API_KEY (needed for chat completions)` | [`openrouter.py`](scripts/ai_ml/openrouter.py) |
| 84 | 🤖 AI & Machine Learning | [UCI Machine Learning Repository](#uci-machine-learning-repository) | 🟢 no key | — | [`uci_ml_repo.py`](scripts/ai_ml/uci_ml_repo.py) |
| 85 | ⚙️ Engineering, Energy & Tech | [US Energy Information Administration (EIA) v2](#us-energy-information-administration-eia-v2) | 🔑 key | `EIA_API_KEY` | [`eia.py`](scripts/engineering_energy/eia.py) |
| 86 | ⚙️ Engineering, Energy & Tech | [GitHub REST API](#github-rest-api) | 🟢 no key | `GITHUB_TOKEN (optional)` | [`github.py`](scripts/engineering_energy/github.py) |
| 87 | ⚙️ Engineering, Energy & Tech | [Hugging Face Hub API](#hugging-face-hub-api) | 🟢 no key | `HF_TOKEN (optional)` | [`huggingface.py`](scripts/engineering_energy/huggingface.py) |
| 88 | ⚙️ Engineering, Energy & Tech | [NLR (formerly NREL) Developer Network — solar, EV charging](#nlr-formerly-nrel-developer-network--solar-ev-charging) | 🟢 no key | `DATA_GOV_API_KEY` | [`nrel.py`](scripts/engineering_energy/nrel.py) |
| 89 | ⚙️ Engineering, Energy & Tech | [NIST National Vulnerability Database (CVE API 2.0)](#nist-national-vulnerability-database-cve-api-20) | 🟢 no key | `NVD_API_KEY (optional)` | [`nvd_cve.py`](scripts/engineering_energy/nvd_cve.py) |
| 90 | ⚙️ Engineering, Energy & Tech | [USGS Water Services (streamflow)](#usgs-water-services-streamflow) | 🟢 no key | — | [`usgs_water.py`](scripts/engineering_energy/usgs_water.py) |
| 91 | 🗺️ Transportation & Geospatial | [Bureau of Transportation Statistics (BTS) Open Data](#bureau-of-transportation-statistics-bts-open-data) | 🟢 no key | `SOCRATA_APP_TOKEN (optional)` | [`bts_open_data.py`](scripts/transportation_geo/bts_open_data.py) |
| 92 | 🗺️ Transportation & Geospatial | [US Census Geocoder](#us-census-geocoder) | 🟢 no key | — | [`census_geocoder.py`](scripts/transportation_geo/census_geocoder.py) |
| 93 | 🗺️ Transportation & Geospatial | [Chicago Taxi & Rideshare Trips](#chicago-taxi--rideshare-trips) | 🟢 no key | `SOCRATA_APP_TOKEN (optional)` | [`chicago_taxi.py`](scripts/transportation_geo/chicago_taxi.py) |
| 94 | 🗺️ Transportation & Geospatial | [GBFS Bike-share Feeds (Capital Bikeshare and 1,000+ systems)](#gbfs-bike-share-feeds-capital-bikeshare-and-1000-systems) | 🟢 no key | — | [`gbfs_bikeshare.py`](scripts/transportation_geo/gbfs_bikeshare.py) |
| 95 | 🗺️ Transportation & Geospatial | [MBTA V3 API (Boston transit)](#mbta-v3-api-boston-transit) | 🟢 no key | `MBTA_API_KEY (optional)` | [`mbta.py`](scripts/transportation_geo/mbta.py) |
| 96 | 🗺️ Transportation & Geospatial | [NHTSA Recalls, Complaints & Safety Ratings](#nhtsa-recalls-complaints--safety-ratings) | 🟢 no key | — | [`nhtsa_safety.py`](scripts/transportation_geo/nhtsa_safety.py) |
| 97 | 🗺️ Transportation & Geospatial | [NHTSA vPIC (vehicle VIN decoder)](#nhtsa-vpic-vehicle-vin-decoder) | 🟢 no key | — | [`nhtsa_vpic.py`](scripts/transportation_geo/nhtsa_vpic.py) |
| 98 | 🗺️ Transportation & Geospatial | [OpenStreetMap Nominatim (geocoding)](#openstreetmap-nominatim-geocoding) | 🟢 no key | — | [`nominatim.py`](scripts/transportation_geo/nominatim.py) |
| 99 | 🗺️ Transportation & Geospatial | [OpenSky Network (live flights)](#opensky-network-live-flights) | 🟢 no key | — | [`opensky.py`](scripts/transportation_geo/opensky.py) |
| 100 | 🗺️ Transportation & Geospatial | [OSRM (routing on OpenStreetMap)](#osrm-routing-on-openstreetmap) | 🟢 no key | — | [`osrm.py`](scripts/transportation_geo/osrm.py) |
| 101 | 🗺️ Transportation & Geospatial | [OpenStreetMap Overpass API](#openstreetmap-overpass-api) | 🟢 no key | — | [`overpass.py`](scripts/transportation_geo/overpass.py) |
| 102 | 🗺️ Transportation & Geospatial | [Transitland (GTFS transit data worldwide)](#transitland-gtfs-transit-data-worldwide) | 🔑 key | `TRANSITLAND_API_KEY` | [`transitland.py`](scripts/transportation_geo/transitland.py) |
| 103 | 🗺️ Transportation & Geospatial | [WMATA (DC Metro) API](#wmata-dc-metro-api) | 🔑 key | `WMATA_API_KEY` | [`wmata.py`](scripts/transportation_geo/wmata.py) |
| 104 | 🏅 Sports | [BALLDONTLIE (NBA, NFL, MLB, ...)](#balldontlie-nba-nfl-mlb-) | 🔑 key | `BALLDONTLIE_API_KEY` | [`balldontlie.py`](scripts/sports/balldontlie.py) |
| 105 | 🏅 Sports | [ESPN Site API (unofficial)](#espn-site-api-unofficial) | 🟢 no key | — | [`espn.py`](scripts/sports/espn.py) |
| 106 | 🏅 Sports | [football-data.org (soccer)](#football-dataorg-soccer) | 🔑 key | `FOOTBALL_DATA_TOKEN` | [`football_data.py`](scripts/sports/football_data.py) |
| 107 | 🏅 Sports | [Jolpica-F1 (Ergast-compatible F1 history)](#jolpica-f1-ergast-compatible-f1-history) | 🟢 no key | — | [`jolpica_f1.py`](scripts/sports/jolpica_f1.py) |
| 108 | 🏅 Sports | [Lichess API (chess)](#lichess-api-chess) | 🟢 no key | — | [`lichess.py`](scripts/sports/lichess.py) |
| 109 | 🏅 Sports | [MLB Stats API](#mlb-stats-api) | 🟢 no key | — | [`mlb_stats.py`](scripts/sports/mlb_stats.py) |
| 110 | 🏅 Sports | [NHL Web API (unofficial)](#nhl-web-api-unofficial) | 🟢 no key | — | [`nhl.py`](scripts/sports/nhl.py) |
| 111 | 🏅 Sports | [OpenF1 (Formula 1 telemetry)](#openf1-formula-1-telemetry) | 🟢 no key | — | [`openf1.py`](scripts/sports/openf1.py) |
| 112 | 🏅 Sports | [TheSportsDB](#thesportsdb) | 🔑 key | `THESPORTSDB_KEY (optional)` | [`thesportsdb.py`](scripts/sports/thesportsdb.py) |
| 113 | 🎬 Culture, Books & Entertainment | [The Met Museum Collection API](#the-met-museum-collection-api) | 🟢 no key | — | [`met_museum.py`](scripts/culture_misc/met_museum.py) |
| 114 | 🎬 Culture, Books & Entertainment | [Open Library (books)](#open-library-books) | 🟢 no key | — | [`open_library.py`](scripts/culture_misc/open_library.py) |
| 115 | 🎬 Culture, Books & Entertainment | [TMDB (The Movie Database)](#tmdb-the-movie-database) | 🔑 key | `TMDB_TOKEN` | [`tmdb.py`](scripts/culture_misc/tmdb.py) |

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

**Endpoint:** `GET https://api.gdeltproject.org/api/v2/doc/doc`

| Parameter | Description | Example |
|:--|:--|:--|
| `query` | Search terms plus operators: sourcecountry:, sourcelang:, domain:, theme:, tone<, "exact phrase" | `"artificial intelligence" sourcecountry:US` |
| `mode` | artlist \| timelinevol \| timelinetone \| timelinesourcecountry \| tonechart | `artlist` |
| `format` | json \| csv \| html \| rss | `json` |
| `maxrecords` | Articles returned in artlist mode (max 250) | `5` |
| `timespan` | Relative window ending now (min, h, d, w, m) | `1w` |
| `startdatetime / enddatetime` | Absolute window, YYYYMMDDHHMMSS (last ~3 months) | `20260901000000` |
| `sort` | datedesc \| dateasc \| tonedesc \| toneasc \| hybridrel | `datedesc` |

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

**Endpoint:** `GET https://content.guardianapis.com/search`

| Parameter | Description | Example |
|:--|:--|:--|
| `q` | Query with AND / OR / NOT and "phrases" | `election` |
| `section` | Section ID (list via /sections) | `politics` |
| `tag` | Tag ID(s), comma = AND, \| = OR (list via /tags) | `politics/politics` |
| `from-date / to-date` | Publication date range, YYYY-MM-DD | `2020-01-01` |
| `order-by` | newest \| oldest \| relevance | `newest` |
| `show-fields` | Extra fields: bodyText, body, headline, byline, wordcount, thumbnail | `wordcount,bodyText` |
| `page-size / page` | Results per page (max 200) and page number | `5` |
| `api-key` | Your key (required) | `<GUARDIAN_API_KEY>` |

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

**Endpoint:** `GET https://hacker-news.firebaseio.com/v0/{resource}.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `topstories / newstories / beststories` | Up to 500 current story IDs | `topstories` |
| `askstories / showstories / jobstories` | Up to 200 IDs of each story type | `askstories` |
| `item/{id}` | A story, comment, poll or job (`kids` = child comment IDs) | `item/8863` |
| `user/{id}` | User profile: karma, created, submitted item IDs | `user/pg` |
| `maxitem` | Largest item ID so far (walk backwards to crawl everything) | `maxitem` |
| `updates` | Recently changed items and profiles | `updates` |

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

**Endpoint:** `GET https://newsapi.org/v2/top-headlines  ·  GET https://newsapi.org/v2/everything`

| Parameter | Description | Example |
|:--|:--|:--|
| `country` | 2-letter country code (top-headlines only) | `us` |
| `category` | business \| entertainment \| general \| health \| science \| sports \| technology (top-headlines) | `business` |
| `sources` | Comma-separated source IDs; can't be combined with country/category | `bbc-news,reuters` |
| `q` | Keywords or phrase; /everything supports AND / OR / NOT | `"climate change" AND policy` |
| `from / to` | ISO-8601 date range (/everything only) | `2026-10-01` |
| `language` | 2-letter language code (/everything) | `en` |
| `sortBy` | relevancy \| popularity \| publishedAt (/everything) | `publishedAt` |
| `pageSize / page` | Results per page (max 100) and page number | `5` |

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

**Endpoint:** `GET https://api.nytimes.com/svc/search/v2/articlesearch.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `q` | Full-text query | `supreme court` |
| `fq` | Filter query on fields such as section, type of material, byline (field names per the docs) | `—` |
| `begin_date / end_date` | Date range, YYYYMMDD | `20260101` |
| `sort` | newest \| oldest \| relevance | `newest` |
| `page` | Page number, 10 results each (max page 100) | `0` |
| `api-key` | Your key (required) | `<NYT_API_KEY>` |

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

**Endpoint:** `GET https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}`

| Parameter | Description | Example |
|:--|:--|:--|
| `project` | Wiki project | `en.wikipedia` |
| `access` | all-access \| desktop \| mobile-app \| mobile-web | `all-access` |
| `agent` | all-agents \| user \| spider \| automated | `user` |
| `article` | Article title with underscores (URL-encoded) | `Artificial_intelligence` |
| `granularity` | daily \| monthly | `daily` |
| `start / end` | YYYYMMDD (or YYYYMMDDHH) | `20260901 / 20260907` |
| `top/{project}/{access}/{year}/{month}/{day}` | Sibling endpoint: 1,000 most-viewed articles that day | `top/en.wikipedia/all-access/2026/09/01` |

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

**Endpoint:** `GET https://public.api.bsky.app/xrpc/{method}`

| Parameter | Description | Example |
|:--|:--|:--|
| `app.bsky.actor.getProfile?actor=` | Handle or DID | `bsky.app` |
| `app.bsky.feed.getAuthorFeed?actor=&limit=&cursor=` | A user's posts; limit 1-100, cursor for paging | `limit=3` |
| `filter (getAuthorFeed)` | posts_with_replies \| posts_no_replies \| posts_with_media \| posts_and_author_threads | `posts_no_replies` |
| `app.bsky.graph.getFollowers?actor=&limit=&cursor=` | Follower list (also getFollows) | `actor=bsky.app` |
| `app.bsky.feed.getPostThread?uri=` | A post and its reply tree (at:// URI) | `at://did:plc:…/app.bsky.feed.post/…` |
| `app.bsky.feed.searchPosts?q=&sort=` | Post search, sort top \| latest (requires an authenticated session) | `q=climate` |

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

**Endpoint:** `GET https://{instance}/api/v1/trends/tags`

| Parameter | Description | Example |
|:--|:--|:--|
| `instance` | Any Mastodon server (host part of the URL) | `mastodon.social` |
| `limit` | Number of tags (max 20) | `5` |
| `offset` | Skip N results for paging | `0` |
| `/api/v1/trends/statuses` | Sibling endpoint: trending posts | `limit=20` |
| `/api/v1/timelines/public?local=&limit=` | Public timeline; local=true for this server only, limit max 40 | `local=true&limit=40` |
| `/api/v1/timelines/tag/{hashtag}` | Posts with a hashtag | `/api/v1/timelines/tag/climate` |

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

**Endpoint:** `GET https://oauth.reddit.com/r/{subreddit}/{sort}  (token: POST https://www.reddit.com/api/v1/access_token)`

| Parameter | Description | Example |
|:--|:--|:--|
| `subreddit` | Subreddit name (path) | `datascience` |
| `sort` | hot \| new \| top \| rising \| controversial (path) | `top` |
| `t` | Time window for top/controversial: hour \| day \| week \| month \| year \| all | `week` |
| `limit` | Items per request (max 100) | `5` |
| `after / before` | Fullname cursor for paging (from the previous response) | `t3_1abc23` |
| `grant_type` | Token request body; client_credentials = read-only app access | `client_credentials` |
| `/r/{sub}/comments/{id}` | Sibling endpoint: a post with its full comment tree | `/r/datascience/comments/1abc23` |

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

**Endpoint:** `GET https://api.stackexchange.com/2.3/questions`

| Parameter | Description | Example |
|:--|:--|:--|
| `site` | Which site: stackoverflow, stats (Cross Validated), datascience, ... | `stackoverflow` |
| `tagged` | Semicolon-separated tags (AND) | `pandas` |
| `sort` | activity \| votes \| creation \| hot \| week \| month | `votes` |
| `order` | desc \| asc | `desc` |
| `fromdate / todate` | Unix epoch seconds | `1704067200` |
| `pagesize / page` | Results per page (max 100) and page number | `5` |
| `filter` | Named filter controlling returned fields; withbody adds question body | `withbody` |
| `key` | Optional app key (raises quota to 10,000/day) | `<key>` |

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

**Endpoint:** `GET https://www.googleapis.com/youtube/v3/search  ·  GET https://www.googleapis.com/youtube/v3/videos`

| Parameter | Description | Example |
|:--|:--|:--|
| `part` | Resource parts: snippet (search); snippet,statistics,contentDetails (videos) | `snippet,statistics` |
| `q` | Search query (search) | `neural network` |
| `type` | video \| channel \| playlist (search) | `video` |
| `maxResults` | 0-50 | `3` |
| `order` | relevance \| date \| viewCount \| rating (search) | `viewCount` |
| `publishedAfter / publishedBefore` | RFC 3339 timestamps (search) | `2026-01-01T00:00:00Z` |
| `regionCode / relevanceLanguage` | Country / language bias (search) | `US / en` |
| `id` | Comma-separated video IDs (videos) | `aircAruvnKk` |
| `key` | Your API key | `<YOUTUBE_API_KEY>` |

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

**Endpoint:** `GET https://www.ncei.noaa.gov/cdo-web/api/v2/data`

| Parameter | Description | Example |
|:--|:--|:--|
| `datasetid` | GHCND (daily), GSOM (monthly), GSOY (yearly), ... | `GHCND` |
| `stationid` | Station ID (find via /stations) | `GHCND:USW00013743` |
| `locationid` | Alternative to stationid: FIPS state, ZIP, city | `FIPS:11` |
| `datatypeid` | TMAX, TMIN, PRCP, SNOW, AWND, ... | `TMAX,TMIN` |
| `startdate / enddate` | YYYY-MM-DD (max 1-year span for daily data) | `2025-07-01 / 2025-07-07` |
| `units` | standard \| metric | `metric` |
| `limit / offset` | Rows per page (max 1000) and offset | `1000` |
| `token (header)` | Your CDO token | `<NOAA_TOKEN>` |

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

**Endpoint:** `GET https://api.weather.gov/points/{lat},{lon}  ·  GET https://api.weather.gov/alerts/active`

| Parameter | Description | Example |
|:--|:--|:--|
| `points/{lat},{lon}` | Resolves a location to its forecast office/grid; returns forecast URLs | `38.8977,-77.0365` |
| `forecast / forecastHourly` | URLs returned in points → properties (12-hour periods / hourly) | `properties.forecast` |
| `alerts/active?area=` | State/marine area code | `VA` |
| `alerts/active?severity=` | Extreme \| Severe \| Moderate \| Minor | `Severe` |
| `alerts/active?event=` | Alert event name | `Flood Warning` |
| `stations/{id}/observations/latest` | Latest observation at a station | `KDCA` |
| `User-Agent (header)` | Required: app name + contact | `(myapp, me@gwu.edu)` |

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

**Endpoint:** `GET https://api.open-meteo.com/v1/forecast  ·  GET https://archive-api.open-meteo.com/v1/archive`

| Parameter | Description | Example |
|:--|:--|:--|
| `latitude / longitude` | Location (comma lists allowed for multiple points) | `38.90 / -77.04` |
| `current` | Current-condition variables | `temperature_2m,wind_speed_10m` |
| `hourly` | Hourly variables | `temperature_2m,precipitation` |
| `daily` | Daily aggregates (requires timezone) | `temperature_2m_max,precipitation_sum` |
| `timezone` | IANA zone or auto | `America/New_York` |
| `forecast_days / past_days` | Forecast length 1-16 days / include recent past days | `3 / 7` |
| `temperature_unit / wind_speed_unit / precipitation_unit` | celsius\|fahrenheit, kmh\|ms\|mph\|kn, mm\|inch | `fahrenheit` |
| `start_date / end_date` | Archive endpoint: historical range back to 1940 | `1990-01-01 / 1990-12-31` |

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

**Endpoint:** `GET https://api.openaq.org/v3/locations`

| Parameter | Description | Example |
|:--|:--|:--|
| `coordinates` | lat,lon centre point | `38.9,-77.04` |
| `radius` | Metres around coordinates (max 25,000) | `20000` |
| `bbox` | minLon,minLat,maxLon,maxLat (alternative to coordinates) | `-77.2,38.8,-76.9,39.0` |
| `iso` | Country ISO code | `US` |
| `parameters_id` | Pollutant ID (e.g. 2 = PM2.5); list via /v3/parameters | `2` |
| `limit / page` | Paging | `5` |
| `/v3/sensors/{id}/measurements?datetime_from=&datetime_to=` | Sibling endpoint: time series for one sensor | `datetime_from=2026-10-01` |
| `X-API-Key (header)` | Your key | `<OPENAQ_API_KEY>` |

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

**Endpoint:** `GET https://api.openweathermap.org/data/2.5/weather`

| Parameter | Description | Example |
|:--|:--|:--|
| `q` | City name, state code, country code | `Washington,DC,US` |
| `lat / lon` | Coordinates (alternative to q) | `38.9 / -77.04` |
| `zip` | ZIP code, country code | `20052,US` |
| `units` | standard (Kelvin) \| metric \| imperial | `metric` |
| `lang` | Description language | `en` |
| `appid` | Your API key | `<OPENWEATHER_API_KEY>` |
| `/data/2.5/forecast` | Sibling endpoint: 5-day/3-hour forecast (same params, cnt= to limit) | `cnt=8` |
| `/data/2.5/air_pollution?lat=&lon=` | Sibling endpoint: AQI and pollutant concentrations | `lat=38.9&lon=-77.04` |

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

**Endpoint:** `GET https://earthquake.usgs.gov/fdsnws/event/1/query`

| Parameter | Description | Example |
|:--|:--|:--|
| `format` | geojson \| csv \| xml \| text \| kml | `geojson` |
| `starttime / endtime` | ISO-8601 dates (default: last 30 days) | `2026-01-01` |
| `minmagnitude / maxmagnitude` | Magnitude range | `5` |
| `latitude / longitude / maxradiuskm` | Circle search | `38.9 / -77.0 / 500` |
| `minlatitude / maxlatitude / minlongitude / maxlongitude` | Bounding-box search | `24 / 50 / -125 / -66` |
| `orderby` | time \| time-asc \| magnitude \| magnitude-asc | `time` |
| `limit / offset` | Max 20,000 events per request | `5` |

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

**Endpoint:** `GET https://www.alphavantage.co/query`

| Parameter | Description | Example |
|:--|:--|:--|
| `function` | TIME_SERIES_DAILY, TIME_SERIES_INTRADAY, GLOBAL_QUOTE, OVERVIEW, INCOME_STATEMENT, NEWS_SENTIMENT, SMA, RSI, CPI, ... | `TIME_SERIES_DAILY` |
| `symbol` | Ticker | `IBM` |
| `interval` | Intraday: 1min \| 5min \| 15min \| 30min \| 60min | `5min` |
| `outputsize` | compact (latest 100 points) \| full | `compact` |
| `datatype` | json \| csv | `json` |
| `tickers / topics` | NEWS_SENTIMENT filters | `AAPL / technology` |
| `apikey` | Your key (demo works for documented IBM examples) | `demo` |

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

**Endpoint:** `GET https://api.bls.gov/publicAPI/v1/timeseries/data/{seriesId}  ·  POST https://api.bls.gov/publicAPI/v2/timeseries/data/`

| Parameter | Description | Example |
|:--|:--|:--|
| `seriesId / seriesid` | Series ID in the path (GET) or a JSON list (POST, up to 50 with v2) | `CUUR0000SA0, LNS14000000` |
| `startyear / endyear` | Year range (POST body; v2 up to 20 years) | `2015 / 2025` |
| `registrationkey` | v2 key (POST body) | `<BLS_API_KEY>` |
| `calculations` | true → net and percent changes (v2) | `true` |
| `annualaverage` | true → include annual averages (v2) | `true` |
| `catalog` | true → series titles and metadata (v2) | `true` |

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

**Endpoint:** `GET https://api.coingecko.com/api/v3/coins/markets`

| Parameter | Description | Example |
|:--|:--|:--|
| `vs_currency` | Quote currency | `usd` |
| `ids` | Comma-separated coin IDs (list via /coins/list) | `bitcoin,ethereum` |
| `order` | market_cap_desc \| volume_desc \| id_asc ... | `market_cap_desc` |
| `per_page / page` | Max 250 per page | `5` |
| `price_change_percentage` | Extra change windows: 1h,24h,7d,14d,30d,200d,1y | `1h,7d` |
| `/simple/price?ids=&vs_currencies=` | Sibling endpoint: quick spot prices | `ids=bitcoin&vs_currencies=usd` |
| `/coins/{id}/market_chart?vs_currency=&days=` | Sibling endpoint: historical price/market cap/volume | `bitcoin, days=365` |
| `x-cg-demo-api-key (header)` | Optional free Demo key | `<key>` |

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

**Endpoint:** `GET https://api.frankfurter.dev/v1/{latest \| YYYY-MM-DD \| start..end}`

| Parameter | Description | Example |
|:--|:--|:--|
| `path` | latest, a single date, or a range start..end (end optional) | `2026-09-01..2026-09-05` |
| `base` | Base currency (default EUR) | `USD` |
| `symbols` | Comma-separated target currencies | `EUR,GBP,JPY` |
| `amount` | Convert this amount | `100` |
| `/v1/currencies` | Sibling endpoint: supported currency codes | `—` |

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

**Endpoint:** `GET https://api.stlouisfed.org/fred/series/observations`

| Parameter | Description | Example |
|:--|:--|:--|
| `series_id` | Series code (UNRATE, GDP, CPIAUCSL, FEDFUNDS, DGS10, ...) | `UNRATE` |
| `api_key / file_type` | Your key; file_type json \| xml | `<FRED_API_KEY> / json` |
| `observation_start / observation_end` | Date range YYYY-MM-DD | `2000-01-01` |
| `frequency` | Aggregate to d \| w \| m \| q \| sa \| a | `q` |
| `aggregation_method` | avg \| sum \| eop (end of period) | `avg` |
| `units` | lin \| chg \| ch1 \| pch \| pc1 \| pca \| log (levels, changes, % changes) | `pc1` |
| `sort_order / limit / offset` | asc \| desc; max 100,000 rows | `desc / 5` |
| `/fred/series/search?search_text=` | Sibling endpoint: find series IDs by keyword | `search_text=housing starts` |

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

**Endpoint:** `GET https://data.sec.gov/…`

| Parameter | Description | Example |
|:--|:--|:--|
| `submissions/CIK{10-digit CIK}.json` | Company profile + recent filings list | `CIK0000320193` |
| `api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json` | One financial concept over time | `CIK0000320193/us-gaap/NetIncomeLoss` |
| `api/xbrl/companyfacts/CIK{cik}.json` | Every XBRL fact a company has reported | `CIK0000320193` |
| `api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json` | One concept across all companies for a period | `us-gaap/Revenues/USD/CY2024` |
| `https://www.sec.gov/files/company_tickers.json` | Ticker → CIK lookup table | `AAPL → 320193` |
| `User-Agent (header)` | Required: name + email | `GWU Capstone you@gwu.edu` |

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

**Endpoint:** `GET https://api.fiscaldata.treasury.gov/services/api/fiscal_service/{endpoint}`

| Parameter | Description | Example |
|:--|:--|:--|
| `endpoint` | Dataset path (see the dataset catalog on the docs page) | `v2/accounting/od/debt_to_penny` |
| `fields` | Comma-separated columns | `record_date,tot_pub_debt_out_amt` |
| `filter` | field:op:value, ops eq gt gte lt lte in; comma = AND | `record_date:gte:2026-01-01` |
| `sort` | Column, prefix - for descending | `-record_date` |
| `page[size] / page[number]` | Paging (default 100) | `5` |
| `format` | json \| csv \| xml | `json` |

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

**Endpoint:** `GET https://api.worldbank.org/v2/country/{codes}/indicator/{indicator}`

| Parameter | Description | Example |
|:--|:--|:--|
| `country codes` | ISO3 codes separated by ;, or all | `USA;CHN;IND` |
| `indicator` | Indicator code (NY.GDP.PCAP.CD, SP.POP.TOTL, SE.XPD.TOTL.GD.ZS, EN.GHG.CO2.PC.CE.AR5, ...) | `NY.GDP.PCAP.CD` |
| `format` | json \| xml | `json` |
| `date` | Year or range | `2000:2023` |
| `mrv` | Most recent N values (instead of date) | `5` |
| `per_page / page` | Paging (default 50) | `50` |
| `/v2/indicator?source=2` | Sibling endpoint: browse indicator codes (source 2 = WDI) | `per_page=100` |

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

**Endpoint:** `GET https://api.ipums.org/metadata/samples  ·  POST https://api.ipums.org/extracts`

| Parameter | Description | Example |
|:--|:--|:--|
| `collection` | usa \| cps \| ipumsi \| atus \| nhgis \| ... | `usa` |
| `version` | API version | `2` |
| `pageSize / pageNumber` | Paging | `5` |
| `extract body: samples` | Sample IDs to include | `{"us2022a": {}}` |
| `extract body: variables` | Variables to include | `{"AGE": {}, "SEX": {}, "INCTOT": {}}` |
| `extract body: dataFormat` | csv \| fixed_width \| stata \| spss | `csv` |
| `Authorization (header)` | Your API key | `<IPUMS_API_KEY>` |

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

**Endpoint:** `GET https://population.un.org/dataportalapi/api/v1/data/indicators/{ids}/locations/{ids}/start/{year}/end/{year}`

| Parameter | Description | Example |
|:--|:--|:--|
| `indicators` | Indicator ID(s); list via /api/v1/indicators | `49 (total population)` |
| `locations` | UN M49 country codes; list via /api/v1/locations | `566 (Nigeria), 840 (USA)` |
| `start / end` | Year range (estimates 1950-2023, projections to 2100) | `2025 / 2050` |
| `pageNumber / pageSize` | Paging | `1 / 100` |
| `Authorization (header)` | Bearer token | `Bearer <UN_POP_TOKEN>` |

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

**Endpoint:** `GET https://api.census.gov/data/{year}/{dataset}`

| Parameter | Description | Example |
|:--|:--|:--|
| `year / dataset` | Vintage + dataset path: acs/acs5, acs/acs1, dec/pl, pep/population, cbp, ... | `2023/acs/acs5` |
| `get` | Comma-separated variables (max 50) or group(TABLE) | `NAME,B01003_001E,B19013_001E` |
| `for` | Target geography with FIPS code or * | `state:11, county:*, tract:*` |
| `in` | Parent geography (required for counties/tracts within a state) | `state:11 county:001` |
| `ucgid` | Alternative geography selector by GEOID | `0400000US11` |
| `key` | Your API key | `<CENSUS_API_KEY>` |
| `/data/{year}/{dataset}/variables.json` | Sibling endpoint: every variable code and label | `2023/acs/acs5/variables.json` |

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

## 🏠 Housing & Real Estate

### CFPB Consumer Complaint Database

| | |
|:--|:--|
| **Docs** | https://cfpb.github.io/api/ccdb/api.html |
| **Auth** | None |
| **Register / token** | No registration. 5M+ complaints about mortgages, credit reports, debt collection, banks, student loans... |
| **Script** | [`scripts/housing/cfpb_complaints.py`](scripts/housing/cfpb_complaints.py) |

**What you can get:**

- Each complaint: product, sub-product, issue, company, state, ZIP, date, company response, timeliness
- Consumer narrative text (when the consumer consented)
- Built-in aggregations by issue, company, state, product

**Capstone ideas:** NLP on complaint narratives, ranking mortgage servicers, linking complaints to local economic stress.

**Endpoint:** `GET https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/`

| Parameter | Description | Example |
|:--|:--|:--|
| `product / sub_product` | Product filter (repeatable) | `Mortgage` |
| `state / zip_code / company` | Location or company filter (repeatable) | `DC` |
| `date_received_min / date_received_max` | Date range YYYY-MM-DD | `2025-01-01` |
| `search_term` | Full-text search across complaint fields, including narratives | `foreclosure` |
| `has_narrative` | true -> only complaints with published narratives | `true` |
| `size / frm / sort` | Page size, offset, created_date_desc \| relevance_desc | `5 / 0 / created_date_desc` |
| `format` | json \| csv (with no_aggs=true for bulk export) | `csv` |

```python
from _common import get, show


def main():
    d = get("https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/", params={
        "product": "Mortgage", "state": "DC", "date_received_min": "2025-01-01", "size": 3, "sort": "created_date_desc"})
    issues = d["aggregations"]["issue"]["issue"]["buckets"]
    show({"mortgage_complaints_DC_since_2025": d["hits"]["total"]["value"],
          "top_issues": [{"issue": b["key"], "complaints": b["doc_count"]} for b in issues[:4]],
          "latest": [{k: h["_source"][k] for k in ("date_received", "company", "issue", "company_response")}
                     for h in d["hits"]["hits"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 21:03</summary>

```json
{
  "mortgage_complaints_DC_since_2025": 198,
  "top_issues": [
    {
      "issue": "Trouble during payment process",
      "complaints": 93
    },
    {
      "issue": "Struggling to pay mortgage",
      "complaints": 54
    },
    {
      "issue": "Applying for a mortgage or refinancing an existing mortgage",
      "complaints": 26
    },
    {
      "issue": "Closing on a mortgage",
      "complaints": 18
    }
  ],
  "latest": [
    {
      "date_received": "2026-09-29T12:57:10.000Z",
      "company": "Rocket Mortgage, LLC",
      "issue": "Struggling to pay mortgage",
      "company_response": "Closed with explanation"
    },
    {
      "date_received": "2026-09-25T18:00:53.000Z",
      "company": "Rocket Mortgage, LLC",
      "issue": "Trouble during payment process",
      "company_response": "Closed with explanation"
    },
    {
      "date_received": "2026-09-22T14:33:55.000Z",
      "company": "CREDIT CARD MANAGEMENT SERVICES INC",
      "issue": "Applying for a mortgage or refinancing an existing mortgage",
      "company_response": "Closed with explanation"
    }
  ]
}
```
</details>

---

### FHFA House Price Index

| | |
|:--|:--|
| **Docs** | https://www.fhfa.gov/data/hpi |
| **Auth** | None (CSV download) |
| **Register / token** | No registration; no REST API. The master CSV (~17 MB) holds every published index; smaller files by geography are on the docs page. Many series are also on FRED (needs a FRED key). |
| **Script** | [`scripts/housing/fhfa_hpi.py`](scripts/housing/fhfa_hpi.py) |

**What you can get:**

- Repeat-sales house price indexes since 1975 (purchase-only, all-transactions, expanded-data)
- National, census division, state, metro (MSA), and (annually) county, ZIP and tract levels
- Seasonally adjusted and not adjusted

**Capstone ideas:** Long-run price appreciation by region, house prices vs. local employment, housing-wealth effects.

**Endpoint:** `GET https://www.fhfa.gov/hpi/download/monthly/hpi_master.csv`

| Parameter | Description | Example |
|:--|:--|:--|
| `hpi_type` | traditional \| distress-free \| ... | `traditional` |
| `hpi_flavor` | purchase-only \| all-transactions \| expanded-data | `purchase-only` |
| `frequency` | monthly \| quarterly | `monthly` |
| `level` | USA or Census Division \| State \| MSA \| ... | `USA or Census Division` |
| `place_name / place_id` | Geography name / code | `United States / USA` |
| `yr / period / index_nsa / index_sa` | Year, month or quarter, index values (1991 = 100 for monthly) | `2026 / 7` |

```python
import csv
import io

from _common import get, show


def main():
    rows = csv.DictReader(io.StringIO(get("https://www.fhfa.gov/hpi/download/monthly/hpi_master.csv", as_json=False)))
    us = [r for r in rows if r["place_id"] == "USA" and r["hpi_flavor"] == "purchase-only" and r["frequency"] == "monthly"]
    latest, year_ago = us[-1], us[-13]
    show({"series": "US purchase-only HPI, seasonally adjusted (Jan 1991 = 100)",
          "latest": [{"month": f'{r["yr"]}-{int(r["period"]):02d}', "index_sa": float(r["index_sa"])} for r in us[-4:]],
          "yoy_change_pct": round(100 * (float(latest["index_sa"]) / float(year_ago["index_sa"]) - 1), 2)})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 21:03</summary>

```json
{
  "series": "US purchase-only HPI, seasonally adjusted (Jan 1991 = 100)",
  "latest": [
    {
      "month": "2026-04",
      "index_sa": 440.89
    },
    {
      "month": "2026-05",
      "index_sa": 442.29
    },
    {
      "month": "2026-06",
      "index_sa": 442.34
    },
    {
      "month": "2026-07",
      "index_sa": 443.52
    }
  ],
  "yoy_change_pct": 2.57
}
```
</details>

---

### HMDA Data Browser (mortgage applications)

| | |
|:--|:--|
| **Docs** | https://ffiec.cfpb.gov/documentation/api/data-browser/ |
| **Auth** | None |
| **Register / token** | No registration. Home Mortgage Disclosure Act data: nearly every US mortgage application since 2018, published each year by the CFPB/FFIEC. |
| **Script** | [`scripts/housing/hmda.py`](scripts/housing/hmda.py) |

**What you can get:**

- Loan-level records: action taken (originated/denied), loan amount, purpose, type, interest rate, property value
- Applicant race, ethnicity, sex, age, income; denial reasons; lender (LEI)
- Census tract of the property

**Capstone ideas:** Fair-lending / mortgage-denial disparities, credit access by neighborhood, lender behavior over the rate cycle.

**Endpoint:** `GET https://ffiec.cfpb.gov/v2/data-browser-api/view/{aggregations \| csv}`

| Parameter | Description | Example |
|:--|:--|:--|
| `years` | Data year(s) | `2023` |
| `states / msamds / counties / leis` | Geography or lender filter (one required); msamds uses metro-division codes | `states=DC, msamds=47894` |
| `actions_taken` | 1 originated, 2 approved not accepted, 3 denied, 4 withdrawn, ... (comma list) | `1,3` |
| `races / ethnicities / sexes` | Applicant demographics (comma lists; each value returns its own row). Aggregations accept at most 2 of these filter variables per query | `White,Black or African American,Asian` |
| `loan_purposes / loan_types` | 1 purchase, 2 improvement, 31 refinance, 32 cash-out; 1 conventional, 2 FHA, 3 VA, 4 USDA | `1` |
| `/view/csv` | Same filters, redirects to a CSV file of the matching loan-level rows | `/view/csv?years=2023&states=DC` |

```python
from collections import defaultdict

from _common import get, show


def main():
    d = get("https://ffiec.cfpb.gov/v2/data-browser-api/view/aggregations", params={
        "years": 2023, "states": "DC", "actions_taken": "1,3",
        "races": "White,Black or African American,Asian"})["aggregations"]
    by_race = defaultdict(dict)
    for row in d:  # one row per (action_taken, race)
        by_race[row["races"]]["originated" if row["actions_taken"] == "1" else "denied"] = row["count"]
    # denial rate among decided applications only: denied / (denied + originated)
    show([{"race": r, **c, "denial_rate": round(c["denied"] / (c["denied"] + c["originated"]), 3)}
          for r, c in by_race.items()])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 21:03</summary>

```json
[
  {
    "race": "White",
    "denied": 703,
    "originated": 3779,
    "denial_rate": 0.157
  },
  {
    "race": "Asian",
    "originated": 423,
    "denied": 119,
    "denial_rate": 0.22
  },
  {
    "race": "Black or African American",
    "denied": 1298,
    "originated": 1995,
    "denial_rate": 0.394
  }
]
```
</details>

---

### HUD User API (Fair Market Rents, Income Limits, ZIP crosswalk)

| | |
|:--|:--|
| **Docs** | https://www.huduser.gov/portal/dataset/fmr-api.html |
| **Auth** | Access token (header `Authorization: Bearer`) |
| **Register / token** | Register at https://www.huduser.gov/hudapi/public/register, log in, and create an access token. Free. |
| **Env var** | `HUD_API_TOKEN` |
| **Script** | [`scripts/housing/hud_user.py`](scripts/housing/hud_user.py) |

**What you can get:**

- Fair Market Rents (FMR) by bedroom count for every metro and county; Small Area FMRs by ZIP
- Income limits (very low / extremely low / low income) by household size
- USPS ZIP crosswalk: ZIP <-> tract, county, CBSA, congressional district with address-share ratios

**Capstone ideas:** Rent burden and voucher policy studies, joining ZIP-level data to Census tracts via the crosswalk.

**Endpoint:** `GET https://www.huduser.gov/hudapi/public/{dataset}`

| Parameter | Description | Example |
|:--|:--|:--|
| `fmr/statedata/{state}` | FMRs for every metro and county in a state | `fmr/statedata/DC` |
| `fmr/data/{entityid}` | FMRs for one area (metro code or 10-digit county code) | `fmr/data/METRO47900M47900` |
| `il/data/{entityid}` | Income limits for one area | `il/data/METRO47900M47900` |
| `usps?type=&query=` | ZIP crosswalk; type 1 ZIP-tract, 2 ZIP-county, 3 ZIP-CBSA, ... query = ZIP, or All | `type=1&query=20052` |
| `year` | Fiscal year (default latest) | `2026` |
| `Authorization (header)` | Bearer token | `Bearer <HUD_API_TOKEN>` |

```python
from _common import get, key, show


def main():
    d = get("https://www.huduser.gov/hudapi/public/fmr/statedata/DC",
            headers={"Authorization": f"Bearer {key('HUD_API_TOKEN')}"})["data"]
    show([{k: m.get(k) for k in ("metro_name", "code", "Efficiency", "One-Bedroom", "Two-Bedroom", "Three-Bedroom",
                                 "Four-Bedroom")} for m in d["metroareas"]])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"metro_name": "Washington-Arlington-Alexandria, DC-VA-MD HUD Metro FMR Area", "code": "METRO47900M47900",
   "Efficiency": 1855, "One-Bedroom": 1906, "Two-Bedroom": 2167, "Three-Bedroom": 2687, "Four-Bedroom": 3192}
]
```
</details>

---

### OpenFEMA — Flood Insurance Claims & Disaster Declarations

| | |
|:--|:--|
| **Docs** | https://www.fema.gov/about/openfema/api |
| **Auth** | None |
| **Register / token** | No registration. Note: the older v2 claims endpoint (FimaNfipClaims) is deprecated and removed after Oct 15, 2026 - use v3 NfipClaims as below. |
| **Script** | [`scripts/housing/openfema_nfip.py`](scripts/housing/openfema_nfip.py) |

**What you can get:**

- 2.7M+ National Flood Insurance Program claims: date of loss, flood zone, building/contents payouts, cause, county
- NFIP policies, disaster declarations (every federal disaster since 1953), housing-assistance grants by ZIP
- OData-style filtering, sorting, selection and counts

**Capstone ideas:** Climate risk and home values, flood-zone mapping accuracy, disaster recovery and housing assistance equity.

**Endpoint:** `GET https://www.fema.gov/api/open/v3/NfipClaims`

| Parameter | Description | Example |
|:--|:--|:--|
| `$filter` | OData filter (eq, gt, lt, and, or); add amountPaidOnBuildingClaim gt 0 before sorting by payout (nulls sort first) | `state eq 'VA' and yearOfLoss eq 2024` |
| `$select` | Columns to return | `dateOfLoss,countyCode,ratedFloodZone,amountPaidOnBuildingClaim` |
| `$orderby` | Sort (desc / asc) | `amountPaidOnBuildingClaim desc` |
| `$top / $skip` | Paging (max 10,000 per call) | `5` |
| `$count` | true -> include the total match count in metadata | `true` |
| `/api/open/v2/DisasterDeclarationsSummaries` | Sibling dataset: every disaster declaration (state, incidentType, dates) | `$filter=state eq 'DC'` |

```python
from _common import get, show


def main():
    d = get("https://www.fema.gov/api/open/v3/NfipClaims", params={
        "$filter": "state eq 'VA' and yearOfLoss eq 2024 and amountPaidOnBuildingClaim gt 0",  # skip unpaid claims
        "$orderby": "amountPaidOnBuildingClaim desc",
        "$select": "dateOfLoss,countyCode,ratedFloodZone,amountPaidOnBuildingClaim,amountPaidOnContentsClaim",
        "$top": 5, "$count": "true"})
    show({"VA_paid_claims_2024": d["metadata"]["count"], "largest_building_payouts": [
        {**c, "dateOfLoss": c["dateOfLoss"][:10]} for c in d["NfipClaims"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 21:03</summary>

```json
{
  "VA_paid_claims_2024": 184,
  "largest_building_payouts": [
    {
      "dateOfLoss": "2024-09-27",
      "countyCode": "51155",
      "ratedFloodZone": "AE",
      "amountPaidOnBuildingClaim": 500000,
      "amountPaidOnContentsClaim": 0
    },
    {
      "dateOfLoss": "2024-08-30",
      "countyCode": "51730",
      "ratedFloodZone": "AE",
      "amountPaidOnBuildingClaim": 397838.64,
      "amountPaidOnContentsClaim": 0
    },
    {
      "dateOfLoss": "2024-09-27",
      "countyCode": "51750",
      "ratedFloodZone": "AE",
      "amountPaidOnBuildingClaim": 379425.03,
      "amountPaidOnContentsClaim": 0
    },
    {
      "dateOfLoss": "2024-09-27",
      "countyCode": "51191",
      "ratedFloodZone": "AE",
      "amountPaidOnBuildingClaim": 231542.96,
      "amountPaidOnContentsClaim": 0
    },
    {
      "dateOfLoss": "2024-09-27",
      "countyCode": "51121",
      "ratedFloodZone": "AE",
      "amountPaidOnBuildingClaim": 220000,
      "amountPaidOnContentsClaim": 100000
    }
  ]
}
```
</details>

---

### Redfin Data Center (housing market tracker)

| | |
|:--|:--|
| **Docs** | https://www.redfin.com/news/data-center/ |
| **Auth** | None (TSV downloads) |
| **Register / token** | No registration; no REST API. Gzipped TSVs on a public S3 bucket, updated monthly (weekly files also available). National file ~0.5 MB; metro/county/ZIP/city files are much larger. Cite Redfin. |
| **Script** | [`scripts/housing/redfin_data_center.py`](scripts/housing/redfin_data_center.py) |

**What you can get:**

- Median sale & list price, price per sq ft, homes sold, pending sales, new listings, inventory, months of supply
- Days on market, sale-to-list ratio, share sold above list, price drops
- By national, state, metro, county, city, ZIP, neighborhood and property type

**Capstone ideas:** Market-cycle detection, inventory vs. price dynamics, mortgage-rate shocks on local markets.

**Endpoint:** `GET https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/{level}_market_tracker.tsv000.gz`

| Parameter | Description | Example |
|:--|:--|:--|
| `level` | us_national \| state \| redfin_metro \| county \| city \| zip_code \| neighborhood | `us_national` |
| `PROPERTY_TYPE` | All Residential \| Single Family Residential \| Condo/Co-op \| Townhouse \| Multi-Family (2-4 Unit) | `All Residential` |
| `IS_SEASONALLY_ADJUSTED` | true \| false | `false` |
| `PERIOD_BEGIN / PERIOD_DURATION` | Month start date; 30 = monthly rows (90-day rolling in some files) | `2026-04-01 / 30` |
| `metrics` | MEDIAN_SALE_PRICE, HOMES_SOLD, INVENTORY, MONTHS_OF_SUPPLY, MEDIAN_DOM, AVG_SALE_TO_LIST, ... (+ _MOM / _YOY) | `—` |

```python
import csv
import gzip
import io

import requests

from _common import USER_AGENT, show

URL = "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/us_national_market_tracker.tsv000.gz"


def main():
    raw = requests.get(URL, headers={"User-Agent": USER_AGENT}, timeout=60).content
    rows = csv.DictReader(io.StringIO(gzip.decompress(raw).decode()), delimiter="\t")
    rows = sorted((r for r in rows if r["PROPERTY_TYPE"] == "All Residential" and r["IS_SEASONALLY_ADJUSTED"] == "false"),
                  key=lambda r: r["PERIOD_BEGIN"])
    show([{"month": r["PERIOD_BEGIN"][:7], "median_sale_price": int(float(r["MEDIAN_SALE_PRICE"])),
           "price_yoy_pct": round(100 * float(r["MEDIAN_SALE_PRICE_YOY"]), 1), "homes_sold": int(float(r["HOMES_SOLD"])),
           "inventory": int(float(r["INVENTORY"])), "months_of_supply": round(float(r["MONTHS_OF_SUPPLY"]), 1)}
          for r in rows[-4:]])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 21:03</summary>

```json
[
  {
    "month": "2026-02",
    "median_sale_price": 429061,
    "price_yoy_pct": 0.8,
    "homes_sold": 320726,
    "inventory": 1336670,
    "months_of_supply": 4.2
  },
  {
    "month": "2026-03",
    "median_sale_price": 436487,
    "price_yoy_pct": 1.1,
    "homes_sold": 417012,
    "inventory": 1393358,
    "months_of_supply": 3.3
  },
  {
    "month": "2026-04",
    "median_sale_price": 442054,
    "price_yoy_pct": 0.9,
    "homes_sold": 446632,
    "inventory": 1468301,
    "months_of_supply": 3.3
  },
  {
    "month": "2026-05",
    "median_sale_price": 449846,
    "price_yoy_pct": 2.2,
    "homes_sold": 464811,
    "inventory": 1460440,
    "months_of_supply": 3.1
  }
]
```
</details>

---

### RentCast (property records, rent & value estimates)

| | |
|:--|:--|
| **Docs** | https://developers.rentcast.io/reference/introduction |
| **Auth** | API key (header `X-Api-Key`) |
| **Register / token** | Create an account at https://app.rentcast.io/app/api and activate the free Developer plan (50 requests/month; paid plans for more). Commercial terms apply - read them before publishing data. |
| **Env var** | `RENTCAST_API_KEY` |
| **Script** | [`scripts/housing/rentcast.py`](scripts/housing/rentcast.py) |

**What you can get:**

- Property records for 140M+ US properties: beds, baths, sq ft, year built, last sale, tax assessments
- Automated rent and value estimates (AVM) with comparable listings
- Active sale/rental listings and ZIP-level market statistics with history

**Capstone ideas:** Rent-prediction models, landlord/investor yield analysis, small-area rent trends.

**Endpoint:** `GET https://api.rentcast.io/v1/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `markets?zipCode=&dataType=&historyRange=` | ZIP market stats; dataType All \| Sale \| Rental; history in months | `zipCode=20052&dataType=Rental` |
| `avm/rent/long-term?address=&propertyType=&bedrooms=` | Rent estimate + comparables for an address | `address=2121 I St NW, Washington, DC 20052` |
| `avm/value?address=` | Value estimate + comparables | `—` |
| `properties?address= / ?zipCode=` | Property records | `zipCode=20052&limit=5` |
| `listings/rental/long-term?city=&state=&status=` | Active rental listings | `city=Washington&state=DC&status=Active` |
| `X-Api-Key (header)` | Your key | `<RENTCAST_API_KEY>` |

```python
from _common import get, key, show


def main():
    d = get("https://api.rentcast.io/v1/markets", headers={"X-Api-Key": key("RENTCAST_API_KEY")},
            params={"zipCode": "20052", "dataType": "All", "historyRange": 6})
    rent, sale = d.get("rentalData", {}), d.get("saleData", {})
    show({"zipCode": d["zipCode"],
          "rental": {k: rent.get(k) for k in ("averageRent", "medianRent", "minRent", "maxRent", "totalListings")},
          "sale": {k: sale.get(k) for k in ("averagePrice", "medianPrice", "totalListings")}})
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "zipCode": "20052",
  "rental": {"averageRent": 2850, "medianRent": 2600, "minRent": 1450, "maxRent": 7900, "totalListings": 214},
  "sale": {"averagePrice": 612000, "medianPrice": 489000, "totalListings": 38}
}
```
</details>

---

### Zillow Research Data (ZHVI home values, ZORI rents)

| | |
|:--|:--|
| **Docs** | https://www.zillow.com/research/data/ |
| **Auth** | None (CSV downloads) |
| **Register / token** | No registration; there is no public REST API - the research page publishes monthly CSVs. Pick a series and geography on the page and copy the CSV link. Cite Zillow per its terms. |
| **Script** | [`scripts/housing/zillow_research.py`](scripts/housing/zillow_research.py) |

**What you can get:**

- ZHVI: typical home value by metro, state, county, city, ZIP, neighborhood (monthly since 2000)
- ZORI: typical observed market rent
- Inventory, new listings, days on market, sale-to-list ratio, forecasts

**Capstone ideas:** Home-price forecasting, rent burden vs. income (join Census ACS), effect of transit or flood risk on values.

**Endpoint:** `GET https://files.zillowstatic.com/research/public_csvs/{series}/{Geography}_{series}_….csv`

| Parameter | Description | Example |
|:--|:--|:--|
| `series folder` | zhvi (home values), zori (rents), invt_fs (inventory), ... | `zhvi` |
| `Geography` | Metro \| State \| County \| City \| Zip \| Neighborhood | `Metro` |
| `file name` | Encodes home type, price tier, smoothing, seasonal adjustment | `Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv` |
| `columns` | RegionID, SizeRank, RegionName, RegionType, StateName, then one column per month | `2026-08-31` |

```python
import csv
import io

from _common import get, show

URL = "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"


def main():
    rows = list(csv.DictReader(io.StringIO(get(URL, as_json=False))))  # ~4.5 MB
    months = [c for c in rows[0] if c[:2] in ("19", "20")]
    out = []
    for r in rows[:6]:  # rows are sorted by SizeRank: US first, then the largest metros
        now, year_ago = float(r[months[-1]]), float(r[months[-13]])
        out.append({"region": r["RegionName"], "typical_home_value": round(now), "yoy_change_pct": round(100 * (now / year_ago - 1), 1)})
    show({"month": months[-1], "metros": out}, max_items=6)
```

<details><summary><b>Sample output</b> — live run 2026-10-11 21:03</summary>

```json
{
  "month": "2026-08-31",
  "metros": [
    {
      "region": "United States",
      "typical_home_value": 368697,
      "yoy_change_pct": 1.2
    },
    {
      "region": "New York, NY",
      "typical_home_value": 735075,
      "yoy_change_pct": 4.7
    },
    {
      "region": "Los Angeles, CA",
      "typical_home_value": 952601,
      "yoy_change_pct": 1.2
    },
    {
      "region": "Chicago, IL",
      "typical_home_value": 357265,
      "yoy_change_pct": 4.9
    },
    {
      "region": "Dallas, TX",
      "typical_home_value": 360437,
      "yoy_change_pct": -2.3
    },
    {
      "region": "Houston, TX",
      "typical_home_value": 305001,
      "yoy_change_pct": -1.9
    }
  ]
}
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

**Endpoint:** `GET https://acleddata.com/api/acled/read  (token: POST https://acleddata.com/oauth/token)`

| Parameter | Description | Example |
|:--|:--|:--|
| `country / iso` | Country name or ISO numeric code | `Kenya / 404` |
| `event_date + event_date_where` | Date filter; BETWEEN takes start\|end | `2026-01-01\|2026-09-30, BETWEEN` |
| `event_type` | Battles \| Protests \| Riots \| Violence against civilians \| Explosions/Remote violence \| Strategic developments | `Protests` |
| `fields` | Pipe-separated columns to return | `event_date\|event_type\|fatalities` |
| `limit / page` | Rows per page (default 5,000) | `5` |
| `_format` | json \| csv | `json` |
| `Authorization (header)` | Bearer token from /oauth/token (valid 24 h) | `Bearer <token>` |

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

**Endpoint:** `GET https://api.congress.gov/v3/bill/{congress}/{type}/{number}`

| Parameter | Description | Example |
|:--|:--|:--|
| `congress` | Congress number (119 = 2025-2026) | `119` |
| `type` | hr \| s \| hjres \| sjres \| hconres \| sconres \| hres \| sres | `hr` |
| `number` | Bill number; add /actions, /cosponsors, /subjects, /text, /summaries | `1/cosponsors` |
| `fromDateTime / toDateTime` | Filter by update time, YYYY-MM-DDTHH:MM:SSZ | `2026-01-01T00:00:00Z` |
| `sort` | updateDate+asc \| updateDate+desc | `updateDate+desc` |
| `limit / offset` | Max 250 per page | `5` |
| `format / api_key` | json \| xml; your api.data.gov key | `json / DEMO_KEY` |
| `/v3/member?currentMember=true` | Sibling endpoint: members of Congress (also /committee, /nomination, /treaty) | `currentMember=true` |

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

<details><summary><b>Sample output</b> — live run 2026-10-12 07:16</summary>

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

### European Parliament Open Data API

| | |
|:--|:--|
| **Docs** | https://data.europarl.europa.eu/en/developer-corner/opendata-api |
| **Auth** | None |
| **Register / token** | No registration. JSON-LD responses (linked data); add format=application/ld+json. |
| **Script** | [`scripts/politics_government/european_parliament.py`](scripts/politics_government/european_parliament.py) |

**What you can get:**

- Members (MEPs): country, political group, national party, mandates
- Plenary documents, adopted texts, votes and roll calls, parliamentary questions
- Meetings, committees, events

**Capstone ideas:** Comparative legislative behavior (EU vs. US Congress), party-group cohesion, cross-national coalitions.

**Endpoint:** `GET https://data.europarl.europa.eu/api/v2/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `resource` | meps \| meps/show-current \| meetings \| plenary-documents \| adopted-texts \| parliamentary-questions \| ... | `meps/show-current` |
| `format` | application/ld+json \| application/rdf+xml \| text/turtle | `application/ld+json` |
| `offset / limit` | Paging | `0 / 500` |
| `country-of-representation` | Filter MEPs by country code (on meps) | `DE` |
| `year` | Filter documents/meetings by year | `2026` |

```python
from collections import Counter

from _common import get, show


def main():
    meps = get("https://data.europarl.europa.eu/api/v2/meps/show-current",
               params={"format": "application/ld+json", "offset": 0, "limit": 1000})["data"]
    show({"current_MEPs": len(meps),
          "seats_by_group": Counter(m.get("api:political-group") for m in meps).most_common(),
          "sample": [{k: m.get(k) for k in ("label", "api:country-of-representation", "api:political-group")} for m in meps[:3]]},
         max_items=10)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:16</summary>

```json
{
  "current_MEPs": 719,
  "seats_by_group": [
    [
      "PPE",
      184
    ],
    [
      "S&D",
      135
    ],
    [
      "PfE",
      85
    ],
    [
      "ECR",
      84
    ],
    [
      "Renew",
      78
    ],
    [
      "Verts/ALE",
      53
    ],
    [
      "The Left",
      45
    ],
    [
      "NI",
      28
    ],
    [
      "ESN",
      27
    ]
  ],
  "sample": [
    {
      "label": "Tobiasz BOCHEŃSKI",
      "api:country-of-representation": "PL",
      "api:political-group": "ECR"
    },
    {
      "label": "Lena DÜPONT",
      "api:country-of-representation": "DE",
      "api:political-group": "PPE"
    },
    {
      "label": "Salvatore DE MEO",
      "api:country-of-representation": "IT",
      "api:political-group": "PPE"
    }
  ]
}
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

**Endpoint:** `GET https://www.federalregister.gov/api/v1/documents.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `conditions[term]` | Full-text search | `artificial intelligence` |
| `conditions[type][]` | RULE \| PRORULE \| NOTICE \| PRESDOCU | `PRORULE` |
| `conditions[agencies][]` | Agency slug | `environmental-protection-agency` |
| `conditions[presidential_document_type]` | executive_order \| memorandum \| proclamation \| ... | `executive_order` |
| `conditions[publication_date][gte] / [lte]` | Date range YYYY-MM-DD | `2025-01-20` |
| `fields[]` | Fields to return (repeatable) | `title, abstract, document_number` |
| `order` | newest \| oldest \| relevance \| executive_order_number | `newest` |
| `per_page / page` | Max 1,000 per page | `5` |

```python
from _common import get, show


def main():
    d = get("https://www.federalregister.gov/api/v1/documents.json", params={
        "conditions[presidential_document_type]": "executive_order", "order": "newest", "per_page": 5,
        "fields[]": ["title", "executive_order_number", "signing_date", "publication_date"]})
    show({"count": d["count"], "results": d["results"]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:16</summary>

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

### GovInfo API (U.S. Government Publishing Office)

| | |
|:--|:--|
| **Docs** | https://api.govinfo.gov/docs/ · https://github.com/usgpo/api |
| **Auth** | api.data.gov key (`DEMO_KEY` works at low volume) |
| **Register / token** | Use your api.data.gov key (https://api.data.gov/signup/). 36,000 req/hour with a personal key. |
| **Env var** | `DATA_GOV_API_KEY` |
| **Script** | [`scripts/politics_government/govinfo.py`](scripts/politics_government/govinfo.py) |

**What you can get:**

- Official full text + metadata of 40+ collections: bills (BILLS), Congressional Record (CREC), US Code (USCODE)
- Code of Federal Regulations (CFR), Federal Register (FR), federal court opinions (USCOURTS), GAO reports, budget
- Every document as PDF, HTML, XML and MODS metadata; search across collections

**Capstone ideas:** Text-as-data on bills and floor speeches, tracking legislative language diffusion, court-opinion corpora.

**Endpoint:** `POST https://api.govinfo.gov/search  ·  GET https://api.govinfo.gov/collections/{collection}/{since}`

| Parameter | Description | Example |
|:--|:--|:--|
| `query (POST body)` | Search with field operators: collection:, congress:, docClass:, publishdate:range(...) | `collection:USCOURTS AND "qualified immunity"` |
| `pageSize / offsetMark (POST body)` | Page size (max 1,000); offsetMark=* then the returned mark for paging | `5 / *` |
| `sorts (POST body)` | List of {field, sortOrder} | `[{"field": "publishdate", "sortOrder": "DESC"}]` |
| `/collections` | List collections with document counts | `—` |
| `/collections/{code}/{lastModifiedStartDate}` | Packages added/changed since a date (pageSize, offsetMark) | `/collections/BILLS/2026-10-01T00:00:00Z` |
| `/packages/{packageId}/summary` | One document's metadata and download links (htm, xml, pdf) | `/packages/BILLS-119hr1enr/summary` |
| `api_key` | api.data.gov key | `DEMO_KEY` |

```python
import os

import requests

from _common import USER_AGENT, show


def main():
    r = requests.post("https://api.govinfo.gov/search", timeout=60, headers={"User-Agent": USER_AGENT},
                      params={"api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY")}, json={
                          "query": 'collection:USCOURTS AND "qualified immunity"', "pageSize": 5, "offsetMark": "*",
                          "sorts": [{"field": "publishdate", "sortOrder": "DESC"}]})
    r.raise_for_status()
    d = r.json()
    show({"matching_court_documents": d["count"], "latest": [
        {"title": x["title"], "court": x["governmentAuthor"][0], "dateIssued": x["dateIssued"], "packageId": x["packageId"]}
        for x in d["results"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:16</summary>

```json
{
  "matching_court_documents": 77082,
  "latest": [
    {
      "title": "Lightfoot et al v. Progressive Direct Insurance Company et al",
      "court": "United States District Court Eastern District of Louisiana",
      "dateIssued": "2026-10-09",
      "packageId": "USCOURTS-laed-2_26-cv-00434"
    },
    {
      "title": "Kushawn Miles-El v. MI, et al",
      "court": "United States Court of Appeals for the Sixth Circuit",
      "dateIssued": "2026-10-09",
      "packageId": "USCOURTS-ca6-25-01140"
    },
    {
      "title": "Pankratz v. The Board of County Commissioners For Roosevelt County et al",
      "court": "United States District Court District of New Mexico",
      "dateIssued": "2026-10-08",
      "packageId": "USCOURTS-nmd-2_21-cv-00655"
    },
    {
      "title": "Edwards v. Bierman et al",
      "court": "United States District Court Northern District of California",
      "dateIssued": "2026-10-08",
      "packageId": "USCOURTS-cand-3_26-cv-05124"
    },
    {
      "title": "Duckett, Alexis et al v. The Board Of Regents Of The University Of Wisconsin System et al",
      "court": "United States District Court Western District of Wisconsin",
      "dateIssued": "2026-10-07",
      "packageId": "USCOURTS-wiwd-3_25-cv-00684"
    }
  ]
}
```
</details>

---

### Grants.gov Search API

| | |
|:--|:--|
| **Docs** | https://www.grants.gov/api/common/search2 |
| **Auth** | None |
| **Register / token** | No registration for search (the applicant-side System-to-System API needs registration). |
| **Script** | [`scripts/politics_government/grants_gov.py`](scripts/politics_government/grants_gov.py) |

**What you can get:**

- Every federal funding opportunity: title, number, agency, open/close dates, status, CFDA/assistance listing
- Filters by eligibility, funding instrument and category
- Opportunity details via fetchOpportunity

**Capstone ideas:** Where federal research money goes by agency/topic over time; matching grants to university strengths.

**Endpoint:** `POST https://api.grants.gov/v1/api/search2`

| Parameter | Description | Example |
|:--|:--|:--|
| `keyword` | Search text | `data science` |
| `oppStatuses` | forecasted \| posted \| closed \| archived (pipe-separated) | `forecasted\|posted` |
| `agencies` | Agency codes (pipe-separated) | `NSF\|HHS-NIH11` |
| `fundingCategories / eligibilities / fundingInstruments` | Category codes (e.g. ST science & technology), eligibility codes, G grant / CA cooperative agreement | `ST` |
| `rows / startRecordNum` | Page size and offset | `5 / 0` |
| `/v1/api/fetchOpportunity` | Sibling endpoint: full details for one opportunity (POST {opportunityId}) | `{"opportunityId": 353475}` |

```python
import requests

from _common import USER_AGENT, show


def main():
    r = requests.post("https://api.grants.gov/v1/api/search2", timeout=60, headers={"User-Agent": USER_AGENT},
                      json={"keyword": "data science", "oppStatuses": "forecasted|posted", "rows": 5})
    r.raise_for_status()
    d = r.json()["data"]
    show({"open_or_forecasted": d["hitCount"], "opportunities": [
        {k: o.get(k) for k in ("number", "title", "agencyCode", "openDate", "closeDate", "oppStatus")} for o in d["oppHits"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:16</summary>

```json
{
  "open_or_forecasted": 884,
  "opportunities": [
    {
      "number": "PD-24-7414",
      "title": "Cyberinfrastructure for Public Access and Open Science",
      "agencyCode": "NSF",
      "openDate": "04/10/2024",
      "closeDate": "",
      "oppStatus": "posted"
    },
    {
      "number": "23-568",
      "title": "Human Networks and Data Science (HNDS)",
      "agencyCode": "NSF",
      "openDate": "05/20/2023",
      "closeDate": "01/14/2027",
      "oppStatus": "posted"
    },
    {
      "number": "PD-24-8084",
      "title": "Computational and Data-Enabled Science and Engineering",
      "agencyCode": "NSF",
      "openDate": "08/15/2024",
      "closeDate": "",
      "oppStatus": "posted"
    },
    {
      "number": "24-587",
      "title": "Research on the Science and Technology Enterprise: Indicators, Statistics, and Methods (NCSES S&amp;T)",
      "agencyCode": "NSF",
      "openDate": "07/01/2024",
      "closeDate": "01/19/2027",
      "oppStatus": "posted"
    },
    {
      "number": "PD-23-277Y",
      "title": "Research Infrastructure in the Social and Behavioral Sciences",
      "agencyCode": "NSF",
      "openDate": "10/05/2023",
      "closeDate": "",
      "oppStatus": "posted"
    }
  ]
}
```
</details>

---

### Harvard Dataverse (MIT Election Lab & replication data)

| | |
|:--|:--|
| **Docs** | https://guides.dataverse.org/en/latest/api/ |
| **Auth** | None for search and public downloads (token to deposit) |
| **Register / token** | No registration to search or download public files. Hosts the MIT Election Data + Science Lab returns (president, Senate, House, county and precinct level) and replication data for thousands of political-science papers. |
| **Env var** | `DATAVERSE_API_TOKEN (optional)` |
| **Script** | [`scripts/politics_government/harvard_dataverse.py`](scripts/politics_government/harvard_dataverse.py) |

**What you can get:**

- Search datasets by keyword, subject, author, publication; citation and DOI for each
- List and download each dataset's files (CSV, Stata, R) via the Data Access API
- Election returns since 1976 from the MIT Election Lab

**Capstone ideas:** Election forecasting, turnout and polarization analysis, replicating published political-science findings.

**Endpoint:** `GET https://dataverse.harvard.edu/api/search`

| Parameter | Description | Example |
|:--|:--|:--|
| `q` | Search text | `MIT Election Data` |
| `type` | dataset \| file \| dataverse | `dataset` |
| `subtree` | Restrict to one collection alias | `medsl` |
| `sort / order` | name \| date; asc \| desc | `date / desc` |
| `per_page / start` | Paging (max 1,000) | `5` |
| `/api/datasets/:persistentId/?persistentId=doi:...` | Sibling endpoint: dataset metadata and file list | `doi:10.7910/DVN/42MVDX` |
| `/api/access/datafile/{fileId}` | Download a file (format=original for the uploaded format) | `format=original` |

```python
from _common import get, show


def main():
    d = get("https://dataverse.harvard.edu/api/search", params={
        "q": "election returns", "type": "dataset", "subtree": "medsl", "sort": "date", "order": "desc", "per_page": 5})["data"]
    show({"datasets": d["total_count"], "latest": [
        {"name": x["name"], "published": x.get("published_at", "")[:10], "doi": x.get("global_id"),
         "files": x.get("fileCount")} for x in d["items"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:16</summary>

```json
{
  "datasets": 55,
  "latest": [
    {
      "name": "Precinct-Level Returns 2022 by Individual State",
      "published": "2026-09-18",
      "doi": "doi:10.7910/DVN/UYQIEP",
      "files": 53
    },
    {
      "name": "State Offices Precinct-Level Returns 2022",
      "published": "2026-09-18",
      "doi": "doi:10.7910/DVN/OAARCY",
      "files": 3
    },
    {
      "name": "U.S. Senate Precinct-Level Returns 2022",
      "published": "2026-06-10",
      "doi": "doi:10.7910/DVN/IAD3XR",
      "files": 3
    },
    {
      "name": "U.S. House of Representatives Precinct-Level Returns 2022",
      "published": "2026-06-10",
      "doi": "doi:10.7910/DVN/EOKNGW",
      "files": 1
    },
    {
      "name": "U.S. Senate statewide 1976–2024",
      "published": "2026-05-11",
      "doi": "doi:10.7910/DVN/PEJ5QU",
      "files": 3
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

**Endpoint:** `GET https://v3.openstates.org/bills`

| Parameter | Description | Example |
|:--|:--|:--|
| `jurisdiction` | State name or OCD jurisdiction ID | `Virginia` |
| `session` | Legislative session identifier | `2026` |
| `q` | Full-text search | `artificial intelligence` |
| `subject / classification` | Subject tag / bill \| resolution \| ... | `Education` |
| `updated_since / created_since / action_since` | Date filters YYYY-MM-DD | `2026-01-01` |
| `sort` | updated_desc \| first_action_desc \| latest_action_desc \| ... | `updated_desc` |
| `include` | Extra data: sponsorships, actions, votes, abstracts, versions (repeatable) | `votes` |
| `per_page / page` | Max 20 per page | `5` |
| `/people?jurisdiction=&name=` | Sibling endpoint: legislators | `jurisdiction=Virginia` |

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

**Endpoint:** `GET https://api.open.fec.gov/v1/candidates/totals/`

| Parameter | Description | Example |
|:--|:--|:--|
| `election_year / cycle` | Election year / two-year period | `2024` |
| `office` | P (president) \| S (senate) \| H (house) | `P` |
| `party / state` | Party code / state | `DEM / VA` |
| `is_active_candidate` | true \| false | `true` |
| `sort` | Field, prefix - for descending | `-receipts` |
| `per_page / page` | Max 100 per page | `5` |
| `/v1/schedules/schedule_a/` | Sibling endpoint: itemized contributions (committee_id, contributor_state, two_year_transaction_period, min_amount; keyset paging with last_index) | `contributor_state=VA` |
| `api_key` | api.data.gov key | `DEMO_KEY` |

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

<details><summary><b>Sample output</b> — live run 2026-10-12 07:16</summary>

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

**Endpoint:** `GET https://api.regulations.gov/v4/documents  ·  /v4/comments  ·  /v4/dockets`

| Parameter | Description | Example |
|:--|:--|:--|
| `filter[searchTerm]` | Keyword search | `artificial intelligence` |
| `filter[documentType]` | Notice \| Rule \| Proposed Rule \| Supporting & Related Material \| Other | `Proposed Rule` |
| `filter[agencyId]` | Agency acronym | `EPA` |
| `filter[postedDate][ge] / [le]` | Date range YYYY-MM-DD | `2026-01-01` |
| `filter[docketId]` | Restrict to one docket | `EPA-HQ-OAR-2021-0317` |
| `sort` | postedDate \| -postedDate \| title \| ... | `-postedDate` |
| `page[size] / page[number]` | 5-250 per page; max 20 pages (narrow by date to go further) | `5` |
| `/v4/comments?filter[commentOnId]=` | Sibling endpoint: comments on a document (objectId from /documents) | `filter[commentOnId]=0900006…` |
| `api_key` | api.data.gov key | `DEMO_KEY` |

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

<details><summary><b>Sample output</b> — live run 2026-10-12 07:16</summary>

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

### Lobbying Disclosure Act (LDA) API

| | |
|:--|:--|
| **Docs** | https://lda.gov/api/redoc/v1/ |
| **Auth** | None (optional free key raises limits) |
| **Register / token** | Works anonymously at low rates. Register at https://lda.gov/api/register/ for a key (header `Authorization: Token <key>`). Formerly lda.senate.gov. |
| **Env var** | `LDA_API_KEY (optional)` |
| **Script** | [`scripts/politics_government/senate_lda.py`](scripts/politics_government/senate_lda.py) |

**What you can get:**

- Every federal lobbying registration and quarterly report since 1999
- Registrant (firm), client, income/expenses, issue codes, specific bills/issues lobbied, agencies contacted
- Individual lobbyists (incl. former government positions), political contributions (LD-203)

**Capstone ideas:** Lobbying spend vs. legislative outcomes, revolving-door networks, which industries lobby on AI or climate.

**Endpoint:** `GET https://lda.gov/api/v1/{resource}/`

| Parameter | Description | Example |
|:--|:--|:--|
| `resource` | filings \| contributions \| registrants \| clients \| lobbyists \| constants/filing/lobbyingactivityissues | `filings` |
| `filing_year / filing_period` | Year; first_quarter \| second_quarter \| third_quarter \| fourth_quarter | `2025 / second_quarter` |
| `filing_specific_lobbying_issues` | Full-text search in the issues/bills lobbied | `artificial intelligence` |
| `client_name / registrant_name` | Name search | `Microsoft` |
| `filing_type` | RR registration, Q1-Q4 quarterly reports, ...  | `Q2` |
| `filing_amount_reported_min / _max` | Reported income (lobbying firms) or expenses (in-house lobbying) range in USD | `100000` |
| `page / page_size` | Paging (max 25 anonymous) | `5` |

```python
from _common import get, show


def main():
    d = get("https://lda.gov/api/v1/filings/", params={
        "filing_year": 2025, "filing_specific_lobbying_issues": "artificial intelligence",
        "filing_amount_reported_min": 1, "page_size": 10})
    unique = {f["filing_uuid"]: f for f in d["results"]}.values()  # the same filing can appear more than once
    show({"filings_mentioning_AI_2025": d["count"], "examples": [
        {"client": f["client"]["name"], "registrant": f["registrant"]["name"], "period": f["filing_period_display"],
         "amount_usd": f["income"] or f["expenses"],  # firms report income, in-house lobbyists report expenses
         "issues": sorted({a["general_issue_code_display"] for a in f["lobbying_activities"]})[:3]}
        for f in list(unique)[:5]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:16</summary>

```json
{
  "filings_mentioning_AI_2025": 4066,
  "examples": [
    {
      "client": "SAP AMERICA, INC.",
      "registrant": "MEHLMAN CONSULTING, INC.",
      "period": "1st Quarter (Jan 1 - Mar 31)",
      "amount_usd": "20000.00",
      "issues": [
        "Computer Industry",
        "Trade (domestic/foreign)"
      ]
    },
    {
      "client": "THE TRUSTEES OF PURDUE UNIVERSITY",
      "registrant": "DEBORAH HOHLT",
      "period": "1st Quarter (Jan 1 - Mar 31)",
      "amount_usd": "90000.00",
      "issues": [
        "Agriculture",
        "Budget/Appropriations",
        "Education"
      ]
    },
    {
      "client": "EPIC SYSTEMS",
      "registrant": "MCDERMOTT+ LLC",
      "period": "1st Quarter (Jan 1 - Mar 31)",
      "amount_usd": "20000.00",
      "issues": [
        "Health Issues"
      ]
    },
    {
      "client": "UNIVERSITY OF TULSA",
      "registrant": "SBL STRATEGIES, LLC",
      "period": "1st Quarter (Jan 1 - Mar 31)",
      "amount_usd": "30000.00",
      "issues": [
        "Defense",
        "Energy/Nuclear",
        "Science/Technology"
      ]
    },
    {
      "client": "VERILY LIFE SCIENCES LLC",
      "registrant": "MCDERMOTT+ LLC",
      "period": "1st Quarter (Jan 1 - Mar 31)",
      "amount_usd": "10000.00",
      "issues": [
        "Health Issues",
        "Medical/Disease Research/Clinical Labs",
        "Medicare/Medicaid"
      ]
    }
  ]
}
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

**Endpoint:** `GET https://commonsvotes-api.parliament.uk/data/divisions.json/search`

| Parameter | Description | Example |
|:--|:--|:--|
| `queryParameters.searchTerm` | Text in the division title | `Budget` |
| `queryParameters.startDate / endDate` | Date range YYYY-MM-DD | `2026-01-01` |
| `queryParameters.memberId` | Only divisions this MP voted in | `172` |
| `queryParameters.take / skip` | Paging (max 25) | `5` |
| `/data/division/{id}.json` | Sibling endpoint: every MP's vote in one division | `/data/division/1234.json` |
| `members-api.parliament.uk/api/Members/Search?Name=` | Members API: find MPs/Lords, party, constituency | `Name=Starmer` |

```python
from _common import get, show


def main():
    divs = get("https://commonsvotes-api.parliament.uk/data/divisions.json/search", params={"queryParameters.take": 5})
    show([{"title": d["Title"], "date": d["Date"][:10], "ayes": d["AyeCount"], "noes": d["NoCount"]} for d in divs])
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:16</summary>

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

### USAJOBS Search API (federal jobs)

| | |
|:--|:--|
| **Docs** | https://developer.usajobs.gov/ |
| **Auth** | API key (header `Authorization-Key`) + your email as `User-Agent` |
| **Register / token** | Request a key at https://developer.usajobs.gov/apirequest/ (emailed). Send headers Host: data.usajobs.gov, User-Agent: <the email you registered>, Authorization-Key: <key>. |
| **Env var** | `USAJOBS_API_KEY, USAJOBS_EMAIL` |
| **Script** | [`scripts/politics_government/usajobs.py`](scripts/politics_government/usajobs.py) |

**What you can get:**

- Every open federal job posting: title, agency, grade/pay range, location, remote/telework, security clearance
- Qualification summaries, hiring paths (public, veterans, students), open/close dates
- Historic job announcements (separate endpoint)

**Capstone ideas:** Federal hiring trends by agency and skill (e.g. demand for data scientists), pay comparisons by location.

**Endpoint:** `GET https://data.usajobs.gov/api/search`

| Parameter | Description | Example |
|:--|:--|:--|
| `Keyword / PositionTitle` | Search text / title | `data scientist` |
| `LocationName / Radius` | City or state, radius in miles | `Washington, DC / 25` |
| `Organization` | Agency subelement codes (semicolon-separated) | `HE38` |
| `RemunerationMinimum / PayGradeLow / PayGradeHigh` | Pay filters | `100000` |
| `HiringPath` | public \| vet \| student \| fed \| ... | `public` |
| `DatePosted / SortField` | Days since posted; sort field (DatePosted, Salary, ...) | `7 / DatePosted` |
| `ResultsPerPage / Page` | Max 500 per page | `5` |

```python
from _common import get, key, show


def main():
    d = get("https://data.usajobs.gov/api/search", headers={
        "Host": "data.usajobs.gov", "User-Agent": key("USAJOBS_EMAIL"), "Authorization-Key": key("USAJOBS_API_KEY")},
        params={"Keyword": "data scientist", "LocationName": "Washington, DC", "ResultsPerPage": 5})["SearchResult"]
    show({"total": d["SearchResultCountAll"], "jobs": [
        {"title": j["PositionTitle"], "agency": j["OrganizationName"], "location": j["PositionLocationDisplay"],
         "pay": f'${j["PositionRemuneration"][0]["MinimumRange"]} - ${j["PositionRemuneration"][0]["MaximumRange"]}',
         "close_date": j["ApplicationCloseDate"][:10]}
        for j in (item["MatchedObjectDescriptor"] for item in d["SearchResultItems"])]})
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "total": 212,
  "jobs": [
    {"title": "Data Scientist", "agency": "Census Bureau", "location": "Suitland, Maryland",
     "pay": "$104,604 - $163,282 Per Year", "close_date": "2026-10-24"}
  ]
}
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

**Endpoint:** `POST https://api.usaspending.gov/api/v2/search/spending_by_category/{category}/`

| Parameter | Description | Example |
|:--|:--|:--|
| `category (path)` | awarding_agency \| awarding_subagency \| recipient \| naics \| psc \| cfda \| state_territory \| county \| district \| ... | `awarding_agency` |
| `filters.time_period` | List of {start_date, end_date} (fiscal year starts Oct 1) | `[{"start_date": "2024-10-01", "end_date": "2025-09-30"}]` |
| `filters.award_type_codes` | A-D = contracts, 02-05 = grants, 07/08 = loans | `["A", "B", "C", "D"]` |
| `filters.agencies` | List of {type, tier, name} | `[{"type": "awarding", "tier": "toptier", "name": "Department of Energy"}]` |
| `filters.recipient_locations` | List of {country, state, county, district} | `[{"country": "USA", "state": "VA"}]` |
| `limit / page` | Paging | `5` |
| `/api/v2/search/spending_by_award/` | Sibling endpoint: individual awards (same filters + fields list) | `fields: ["Award ID", "Recipient Name", "Award Amount"]` |

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

<details><summary><b>Sample output</b> — live run 2026-10-12 07:16</summary>

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

## ⚖️ Law & Justice

### CourtListener (Free Law Project)

| | |
|:--|:--|
| **Docs** | https://wiki.free.law/c/courtlistener/help/api/rest/v4/rest-api-v47 |
| **Auth** | None for search (token for higher limits and most other endpoints) |
| **Register / token** | Search works anonymously. Create a free account, then copy your token from https://www.courtlistener.com/profile/api/ (header `Authorization: Token <token>`; 5,000 req/hour). |
| **Env var** | `COURTLISTENER_TOKEN (optional)` |
| **Script** | [`scripts/law/courtlistener.py`](scripts/law/courtlistener.py) |

**What you can get:**

- Millions of court opinions (federal and state, incl. the full Caselaw Access Project history) with full text
- PACER dockets and filings from the RECAP archive, oral-argument audio, judges and their careers
- Citation graph: which opinions cite which (citeCount, cited opinions)

**Capstone ideas:** Legal NLP (outcome prediction, topic models), citation-network influence of precedents, judge behavior.

**Endpoint:** `GET https://www.courtlistener.com/api/rest/v4/search/`

| Parameter | Description | Example |
|:--|:--|:--|
| `q` | Query; supports "phrases", AND/OR, field searches like caseName:(...) | `"qualified immunity"` |
| `type` | o opinions \| r RECAP dockets \| d dockets \| p judges \| oa oral arguments | `o` |
| `court` | Space-separated court IDs | `scotus ca4 dcd` |
| `filed_after / filed_before` | Date range YYYY-MM-DD | `2020-01-01` |
| `order_by` | score desc \| dateFiled desc \| citeCount desc | `citeCount desc` |
| `cursor` | Paging cursor from the 'next' URL (20 results per page) | `—` |
| `/api/rest/v4/opinions/{id}/` | Sibling endpoint (token): an opinion's full text (plain_text, html_with_citations) | `—` |

```python
from _common import get, show


def main():
    d = get("https://www.courtlistener.com/api/rest/v4/search/", params={
        "q": '"qualified immunity"', "type": "o", "court": "scotus", "order_by": "citeCount desc"})
    show({"scotus_opinions_mentioning_qualified_immunity": d["count"], "most_cited": [
        {"case": r["caseName"], "date": r["dateFiled"], "cited_by": r["citeCount"], "citation": r["citation"][:1],
         "url": "https://www.courtlistener.com" + r["absolute_url"]} for r in d["results"][:5]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:16</summary>

```json
{
  "scotus_opinions_mentioning_qualified_immunity": 231,
  "most_cited": [
    {
      "case": "Ashcroft v. Iqbal",
      "date": "2009-05-18",
      "cited_by": 160249,
      "citation": [
        "173 L. Ed. 2d 868"
      ],
      "url": "https://www.courtlistener.com/opinion/145875/ashcroft-v-iqbal/"
    },
    {
      "case": "Monell v. New York City Dept. of Social Servs.",
      "date": "1978-06-06",
      "cited_by": 43435,
      "citation": [
        "56 L. Ed. 2d 611"
      ],
      "url": "https://www.courtlistener.com/opinion/109881/monell-v-new-york-city-dept-of-social-servs/"
    },
    {
      "case": "Farmer v. Brennan",
      "date": "1994-06-06",
      "cited_by": 29755,
      "citation": [
        "128 L. Ed. 2d 811"
      ],
      "url": "https://www.courtlistener.com/opinion/1087956/farmer-v-brennan/"
    },
    {
      "case": "Harlow v. Fitzgerald",
      "date": "1982-06-24",
      "cited_by": 23283,
      "citation": [
        "73 L. Ed. 2d 396"
      ],
      "url": "https://www.courtlistener.com/opinion/110763/harlow-v-fitzgerald/"
    },
    {
      "case": "Graham v. Connor",
      "date": "1989-05-15",
      "cited_by": 17104,
      "citation": [
        "104 L. Ed. 2d 443"
      ],
      "url": "https://www.courtlistener.com/opinion/112257/graham-v-connor/"
    }
  ]
}
```
</details>

---

### U.S. Department of Justice News API

| | |
|:--|:--|
| **Docs** | https://www.justice.gov/developer |
| **Auth** | None |
| **Register / token** | No registration. Covers DOJ press releases (incl. all 94 U.S. Attorney's Offices) and blog posts. |
| **Script** | [`scripts/law/doj_press_releases.py`](scripts/law/doj_press_releases.py) |

**What you can get:**

- 270,000+ press releases: title, date, full HTML body, DOJ components (office/division), topics
- Prosecutions, sentencings, settlements and indictments across fraud, cybercrime, civil rights, environment...

**Capstone ideas:** Enforcement-priority shifts across administrations, NLP extraction of charges and sentences, topic trends.

**Endpoint:** `GET https://www.justice.gov/api/v1/{press_releases \| blog_entries}.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `pagesize / page` | Page size and page number (0-based) | `5 / 0` |
| `sort / direction` | Sort field and order | `date / DESC` |
| `parameters[title]` | Filter by words in the title (other fields: parameters[{field}]) | `fraud` |
| `fields` | date = Unix timestamp; component[].name = DOJ office; topic[].name = subject | `—` |

```python
import datetime as dt
import html
import re

from _common import get, show


def main():
    d = get("https://www.justice.gov/api/v1/press_releases.json", params={
        "pagesize": 5, "sort": "date", "direction": "DESC", "parameters[title]": "artificial intelligence"})
    show({"matching_releases": d["metadata"]["resultset"]["count"], "latest": [
        {"date": dt.date.fromtimestamp(int(r["date"])).isoformat(), "title": r["title"],
         "offices": [c["name"] for c in r.get("component") or []][:2],
         "topics": [t.get("name") for t in r.get("topic") or []],
         "teaser": html.unescape(re.sub("<[^>]+>", "", r.get("teaser") or "")).strip()}
        for r in d["results"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:16</summary>

```json
{
  "matching_releases": "29",
  "latest": [
    {
      "date": "2026-09-11",
      "title": "Founder And Former CEO Of Artificial Intelligence Company Pleads Guilty To Securities Fraud",
      "offices": [
        "USAO - New York, Southern"
      ],
      "topics": [
        "Securities, Commodities, & Investment Fraud"
      ],
      "teaser": "United States Attorney for the Southern District of New York, Jamie McDonald, and Assistant Director in Charge of the New York Field Office of the Federal Burea..."
    },
    {
      "date": "2026-03-25",
      "title": "Chinese national and two U.S. citizens charged with conspiring to smuggle artificial intelligence technology to China",
      "offices": [
        "USAO - Georgia, Northern"
      ],
      "topics": [
        "National Security"
      ],
      "teaser": "Stanley Yi Zheng, Matthew Kelly, and Tommy Shad English have been charged with conspiring to commit smuggling and export control violations. The three defendant..."
    },
    {
      "date": "2026-03-25",
      "title": "Chinese National and Two U.S. Citizens Charged with Conspiring to Smuggle Artificial Intelligence Technology to China",
      "offices": [
        "National Security Division (NSD)",
        "USAO - Georgia, Northern"
      ],
      "topics": [
        "Export Control",
        "National Security"
      ],
      "teaser": "Stanley Yi Zheng, Matthew Kelly, and Tommy Shad English have been charged with conspiring to commit smuggling and export control violations. The three defendant..."
    },
    {
      "date": "2026-03-19",
      "title": "Three Charged With Conspiring To Unlawfully Divert U.S. Artificial Intelligence Technology To China",
      "offices": [
        "USAO - New York, Southern"
      ],
      "topics": [
        "National Security"
      ],
      "teaser": "United States Attorney for the Southern District of New York, Jay Clayton, Assistant Director of the Counterintelligence and Espionage Division of the Federal B..."
    },
    {
      "date": "2026-03-19",
      "title": "Three Charged with Conspiring to Unlawfully Divert Cutting Edge U.S. Artificial Intelligence Technology to China",
      "offices": [
        "Office of the Attorney General",
        "Federal Bureau of Investigation (FBI)"
      ],
      "topics": [],
      "teaser": "Today, an indictment was unsealed charging Yih-Shyan “Wally” Liaw, Ruei-Tsang “Steven” Chang, and Ting-Wei “Willy” Sun, for allegedly conspiring to divert high-..."
    }
  ]
}
```
</details>

---

### eCFR API (Code of Federal Regulations)

| | |
|:--|:--|
| **Docs** | https://www.ecfr.gov/developers/documentation/api/v1 |
| **Auth** | None |
| **Register / token** | No registration. The eCFR is updated daily; the API also returns any past version of a section. |
| **Script** | [`scripts/law/ecfr.py`](scripts/law/ecfr.py) |

**What you can get:**

- Full text of all 50 CFR titles (structure, XML) as of any date since 2017
- Full-text search with section hierarchy and headings
- Change history: which sections were amended when

**Capstone ideas:** Measuring regulatory growth and complexity by agency, tracking how rules change over time, regulation NLP.

**Endpoint:** `GET https://www.ecfr.gov/api/{service}/v1/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `search/v1/results?query=` | Full-text search (per_page, page, date, title filters) | `query=artificial intelligence` |
| `search/v1/count?query=` | Number of matching sections | `query=privacy` |
| `versioner/v1/titles.json` | All titles with latest amended / issue dates | `—` |
| `versioner/v1/full/{date}/title-{n}.xml` | Full XML of a title (part=, section= to narrow); date must be <= the title's up_to_date_as_of, and the client must accept gzip (requests does; curl needs --compressed) | `/full/2026-10-08/title-45.xml?part=164` |
| `versioner/v1/versions/title-{n}.json` | Version history of sections in a title | `title-45.json` |
| `admin/v1/agencies.json` | Agencies and the CFR references they own | `—` |

```python
import re

from _common import get, show


def main():
    d = get("https://www.ecfr.gov/api/search/v1/results", params={"query": "artificial intelligence", "per_page": 50})
    sections = {}  # results are per matching paragraph, so keep one row per section
    for r in d["results"]:
        if r["hierarchy"]["section"]:
            sections.setdefault((r["hierarchy"]["title"], r["hierarchy"]["section"]), r)
    titles = get("https://www.ecfr.gov/api/versioner/v1/titles.json")["titles"]
    show({"matching_paragraphs": d["meta"]["total_count"], "example_sections": [
        {"title": r["hierarchy_headings"]["title"], "section": r["hierarchy_headings"]["section"],
         "heading": re.sub("<[^>]+>", "", r["headings"]["section"] or ""),  # strip search-highlight tags
         "agency_chapter": (r["headings"]["chapter"] or "").strip()}
        for r in list(sections.values())[:5]],
          "recently_amended_titles": [{k: t[k] for k in ("number", "name", "latest_amended_on")}
                                      for t in sorted(titles, key=lambda t: t.get("latest_amended_on") or "", reverse=True)[:3]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:16</summary>

```json
{
  "matching_paragraphs": 214,
  "example_sections": [
    {
      "title": "Title 15",
      "section": "§ 740.27",
      "heading": "License Exception Artificial Intelligence Authorization (AIA).",
      "agency_chapter": "Bureau of Industry and Security, Department of Commerce"
    },
    {
      "title": "Title 15",
      "section": "§ 748.3",
      "heading": "Classification requests and advisory opinions.",
      "agency_chapter": "Bureau of Industry and Security, Department of Commerce"
    },
    {
      "title": "Title 28",
      "section": "§ 202.301",
      "heading": "Prohibited data-brokerage transactions.",
      "agency_chapter": "Department of Justice"
    },
    {
      "title": "Title 28",
      "section": "§ 202.217",
      "heading": "Employment agreement.",
      "agency_chapter": "Department of Justice"
    },
    {
      "title": "Title 15",
      "section": "§ 742.6",
      "heading": "Regional stability.",
      "agency_chapter": "Bureau of Industry and Security, Department of Commerce"
    }
  ],
  "recently_amended_titles": [
    {
      "number": 7,
      "name": "Agriculture",
      "latest_amended_on": "2026-10-08"
    },
    {
      "number": 17,
      "name": "Commodity and Securities Exchanges",
      "latest_amended_on": "2026-10-08"
    },
    {
      "number": 34,
      "name": "Education",
      "latest_amended_on": "2026-10-08"
    }
  ]
}
```
</details>

---

### FBI Crime Data Explorer API

| | |
|:--|:--|
| **Docs** | https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/docApi |
| **Auth** | api.data.gov key (`DEMO_KEY` works but is often rate-limited here) |
| **Register / token** | Use your api.data.gov key (https://api.data.gov/signup/), passed as API_KEY. The shared DEMO_KEY hits HTTP 429 quickly on this API, so get a personal key. |
| **Env var** | `DATA_GOV_API_KEY` |
| **Script** | [`scripts/law/fbi_crime_data.py`](scripts/law/fbi_crime_data.py) |

**What you can get:**

- Monthly offense and clearance rates (per 100k) for the nation, states and individual agencies
- Reported counts by offense: violent, property, homicide, rape, robbery, aggravated assault, burglary, ...
- Agency directory (ORI codes), NIBRS incident detail, hate crime, use-of-force and LEOKA data

**Capstone ideas:** Crime trends and policy evaluation, clearance-rate disparities, linking crime to housing/transport data.

**Endpoint:** `GET https://api.usa.gov/crime/fbi/cde/summarized/{level}/{offense}`

| Parameter | Description | Example |
|:--|:--|:--|
| `level` | state/{abbr}; the docs also list national and agency/{ORI} | `state/VA` |
| `offense` | V violent crime, P property crime, RPE rape, ROB robbery, ASS aggravated assault (full code list in the docs) | `V` |
| `from / to` | Month range MM-YYYY | `01-2023 / 12-2023` |
| `API_KEY` | api.data.gov key (query parameter) | `DEMO_KEY` |
| `/agency/byStateAbbr/{abbr}` | Sibling endpoint (per the docs): agencies and ORI codes in a state | `/agency/byStateAbbr/DC` |

```python
import os

from _common import get, show


def main():
    d = get("https://api.usa.gov/crime/fbi/cde/summarized/state/VA/V", params={
        "from": "01-2023", "to": "12-2023", "API_KEY": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY")})
    rates = d["offenses"]["rates"]
    show({"offense": "violent crime, monthly rate per 100k (2023)", "annual_rate_per_100k": {
        name: round(sum(months.values()), 1) for name, months in rates.items()}})
```

<details><summary><b>Sample output</b> — live run 2026-10-12</summary>

```json
{
  "offense": "violent crime, monthly rate per 100k (2023)",
  "annual_rate_per_100k": {
    "Virginia Offenses": 243.5,
    "Virginia Clearances": 130.1,
    "United States Offenses": 385.6,
    "United States Clearances": 159.1
  }
}
```
</details>

---

### Oyez (U.S. Supreme Court cases)

| | |
|:--|:--|
| **Docs** | https://github.com/walkerdb/supreme_court_transcripts (community notes; no official docs) |
| **Auth** | None |
| **Register / token** | No registration. Unofficial API behind oyez.org (Cornell LII / Justia): be gentle and cache results. |
| **Script** | [`scripts/law/oyez.py`](scripts/law/oyez.py) |

**What you can get:**

- Every Supreme Court case: question presented, facts, conclusion, parties, lower court
- Decision: majority/minority vote counts, winning party, how each justice voted
- Timeline (granted/argued/decided), advocates, oral-argument audio and transcripts

**Capstone ideas:** Predicting Supreme Court outcomes, oral-argument transcript analysis, justice voting blocs.

**Endpoint:** `GET https://api.oyez.org/cases`

| Parameter | Description | Example |
|:--|:--|:--|
| `filter` | term:{year} (the October Term) | `term:2024` |
| `per_page` | Results per page; 0 = all cases in the term | `0` |
| `/cases/{term}/{docket}` | Full case record with decisions, votes, advocates | `/cases/2024/23-191` |
| `decisions[].votes[]` | Field in the case record: each justice's vote (majority / minority) | `—` |
| `oral_argument_audio[].href` | Field in the case record: link to the transcript and audio record | `—` |

```python
import re

from _common import get, show


def main():
    cases = get("https://api.oyez.org/cases", params={"filter": "term:2024", "per_page": 0})
    out = []
    for c in cases[:4]:
        full = get(c["href"])
        dec = (full.get("decisions") or [{}])[0]
        out.append({"case": full["name"], "docket": full["docket_number"].strip(),
                    "question": re.sub("<[^>]+>", "", full.get("question") or "").strip(),  # strip HTML tags
                    "vote": f'{dec["majority_vote"]}-{dec["minority_vote"]}' if dec.get("majority_vote") is not None else None,
                    "winner": dec.get("winning_party")})
    show({"term_2024_cases": len(cases), "examples": out}, max_str=200)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:17</summary>

```json
{
  "term_2024_cases": 62,
  "examples": [
    {
      "case": "Williams v. Reed",
      "docket": "23-191",
      "question": "Does a Section 1983 claim brought in state court require the plaintiffs to first exhaust state administrative remedies?",
      "vote": "5-4",
      "winner": "Williams"
    },
    {
      "case": "Glossip v. Oklahoma",
      "docket": "22-7466",
      "question": "May Oklahoma carry out the execution of Richard Glossip in light of the prosecutorial misconduct and other errors that affected his conviction and sentencing?",
      "vote": "6-2",
      "winner": "Glossip"
    },
    {
      "case": "Lackey v. Stinnie",
      "docket": "23-621",
      "question": "Is a party who obtains a preliminary injunction a “prevailing party” for purposes of being entitled to attorney’s fees under 42 U.S.C § 1988?",
      "vote": null,
      "winner": null
    },
    {
      "case": "Bondi v. VanDerStok",
      "docket": "23-852",
      "question": "Did the ATF exceed its statutory authority in promulgating its Final Rule purporting to regulate so-called “ghost guns”?",
      "vote": null,
      "winner": null
    }
  ]
}
```
</details>

---

### Supreme Court Database (SCDB)

| | |
|:--|:--|
| **Docs** | http://scdb.wustl.edu/documentation.php |
| **Auth** | None (CSV download) |
| **Register / token** | No registration; no REST API. Download zipped CSVs from http://scdb.wustl.edu/data.php - a new release each year (file names carry the release, e.g. 2026_01). Case-centered and justice-centered versions. |
| **Script** | [`scripts/law/supreme_court_database.py`](scripts/law/supreme_court_database.py) |

**What you can get:**

- Every Supreme Court decision since 1791 coded on 200+ variables: issue area, decision direction (liberal/conservative), vote split, parties, lower court, precedent alteration
- Justice-centered file: how each justice voted and whether they wrote an opinion

**Capstone ideas:** The standard dataset for judicial-politics research: ideal points, court ideology over time, outcome prediction.

**Endpoint:** `GET http://scdb.wustl.edu/_brickFiles/{release}/SCDB_{release}_{caseCentered \| justiceCentered}_Citation.csv.zip`

| Parameter | Description | Example |
|:--|:--|:--|
| `release` | Yearly release ID | `2026_01` |
| `caseCentered / justiceCentered` | One row per case, or one row per justice vote | `caseCentered` |
| `Citation / Docket / LegalProvision` | Unit of analysis inside the file name | `Citation` |
| `key columns` | term, caseName, dateDecision, issueArea, decisionDirection (1 conservative, 2 liberal), majVotes, minVotes, chief | `—` |

```python
import csv
import io
import zipfile
from collections import Counter

import requests

from _common import USER_AGENT, show

URL = "http://scdb.wustl.edu/_brickFiles/2026_01/SCDB_2026_01_caseCentered_Citation.csv.zip"


def main():
    z = zipfile.ZipFile(io.BytesIO(requests.get(URL, headers={"User-Agent": USER_AGENT}, timeout=120).content))
    rows = list(csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="latin-1")))
    terms = sorted({r["term"] for r in rows})[-5:]
    out = []
    for t in terms:
        cases = [r for r in rows if r["term"] == t]
        direction = Counter(r["decisionDirection"] for r in cases)
        out.append({"term": t, "cases": len(cases), "liberal_share": round(direction["2"] / (direction["1"] + direction["2"]), 2),
                    "unanimous": sum(r["minVotes"] == "0" for r in cases)})
    show({"cases_in_database": len(rows), "last_5_terms": out})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:17</summary>

```json
{
  "cases_in_database": 9409,
  "last_5_terms": [
    {
      "term": "2021",
      "cases": 65,
      "liberal_share": 0.35,
      "unanimous": 19
    },
    {
      "term": "2022",
      "cases": 58,
      "liberal_share": 0.45,
      "unanimous": 29
    },
    {
      "term": "2023",
      "cases": 59,
      "liberal_share": 0.51,
      "unanimous": 26
    },
    {
      "term": "2024",
      "cases": 64,
      "liberal_share": 0.51,
      "unanimous": 28
    },
    {
      "term": "2025",
      "cases": 66,
      "liberal_share": 0.52,
      "unanimous": 29
    }
  ]
}
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

**Endpoint:** `GET https://data.cdc.gov/resource/{dataset-id}.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `dataset-id` | 4x4 ID from the dataset page (API button) | `swc5-untb (PLACES, county)` |
| `column=value` | Simple equality filter on any column | `measureid=DEPRESSION` |
| `$select` | Columns or aggregates | `stateabbr,avg(data_value)` |
| `$where` | SQL-like filter | `stateabbr in ('DC','MD','VA')` |
| `$group` | Group-by for aggregates | `stateabbr` |
| `$order` | Sort | `data_value DESC` |
| `$limit / $offset` | Paging (default limit 1,000) | `5` |
| `$q` | Full-text search across columns | `depression` |
| `X-App-Token (header)` | Optional app token | `<SOCRATA_APP_TOKEN>` |

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

**Endpoint:** `GET https://clinicaltrials.gov/api/v2/studies`

| Parameter | Description | Example |
|:--|:--|:--|
| `query.cond / query.intr / query.term` | Condition / intervention / general search | `Alzheimer Disease` |
| `query.locn / query.spons` | Location / sponsor search | `Washington, DC` |
| `filter.overallStatus` | RECRUITING \| COMPLETED \| ACTIVE_NOT_RECRUITING \| ... (comma list) | `RECRUITING` |
| `filter.geo` | distance(lat,lon,radius) | `distance(38.9,-77.04,50mi)` |
| `fields` | Fields to return (piece names) | `NCTId,BriefTitle,Phase` |
| `pageSize / pageToken` | Max 1,000 per page; nextPageToken for paging | `5` |
| `countTotal` | true → include totalCount | `true` |
| `format` | json \| csv | `json` |
| `/api/v2/studies/{nctId}` | Sibling endpoint: one full study record | `NCT05123456` |

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

**Endpoint:** `GET https://api.fda.gov/{category}/{endpoint}.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `category/endpoint` | drug/event, drug/label, drug/enforcement, device/event, food/enforcement, ... | `drug/event` |
| `search` | field:term with AND/OR (+), ranges [A TO B] | `patient.drug.openfda.generic_name:"semaglutide"` |
| `count` | Aggregate counts by a field (.exact for whole phrases) | `patient.reaction.reactionmeddrapt.exact` |
| `limit / skip` | Max 1,000 per request; skip up to 25,000 | `6` |
| `sort` | field:asc\|desc | `receivedate:desc` |
| `api_key` | Optional key | `<OPENFDA_API_KEY>` |

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

**Endpoint:** `GET https://ourworldindata.org/grapher/{chart-slug}.csv`

| Parameter | Description | Example |
|:--|:--|:--|
| `chart-slug` | The slug from any OWID chart URL | `happiness-cantril-ladder` |
| `useColumnShortNames` | true → snake_case column names | `true` |
| `csvType` | full (all countries/years) \| filtered (as currently shown in the chart) | `full` |
| `{chart-slug}.metadata.json` | Sibling endpoint: sources, units, descriptions | `happiness-cantril-ladder.metadata.json` |
| `{chart-slug}.zip` | CSV + metadata + README in one download | `happiness-cantril-ladder.zip` |

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

**Endpoint:** `GET https://ghoapi.azureedge.net/api/{IndicatorCode}`

| Parameter | Description | Example |
|:--|:--|:--|
| `IndicatorCode` | Indicator; browse via /api/Indicator | `WHOSIS_000001 (life expectancy)` |
| `$filter` | OData filter on SpatialDim (country), TimeDim (year), Dim1 (sex/other) | `SpatialDim eq 'USA' and TimeDim eq 2021` |
| `$select` | Columns to return | `SpatialDim,TimeDim,NumericValue` |
| `$top` | Limit rows | `10` |
| `/api/Indicator?$filter=contains(IndicatorName,'…')` | Sibling endpoint: search indicator codes | `contains(IndicatorName,'obesity')` |
| `/api/DIMENSION/COUNTRY/DimensionValues` | Sibling endpoint: country codes and names | `—` |

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

## 🎓 Education

### College Scorecard (US Dept. of Education)

| | |
|:--|:--|
| **Docs** | https://collegescorecard.ed.gov/data/api-documentation/ |
| **Auth** | api.data.gov key (`DEMO_KEY` works at low volume) |
| **Register / token** | Use your api.data.gov key (https://api.data.gov/signup/). 1,000 req/hour; DEMO_KEY 30/hour. |
| **Env var** | `DATA_GOV_API_KEY` |
| **Script** | [`scripts/education/college_scorecard.py`](scripts/education/college_scorecard.py) |

**What you can get:**

- Every US college: admissions rate, SAT/ACT ranges, enrollment, demographics, tuition & net price
- Completion and retention rates, student debt, loan repayment
- Earnings of graduates by institution and by field of study

**Capstone ideas:** College ROI / value-added models, equity in access and outcomes, predicting graduation rates.

**Endpoint:** `GET https://api.data.gov/ed/collegescorecard/v1/schools.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `school.name / school.state / id` | Filters (id = IPEDS unit id) | `school.state=DC` |
| `{field}__range` | Numeric range filter | `latest.student.size__range=5000..` |
| `fields` | Comma-separated dotted field names (see the data dictionary) | `school.name,latest.admissions.admission_rate.overall` |
| `sort` | field:asc \| field:desc | `latest.student.size:desc` |
| `per_page / page` | Max 100 per page | `5` |
| `latest.programs.cip_4_digit` | Field-of-study level data (earnings, debt by program) | `fields=latest.programs.cip_4_digit` |
| `api_key` | api.data.gov key | `DEMO_KEY` |

```python
import os

from _common import get, show

FIELDS = {
    "school.name": "name",
    "latest.student.size": "undergrads",
    "latest.admissions.admission_rate.overall": "admit_rate",
    "latest.cost.tuition.out_of_state": "tuition_out_of_state",
    "latest.completion.consumer_rate": "completion_rate",
    "latest.earnings.10_yrs_after_entry.median": "median_earnings_10yr",
}


def main():
    d = get("https://api.data.gov/ed/collegescorecard/v1/schools.json", params={
        "api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY"), "school.state": "DC",
        "school.degrees_awarded.predominant": 3, "fields": ",".join(FIELDS),
        "sort": "latest.student.size:desc", "per_page": 5})
    show({"total": d["metadata"]["total"],
          "schools": [{FIELDS[k]: s.get(k) for k in FIELDS} for s in d["results"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:46</summary>

```json
{
  "total": 10,
  "schools": [
    {
      "name": "George Washington University",
      "undergrads": 11182,
      "admit_rate": 0.4709,
      "tuition_out_of_state": 67710,
      "completion_rate": 0.8433,
      "median_earnings_10yr": 90873
    },
    {
      "name": "Howard University",
      "undergrads": 10108,
      "admit_rate": 0.4134,
      "tuition_out_of_state": 35810,
      "completion_rate": 0.6426,
      "median_earnings_10yr": 63066
    },
    {
      "name": "Strayer University-Global Region",
      "undergrads": 9376,
      "admit_rate": null,
      "tuition_out_of_state": 13920,
      "completion_rate": 0.1254,
      "median_earnings_10yr": 40092
    },
    {
      "name": "Georgetown University",
      "undergrads": 7569,
      "admit_rate": 0.1291,
      "tuition_out_of_state": 68017,
      "completion_rate": 0.9493,
      "median_earnings_10yr": 103494
    },
    {
      "name": "American University",
      "undergrads": 7266,
      "admit_rate": 0.6196,
      "tuition_out_of_state": 58771,
      "completion_rate": 0.7873,
      "median_earnings_10yr": 77370
    }
  ]
}
```
</details>

---

### ERIC (Education Resources Information Center)

| | |
|:--|:--|
| **Docs** | https://eric.ed.gov/?api |
| **Auth** | None |
| **Register / token** | No registration. Run by the Institute of Education Sciences (IES). |
| **Script** | [`scripts/education/eric.py`](scripts/education/eric.py) |

**What you can get:**

- 1.9M+ education research records: title, authors, abstract, descriptors (subject terms), publication type
- Peer-review flag, full-text availability (ERIC-hosted PDFs), education level, audience, source journal

**Capstone ideas:** Topic modeling of education research trends, systematic-review automation, citation-free literature mapping.

**Endpoint:** `GET https://api.ies.ed.gov/eric/`

| Parameter | Description | Example |
|:--|:--|:--|
| `search` | Query; supports field syntax (title:, author:, subject:, source:) and AND / OR | `subject:"mathematics anxiety"` |
| `fields` | Comma-separated fields to return | `id,title,author,publicationdateyear,peerreviewed` |
| `format` | json \| xml \| csv | `json` |
| `rows / start` | Max 2,000 per request; offset for paging | `5` |
| `Filter terms in search` | peerreviewed:T, e_fulltextauth:1, publicationdateyear:2024, educationlevel:"Higher Education" | `peerreviewed:T` |

```python
from _common import get, show


def main():
    d = get("https://api.ies.ed.gov/eric/", params={
        "search": 'subject:"artificial intelligence" AND peerreviewed:T AND publicationdateyear:2025',
        "fields": "id,title,author,source,publicationdateyear", "format": "json", "rows": 5})["response"]
    show({"numFound": d["numFound"], "docs": d["docs"]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:46</summary>

```json
{
  "numFound": 3226,
  "docs": [
    {
      "id": "EJ1475514",
      "title": "The Role of the Expressive Therapies Continuum in Human Creativity in the Age of AI",
      "author": [
        "Nancy S. Choe",
        "Lisa D. Hinz"
      ],
      "source": "Art Therapy: Journal of the American Art Therapy Association",
      "publicationdateyear": 2025
    },
    {
      "id": "EJ1493662",
      "title": "Information Literacy in the Artificial Intelligence Era: A Proposed Framework",
      "author": [
        "Radwan Ali",
        "Matthew L. Wilson"
      ],
      "source": "Communications in Information Literacy",
      "publicationdateyear": 2025
    },
    {
      "id": "EJ1494133",
      "title": "&quot;The Cat Sat on the …?&quot; Why Generative AI Has Limited Creativity",
      "author": [
        "David H. Cropley"
      ],
      "source": "Journal of Creative Behavior",
      "publicationdateyear": 2025
    },
    {
      "id": "EJ1475713",
      "title": "Machine Bias. How Do Generative Language Models Answer Opinion Polls?",
      "author": [
        "Julien Boelaert",
        "Samuel Coavoux",
        "Étienne Ollion",
        "Ivaylo Petev",
        "Patrick Präg"
      ],
      "source": "Sociological Methods & Research",
      "publicationdateyear": 2025
    },
    {
      "id": "EJ1470444",
      "title": "Use of Artificial Neural Network in Educational Laboratory Applications: Low-Velocity Impact Test",
      "author": [
        "Yusuf Uzun",
        "Mehmet Kayrici"
      ],
      "source": "Journal of Education in Science, Environment and Health",
      "publicationdateyear": 2025
    }
  ]
}
```
</details>

---

### Hipolabs Universities List

| | |
|:--|:--|
| **Docs** | https://github.com/Hipo/university-domains-list |
| **Auth** | None |
| **Register / token** | No registration. Community-maintained list (~10k universities); download the JSON from GitHub for bulk use. |
| **Script** | [`scripts/education/hipolabs_universities.py`](scripts/education/hipolabs_universities.py) |

**What you can get:**

- University name, country, state/province, website, and email domains worldwide

**Capstone ideas:** Mapping .edu email domains to institutions, linking institutions across datasets.

**Endpoint:** `GET http://universities.hipolabs.com/search`

| Parameter | Description | Example |
|:--|:--|:--|
| `name` | Substring of the university name | `George Washington` |
| `country` | Country name | `United States` |
| `domain` | Web domain | `gwu.edu` |
| `limit / offset` | Paging | `5` |

```python
from _common import get, show


def main():
    d = get("http://universities.hipolabs.com/search", params={"name": "Washington", "country": "United States", "limit": 5})
    show([{"name": u["name"], "domains": u["domains"], "web": u["web_pages"][0]} for u in d])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:46</summary>

```json
[
  {
    "name": "Washington University, Saint Louis",
    "domains": [
      "wustl.edu"
    ],
    "web": "http://www.wustl.edu/"
  },
  {
    "name": "Washington & Jefferson College",
    "domains": [
      "washjeff.edu"
    ],
    "web": "http://www.washjeff.edu/"
  },
  {
    "name": "Washington & Lee University",
    "domains": [
      "wlu.edu"
    ],
    "web": "http://www.wlu.edu/"
  },
  {
    "name": "Washington State University",
    "domains": [
      "wsu.edu"
    ],
    "web": "http://www.wsu.edu/"
  },
  {
    "name": "Washington State University at Vancouver",
    "domains": [
      "vancouver.wsu.edu"
    ],
    "web": "http://www.vancouver.wsu.edu/"
  }
]
```
</details>

---

### NAEP — The Nation's Report Card

| | |
|:--|:--|
| **Docs** | https://www.nationsreportcard.gov/api_documentation.aspx |
| **Auth** | None |
| **Register / token** | No registration. |
| **Script** | [`scripts/education/naep.py`](scripts/education/naep.py) |

**What you can get:**

- National, state and large-district (TUDA) assessment results in math, reading, science, writing, civics, history
- Average scale scores, achievement-level percentages, percentiles, gaps by race, gender, school lunch eligibility
- Trends back to the 1990s with standard errors

**Capstone ideas:** Pandemic learning-loss analysis, state policy evaluation, achievement-gap trends.

**Endpoint:** `GET https://www.nationsreportcard.gov/DataService/GetAdhocData.aspx?type=data`

| Parameter | Description | Example |
|:--|:--|:--|
| `subject` | mathematics \| reading \| science \| writing \| civics \| ushistory \| ... | `mathematics` |
| `grade` | 4 \| 8 \| 12 | `8` |
| `subscale` | Composite scale code: MRPCM (math), RRPCM (reading), SRPUV (science) | `MRPCM` |
| `variable` | Reporting group: TOTAL, GENDER, SDRACE, SLUNCH3, ... | `TOTAL` |
| `jurisdiction` | NP (national public), state codes, TUDA districts (comma list) | `NP,DC,MD,VA` |
| `stattype` | MN:MN (mean), ALC:BB/AB/PR/AD (achievement levels), RP:… (percentiles) | `MN:MN` |
| `Year` | Assessment year(s), comma list | `2019,2022,2024` |

```python
from _common import get, show


def main():
    d = get("https://www.nationsreportcard.gov/DataService/GetAdhocData.aspx", params={
        "type": "data", "subject": "mathematics", "grade": 8, "subscale": "MRPCM", "variable": "TOTAL",
        "jurisdiction": "NP,DC,MD,VA", "stattype": "MN:MN", "Year": "2019,2024"})
    show([{"year": r["year"], "jurisdiction": r["jurisLabel"], "grade8_math_avg_score": round(r["value"], 1)}
          for r in d["result"]], max_items=8)
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:46</summary>

```json
[
  {
    "year": 2019,
    "jurisdiction": "District of Columbia",
    "grade8_math_avg_score": 269.2
  },
  {
    "year": 2019,
    "jurisdiction": "Maryland",
    "grade8_math_avg_score": 280.1
  },
  {
    "year": 2019,
    "jurisdiction": "National public",
    "grade8_math_avg_score": 281.0
  },
  {
    "year": 2019,
    "jurisdiction": "Virginia",
    "grade8_math_avg_score": 287.1
  },
  {
    "year": 2024,
    "jurisdiction": "District of Columbia",
    "grade8_math_avg_score": 261.5
  },
  {
    "year": 2024,
    "jurisdiction": "Maryland",
    "grade8_math_avg_score": 268.2
  },
  {
    "year": 2024,
    "jurisdiction": "National public",
    "grade8_math_avg_score": 272.2
  },
  {
    "year": 2024,
    "jurisdiction": "Virginia",
    "grade8_math_avg_score": 275.0
  }
]
```
</details>

---

### UNESCO Institute for Statistics (UIS) Data API

| | |
|:--|:--|
| **Docs** | https://api.uis.unesco.org/api/public/documentation/ |
| **Auth** | None |
| **Register / token** | No registration (fair-use limits). The official global source for SDG 4 education indicators. |
| **Script** | [`scripts/education/unesco_uis.py`](scripts/education/unesco_uis.py) |

**What you can get:**

- Enrollment, completion, out-of-school rates, literacy, learning outcomes, teachers, education spending
- Every country and region, ~1970-today, plus science/culture indicators

**Capstone ideas:** Global education gaps, spending vs. outcomes, gender parity in schooling.

**Endpoint:** `GET https://api.uis.unesco.org/api/public/data/indicators`

| Parameter | Description | Example |
|:--|:--|:--|
| `indicator` | Indicator ID(s); browse via /api/public/definitions/indicators | `CR.1 (completion rate, primary)` |
| `geoUnit` | ISO3 country or region code(s) | `USA` |
| `geoUnitType` | NATIONAL \| REGIONAL (instead of geoUnit) | `NATIONAL` |
| `start / end` | Year range | `2015 / 2024` |
| `indicatorMetadata / footnotes` | true → include definitions and footnotes | `true` |
| `/api/public/versions` | Data release versions (pin a version for reproducibility) | `—` |

```python
from _common import get, show


def main():
    d = get("https://api.uis.unesco.org/api/public/data/indicators", params={
        "indicator": "CR.1", "geoUnit": ["USA", "IND", "NGA", "BRA", "DEU"], "start": 2015})
    latest = {}
    for r in d["records"]:  # keep the most recent year per country
        if r["value"] is not None and r["year"] >= latest.get(r["geoUnit"], {}).get("year", 0):
            latest[r["geoUnit"]] = r
    show([{"country": g, "year": r["year"], "primary_completion_rate_pct": round(r["value"], 1)}
          for g, r in latest.items()])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:46</summary>

```json
[
  {
    "country": "BRA",
    "year": 2024,
    "primary_completion_rate_pct": 97.6
  },
  {
    "country": "IND",
    "year": 2019,
    "primary_completion_rate_pct": 94.2
  },
  {
    "country": "NGA",
    "year": 2024,
    "primary_completion_rate_pct": 77.4
  },
  {
    "country": "USA",
    "year": 2024,
    "primary_completion_rate_pct": 99.6
  }
]
```
</details>

---

### Urban Institute Education Data Portal

| | |
|:--|:--|
| **Docs** | https://educationdata.urban.org/documentation/ |
| **Auth** | None |
| **Register / token** | No registration. Cleaned, harmonized federal education data (CCD, IPEDS, CRDC, EDFacts, Scorecard, SAIPE). R/Stata packages: `educationdata`. |
| **Script** | [`scripts/education/urban_education_data.py`](scripts/education/urban_education_data.py) |

**What you can get:**

- K-12: every public school and district — enrollment by race/grade, teachers, free/reduced lunch, finance
- Civil Rights Data Collection: discipline, AP/IB access, chronic absenteeism
- Higher ed (IPEDS): admissions, enrollment, aid, graduation, finance

**Capstone ideas:** School segregation, discipline disparities, funding vs. outcomes, district-level panel studies.

**Endpoint:** `GET https://educationdata.urban.org/api/v1/{level}/{source}/{topic}/{year}/[{subtopic}]/`

| Parameter | Description | Example |
|:--|:--|:--|
| `level` | schools \| school-districts \| college-university | `schools` |
| `source` | ccd \| crdc \| edfacts \| ipeds \| scorecard \| saipe \| nhgis | `ccd` |
| `topic / subtopic` | e.g. directory, enrollment/grade-{n}/race, finance, admissions-enrollment, grad-rates | `enrollment/2022/grade-99` |
| `year` | School year (fall) | `2022` |
| `fips / leaid / ncessch / unitid` | Filters: state FIPS, district ID, school ID, college ID | `fips=11` |
| `/api/v1/{…}/summaries?var=&stat=&by=` | Server-side aggregation (sum, avg, count...) grouped by a variable | `var=enrollment&stat=sum&by=fips` |
| `page / per_page` | Results paged (default 10,000; follow the next URL) | `—` |

```python
from _common import get, show


def main():
    d = get("https://educationdata.urban.org/api/v1/school-districts/ccd/directory/2022/", params={"fips": 11})
    rows = sorted(d["results"], key=lambda r: -(r.get("enrollment") or 0))
    show({"count": d["count"], "largest_DC_districts": [
        {k: r.get(k) for k in ("lea_name", "enrollment", "number_of_schools", "teachers_total_fte")} for r in rows]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:46</summary>

```json
{
  "count": 71,
  "largest_DC_districts": [
    {
      "lea_name": "District of Columbia Public Schools",
      "enrollment": 50131,
      "number_of_schools": 116,
      "teachers_total_fte": 4389
    },
    {
      "lea_name": "KIPP DC PCS",
      "enrollment": 7361,
      "number_of_schools": 20,
      "teachers_total_fte": 596
    },
    {
      "lea_name": "Friendship PCS",
      "enrollment": 4609,
      "number_of_schools": 15,
      "teachers_total_fte": 419
    },
    {
      "lea_name": "DC Prep PCS",
      "enrollment": 2128,
      "number_of_schools": 6,
      "teachers_total_fte": 177
    },
    {
      "lea_name": "Carlos Rosario International PCS",
      "enrollment": 1984,
      "number_of_schools": 0,
      "teachers_total_fte": 0
    },
    "... (66 more)"
  ]
}
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

**Endpoint:** `GET http://export.arxiv.org/api/query`

| Parameter | Description | Example |
|:--|:--|:--|
| `search_query` | Field prefixes ti: au: abs: cat: all: combined with AND / OR / ANDNOT | `cat:cs.CL AND abs:retrieval` |
| `id_list` | Comma-separated arXiv IDs | `1706.03762` |
| `start / max_results` | Offset and page size (≤ 2,000 per call) | `0 / 3` |
| `sortBy` | relevance \| lastUpdatedDate \| submittedDate | `submittedDate` |
| `sortOrder` | ascending \| descending | `descending` |

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

**Endpoint:** `GET https://api.crossref.org/works`

| Parameter | Description | Example |
|:--|:--|:--|
| `query.bibliographic / query.author / query` | Citation-style, author, or general search | `data science education` |
| `filter` | Comma list: from-pub-date, until-pub-date, type, has-funder, issn, ... | `from-pub-date:2020-01-01,type:journal-article` |
| `sort / order` | is-referenced-by-count \| published \| relevance; asc \| desc | `is-referenced-by-count / desc` |
| `rows / offset` | Max 1,000 per page | `3` |
| `cursor` | Deep paging (start with *) | `*` |
| `select` | Fields to return | `DOI,title,author` |
| `mailto` | Your email → polite pool | `you@gwu.edu` |
| `/works/{doi}` | Sibling endpoint: one DOI's metadata | `/works/10.1038/nature14539` |

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

**Endpoint:** `GET https://api.nasa.gov/neo/rest/v1/feed`

| Parameter | Description | Example |
|:--|:--|:--|
| `start_date / end_date` | NeoWs date range, max 7 days | `2026-10-01` |
| `api_key` | api.data.gov key | `DEMO_KEY` |
| `/planetary/apod?date=&count=` | Sibling endpoint: Astronomy Picture of the Day | `date=2026-10-01` |
| `/DONKI/FLR?startDate=&endDate=` | Sibling endpoint: solar flares (also CME, GST, ...) | `startDate=2026-09-01` |
| `eonet.gsfc.nasa.gov/api/v3/events?category=&status=` | EONET natural events (no key) | `category=wildfires&status=open` |

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

**Endpoint:** `GET https://api.openalex.org/works`

| Parameter | Description | Example |
|:--|:--|:--|
| `search` | Full-text search over title, abstract, fulltext | `large language models` |
| `filter` | Comma-separated (AND) attribute filters | `from_publication_date:2024-01-01,is_oa:true` |
| `sort` | cited_by_count \| publication_date \| relevance_score, with :desc | `cited_by_count:desc` |
| `per-page / page` | Max 200 per page | `5` |
| `cursor` | Deep paging beyond 10,000 results (start with *) | `*` |
| `select` | Return only these fields | `id,title,cited_by_count` |
| `group_by` | Counts by a field | `publication_year` |
| `mailto` | Your email → faster polite pool | `you@gwu.edu` |
| `/authors · /institutions · /sources · /topics · /funders` | Other entity endpoints with the same params | `/institutions?search=george washington` |

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

**Endpoint:** `GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{namespace}/{identifiers}/{operation}/{output}`

| Parameter | Description | Example |
|:--|:--|:--|
| `namespace` | cid \| name \| smiles \| inchikey \| formula | `cid` |
| `identifiers` | Comma list of CIDs, or one name/SMILES | `2519,2244,3672` |
| `operation` | property/<list> \| synonyms \| cids \| description \| record | `property/MolecularWeight,XLogP` |
| `output` | JSON \| CSV \| XML \| TXT \| PNG | `JSON` |
| `name → CID` | Look up a compound ID by name | `/compound/name/caffeine/cids/JSON` |
| `fastsimilarity_2d` | Similar structures; Threshold= 0-100 | `/compound/fastsimilarity_2d/cid/2244/cids/JSON?Threshold=95` |

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

**Endpoint:** `GET https://api.semanticscholar.org/graph/v1/paper/search`

| Parameter | Description | Example |
|:--|:--|:--|
| `query` | Plain-text search | `attention is all you need` |
| `fields` | title, abstract, year, authors, citationCount, tldr, embedding, openAccessPdf, ... | `title,year,citationCount,tldr` |
| `limit / offset` | Max 100 per page | `3` |
| `year` | Year or range | `2020-2024` |
| `fieldsOfStudy` | Comma list | `Computer Science` |
| `/graph/v1/paper/{id}` | Sibling endpoint: one paper; id can be DOI:, ARXIV:, CorpusId: | `ARXIV:1706.03762` |
| `/graph/v1/paper/{id}/citations` | Sibling endpoint: citing papers (also /references) | `fields=title,year` |
| `x-api-key (header)` | Optional key | `<S2_API_KEY>` |

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

## 🤖 AI & Machine Learning

### Anthropic Claude API

| | |
|:--|:--|
| **Docs** | https://platform.claude.com/docs/en/api/overview · models & pricing: https://platform.claude.com/docs/en/about-claude/pricing |
| **Auth** | API key (header `x-api-key`; the Python SDK reads `ANTHROPIC_API_KEY`) |
| **Register / token** | Create an account at https://platform.claude.com -> Settings -> API Keys -> Create Key, then add credits under Billing (pay-as-you-go). Example prices per million tokens: Claude Opus 5.5 $4 in / $20 out, Claude Haiku 5.5 $0.10 / $0.50. Install the SDK with `pip install anthropic`. |
| **Env var** | `ANTHROPIC_API_KEY` |
| **Script** | [`scripts/ai_ml/anthropic_claude.py`](scripts/ai_ml/anthropic_claude.py) |

**What you can get:**

- Text generation, summarization, classification, extraction, and Q&A over long documents (1M-token context)
- Vision (images, PDFs), tool use / function calling, structured JSON outputs
- Batch API (50% cheaper, asynchronous), prompt caching, token counting, Models API

**Capstone ideas:** LLM-assisted labeling of text data, summarizing documents pulled from the other APIs here, LLM-as-judge evaluation.

**Endpoint:** `POST https://api.anthropic.com/v1/messages  (Python SDK: client.messages.create / client.beta.messages.create)`

| Parameter | Description | Example |
|:--|:--|:--|
| `x-api-key / anthropic-version (headers)` | Your key; API version header for raw HTTP (the SDK sets both) | `anthropic-version: 2023-06-01` |
| `model` | claude-opus-5-5 (default, most capable Opus), claude-sonnet-5-5, claude-haiku-5-5 (fastest, cheapest) | `claude-opus-5-5` |
| `max_tokens` | Required cap on output tokens; ~16,000 for non-streaming calls (stream for larger) | `16000` |
| `messages` | List of {role: user \| assistant, content}; the first must be user | `[{"role": "user", "content": "..."}]` |
| `system` | System prompt | `You are a concise research assistant.` |
| `output_config.effort` | low \| medium \| high \| xhigh \| max - how much the model thinks (default medium on Opus 5.5) | `low` |
| `output_config.format` | Structured outputs: force JSON that matches a schema | `{"type": "json_schema", "schema": {...}}` |
| `stream` | true -> server-sent events (SDK: client.messages.stream) | `true` |
| `tools` | Tool / function definitions the model can call | `[{"name": ..., "input_schema": {...}}]` |
| `fallbacks + anthropic-beta header` | "default" retries a safety-classifier refusal on a recommended fallback model | `server-side-fallback-2026-07-01` |
| `temperature / top_p / top_k` | Not accepted on current models (returns 400) - steer with the prompt and effort instead | `—` |

```python
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
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "model": "claude-opus-5-5",
  "stop_reason": "end_turn",
  "text": "1. Policy-text classifier: label Federal Register rules by topic with an LLM, then train a cheap model on the labels.\n2. News-to-markets: summarize GDELT coverage per company and test whether sentiment predicts SEC-filing events.\n3. Education Q&A: answer questions over College Scorecard data with tool calls that query the API live.",
  "usage": {"input_tokens": 41, "output_tokens": 96}
}
```
</details>

---

### Epoch AI — Data on AI Models

| | |
|:--|:--|
| **Docs** | https://epoch.ai/data/ai-models |
| **Auth** | None |
| **Register / token** | No registration. CSV downloads updated continuously; licensed CC-BY (cite Epoch AI). |
| **Script** | [`scripts/ai_ml/epoch_ai.py`](scripts/ai_ml/epoch_ai.py) |

**What you can get:**

- 3,000+ AI models: organization, release date, domain, task, parameters, training compute (FLOP)
- Training dataset size, hardware, cost estimates, open-weights status, country
- Separate 'notable' and 'frontier' subsets

**Capstone ideas:** Scaling-trend analysis (compute over time), forecasting training costs, open vs closed model studies.

**Endpoint:** `GET https://epoch.ai/data/{file}.csv`

| Parameter | Description | Example |
|:--|:--|:--|
| `all_ai_models.csv` | Every tracked model | `all_ai_models.csv` |
| `notable_ai_models.csv` | Models meeting notability criteria (state of the art, highly cited, ...) | `notable_ai_models.csv` |
| `frontier_ai_models.csv` | Models at the frontier of training compute when released | `frontier_ai_models.csv` |
| `Columns` | Model, Organization, Publication date, Domain, Parameters, Training compute (FLOP), Open model weights?, ... | `—` |

```python
import csv
import io

from _common import get, show


def main():
    rows = list(csv.DictReader(io.StringIO(get("https://epoch.ai/data/notable_ai_models.csv", as_json=False))))
    with_compute = [r for r in rows if r["Training compute (FLOP)"]]
    largest = sorted(with_compute, key=lambda r: -float(r["Training compute (FLOP)"]))[:5]
    show({"notable_models": len(rows), "largest_training_compute": [
        {"model": r["Model"], "org": r["Organization"], "date": r["Publication date"],
         "training_FLOP": f'{float(r["Training compute (FLOP)"]):.1e}', "parameters": r["Parameters"] or None}
        for r in largest]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:51</summary>

```json
{
  "notable_models": 1078,
  "largest_training_compute": [
    {
      "model": "GPT-6 Astra",
      "org": "OpenAI",
      "date": "2026-09-03",
      "training_FLOP": "1.0e+27",
      "parameters": null
    },
    {
      "model": "Grok 4",
      "org": "xAI",
      "date": "2025-07-09",
      "training_FLOP": "5.0e+26",
      "parameters": "3000000000000.0"
    },
    {
      "model": "GPT-4.5",
      "org": "OpenAI",
      "date": "2025-02-27",
      "training_FLOP": "3.8e+26",
      "parameters": null
    },
    {
      "model": "Grok 3",
      "org": "xAI",
      "date": "2025-02-17",
      "training_FLOP": "3.5e+26",
      "parameters": "3000000000000.0"
    },
    {
      "model": "GPT-5",
      "org": "OpenAI",
      "date": "2025-08-07",
      "training_FLOP": "6.6e+25",
      "parameters": null
    }
  ]
}
```
</details>

---

### Google Gemini API

| | |
|:--|:--|
| **Docs** | https://ai.google.dev/gemini-api/docs |
| **Auth** | API key (header `x-goog-api-key`) |
| **Register / token** | Get a key at https://aistudio.google.com/apikey (Google account). A free tier with lower rate limits is available for some models; see https://ai.google.dev/gemini-api/docs/rate-limits. |
| **Env var** | `GEMINI_API_KEY` |
| **Script** | [`scripts/ai_ml/google_gemini.py`](scripts/ai_ml/google_gemini.py) |

**What you can get:**

- Text and multimodal generation (gemini-3.8-flash; gemini-3.5-flash-lite for high volume)
- Image, audio and video understanding; image generation and text-to-speech models
- Embeddings, Live API (real-time audio)

**Capstone ideas:** Multimodal capstones (video/audio analysis), low-cost bulk labeling with Flash-Lite.

**Endpoint:** `POST https://generativelanguage.googleapis.com/v1beta/interactions`

| Parameter | Description | Example |
|:--|:--|:--|
| `model` | gemini-3.8-flash \| gemini-3.5-flash-lite \| ... (GET /v1beta/models lists them) | `gemini-3.8-flash` |
| `input` | A string, or a list of content objects ({type: text \| image, ...}) | `Suggest three capstone ideas.` |
| `system_instruction` | System prompt | `You are a concise research assistant.` |
| `generation_config.temperature` | Sampling temperature | `1.0` |
| `generation_config.thinking_level` | How much the model thinks | `low` |
| `store / previous_interaction_id` | store=false for stateless calls; chain turns server-side by ID | `false` |
| `x-goog-api-key (header)` | Your key | `<GEMINI_API_KEY>` |

```python
import os

import requests

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
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "model": "gemini-3.8-flash",
  "text": "1. Predict wildfire smoke exposure by joining NASA EONET events with OpenAQ readings...\n2. ...\n3. ..."
}
```
</details>

---

### Hugging Face Daily Papers

| | |
|:--|:--|
| **Docs** | https://huggingface.co/docs/hub/en/api |
| **Auth** | None |
| **Register / token** | No registration. |
| **Script** | [`scripts/ai_ml/hf_daily_papers.py`](scripts/ai_ml/hf_daily_papers.py) |

**What you can get:**

- Curated daily list of new AI papers (arXiv) with upvotes, comment counts, AI-generated summary, keywords
- Linked models, datasets and Spaces for each paper
- Paper search by keyword

**Capstone ideas:** Tracking what the AI community pays attention to, predicting paper popularity from abstracts.

**Endpoint:** `GET https://huggingface.co/api/daily_papers`

| Parameter | Description | Example |
|:--|:--|:--|
| `limit` | Number of papers | `5` |
| `date` | A specific day, YYYY-MM-DD | `2026-10-09` |
| `sort` | publishedAt \| trending | `trending` |
| `/api/papers/search?q=` | Sibling endpoint: search papers | `q=diffusion` |
| `/api/papers/{arxiv_id}` | Sibling endpoint: one paper's metadata | `/api/papers/1706.03762` |

```python
from _common import get, show


def main():
    d = get("https://huggingface.co/api/daily_papers", params={"limit": 5})
    show([{"arxiv_id": p["paper"]["id"], "title": p["paper"]["title"], "upvotes": p["paper"].get("upvotes"),
           "comments": p.get("numComments"), "published": p["paper"]["publishedAt"][:10],
           "summary": p["paper"].get("summary")} for p in d], max_str=200)
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:51</summary>

```json
[
  {
    "arxiv_id": "2610.11216",
    "title": "The Lattice of Transition Laws",
    "upvotes": 5,
    "comments": 2,
    "published": "2026-10-08",
    "summary": "Diffusion and autoregression (AR) have long been seen as different categories of generative models, with diffusion specialising in continuous fields and AR specialising in discrete tokens. Recent work..."
  },
  {
    "arxiv_id": "2609.38527",
    "title": "Behavioral Persistence and Incomplete Functional Transfer of Co-evolved Communication in Evolutionary Robotics",
    "upvotes": 4,
    "comments": 2,
    "published": "2026-09-29",
    "summary": "This work evaluates the direct transfer of a co-evolved communication protocol from a 2D simulation to a 3D physical environment, without retraining the network weights. Two e-puck-type robots, contro..."
  },
  {
    "arxiv_id": "2610.09280",
    "title": "Evaluating the Transfer of Co-Evolved Communication from 2D to 3D Simulation",
    "upvotes": 4,
    "comments": 2,
    "published": "2026-10-07",
    "summary": "This work examines the transfer of a co-evolved communication mechanism between two robotic agents from a discrete two-dimensional (2D) simulator to a three-dimensional simulator with real physics (3D..."
  },
  {
    "arxiv_id": "2609.33987",
    "title": "Opera: A Verbal Critic Framework for Long-horizon Coding Agents",
    "upvotes": 14,
    "comments": 2,
    "published": "2026-10-08",
    "summary": "Long-horizon coding agents need timely corrections, yet feedback can be ineffective or even harmful when it misjudges ongoing work or fails to address the underlying problem. Existing critics focus on..."
  },
  {
    "arxiv_id": "2610.11169",
    "title": "Skill Constellations: Tracing the Supply Chain of Agent Skills on GitHub",
    "upvotes": 5,
    "comments": 3,
    "published": "2026-10-08",
    "summary": "Agent skills are SKILL.md instructions and scripts that AI coding agents such as Claude Code and Codex run with the permissions of their user. Developers share skills by copying them between repositor..."
  }
]
```
</details>

---

### Hugging Face Inference Providers

| | |
|:--|:--|
| **Docs** | https://huggingface.co/docs/inference-providers |
| **Auth** | None to list models; Hugging Face token (Bearer) to run them |
| **Register / token** | Create a token at https://huggingface.co/settings/tokens (enable 'Make calls to Inference Providers'). Free accounts get a small monthly credit; PRO accounts more. One token reaches many providers. |
| **Env var** | `HF_TOKEN (needed for chat completions)` |
| **Script** | [`scripts/ai_ml/hf_inference_providers.py`](scripts/ai_ml/hf_inference_providers.py) |

**What you can get:**

- One OpenAI-compatible endpoint for open-weight models (Qwen, Llama, DeepSeek, Mistral, ...) served by many providers
- Live model list with per-provider price, context length, latency, throughput, tool-use support
- Also text-to-image, speech, embeddings via provider-specific routes

**Capstone ideas:** Compare open-weight models on your task, cost/latency benchmarking across providers.

**Endpoint:** `GET https://router.huggingface.co/v1/models  ·  POST https://router.huggingface.co/v1/chat/completions`

| Parameter | Description | Example |
|:--|:--|:--|
| `model` | Hub model ID; append :provider or :fastest / :cheapest to pick a provider | `Qwen/Qwen3.8-27B:cheapest` |
| `messages` | OpenAI-style list of {role, content} | `[{"role": "user", "content": "..."}]` |
| `max_tokens` | Cap on generated tokens | `300` |
| `temperature / top_p` | Sampling controls | `0.7` |
| `stream` | true -> server-sent events | `true` |
| `tools / response_format` | Function calling / JSON output (where the provider supports it) | `{"type": "json_object"}` |
| `Authorization (header)` | Bearer token | `Bearer <HF_TOKEN>` |

```python
import os

import requests

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
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:51</summary>

```json
{
  "models_available": 127,
  "sample_models": [
    {
      "id": "Qwen/Qwen3.8-27B",
      "providers": 5,
      "cheapest_input_usd_per_M": 0.2
    },
    {
      "id": "deepseek-ai/DeepSeek-V4.1-Flash",
      "providers": 5,
      "cheapest_input_usd_per_M": 0.2
    },
    {
      "id": "zai-org/GLM-5.3-Flash",
      "providers": 7,
      "cheapest_input_usd_per_M": 0.15
    },
    {
      "id": "zai-org/GLM-5.3",
      "providers": 7,
      "cheapest_input_usd_per_M": 0.9
    },
    {
      "id": "meta-llama/Llama-3.1-8B-Instruct",
      "providers": 4,
      "cheapest_input_usd_per_M": 0.02
    }
  ]
}
```
</details>

---

### Kaggle API

| | |
|:--|:--|
| **Docs** | https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md |
| **Auth** | Basic auth with username + API key (dataset search currently answers without auth) |
| **Register / token** | Kaggle account -> Settings -> API -> 'Create New Token' downloads kaggle.json (username + key). CLI: `pip install kaggle`, then `kaggle datasets download -d owner/dataset`. Accept each competition's rules on the website first. |
| **Env var** | `KAGGLE_USERNAME, KAGGLE_KEY` |
| **Script** | [`scripts/ai_ml/kaggle.py`](scripts/ai_ml/kaggle.py) |

**What you can get:**

- Search 400k+ public datasets: title, owner, size, licence, usability rating, downloads, votes
- Download datasets and competition files
- Competitions, leaderboards, notebooks (kernels), models

**Capstone ideas:** Finding training data fast; studying which dataset properties predict popularity.

**Endpoint:** `GET https://www.kaggle.com/api/v1/datasets/list`

| Parameter | Description | Example |
|:--|:--|:--|
| `search` | Search term | `titanic` |
| `sortBy` | hottest \| votes \| updated \| active \| published | `votes` |
| `filetype` | all \| csv \| sqlite \| json \| bigQuery \| parquet | `csv` |
| `license / tagids / user` | Filters: licence group, tag IDs, owner | `license=cc` |
| `page` | Page number (20 per page) | `1` |
| `/api/v1/datasets/download/{owner}/{dataset}` | Download a dataset as a zip (auth required) | `/datasets/download/heptapod/titanic` |
| `/api/v1/competitions/list?search=` | Sibling endpoint: competitions (auth required) | `search=forecasting` |

```python
import os

import requests

from _common import USER_AGENT, show


def main():
    auth = (os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]) if os.getenv("KAGGLE_KEY") else None
    r = requests.get("https://www.kaggle.com/api/v1/datasets/list", auth=auth, timeout=30, headers={"User-Agent": USER_AGENT},
                     params={"search": "climate", "sortBy": "votes", "filetype": "csv"})
    r.raise_for_status()
    # Most field names in this response carry a "Nullable" suffix (ref does not)
    show([{"ref": d.get("ref"), "title": d.get("titleNullable"),
           "size_bytes": d.get("totalBytesNullable"), "license": d.get("licenseNameNullable"),
           "usability": round(d.get("usabilityRatingNullable") or 0, 2), "downloads": d.get("downloadCount"),
           "votes": d.get("voteCount")} for d in r.json()[:5]])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:51</summary>

```json
[
  {
    "ref": "jsphyg/weather-dataset-rattle-package",
    "title": "Rain in Australia",
    "size_bytes": 4020790,
    "license": "Other (specified in description)",
    "usability": 1.0,
    "downloads": 184449,
    "votes": 1894
  },
  {
    "ref": "nelgiriyewithana/countries-of-the-world-2023",
    "title": "Global Country Information Dataset 2023",
    "size_bytes": 24063,
    "license": "Attribution 4.0 International (CC BY 4.0)",
    "usability": 1.0,
    "downloads": 48059,
    "votes": 877
  },
  {
    "ref": "atharvaingle/crop-recommendation-dataset",
    "title": "Crop Recommendation Dataset",
    "size_bytes": 65234,
    "license": "Apache 2.0",
    "usability": 0.71,
    "downloads": 110597,
    "votes": 670
  },
  {
    "ref": "rohanrao/air-quality-data-in-india",
    "title": "Air Quality Data in India (2015 - 2020)",
    "size_bytes": 76469579,
    "license": "CC0: Public Domain",
    "usability": 1.0,
    "downloads": 98616,
    "votes": 667
  },
  {
    "ref": "nehaprabhavalkar/indian-food-101",
    "title": "Indian Food 101",
    "size_bytes": 6951,
    "license": "Data files © Original Authors",
    "usability": 1.0,
    "downloads": 40478,
    "votes": 628
  }
]
```
</details>

---

### Ollama (local LLMs)

| | |
|:--|:--|
| **Docs** | https://docs.ollama.com/api/introduction |
| **Auth** | None (runs on your own machine) |
| **Register / token** | Install from https://ollama.com/download, then `ollama pull llama3.2` (or any model from https://ollama.com/library). Free; speed depends on your hardware. Data never leaves your machine. |
| **Env var** | `OLLAMA_MODEL (optional, default llama3.2)` |
| **Script** | [`scripts/ai_ml/ollama.py`](scripts/ai_ml/ollama.py) |

**What you can get:**

- Chat / generation with open-weight models running locally (Llama, Qwen, Gemma, Mistral, DeepSeek...)
- Embeddings (/api/embed), JSON-schema structured output, tool calling
- OpenAI-compatible endpoint at /v1/chat/completions

**Capstone ideas:** Private-data projects (no data sent to a vendor), reproducible experiments with pinned local models.

**Endpoint:** `POST http://localhost:11434/api/chat`

| Parameter | Description | Example |
|:--|:--|:--|
| `model` | Any pulled model (GET /api/tags lists them) | `llama3.2` |
| `messages` | List of {role, content} | `[{"role": "user", "content": "..."}]` |
| `stream` | false -> one JSON response (default streams JSON lines) | `false` |
| `format` | "json" or a JSON schema for structured output | `json` |
| `options` | Sampling/runtime options: temperature, num_ctx, seed, ... | `{"temperature": 0, "seed": 42}` |
| `/api/embed` | Sibling endpoint: embeddings (model, input) | `model=nomic-embed-text` |

```python
import os
import sys

import requests

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
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "model": "llama3.2",
  "reply": "The EPA Air Quality System (AQS) and OpenAQ are two widely used public air-quality datasets.",
  "eval_count": 24
}
```
</details>

---

### OpenAI API

| | |
|:--|:--|
| **Docs** | https://developers.openai.com/api/docs |
| **Auth** | API key (header `Authorization: Bearer`) |
| **Register / token** | Create an account at https://platform.openai.com -> API keys -> Create new secret key, and add billing credits (pay-as-you-go). Small model example: gpt-6-luna at $0.10 in / $0.50 out per million tokens. |
| **Env var** | `OPENAI_API_KEY` |
| **Script** | [`scripts/ai_ml/openai.py`](scripts/ai_ml/openai.py) |

**What you can get:**

- Text generation via the Responses API (gpt-6-astra flagship, gpt-6.1-sol balanced, gpt-6-luna small)
- Embeddings, image generation, speech-to-text and text-to-speech, moderation
- Batch API, fine-tuning, file search

**Capstone ideas:** Embedding-based clustering of text datasets, zero-shot classification baselines, transcription of audio data.

**Endpoint:** `POST https://api.openai.com/v1/responses`

| Parameter | Description | Example |
|:--|:--|:--|
| `model` | gpt-6-astra \| gpt-6.1-sol \| gpt-6-luna (GET /v1/models lists what your key can use) | `gpt-6-luna` |
| `input` | A string, or a list of {role, content} messages | `Suggest three capstone ideas.` |
| `instructions` | System-style instructions | `You are a concise research assistant.` |
| `max_output_tokens` | Cap on generated tokens | `500` |
| `stream` | true -> server-sent events | `true` |
| `/v1/embeddings` | Sibling endpoint: vector embeddings (model, input) | `—` |
| `Authorization (header)` | Bearer token | `Bearer <OPENAI_API_KEY>` |

```python
import os

import requests

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
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "model": "gpt-6-luna",
  "text": "1. Track how news sentiment (GDELT) leads stock volatility...\n2. ...\n3. ...",
  "usage": {"input_tokens": 32, "output_tokens": 88}
}
```
</details>

---

### OpenML

| | |
|:--|:--|
| **Docs** | https://www.openml.org/apis |
| **Auth** | None for reading (free key to upload runs) |
| **Register / token** | No registration to read. Create an account at https://www.openml.org for a key to upload results. Python package: `pip install openml`. |
| **Env var** | `OPENML_API_KEY (optional)` |
| **Script** | [`scripts/ai_ml/openml.py`](scripts/ai_ml/openml.py) |

**What you can get:**

- 20,000+ ML datasets with meta-features (instances, features, classes, missing values)
- Tasks (dataset + target + evaluation split), benchmark suites (e.g. OpenML-CC18)
- Millions of runs: model + hyperparameters + evaluation scores

**Capstone ideas:** Meta-learning (predict which model wins from dataset features), AutoML benchmarking, reproducible baselines.

**Endpoint:** `GET https://www.openml.org/api/v1/json/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `data/list/{filters}` | Dataset list; path filters limit/N, offset/N, status/active, tag/X, number_instances/A..B | `data/list/limit/5/number_instances/1000..5000` |
| `data/{id}` | One dataset's description, licence, default target, file URL | `data/61 (iris)` |
| `data/qualities/{id}` | Meta-features for a dataset | `data/qualities/61` |
| `task/list/{filters}` | Tasks; filters type/1 (classification), tag/, data_id/ | `task/list/type/1/limit/5` |
| `evaluation/list/function/{metric}/task/{id}` | Best runs on a task by a metric | `function/predictive_accuracy/task/59` |
| `study/{id}` | Benchmark suite contents | `study/99 (OpenML-CC18)` |

```python
from _common import get, show


def main():
    d = get("https://www.openml.org/api/v1/json/data/list/limit/200/status/active/number_instances/1000..5000")
    rows = []
    for ds in d["data"]["dataset"][:5]:
        q = {x["name"]: int(float(x["value"])) for x in ds.get("quality", [])}  # values arrive as "3196.0"
        rows.append({"id": ds["did"], "name": ds["name"], "instances": q.get("NumberOfInstances"),
                     "features": q.get("NumberOfFeatures"), "classes": q.get("NumberOfClasses"),
                     "missing_values": q.get("NumberOfMissingValues")})
    show(rows)
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:51</summary>

```json
[
  {
    "id": 3,
    "name": "kr-vs-kp",
    "instances": 3196,
    "features": 37,
    "classes": 2,
    "missing_values": 0
  },
  {
    "id": 12,
    "name": "mfeat-factors",
    "instances": 2000,
    "features": 217,
    "classes": 10,
    "missing_values": 0
  },
  {
    "id": 14,
    "name": "mfeat-fourier",
    "instances": 2000,
    "features": 77,
    "classes": 10,
    "missing_values": 0
  },
  {
    "id": 16,
    "name": "mfeat-karhunen",
    "instances": 2000,
    "features": 65,
    "classes": 10,
    "missing_values": 0
  },
  {
    "id": 18,
    "name": "mfeat-morphological",
    "instances": 2000,
    "features": 7,
    "classes": 10,
    "missing_values": 0
  }
]
```
</details>

---

### OpenRouter

| | |
|:--|:--|
| **Docs** | https://openrouter.ai/docs/api_reference/overview |
| **Auth** | None to list models; API key (Bearer) to run them |
| **Register / token** | Sign in at https://openrouter.ai -> Keys -> Create Key. Pay-as-you-go credits; some models are free (IDs ending in ':free', with low rate limits). |
| **Env var** | `OPENROUTER_API_KEY (needed for chat completions)` |
| **Script** | [`scripts/ai_ml/openrouter.py`](scripts/ai_ml/openrouter.py) |

**What you can get:**

- One OpenAI-compatible API for hundreds of models from many labs (OpenAI, Anthropic, Google, Meta, Qwen, ...)
- Public model catalog: context length, modalities, per-token prices, max output
- Automatic provider fallback and routing by price or latency

**Capstone ideas:** Model-comparison studies at fixed cost, tracking LLM price trends over time from the public catalog.

**Endpoint:** `GET https://openrouter.ai/api/v1/models  ·  POST https://openrouter.ai/api/v1/chat/completions`

| Parameter | Description | Example |
|:--|:--|:--|
| `model` | Model ID from /models (add :free for free variants where offered) | `qwen/qwen3.8-27b` |
| `messages` | OpenAI-style list of {role, content} | `[{"role": "user", "content": "..."}]` |
| `max_tokens / temperature` | Output cap and sampling | `300 / 0.7` |
| `models` | Fallback list tried in order if the first model fails | `["model-a", "model-b"]` |
| `provider.sort` | Route by price \| throughput \| latency | `price` |
| `stream` | true -> server-sent events | `true` |
| `Authorization (header)` | Bearer key | `Bearer <OPENROUTER_API_KEY>` |

```python
import os

import requests

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
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:51</summary>

```json
{
  "models_available": 458,
  "free_models": 19,
  "newest": [
    {
      "id": "stepfun/step-5-preview",
      "context_length": 1000000,
      "usd_per_M_input": 1.0,
      "usd_per_M_output": 2.7
    },
    {
      "id": "anthropic/claude-haiku-5.5",
      "context_length": 1000000,
      "usd_per_M_input": 0.1,
      "usd_per_M_output": 0.5
    },
    {
      "id": "anthropic/claude-haiku-5.5:batch",
      "context_length": 1000000,
      "usd_per_M_input": 0.05,
      "usd_per_M_output": 0.25
    },
    {
      "id": "google/gemini-nano-banana-2.1",
      "context_length": 65536,
      "usd_per_M_input": 1.5,
      "usd_per_M_output": 7.5
    },
    {
      "id": "mistralai/mistral-large-4-0",
      "context_length": 1048576,
      "usd_per_M_input": 0.68,
      "usd_per_M_output": 2.09
    }
  ]
}
```
</details>

---

### UCI Machine Learning Repository

| | |
|:--|:--|
| **Docs** | https://github.com/uci-ml-repo/ucimlrepo |
| **Auth** | None |
| **Register / token** | No registration. Python package: `pip install ucimlrepo` (`fetch_ucirepo(id=53)` returns pandas DataFrames). |
| **Script** | [`scripts/ai_ml/uci_ml_repo.py`](scripts/ai_ml/uci_ml_repo.py) |

**What you can get:**

- 600+ classic ML datasets: abstract, task type, number of instances/features, target column, DOI, citation
- Direct CSV download URL for each dataset
- Variable-level metadata (type, role, units, missing values)

**Capstone ideas:** Classic benchmarks for teaching and baselines; fairness studies on Adult/German Credit.

**Endpoint:** `GET https://archive.ics.uci.edu/api/dataset`

| Parameter | Description | Example |
|:--|:--|:--|
| `id` | Dataset ID (from the dataset URL) | `53 (Iris)` |
| `name` | Dataset name instead of id | `Iris` |
| `data.data_url` | Field in the response: direct CSV download | `https://archive.ics.uci.edu/static/public/53/data.csv` |
| `data.variables` | Field in the response: name, role (Feature/Target), type, units | `—` |

```python
from _common import get, show


def main():
    out = []
    for i in (2, 186, 53):  # Adult, Wine Quality, Iris
        d = get("https://archive.ics.uci.edu/api/dataset", params={"id": i})["data"]
        out.append({k: d.get(k) for k in ("uci_id", "name", "area", "tasks", "num_instances", "num_features",
                                          "target_col", "data_url")})
    show(out)
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:51</summary>

```json
[
  {
    "uci_id": 2,
    "name": "Adult",
    "area": "Social Science",
    "tasks": [
      "Classification"
    ],
    "num_instances": 48842,
    "num_features": 14,
    "target_col": [
      "income"
    ],
    "data_url": "https://archive.ics.uci.edu/static/public/2/data.csv"
  },
  {
    "uci_id": 186,
    "name": "Wine Quality",
    "area": "Business",
    "tasks": [
      "Classification",
      "Regression"
    ],
    "num_instances": 4898,
    "num_features": 11,
    "target_col": [
      "quality"
    ],
    "data_url": "https://archive.ics.uci.edu/static/public/186/data.csv"
  },
  {
    "uci_id": 53,
    "name": "Iris",
    "area": "Biology",
    "tasks": [
      "Classification"
    ],
    "num_instances": 150,
    "num_features": 4,
    "target_col": [
      "class"
    ],
    "data_url": "https://archive.ics.uci.edu/static/public/53/data.csv"
  }
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

**Endpoint:** `GET https://api.eia.gov/v2/{route}/data/`

| Parameter | Description | Example |
|:--|:--|:--|
| `route` | Dataset route; GET /v2/{route} (no /data) to browse facets and children | `electricity/rto/region-data` |
| `frequency` | hourly \| local-hourly \| daily \| monthly \| quarterly \| annual (depends on route) | `hourly` |
| `data[0]` | Data column(s) to return | `value` |
| `facets[{name}][]` | Facet filters, e.g. respondent (balancing authority), type (D = demand, NG = net generation, DF = forecast) | `facets[respondent][]=PJM` |
| `start / end` | Period range in the route's format | `2026-10-01T00` |
| `sort[0][column] / sort[0][direction]` | Sort | `period / desc` |
| `length / offset` | Max 5,000 rows per request | `5` |
| `api_key` | Your key | `<EIA_API_KEY>` |

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

**Endpoint:** `GET https://api.github.com/search/repositories`

| Parameter | Description | Example |
|:--|:--|:--|
| `q` | Keywords + qualifiers: language:, topic:, stars:>N, created:>DATE, user:, org: | `topic:machine-learning language:python` |
| `sort` | stars \| forks \| updated \| help-wanted-issues | `stars` |
| `order` | desc \| asc | `desc` |
| `per_page / page` | Max 100 per page; search capped at 1,000 results | `5` |
| `/repos/{owner}/{repo}` | Sibling endpoint: repo detail (+ /commits, /issues?state=all, /contributors, /releases) | `/repos/pandas-dev/pandas` |
| `Authorization (header)` | Optional token (5,000 req/hour) | `Bearer <GITHUB_TOKEN>` |

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

**Endpoint:** `GET https://huggingface.co/api/models`

| Parameter | Description | Example |
|:--|:--|:--|
| `search` | Substring of the model ID | `bert` |
| `author` | Organization or user | `google` |
| `pipeline_tag` | Task: text-classification, text-generation, image-classification, ... | `text-classification` |
| `filter` | Tag filter (library, license, language, dataset) | `pytorch` |
| `sort / direction` | downloads \| likes \| lastModified \| trendingScore; -1 = descending | `downloads / -1` |
| `limit` | Max results | `5` |
| `full` | true → more metadata (files, card data) | `true` |
| `/api/datasets?search=` | Sibling endpoint: dataset search (same params) | `search=imdb` |
| `datasets-server.huggingface.co/rows?dataset=&config=&split=&offset=&length=` | Read rows of any public dataset (length ≤ 100) | `dataset=stanfordnlp/imdb&config=plain_text&split=train` |

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

**Endpoint:** `GET https://developer.nlr.gov/api/alt-fuel-stations/v1/nearest.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `latitude / longitude or location` | Coordinates or a free-text address | `38.9 / -77.04` |
| `radius` | Miles (default 5; or infinite) | `5` |
| `fuel_type` | ELEC \| E85 \| CNG \| LPG \| HY \| BD \| LNG \| RD (comma list) | `ELEC` |
| `ev_network` | Charging network | `Tesla` |
| `ev_charging_level` | 1 \| 2 \| dc_fast | `dc_fast` |
| `status` | E (open) \| P (planned) \| T (temporarily unavailable) | `E` |
| `limit` | Max results | `5` |
| `/api/pvwatts/v8.json` | Sibling endpoint: solar PV output (lat, lon, system_capacity, azimuth, tilt, array_type, module_type, losses) | `system_capacity=4&tilt=20` |
| `api_key` | api.data.gov key | `DEMO_KEY` |

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

**Endpoint:** `GET https://services.nvd.nist.gov/rest/json/cves/2.0`

| Parameter | Description | Example |
|:--|:--|:--|
| `keywordSearch / keywordExactMatch` | Words in the description; flag for exact phrase | `pytorch` |
| `cveId` | A single CVE | `CVE-2021-44228` |
| `cvssV3Severity` | LOW \| MEDIUM \| HIGH \| CRITICAL | `CRITICAL` |
| `cweId` | Weakness type | `CWE-79` |
| `pubStartDate / pubEndDate` | Published range, ISO-8601 (max 120 days) | `2026-01-01T00:00:00.000` |
| `cpeName` | Affected product (CPE 2.3 name) | `cpe:2.3:a:pytorch:pytorch:*:*:*:*:*:*:*:*` |
| `resultsPerPage / startIndex` | Max 2,000 per page | `5` |
| `apiKey (header)` | Optional key | `<NVD_API_KEY>` |

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

**Endpoint:** `GET https://waterservices.usgs.gov/nwis/iv/  (instantaneous)  ·  /nwis/dv/  (daily)`

| Parameter | Description | Example |
|:--|:--|:--|
| `sites` | Comma list of site numbers | `01646500 (Potomac at Little Falls)` |
| `stateCd / bBox / huc` | Alternative major filters (one required) | `va` |
| `parameterCd` | 00060 discharge (cfs), 00065 gage height (ft), 00010 water temp (°C) | `00060,00065` |
| `period` | ISO-8601 duration back from now | `PT2H, P7D` |
| `startDT / endDT` | Absolute range (instead of period) | `2026-10-01` |
| `siteStatus` | active \| inactive \| all | `active` |
| `format` | json \| rdb (tab-delimited) | `json` |

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

### Bureau of Transportation Statistics (BTS) Open Data

| | |
|:--|:--|
| **Docs** | https://data.bts.gov/ (Socrata SODA API; same query language as the CDC entry) |
| **Auth** | None (optional Socrata app token) |
| **Register / token** | No registration. Every dataset page has an API button with its 4x4 ID. Airline on-time performance is a separate bulk download at https://www.transtats.bts.gov. |
| **Env var** | `SOCRATA_APP_TOKEN (optional)` |
| **Script** | [`scripts/transportation_geo/bts_open_data.py`](scripts/transportation_geo/bts_open_data.py) |

**What you can get:**

- Border crossings by port, month and mode (trucks, buses, pedestrians, personal vehicles)
- Monthly Transportation Statistics: air/transit ridership, freight, fuel prices, safety
- Freight flows, transportation employment, pipeline and rail statistics

**Capstone ideas:** Trade/mobility shocks at the border, transportation as an economic indicator, post-pandemic travel recovery.

**Endpoint:** `GET https://data.bts.gov/resource/{dataset-id}.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `dataset-id` | 4x4 ID from the dataset page | `keg4-3bc2 (Border Crossing Entry Data)` |
| `$select / $where / $group / $order` | SQL-like query clauses (see the CDC entry) | `$select=port_name,sum(value)&$group=port_name` |
| `column=value` | Equality filter | `measure=Trucks` |
| `$limit / $offset` | Paging (default 1,000) | `5` |
| `X-App-Token (header)` | Optional app token | `<SOCRATA_APP_TOKEN>` |

```python
from _common import get, show


def main():
    d = get("https://data.bts.gov/resource/keg4-3bc2.json", params={
        "$select": "port_name,state,sum(value) as trucks", "measure": "Trucks", "border": "US-Mexico Border",
        "$where": "date >= '2025-01-01T00:00:00' AND date < '2026-01-01T00:00:00'",
        "$group": "port_name,state", "$order": "trucks DESC", "$limit": 5})
    show({"busiest_US_Mexico_truck_crossings_2025": d})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 21:03</summary>

```json
{
  "busiest_US_Mexico_truck_crossings_2025": [
    {
      "port_name": "Laredo",
      "state": "Texas",
      "trucks": "2945388"
    },
    {
      "port_name": "Otay Mesa",
      "state": "California",
      "trucks": "990111"
    },
    {
      "port_name": "Hidalgo",
      "state": "Texas",
      "trucks": "719184"
    },
    {
      "port_name": "Ysleta",
      "state": "Texas",
      "trucks": "641729"
    },
    {
      "port_name": "Calexico East",
      "state": "California",
      "trucks": "446835"
    }
  ]
}
```
</details>

---

### US Census Geocoder

| | |
|:--|:--|
| **Docs** | https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.html |
| **Auth** | None |
| **Register / token** | No registration. Batch mode geocodes up to 10,000 addresses per uploaded CSV. |
| **Script** | [`scripts/transportation_geo/census_geocoder.py`](scripts/transportation_geo/census_geocoder.py) |

**What you can get:**

- US address -> matched address, lat/lon
- Census geographies for the point: state, county, tract, block group, block (GEOIDs)
- Congressional and state legislative districts

**Capstone ideas:** The bridge between addresses and Census data: geocode once, then join ACS demographics by tract GEOID.

**Endpoint:** `GET https://geocoding.geo.census.gov/geocoder/{returntype}/{searchtype}`

| Parameter | Description | Example |
|:--|:--|:--|
| `returntype` | locations (coordinates only) \| geographies (coordinates + census areas) | `geographies` |
| `searchtype` | onelineaddress \| address \| coordinates \| addressbatch (POST CSV) | `onelineaddress` |
| `address / street, city, state, zip` | Address input | `2121 I St NW, Washington, DC 20052` |
| `x / y` | Longitude / latitude for searchtype=coordinates | `-77.0486 / 38.8997` |
| `benchmark / vintage` | Address and boundary versions | `Public_AR_Current / Current_Current` |
| `layers` | Which geographies to return | `Census Tracts,Counties` |
| `format` | json \| html | `json` |

```python
from _common import get, show


def main():
    d = get("https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress", params={
        "address": "2121 I St NW, Washington, DC 20052", "benchmark": "Public_AR_Current",
        "vintage": "Current_Current", "format": "json"})["result"]["addressMatches"][0]
    geo = d["geographies"]
    block = next(v[0] for k, v in geo.items() if k.endswith("Census Blocks"))  # layer names carry the vintage year
    district = next(v[0] for k, v in geo.items() if k.endswith("Congressional Districts"))
    show({"matched": d["matchedAddress"], "lon": d["coordinates"]["x"], "lat": d["coordinates"]["y"],
          "county": geo["Counties"][0]["NAME"], "tract_GEOID": geo["Census Tracts"][0]["GEOID"],
          "block_group_GEOID": block["GEOID"][:12],  # a block GEOID's first 12 digits are its block group
          "congressional_district": district["NAME"]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 21:03</summary>

```json
{
  "matched": "2121 I ST NW, WASHINGTON, DC, 20037",
  "lon": -77.0467239318,
  "lat": 38.900750817068,
  "county": "District of Columbia",
  "tract_GEOID": "11001010800",
  "block_group_GEOID": "110010108001",
  "congressional_district": "Delegate District (at Large)"
}
```
</details>

---

### Chicago Taxi & Rideshare Trips

| | |
|:--|:--|
| **Docs** | https://data.cityofchicago.org/Transportation/Taxi-Trips-2024-/ajtu-isnz |
| **Auth** | None (optional Socrata app token) |
| **Register / token** | No registration. Large tables: always filter by date and aggregate server-side ($select + $group) rather than pulling raw rows. Rideshare (TNP) trips are a separate dataset on the same portal. |
| **Env var** | `SOCRATA_APP_TOKEN (optional)` |
| **Script** | [`scripts/transportation_geo/chicago_taxi.py`](scripts/transportation_geo/chicago_taxi.py) |

**What you can get:**

- Every taxi trip since 2024: start/end time (15-min), duration, miles, community areas, fare, tip, payment type, company
- Pickup/dropoff centroids (lat/lon), hashed taxi IDs
- Older trips (2013-2023) in a separate dataset

**Capstone ideas:** Fare and tip prediction, demand forecasting by area/hour, taxi vs. rideshare competition.

**Endpoint:** `GET https://data.cityofchicago.org/resource/ajtu-isnz.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `$where` | Filter rows - always bound trip_start_timestamp | `trip_start_timestamp between '2026-09-01T00:00:00' and '2026-09-08T00:00:00'` |
| `$select + $group` | Server-side aggregation | `payment_type, count(*) as trips, avg(fare)` |
| `$order / $limit` | Sort and limit | `trips DESC / 5` |
| `columns` | trip_seconds, trip_miles, pickup_community_area, fare, tips, payment_type, company, ... | `—` |
| `X-App-Token (header)` | Optional app token | `<SOCRATA_APP_TOKEN>` |

```python
from _common import get, show


def main():
    d = get("https://data.cityofchicago.org/resource/ajtu-isnz.json", timeout=120, params={
        "$select": "payment_type, count(*) as trips, avg(fare) as avg_fare, avg(tips) as avg_tip, avg(trip_miles) as avg_miles",
        "$where": "trip_start_timestamp between '2026-09-01T00:00:00' and '2026-09-08T00:00:00'",
        "$group": "payment_type", "$order": "trips DESC", "$limit": 5})
    show([{k: (round(float(v), 2) if k.startswith("avg") else v) for k, v in r.items()} for r in d])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 21:03</summary>

```json
[
  {
    "payment_type": "Mobile",
    "trips": "47705",
    "avg_fare": 15.77,
    "avg_tip": 1.32,
    "avg_miles": 3.68
  },
  {
    "payment_type": "Credit Card",
    "trips": "41964",
    "avg_fare": 30.13,
    "avg_tip": 6.16,
    "avg_miles": 8.28
  },
  {
    "payment_type": "Cash",
    "trips": "27946",
    "avg_fare": 23.03,
    "avg_tip": 0.0,
    "avg_miles": 5.26
  },
  {
    "payment_type": "Prcard",
    "trips": "9185",
    "avg_fare": 25.65,
    "avg_tip": 0.09,
    "avg_miles": 7.23
  },
  {
    "payment_type": "Unknown",
    "trips": "2580",
    "avg_fare": 26.14,
    "avg_tip": 0.01,
    "avg_miles": 7.31
  }
]
```
</details>

---

### GBFS Bike-share Feeds (Capital Bikeshare and 1,000+ systems)

| | |
|:--|:--|
| **Docs** | https://github.com/MobilityData/gbfs (spec) · system list: https://github.com/MobilityData/gbfs/blob/master/systems.csv |
| **Auth** | None |
| **Register / token** | No registration. GBFS is an open standard: every public bike/scooter-share system publishes the same JSON feeds. Trip-history CSVs for Capital Bikeshare: https://capitalbikeshare.com/system-data. |
| **Script** | [`scripts/transportation_geo/gbfs_bikeshare.py`](scripts/transportation_geo/gbfs_bikeshare.py) |

**What you can get:**

- Real-time station status: bikes / e-bikes / docks available, renting/returning flags
- Station information: name, lat/lon, capacity
- Free-floating vehicles, pricing plans, system alerts

**Capstone ideas:** Rebalancing / demand forecasting, equity of station access, weather effects on micromobility.

**Endpoint:** `GET https://gbfs.lyft.com/gbfs/2.3/dca-cabi/en/{feed}.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `gbfs.json` | Auto-discovery: lists every feed URL for the system | `gbfs/2.3/dca-cabi/gbfs.json` |
| `station_information` | Static: station_id, name, lat, lon, capacity | `station_information.json` |
| `station_status` | Real time: num_bikes_available, num_ebikes_available, num_docks_available | `station_status.json` |
| `free_bike_status / vehicle_status` | Dockless vehicles with location | `free_bike_status.json` |
| `system_id` | Other systems: bay-wheels, cdc-divvy (Chicago), bkn (Citi Bike NYC)... | `dca-cabi` |

```python
from _common import get, show

BASE = "https://gbfs.lyft.com/gbfs/2.3/dca-cabi/en"


def main():
    info = {s["station_id"]: s for s in get(f"{BASE}/station_information.json")["data"]["stations"]}
    status = get(f"{BASE}/station_status.json")
    stations = status["data"]["stations"]
    top = sorted(stations, key=lambda s: -s["num_bikes_available"])[:5]
    show({"stations": len(stations), "bikes_available_total": sum(s["num_bikes_available"] for s in stations),
          "fullest_stations": [{"name": info[s["station_id"]]["name"], "bikes": s["num_bikes_available"],
                                "ebikes": s.get("num_ebikes_available"), "docks_free": s["num_docks_available"]}
                               for s in top if s["station_id"] in info]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 21:03</summary>

```json
{
  "stations": 865,
  "bikes_available_total": 5937,
  "fullest_stations": [
    {
      "name": "13th & H St NE",
      "bikes": 31,
      "ebikes": 8,
      "docks_free": 0
    },
    {
      "name": "13th & D St NE",
      "bikes": 29,
      "ebikes": 9,
      "docks_free": 5
    },
    {
      "name": "10th St & Constitution Ave NW",
      "bikes": 28,
      "ebikes": 2,
      "docks_free": 19
    },
    {
      "name": "31st & K St NW",
      "bikes": 28,
      "ebikes": 2,
      "docks_free": 0
    },
    {
      "name": "Wisconsin Ave & K St NW",
      "bikes": 27,
      "ebikes": 15,
      "docks_free": 7
    }
  ]
}
```
</details>

---

### MBTA V3 API (Boston transit)

| | |
|:--|:--|
| **Docs** | https://api-v3.mbta.com/docs/swagger/index.html |
| **Auth** | None (20 req/min) / free key (1,000 req/min, header `x-api-key`) |
| **Register / token** | Works keyless at low volume. Register at https://api-v3.mbta.com/register for a key. A good example of a modern JSON:API transit API (WMATA's needs a key; GTFS-Realtime feeds exist for most US agencies). |
| **Env var** | `MBTA_API_KEY (optional)` |
| **Script** | [`scripts/transportation_geo/mbta.py`](scripts/transportation_geo/mbta.py) |

**What you can get:**

- Real-time predictions and vehicle positions for subway, bus, commuter rail, ferry
- Routes, stops, schedules, service alerts
- include= joins related resources in one call

**Capstone ideas:** Transit reliability (predicted vs. scheduled), headway analysis, alert text mining.

**Endpoint:** `GET https://api-v3.mbta.com/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `resource` | predictions \| vehicles \| schedules \| alerts \| routes \| stops \| trips | `predictions` |
| `filter[stop] / filter[route]` | Filter by stop or route ID (comma lists) | `filter[stop]=place-pktrm` |
| `filter[route_type]` | 0 light rail, 1 subway, 2 commuter rail, 3 bus, 4 ferry | `1` |
| `include` | Embed related resources | `route,trip` |
| `sort` | Attribute to sort by (prefix - for descending) | `arrival_time` |
| `page[limit] / page[offset]` | Paging | `5` |

```python
from _common import get, show


def main():
    d = get("https://api-v3.mbta.com/predictions", params={
        "filter[stop]": "place-pktrm", "sort": "arrival_time", "page[limit]": 12, "include": "trip"})
    trips = {t["id"]: t["attributes"]["headsign"] for t in d.get("included", []) if t["type"] == "trip"}
    show([{"route": p["relationships"]["route"]["data"]["id"],
           "headsign": trips.get(p["relationships"]["trip"]["data"]["id"]),
           "arrival_time": p["attributes"]["arrival_time"],  # null at a trip's first stop
           "departure_time": p["attributes"]["departure_time"]}
          for p in d["data"] if p["attributes"]["arrival_time"] or p["attributes"]["departure_time"]],  # drop skipped stops
         max_items=6)
```

<details><summary><b>Sample output</b> — live run 2026-10-11 21:03</summary>

```json
[
  {
    "route": "Green-C",
    "headsign": "Cleveland Circle",
    "arrival_time": "2026-10-11T21:03:05-04:00",
    "departure_time": "2026-10-11T21:04:22-04:00"
  },
  {
    "route": "Green-D",
    "headsign": "Union Square",
    "arrival_time": "2026-10-11T21:03:39-04:00",
    "departure_time": "2026-10-11T21:04:47-04:00"
  },
  {
    "route": "Green-C",
    "headsign": "Government Center",
    "arrival_time": "2026-10-11T21:04:24-04:00",
    "departure_time": "2026-10-11T21:05:32-04:00"
  },
  {
    "route": "Red",
    "headsign": "Ashmont",
    "arrival_time": "2026-10-11T21:06:25-04:00",
    "departure_time": "2026-10-11T21:07:47-04:00"
  },
  {
    "route": "Green-B",
    "headsign": "Government Center",
    "arrival_time": "2026-10-11T21:06:26-04:00",
    "departure_time": "2026-10-11T21:07:34-04:00"
  },
  {
    "route": "Green-D",
    "headsign": "Riverside",
    "arrival_time": "2026-10-11T21:07:40-04:00",
    "departure_time": "2026-10-11T21:08:58-04:00"
  },
  "... (5 more)"
]
```
</details>

---

### NHTSA Recalls, Complaints & Safety Ratings

| | |
|:--|:--|
| **Docs** | https://www.nhtsa.gov/nhtsa-datasets-and-apis |
| **Auth** | None |
| **Register / token** | No registration. |
| **Script** | [`scripts/transportation_geo/nhtsa_safety.py`](scripts/transportation_geo/nhtsa_safety.py) |

**What you can get:**

- Recalls by vehicle: campaign number, component, summary, consequence, remedy
- Owner complaints: crash/fire/injury flags, component, free-text description
- NCAP 5-star crash-test ratings (overall, frontal, side, rollover)

**Capstone ideas:** NLP on complaint narratives to detect defects early, recall timing analysis, safety-rating vs. crash outcomes.

**Endpoint:** `GET https://api.nhtsa.gov/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `recalls/recallsByVehicle?make=&model=&modelYear=` | Recalls for a vehicle | `make=honda&model=accord&modelYear=2020` |
| `complaints/complaintsByVehicle?make=&model=&modelYear=` | Owner complaints for a vehicle | `make=tesla&model=model 3&modelYear=2022` |
| `SafetyRatings/modelyear/{year}/make/{make}/model/{model}` | Returns vehicle variants with VehicleId | `modelyear/2024/make/toyota/model/camry` |
| `SafetyRatings/VehicleId/{id}` | Star ratings for one variant | `VehicleId/19000` |
| `recalls/campaignNumber?campaignNumber=` | One recall campaign | `campaignNumber=20V682000` |

```python
from collections import Counter

from _common import get, show


def main():
    q = {"make": "tesla", "model": "model 3", "modelYear": 2022}
    recalls = get("https://api.nhtsa.gov/recalls/recallsByVehicle", params=q)["results"]
    complaints = get("https://api.nhtsa.gov/complaints/complaintsByVehicle", params=q)["results"]
    variants = get("https://api.nhtsa.gov/SafetyRatings/modelyear/2022/make/tesla/model/model%203")["Results"]
    rating = get(f'https://api.nhtsa.gov/SafetyRatings/VehicleId/{variants[0]["VehicleId"]}')["Results"][0]
    show({"vehicle": "2022 Tesla Model 3",
          "recalls": [{"campaign": r["NHTSACampaignNumber"], "component": r["Component"]} for r in recalls],
          "complaints": len(complaints),
          "top_complaint_components": Counter(c["components"] for c in complaints).most_common(4),
          "ncap": {k: rating.get(k) for k in ("VehicleDescription", "OverallRating", "OverallFrontCrashRating",
                                              "OverallSideCrashRating", "RolloverRating")}})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 21:03</summary>

```json
{
  "vehicle": "2022 Tesla Model 3",
  "recalls": [
    {
      "campaign": "22V037000",
      "component": "ELECTRICAL SYSTEM:ADAS:AUTONOMOUS/SELF DRIVING:SOFTWARE"
    },
    {
      "campaign": "22V045000",
      "component": "ELECTRICAL SYSTEM:SOFTWARE"
    },
    {
      "campaign": "22V050000",
      "component": "ELECTRICAL SYSTEM:SOFTWARE"
    },
    {
      "campaign": "22V063000",
      "component": "ELECTRICAL SYSTEM:SOFTWARE"
    },
    {
      "campaign": "22V235000",
      "component": "ELECTRICAL SYSTEM:SOFTWARE"
    },
    "... (13 more)"
  ],
  "complaints": 766,
  "top_complaint_components": [
    [
      "FORWARD COLLISION AVOIDANCE",
      263
    ],
    [
      "VEHICLE SPEED CONTROL,FORWARD COLLISION AVOIDANCE",
      77
    ],
    [
      "SERVICE BRAKES,FORWARD COLLISION AVOIDANCE",
      72
    ],
    [
      "VEHICLE SPEED CONTROL,SERVICE BRAKES,FORWARD COLLISION AVOIDANCE",
      48
    ]
  ],
  "ncap": {
    "VehicleDescription": "2022 Tesla Model 3 4 DR AWD",
    "OverallRating": "5",
    "OverallFrontCrashRating": "5",
    "OverallSideCrashRating": "5",
    "RolloverRating": "5"
  }
}
```
</details>

---

### NHTSA vPIC (vehicle VIN decoder)

| | |
|:--|:--|
| **Docs** | https://vpic.nhtsa.dot.gov/api/ |
| **Auth** | None |
| **Register / token** | No registration. Batch decoding of up to 50 VINs per request via POST DecodeVINValuesBatch. |
| **Script** | [`scripts/transportation_geo/nhtsa_vpic.py`](scripts/transportation_geo/nhtsa_vpic.py) |

**What you can get:**

- Decode any VIN: make, model, year, body class, engine, fuel type, drive type, plant, safety features
- All makes, models per make/year, manufacturer details, vehicle types

**Capstone ideas:** Enriching crash/registration/insurance data with vehicle attributes; EV adoption studies from VIN registries.

**Endpoint:** `GET https://vpic.nhtsa.dot.gov/api/vehicles/{operation}`

| Parameter | Description | Example |
|:--|:--|:--|
| `DecodeVinValues/{VIN}?format=json&modelyear=` | Flat key/value decode of one VIN (partial VINs with * allowed) | `DecodeVinValues/1HGCV1F34LA000000` |
| `DecodeVINValuesBatch/ (POST)` | Body DATA=VIN1,year;VIN2,year (up to 50) | `DATA=5UXWX7C5*BA,2011;...` |
| `GetModelsForMakeYear/make/{make}/modelyear/{year}` | Models a maker sold in a year; make matches substrings, so filter on Make_Name | `make/tesla/modelyear/2024` |
| `GetAllMakes` | Every make in the database | `—` |
| `format` | json \| csv \| xml | `json` |

```python
from _common import get, show

FIELDS = ["Make", "Model", "ModelYear", "BodyClass", "FuelTypePrimary", "ElectrificationLevel", "DriveType", "PlantCountry"]


def main():
    vin = get("https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/5YJ3E1EA7KF000000", params={"format": "json"})
    models = get("https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMakeYear/make/ford/modelyear/2025",
                 params={"format": "json"})
    show({"decoded_vin": {k: vin["Results"][0][k] for k in FIELDS},
          # make= matches substrings (e.g. "Bradford"), so keep exact matches only
          "ford_2025_models": sorted(m["Model_Name"] for m in models["Results"] if m["Make_Name"] == "FORD")},
         max_items=8)
```

<details><summary><b>Sample output</b> — live run 2026-10-11 21:04</summary>

```json
{
  "decoded_vin": {
    "Make": "TESLA",
    "Model": "Model 3",
    "ModelYear": "2019",
    "BodyClass": "Sedan/Saloon",
    "FuelTypePrimary": "Electric",
    "ElectrificationLevel": "BEV (Battery Electric Vehicle)",
    "DriveType": "",
    "PlantCountry": "UNITED STATES (USA)"
  },
  "ford_2025_models": [
    "'34",
    "Bronco",
    "Bronco Sport",
    "Commercial Chassis",
    "E-350",
    "E-450",
    "Escape",
    "Expedition",
    "... (26 more)"
  ]
}
```
</details>

---

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

**Endpoint:** `GET https://nominatim.openstreetmap.org/search  ·  /reverse`

| Parameter | Description | Example |
|:--|:--|:--|
| `q` | Free-text address or place | `George Washington University, Washington DC` |
| `street / city / state / country / postalcode` | Structured alternative to q | `city=Washington` |
| `format` | json \| jsonv2 \| geojson \| geocodejson \| xml | `jsonv2` |
| `limit` | Max results (max 40) | `2` |
| `countrycodes` | Restrict to ISO 3166-1 alpha-2 codes | `us` |
| `addressdetails` | 1 → split the address into parts | `1` |
| `/reverse?lat=&lon=&zoom=` | Coordinates → address (zoom 3 = country … 18 = building) | `lat=38.8997&lon=-77.0486` |

```python
from _common import get, show


def main():
    d = get("https://nominatim.openstreetmap.org/search",
            params={"q": "George Washington University, Washington DC", "format": "jsonv2", "limit": 2})
    show([{k: r[k] for k in ("display_name", "lat", "lon", "category", "type")} for r in d])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 21:04</summary>

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

**Endpoint:** `GET https://opensky-network.org/api/states/all`

| Parameter | Description | Example |
|:--|:--|:--|
| `lamin / lomin / lamax / lomax` | Bounding box | `38.6 / -77.6 / 39.2 / -76.6` |
| `icao24` | One or more aircraft transponder addresses | `a0b1c2` |
| `time` | Unix seconds (past states need an account) | `1760200000` |
| `extended` | 1 → include aircraft category | `1` |
| `/flights/arrival?airport=&begin=&end=` | Sibling endpoint (account): arrivals at an airport (also /flights/departure) | `airport=KIAD` |
| `/flights/aircraft?icao24=&begin=&end=` | Sibling endpoint (account): flights by aircraft | `icao24=a0b1c2` |

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

<details><summary><b>Sample output</b> — live run 2026-10-11 21:04</summary>

```json
{
  "aircraft_in_box": 51,
  "sample": [
    {
      "callsign": "AAL3238",
      "origin_country": "United States",
      "lat": 38.7546,
      "lon": -77.4392,
      "baro_alt_m": 952.5,
      "velocity_mps": 110.62
    },
    {
      "callsign": "SWA1469",
      "origin_country": "United States",
      "lat": 39.159,
      "lon": -76.6585,
      "baro_alt_m": 304.8,
      "velocity_mps": 86.18
    },
    {
      "callsign": "SWA1044",
      "origin_country": "United States",
      "lat": 39.1083,
      "lon": -76.6769,
      "baro_alt_m": 830.58,
      "velocity_mps": 125.16
    },
    {
      "callsign": "N416KC",
      "origin_country": "United States",
      "lat": 38.962,
      "lon": -77.4725,
      "baro_alt_m": null,
      "velocity_mps": 9.77
    },
    {
      "callsign": "SCX3053",
      "origin_country": "United States",
      "lat": 39.1867,
      "lon": -76.6832,
      "baro_alt_m": 106.68,
      "velocity_mps": 77.39
    },
    "... (46 more)"
  ]
}
```
</details>

---

### OSRM (routing on OpenStreetMap)

| | |
|:--|:--|
| **Docs** | https://project-osrm.org/docs/v5.24.0/api/ |
| **Auth** | None |
| **Register / token** | The public demo server allows light use only (~1 request/second, no bulk jobs). For real projects run your own OSRM with Docker on an OSM extract (car, bike or foot profiles). |
| **Script** | [`scripts/transportation_geo/osrm.py`](scripts/transportation_geo/osrm.py) |

**What you can get:**

- Fastest route between points: distance, duration, turn-by-turn steps, geometry
- Travel-time / distance matrices between many points (table service)
- Snap GPS traces to roads (match), nearest road, travelling-salesman trips

**Capstone ideas:** Accessibility measures (travel time to hospitals/jobs), commute-time features for housing models.

**Endpoint:** `GET https://router.project-osrm.org/{service}/v1/{profile}/{lon,lat;lon,lat;...}`

| Parameter | Description | Example |
|:--|:--|:--|
| `service` | route \| table \| nearest \| match \| trip | `route` |
| `profile` | driving on the demo server (car \| bike \| foot on self-hosted) | `driving` |
| `coordinates` | lon,lat pairs separated by ; (note: longitude first) | `-77.0486,38.8997;-77.0091,38.8899` |
| `overview` | full \| simplified \| false (route geometry) | `false` |
| `steps` | true -> turn-by-turn instructions | `true` |
| `annotations (table)` | duration \| distance \| duration,distance | `duration,distance` |
| `sources / destinations (table)` | Indexes of coordinates to use as origins / destinations | `sources=0` |

```python
from _common import get, show

PLACES = {"GWU": (-77.0486, 38.8997), "Union Station": (-77.0063, 38.8973),
          "Reagan Airport": (-77.0402, 38.8512), "Dulles Airport": (-77.4565, 38.9531)}


def main():
    coords = ";".join(f"{lon},{lat}" for lon, lat in PLACES.values())
    d = get(f"https://router.project-osrm.org/table/v1/driving/{coords}",
            params={"sources": 0, "annotations": "duration,distance"})
    names = list(PLACES)
    show([{"from": "GWU", "to": names[i], "drive_min": round(d["durations"][0][i] / 60, 1),
           "km": round(d["distances"][0][i] / 1000, 1)} for i in range(1, len(names))])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 21:04</summary>

```json
[
  {
    "from": "GWU",
    "to": "Union Station",
    "drive_min": 7.9,
    "km": 4.6
  },
  {
    "from": "GWU",
    "to": "Reagan Airport",
    "drive_min": 12.1,
    "km": 9.3
  },
  {
    "from": "GWU",
    "to": "Dulles Airport",
    "drive_min": 52.3,
    "km": 45.0
  }
]
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

**Endpoint:** `POST https://overpass-api.de/api/interpreter  (body: data=<Overpass QL query>)`

| Parameter | Description | Example |
|:--|:--|:--|
| `[out:json]` | Output format (json \| xml \| csv(...)) | `[out:json]` |
| `[timeout:N]` | Server timeout in seconds | `[timeout:25]` |
| `node / way / relation / nwr` | Element type to query | `nwr` |
| `["key"="value"]` | Tag filter (also ["key"], ["key"~"regex"]) | `["amenity"="cafe"]` |
| `(around:radius,lat,lon)` | Within radius (metres) of a point | `(around:600,38.8997,-77.0486)` |
| `(south,west,north,east)` | Bounding box | `(38.88,-77.06,38.91,-77.02)` |
| `area[...]->.a; node(area.a)` | Within a named area | `area["name"="District of Columbia"]` |
| `out [N] [geom\|center]` | Return results (limit, geometry) | `out 5` |

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

<details><summary><b>Sample output</b> — live run 2026-10-11 21:04</summary>

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

### Transitland (GTFS transit data worldwide)

| | |
|:--|:--|
| **Docs** | https://www.transit.land/documentation/rest-api/ |
| **Auth** | API key (header `apikey`) |
| **Register / token** | Sign up for a free key at https://www.transit.land/documentation#signing-up-for-an-api-key (free tier for non-commercial use). Aggregates 2,500+ GTFS feeds. |
| **Env var** | `TRANSITLAND_API_KEY` |
| **Script** | [`scripts/transportation_geo/transitland.py`](scripts/transportation_geo/transitland.py) |

**What you can get:**

- Transit agencies, routes, stops and schedules from GTFS feeds worldwide in one API
- Stop departures, route shapes (GeoJSON), feed versions over time
- Search near a point or inside a bounding box

**Capstone ideas:** Transit-access scores for neighborhoods, comparing service levels across cities, service-change history.

**Endpoint:** `GET https://transit.land/api/v2/rest/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `resource` | agencies \| routes \| stops \| feeds \| stops/{key}/departures | `routes` |
| `search` | Text search on names | `metro` |
| `lat / lon / radius` | Near a point (radius in metres) | `38.8997 / -77.0486 / 500` |
| `bbox` | min_lon,min_lat,max_lon,max_lat | `-77.12,38.80,-76.91,38.99` |
| `route_type` | GTFS route type: 1 subway, 3 bus, ... | `1` |
| `limit / after` | Paging (cursor in meta.after) | `5` |
| `apikey (header)` | Your key | `<TRANSITLAND_API_KEY>` |

```python
from _common import get, key, show


def main():
    d = get("https://transit.land/api/v2/rest/routes", headers={"apikey": key("TRANSITLAND_API_KEY")},
            params={"lat": 38.8997, "lon": -77.0486, "radius": 1000, "route_type": 1, "limit": 5})
    show([{"route": r["route_short_name"], "long_name": r["route_long_name"], "agency": r["agency"]["agency_name"],
           "route_type": r["route_type"]} for r in d["routes"]])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"route": "RD", "long_name": "Red Line", "agency": "Washington Metropolitan Area Transit Authority", "route_type": 1},
  {"route": "BL", "long_name": "Blue Line", "agency": "Washington Metropolitan Area Transit Authority", "route_type": 1}
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

**Endpoint:** `GET https://api.wmata.com/StationPrediction.svc/json/GetPrediction/{StationCodes}`

| Parameter | Description | Example |
|:--|:--|:--|
| `StationCodes` | Comma list of station codes, or All | `A01 (Metro Center)` |
| `/Rail.svc/json/jStations?LineCode=` | Sibling endpoint: stations (RD, BL, OR, SV, GR, YL) | `LineCode=RD` |
| `/Incidents.svc/json/Incidents · /ElevatorIncidents` | Sibling endpoints: rail disruptions, elevator/escalator outages | `—` |
| `/NextBusService.svc/json/jPredictions?StopID=` | Sibling endpoint: bus arrival predictions | `StopID=1001195` |
| `/Bus.svc/json/jBusPositions?RouteID=&Lat=&Lon=&Radius=` | Sibling endpoint: live bus locations | `RouteID=30N` |
| `api_key (header)` | Your primary key | `<WMATA_API_KEY>` |

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

## 🏅 Sports

### BALLDONTLIE (NBA, NFL, MLB, ...)

| | |
|:--|:--|
| **Docs** | https://docs.balldontlie.io/ |
| **Auth** | API key (header `Authorization`) |
| **Register / token** | Sign up at https://app.balldontlie.io -> key on the dashboard. Free tier: teams, players and games (5 req/min); paid tiers add box scores, season averages, odds and more sports. |
| **Env var** | `BALLDONTLIE_API_KEY` |
| **Script** | [`scripts/sports/balldontlie.py`](scripts/sports/balldontlie.py) |

**What you can get:**

- NBA teams, players, games back to 1946 (free)
- Box scores, player stats, season averages, injuries, betting odds (paid)
- Same API shape for NFL, MLB, NHL, EPL, WNBA, NCAA

**Capstone ideas:** Player-performance forecasting, schedule-fatigue effects, team-strength ratings.

**Endpoint:** `GET https://api.balldontlie.io/v1/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `resource` | teams \| players \| games \| stats \| season_averages \| box_scores (some paid) | `games` |
| `seasons[]` | Season start year (repeatable) | `2025` |
| `team_ids[] / player_ids[]` | Filter by IDs (repeatable) | `14` |
| `dates[] / start_date / end_date` | YYYY-MM-DD | `2026-01-15` |
| `search` | Player name search (players) | `curry` |
| `per_page / cursor` | Max 100; cursor-based paging (meta.next_cursor) | `25` |
| `Authorization (header)` | Your API key | `<BALLDONTLIE_API_KEY>` |

```python
from _common import get, key, show


def main():
    d = get("https://api.balldontlie.io/v1/games", headers={"Authorization": key("BALLDONTLIE_API_KEY")},
            params={"seasons[]": 2025, "team_ids[]": 30, "per_page": 5})  # 30 = Washington Wizards
    show([{"date": g["date"], "home": g["home_team"]["full_name"], "home_score": g["home_team_score"],
           "visitor": g["visitor_team"]["full_name"], "visitor_score": g["visitor_team_score"], "status": g["status"]}
          for g in d["data"]])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"date": "2026-04-12", "home": "Boston Celtics", "home_score": 112, "visitor": "Washington Wizards", "visitor_score": 98, "status": "Final"},
  {"date": "2026-04-10", "home": "Washington Wizards", "home_score": 104, "visitor": "Miami Heat", "visitor_score": 109, "status": "Final"}
]
```
</details>

---

### ESPN Site API (unofficial)

| | |
|:--|:--|
| **Docs** | https://github.com/pseudo-r/Public-ESPN-API (community docs; no official documentation) |
| **Auth** | None |
| **Register / token** | No registration. Unofficial and undocumented: endpoints can change without notice and ESPN's terms apply. It rejects some custom User-Agents, so this example sends the default `requests` one. |
| **Script** | [`scripts/sports/espn.py`](scripts/sports/espn.py) |

**What you can get:**

- Scoreboards with live/final scores for NFL, NBA, MLB, NHL, college sports, soccer, ...
- Teams, rosters, schedules, standings, news headlines
- Game summaries: box scores, play-by-play, win probability

**Capstone ideas:** Win-probability models, home-advantage analysis, sports-news NLP.

**Endpoint:** `GET https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `sport / league` | football/nfl, basketball/nba, baseball/mlb, hockey/nhl, football/college-football, soccer/eng.1, ... | `football/nfl` |
| `resource` | scoreboard \| teams \| teams/{id} \| teams/{id}/roster \| teams/{id}/schedule \| news \| summary | `scoreboard` |
| `dates` | Scoreboard date or range, YYYYMMDD or YYYYMMDD-YYYYMMDD | `20261011` |
| `week / seasontype` | NFL/college week number; seasontype 1 = pre, 2 = regular, 3 = post | `6 / 2` |
| `summary?event=` | Box score + play-by-play for one game (event id from the scoreboard) | `event=401872993` |
| `limit` | Max events/items | `100` |

```python
import requests

from _common import get, show


def main():
    d = get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
            headers={"User-Agent": requests.utils.default_user_agent()})
    show({"season": d["season"]["year"], "week": d.get("week", {}).get("number"), "games": [
        {"name": e["shortName"], "status": e["status"]["type"]["description"],
         "score": " - ".join(f'{c["team"]["abbreviation"]} {c.get("score")}' for c in e["competitions"][0]["competitors"])}
        for e in d["events"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:46</summary>

```json
{
  "season": 2026,
  "week": 5,
  "games": [
    {
      "name": "BAL @ ATL",
      "status": "In Progress",
      "score": "ATL 7 - BAL 3"
    },
    {
      "name": "PHI VS JAX",
      "status": "Final",
      "score": "JAX 19 - PHI 17"
    },
    {
      "name": "CHI @ GB",
      "status": "Final",
      "score": "GB 31 - CHI 7"
    },
    {
      "name": "HOU @ TEN",
      "status": "Final",
      "score": "TEN 3 - HOU 26"
    },
    {
      "name": "CIN @ MIA",
      "status": "Final",
      "score": "MIA 24 - CIN 31"
    },
    "... (10 more)"
  ]
}
```
</details>

---

### football-data.org (soccer)

| | |
|:--|:--|
| **Docs** | https://docs.football-data.org/general/v4/index.html |
| **Auth** | API token (header `X-Auth-Token`) |
| **Register / token** | Register at https://www.football-data.org/client/register (token emailed). Free tier: 12 top competitions (Premier League, La Liga, Bundesliga, Serie A, Champions League, World Cup...), 10 req/min. |
| **Env var** | `FOOTBALL_DATA_TOKEN` |
| **Script** | [`scripts/sports/football_data.py`](scripts/sports/football_data.py) |

**What you can get:**

- Competitions, standings (total/home/away), matches with scores and status
- Teams with squads, top scorers, head-to-head

**Capstone ideas:** Match-outcome prediction (Poisson / Elo), home-advantage studies, league-table simulation.

**Endpoint:** `GET https://api.football-data.org/v4/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `competitions/{code}/standings` | League table; codes PL, PD, BL1, SA, FL1, CL, WC, ... | `competitions/PL/standings` |
| `competitions/{code}/matches` | Fixtures/results; filter with matchday, status, dateFrom, dateTo, season | `matchday=7` |
| `competitions/{code}/scorers` | Top scorers (limit=) | `limit=10` |
| `teams/{id}/matches` | A team's matches (status, venue HOME \| AWAY, limit) | `teams/57/matches?status=FINISHED` |
| `season` | Season start year | `2025` |
| `X-Auth-Token (header)` | Your token | `<FOOTBALL_DATA_TOKEN>` |

```python
from _common import get, key, show


def main():
    d = get("https://api.football-data.org/v4/competitions/PL/standings",
            headers={"X-Auth-Token": key("FOOTBALL_DATA_TOKEN")})
    table = d["standings"][0]["table"]
    show([{"pos": r["position"], "team": r["team"]["name"], "played": r["playedGames"], "won": r["won"],
           "draw": r["draw"], "lost": r["lost"], "points": r["points"], "goal_diff": r["goalDifference"]} for r in table])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"pos": 1, "team": "Arsenal FC", "played": 7, "won": 5, "draw": 1, "lost": 1, "points": 16, "goal_diff": 10},
  {"pos": 2, "team": "Liverpool FC", "played": 7, "won": 5, "draw": 0, "lost": 2, "points": 15, "goal_diff": 6}
]
```
</details>

---

### Jolpica-F1 (Ergast-compatible F1 history)

| | |
|:--|:--|
| **Docs** | https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md |
| **Auth** | None |
| **Register / token** | No registration. Drop-in replacement for the retired Ergast API; ~4 req/sec, 500/hour. |
| **Script** | [`scripts/sports/jolpica_f1.py`](scripts/sports/jolpica_f1.py) |

**What you can get:**

- Every F1 season since 1950: race results, qualifying, sprint, driver & constructor standings
- Lap times and pit stops (1996+/2011+), circuits, drivers, constructors

**Capstone ideas:** Long-run analysis of team dominance, driver-rating models, era comparisons.

**Endpoint:** `GET https://api.jolpi.ca/ergast/f1/{season}/{round}/{resource}.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `season` | Year or current | `2025` |
| `round` | Race number or last (optional) | `last` |
| `resource` | results \| qualifying \| sprint \| driverStandings \| constructorStandings \| laps \| pitstops \| drivers \| circuits | `driverStandings` |
| `drivers/{id} / constructors/{id} / circuits/{id}` | Filter path segments (combinable) | `drivers/max_verstappen/results` |
| `limit / offset` | Paging (max 100) | `30` |

```python
from _common import get, show


def main():
    d = get("https://api.jolpi.ca/ergast/f1/2025/driverStandings.json")["MRData"]["StandingsTable"]["StandingsLists"][0]
    show({"season": d["season"], "after_round": d["round"], "standings": [
        {"pos": s["position"], "driver": f'{s["Driver"]["givenName"]} {s["Driver"]["familyName"]}',
         "team": s["Constructors"][0]["name"], "points": s["points"], "wins": s["wins"]}
        for s in d["DriverStandings"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:46</summary>

```json
{
  "season": "2025",
  "after_round": "24",
  "standings": [
    {
      "pos": "1",
      "driver": "Lando Norris",
      "team": "McLaren",
      "points": "423",
      "wins": "7"
    },
    {
      "pos": "2",
      "driver": "Max Verstappen",
      "team": "Red Bull",
      "points": "421",
      "wins": "8"
    },
    {
      "pos": "3",
      "driver": "Oscar Piastri",
      "team": "McLaren",
      "points": "410",
      "wins": "7"
    },
    {
      "pos": "4",
      "driver": "George Russell",
      "team": "Mercedes",
      "points": "319",
      "wins": "2"
    },
    {
      "pos": "5",
      "driver": "Charles Leclerc",
      "team": "Ferrari",
      "points": "242",
      "wins": "0"
    },
    "... (16 more)"
  ]
}
```
</details>

---

### Lichess API (chess)

| | |
|:--|:--|
| **Docs** | https://lichess.org/api |
| **Auth** | None for public data (OAuth token for account actions) |
| **Register / token** | No registration for public endpoints; one request at a time, back off on HTTP 429. The full game database (billions of games, PGN) is free at https://database.lichess.org. |
| **Script** | [`scripts/sports/lichess.py`](scripts/sports/lichess.py) |

**What you can get:**

- Leaderboards by time control, user profiles & rating histories
- Game export as PGN/NDJSON with moves, clocks and engine evals
- Opening explorer statistics (token required), puzzles

**Capstone ideas:** Rating-dynamics modeling, opening-success analysis, cheating/anomaly detection.

**Endpoint:** `GET https://lichess.org/api/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `player/top/{nb}/{perfType}` | Leaderboard; nb ≤ 200; perfType bullet \| blitz \| rapid \| classical \| ... | `player/top/10/blitz` |
| `user/{username}` | Profile with ratings per time control | `user/DrNykterstein` |
| `user/{username}/rating-history` | Rating over time | `user/DrNykterstein/rating-history` |
| `games/user/{username}?max=&perfType=&evals=&clocks=` | Export games (Accept: application/x-ndjson for JSON lines) | `max=10&perfType=blitz` |
| `explorer.lichess.ovh/lichess?fen=&speeds=&ratings=` | Opening explorer statistics for a position (now requires a Lichess OAuth token) | `speeds=blitz&ratings=2000` |

```python
from _common import get, show


def main():
    d = get("https://lichess.org/api/player/top/5/blitz", headers={"Accept": "application/vnd.lichess.v3+json"})
    show([{"username": u["username"], "title": u.get("title"), "blitz_rating": u["perfs"]["blitz"]["rating"],
           "progress": u["perfs"]["blitz"]["progress"]} for u in d["users"]])
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:46</summary>

```json
[
  {
    "username": "athena-pallada",
    "title": "GM",
    "blitz_rating": 2991,
    "progress": -10
  },
  {
    "username": "cutemouse83",
    "title": "GM",
    "blitz_rating": 2985,
    "progress": -8
  },
  {
    "username": "Real_G13M37",
    "title": null,
    "blitz_rating": 2955,
    "progress": 9
  },
  {
    "username": "tarzansmurf",
    "title": null,
    "blitz_rating": 2955,
    "progress": 47
  },
  {
    "username": "Lintchevski_Daniil",
    "title": "GM",
    "blitz_rating": 2944,
    "progress": 4
  }
]
```
</details>

---

### MLB Stats API

| | |
|:--|:--|
| **Docs** | https://github.com/toddrob99/MLB-StatsAPI/wiki (community docs) · terms: http://gdx.mlb.com/components/copyright.txt |
| **Auth** | None |
| **Register / token** | No registration. Public but not formally documented; MLB's copyright notice limits use to individual, non-commercial, non-bulk purposes. Python wrapper: `pip install MLB-StatsAPI`. |
| **Script** | [`scripts/sports/mlb_stats.py`](scripts/sports/mlb_stats.py) |

**What you can get:**

- Schedules, live game feeds with pitch-by-pitch data (velocity, spin, location)
- Standings, rosters, player season & career stats, team stats
- Minor-league and historical data

**Capstone ideas:** Pitch-outcome prediction, player-valuation models, in-game win probability.

**Endpoint:** `GET https://statsapi.mlb.com/api/v1/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `standings?leagueId=&season=` | 103 = American League, 104 = National League | `leagueId=103,104&season=2026` |
| `schedule?sportId=&date= / startDate=&endDate=` | sportId 1 = MLB; dates MM/DD/YYYY or YYYY-MM-DD | `sportId=1&date=2026-09-28` |
| `game/{gamePk}/feed/live (v1.1)` | Full live feed incl. every pitch (/api/v1.1/game/{gamePk}/feed/live) | `gamePk=822679 (from /schedule)` |
| `people/{id}/stats?stats=&group=&season=` | Player stats; stats season \| career \| gameLog, group hitting \| pitching | `stats=season&group=hitting` |
| `teams/{id}/roster` | Active roster (rosterType=40Man for the 40-man roster) | `teams/120/roster` |
| `hydrate` | Embed related objects in one call | `hydrate=team,venue` |

```python
from _common import get, show


def main():
    d = get("https://statsapi.mlb.com/api/v1/standings", params={"leagueId": "103,104", "season": 2026})
    rows = [t for rec in d["records"] for t in rec["teamRecords"]]
    rows.sort(key=lambda t: -float(t["winningPercentage"]))
    show([{"team": t["team"]["name"], "W": t["wins"], "L": t["losses"], "pct": t["winningPercentage"],
           "run_diff": t["runDifferential"]} for t in rows], max_items=6)
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:46</summary>

```json
[
  {
    "team": "Brewers",
    "W": 103,
    "L": 59,
    "pct": ".636",
    "run_diff": 214
  },
  {
    "team": "Dodgers",
    "W": 100,
    "L": 62,
    "pct": ".617",
    "run_diff": 201
  },
  {
    "team": "Rays",
    "W": 98,
    "L": 64,
    "pct": ".605",
    "run_diff": 86
  },
  {
    "team": "Braves",
    "W": 94,
    "L": 68,
    "pct": ".580",
    "run_diff": 116
  },
  {
    "team": "Yankees",
    "W": 93,
    "L": 68,
    "pct": ".578",
    "run_diff": 138
  },
  {
    "team": "Padres",
    "W": 91,
    "L": 71,
    "pct": ".562",
    "run_diff": 41
  },
  "... (24 more)"
]
```
</details>

---

### NHL Web API (unofficial)

| | |
|:--|:--|
| **Docs** | https://github.com/Zmalski/NHL-API-Reference (community docs) |
| **Auth** | None |
| **Register / token** | No registration. Undocumented public API behind nhl.com; endpoints can change. |
| **Script** | [`scripts/sports/nhl.py`](scripts/sports/nhl.py) |

**What you can get:**

- Standings, schedules, scores
- Play-by-play with shot coordinates and event types
- Player game logs and season stats, team rosters

**Capstone ideas:** Expected-goals (xG) models from shot locations, goalie performance analysis.

**Endpoint:** `GET https://api-web.nhle.com/v1/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `standings/{date \| now}` | League standings on a date (YYYY-MM-DD) | `standings/now` |
| `schedule/{date \| now}` | Week of games starting at the date | `schedule/2026-10-11` |
| `score/{date \| now}` | Scores for a date | `score/now` |
| `gamecenter/{gameId}/play-by-play` | Every event with x/y coordinates | `gamecenter/2026020001/play-by-play` |
| `roster/{team}/current` | Team roster (3-letter team code) | `roster/WSH/current` |
| `player/{playerId}/landing` | Player bio + career stats | `player/8471214/landing` |

```python
from _common import get, show


def main():
    d = get("https://api-web.nhle.com/v1/standings/now")
    rows = sorted(d["standings"], key=lambda t: t["leagueSequence"])
    show({"as_of": d["standingsDateTimeUtc"], "top_teams": [
        {"team": t["teamName"]["default"], "GP": t["gamesPlayed"], "W": t["wins"], "L": t["losses"],
         "PTS": t["points"], "GF": t["goalFor"], "GA": t["goalAgainst"]} for t in rows]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:46</summary>

```json
{
  "as_of": "2026-10-12T00:46:30Z",
  "top_teams": [
    {
      "team": "Vegas Golden Knights",
      "GP": 6,
      "W": 5,
      "L": 1,
      "PTS": 10,
      "GF": 23,
      "GA": 14
    },
    {
      "team": "New York Rangers",
      "GP": 7,
      "W": 5,
      "L": 2,
      "PTS": 10,
      "GF": 23,
      "GA": 13
    },
    {
      "team": "Edmonton Oilers",
      "GP": 5,
      "W": 4,
      "L": 0,
      "PTS": 9,
      "GF": 25,
      "GA": 17
    },
    {
      "team": "Carolina Hurricanes",
      "GP": 6,
      "W": 4,
      "L": 1,
      "PTS": 9,
      "GF": 23,
      "GA": 16
    },
    {
      "team": "Tampa Bay Lightning",
      "GP": 5,
      "W": 4,
      "L": 1,
      "PTS": 8,
      "GF": 16,
      "GA": 12
    },
    "... (27 more)"
  ]
}
```
</details>

---

### OpenF1 (Formula 1 telemetry)

| | |
|:--|:--|
| **Docs** | https://openf1.org/ |
| **Auth** | None for historical data (paid tier for live sessions) |
| **Register / token** | No registration for historical data since 2023. Rate-limited to a few requests/sec. |
| **Script** | [`scripts/sports/openf1.py`](scripts/sports/openf1.py) |

**What you can get:**

- Sessions, meetings, drivers, lap times, sector times, pit stops, stints/tyres, race control messages
- Car telemetry (~3.7 Hz): speed, RPM, throttle, brake, gear, DRS
- Car position (x, y, z) on track, intervals, weather

**Capstone ideas:** Tyre-degradation models, pit-strategy optimization, driver comparison from telemetry.

**Endpoint:** `GET https://api.openf1.org/v1/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `resource` | sessions \| meetings \| drivers \| laps \| pit \| stints \| car_data \| location \| position \| intervals \| weather \| race_control | `laps` |
| `session_key / meeting_key` | Identify a session (use latest for the most recent) | `session_key=9693` |
| `year / session_type / country_name` | Filters on sessions/meetings | `year=2025&session_type=Race` |
| `driver_number` | Car number | `1` |
| `field>=value / field<=value` | Comparison filters on any numeric/date field | `lap_number>=10&speed>=300` |
| `csv` | true → CSV instead of JSON | `true` |

```python
from _common import get, show


def main():
    race = get("https://api.openf1.org/v1/sessions", params={"year": 2025, "session_type": "Race",
                                                            "country_name": "Australia"})[0]
    laps = get("https://api.openf1.org/v1/laps", params={"session_key": race["session_key"], "driver_number": 1})
    show({"session": {k: race[k] for k in ("session_key", "location", "date_start")},
          "laps_driver_1": [{k: l.get(k) for k in ("lap_number", "lap_duration", "duration_sector_1", "st_speed")}
                            for l in laps[1:]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:46</summary>

```json
{
  "session": {
    "session_key": 9693,
    "location": "Melbourne",
    "date_start": "2025-03-16T04:00:00+00:00"
  },
  "laps_driver_1": [
    {
      "lap_number": 2,
      "lap_duration": 169.322,
      "duration_sector_1": 58.141,
      "st_speed": 210
    },
    {
      "lap_number": 3,
      "lap_duration": 162.145,
      "duration_sector_1": 56.23,
      "st_speed": 180
    },
    {
      "lap_number": 4,
      "lap_duration": 155.775,
      "duration_sector_1": 54.351,
      "st_speed": 187
    },
    {
      "lap_number": 5,
      "lap_duration": 143.609,
      "duration_sector_1": 53.513,
      "st_speed": 194
    },
    {
      "lap_number": 6,
      "lap_duration": 139.654,
      "duration_sector_1": 47.887,
      "st_speed": 127
    },
    "... (51 more)"
  ]
}
```
</details>

---

### TheSportsDB

| | |
|:--|:--|
| **Docs** | https://www.thesportsdb.com/documentation |
| **Auth** | API key in the URL path (free test key `123`) |
| **Register / token** | The free key `123` works for v1 lookups (~30 req/min). A paid Patreon key unlocks v2, livescores and higher limits. |
| **Env var** | `THESPORTSDB_KEY (optional)` |
| **Script** | [`scripts/sports/thesportsdb.py`](scripts/sports/thesportsdb.py) |

**What you can get:**

- Teams, players, leagues across 50+ sports (soccer, NFL, NBA, cricket, rugby, motorsport, esports...)
- Past & upcoming events with scores, venues, stadium/badge/kit artwork
- League tables for major soccer leagues

**Capstone ideas:** Multi-sport metadata joins, image datasets (badges/kits), schedule analysis.

**Endpoint:** `GET https://www.thesportsdb.com/api/v1/json/{key}/{endpoint}.php`

| Parameter | Description | Example |
|:--|:--|:--|
| `key` | 123 (free) or your premium key | `123` |
| `searchteams.php?t=` | Search teams by name | `t=Arsenal` |
| `searchplayers.php?p=` | Search players by name | `p=Bukayo Saka` |
| `lookuptable.php?l=&s=` | League table (league id + season) | `l=4328&s=2025-2026` |
| `eventsnextleague.php?id= / eventspastleague.php?id=` | Upcoming / recent events in a league | `id=4328` |
| `eventsseason.php?id=&s=` | All events in a league season | `id=4328&s=2025-2026` |

```python
import os

from _common import get, show


def main():
    k = os.getenv("THESPORTSDB_KEY", "123")
    team = get(f"https://www.thesportsdb.com/api/v1/json/{k}/searchteams.php", params={"t": "Arsenal"})["teams"][0]
    past = get(f"https://www.thesportsdb.com/api/v1/json/{k}/eventslast.php", params={"id": team["idTeam"]})
    show({"team": {k2: team[k2] for k2 in ("strTeam", "strLeague", "intFormedYear", "strStadium")},
          "last_results": [{k2: e[k2] for k2 in ("dateEvent", "strEvent", "intHomeScore", "intAwayScore")}
                           for e in (past.get("results") or [])]})
```

<details><summary><b>Sample output</b> — live run 2026-10-11 20:46</summary>

```json
{
  "team": {
    "strTeam": "Arsenal",
    "strLeague": "English Premier League",
    "intFormedYear": "1892",
    "strStadium": "Emirates Stadium"
  },
  "last_results": [
    {
      "dateEvent": "2026-10-10",
      "strEvent": "Arsenal vs Leeds United",
      "intHomeScore": "2",
      "intAwayScore": "1"
    }
  ]
}
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

**Endpoint:** `GET https://collectionapi.metmuseum.org/public/collection/v1/objects/{objectID}`

| Parameter | Description | Example |
|:--|:--|:--|
| `objectID` | Object to fetch | `436524` |
| `/objects` | All object IDs; filter with departmentIds= and metadataDate= (changed since) | `departmentIds=11` |
| `/departments` | Department IDs and names | `—` |

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

**Endpoint:** `GET https://openlibrary.org/search.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `q / title / author / subject` | General or field-specific search | `data science` |
| `fields` | Fields to return | `title,author_name,first_publish_year` |
| `sort` | new \| old \| rating \| editions \| random | `new` |
| `limit / page` | Paging | `5` |
| `lang` | Preferred language (ISO 639-1) | `en` |
| `/works/{id}.json · /isbn/{isbn}.json · /authors/{id}.json` | Sibling endpoints: work, edition by ISBN, author | `/isbn/9781492041139.json` |
| `covers.openlibrary.org/b/isbn/{isbn}-{S\|M\|L}.jpg` | Cover images | `9781492041139-M.jpg` |

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

**Endpoint:** `GET https://api.themoviedb.org/3/trending/movie/{time_window}`

| Parameter | Description | Example |
|:--|:--|:--|
| `time_window` | day \| week | `week` |
| `/3/search/movie?query=&year=` | Sibling endpoint: search by title | `query=Inception` |
| `/3/discover/movie` | Filter: with_genres, primary_release_year, vote_count.gte, sort_by (popularity.desc, revenue.desc, ...) | `sort_by=revenue.desc&primary_release_year=2025` |
| `/3/movie/{id}?append_to_response=` | Details + credits, keywords, reviews in one call | `append_to_response=credits,keywords` |
| `language / page` | Locale and paging | `en-US / 1` |
| `Authorization (header)` | Read-access token | `Bearer <TMDB_TOKEN>` |

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
