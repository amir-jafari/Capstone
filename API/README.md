# Public API Catalog for Data Science Capstones

> A curated, runnable list of free APIs across news, social media, weather & environment, finance,
> census, housing, political science, law, health, agriculture & food, education, jobs & labor, science & space, AI & machine learning,
> engineering & energy, transportation, sports, and culture, media & entertainment.
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
| 12 | 🌦️ Weather, Climate & Environment | [EPA AirNow API (real-time air quality)](#epa-airnow-api-real-time-air-quality) | 🔑 key | `AIRNOW_API_KEY` | [`airnow.py`](scripts/weather_environment/airnow.py) |
| 13 | 🌦️ Weather, Climate & Environment | [Climate TRACE (global emissions inventory)](#climate-trace-global-emissions-inventory) | 🟢 no key | — | [`climate_trace.py`](scripts/weather_environment/climate_trace.py) |
| 14 | 🌦️ Weather, Climate & Environment | [U.S. Drought Monitor](#us-drought-monitor) | 🟢 no key | — | [`drought_monitor.py`](scripts/weather_environment/drought_monitor.py) |
| 15 | 🌦️ Weather, Climate & Environment | [EPA Air Quality System (AQS) API](#epa-air-quality-system-aqs-api) | 🔑 key | `AQS_EMAIL, AQS_KEY` | [`epa_aqs.py`](scripts/weather_environment/epa_aqs.py) |
| 16 | 🌦️ Weather, Climate & Environment | [EPA ECHO (Enforcement & Compliance History)](#epa-echo-enforcement--compliance-history) | 🟢 no key | — | [`epa_echo.py`](scripts/weather_environment/epa_echo.py) |
| 17 | 🌦️ Weather, Climate & Environment | [EPA Envirofacts (Toxics Release Inventory and more)](#epa-envirofacts-toxics-release-inventory-and-more) | 🟢 no key | — | [`epa_envirofacts.py`](scripts/weather_environment/epa_envirofacts.py) |
| 18 | 🌦️ Weather, Climate & Environment | [GBIF (Global Biodiversity Information Facility)](#gbif-global-biodiversity-information-facility) | 🟢 no key | — | [`gbif.py`](scripts/weather_environment/gbif.py) |
| 19 | 🌦️ Weather, Climate & Environment | [iNaturalist API](#inaturalist-api) | 🟢 no key | — | [`inaturalist.py`](scripts/weather_environment/inaturalist.py) |
| 20 | 🌦️ Weather, Climate & Environment | [NASA FIRMS (active fire detections)](#nasa-firms-active-fire-detections) | 🔑 key | `FIRMS_MAP_KEY` | [`nasa_firms.py`](scripts/weather_environment/nasa_firms.py) |
| 21 | 🌦️ Weather, Climate & Environment | [NOAA Climate Data Online (CDO)](#noaa-climate-data-online-cdo) | 🔑 key | `NOAA_TOKEN` | [`noaa_cdo.py`](scripts/weather_environment/noaa_cdo.py) |
| 22 | 🌦️ Weather, Climate & Environment | [NOAA Climate at a Glance](#noaa-climate-at-a-glance) | 🟢 no key | — | [`noaa_climate_at_a_glance.py`](scripts/weather_environment/noaa_climate_at_a_glance.py) |
| 23 | 🌦️ Weather, Climate & Environment | [NOAA Global Monitoring Lab — Mauna Loa CO2](#noaa-global-monitoring-lab--mauna-loa-co2) | 🟢 no key | — | [`noaa_co2.py`](scripts/weather_environment/noaa_co2.py) |
| 24 | 🌦️ Weather, Climate & Environment | [NOAA Tides & Currents (CO-OPS)](#noaa-tides--currents-co-ops) | 🟢 no key | — | [`noaa_tides.py`](scripts/weather_environment/noaa_tides.py) |
| 25 | 🌦️ Weather, Climate & Environment | [US National Weather Service (api.weather.gov)](#us-national-weather-service-apiweathergov) | 🟢 no key | — | [`nws.py`](scripts/weather_environment/nws.py) |
| 26 | 🌦️ Weather, Climate & Environment | [Open-Meteo](#open-meteo) | 🟢 no key | — | [`open_meteo.py`](scripts/weather_environment/open_meteo.py) |
| 27 | 🌦️ Weather, Climate & Environment | [OpenAQ v3 (Air Quality)](#openaq-v3-air-quality) | 🔑 key | `OPENAQ_API_KEY` | [`openaq.py`](scripts/weather_environment/openaq.py) |
| 28 | 🌦️ Weather, Climate & Environment | [OpenWeatherMap](#openweathermap) | 🔑 key | `OPENWEATHER_API_KEY` | [`openweathermap.py`](scripts/weather_environment/openweathermap.py) |
| 29 | 🌦️ Weather, Climate & Environment | [USGS Earthquake Catalog](#usgs-earthquake-catalog) | 🟢 no key | — | [`usgs_earthquakes.py`](scripts/weather_environment/usgs_earthquakes.py) |
| 30 | 💵 Finance & Economics | [Alpha Vantage](#alpha-vantage) | 🟢 no key | `ALPHAVANTAGE_API_KEY` | [`alpha_vantage.py`](scripts/finance_economics/alpha_vantage.py) |
| 31 | 💵 Finance & Economics | [US Bureau of Labor Statistics (BLS)](#us-bureau-of-labor-statistics-bls) | 🟢 no key | `BLS_API_KEY (optional)` | [`bls.py`](scripts/finance_economics/bls.py) |
| 32 | 💵 Finance & Economics | [CoinGecko](#coingecko) | 🟢 no key | — | [`coingecko.py`](scripts/finance_economics/coingecko.py) |
| 33 | 💵 Finance & Economics | [Frankfurter (ECB exchange rates)](#frankfurter-ecb-exchange-rates) | 🟢 no key | — | [`frankfurter.py`](scripts/finance_economics/frankfurter.py) |
| 34 | 💵 Finance & Economics | [FRED (Federal Reserve Economic Data)](#fred-federal-reserve-economic-data) | 🔑 key | `FRED_API_KEY` | [`fred.py`](scripts/finance_economics/fred.py) |
| 35 | 💵 Finance & Economics | [SEC EDGAR (data.sec.gov)](#sec-edgar-datasecgov) | 🟢 no key | — | [`sec_edgar.py`](scripts/finance_economics/sec_edgar.py) |
| 36 | 💵 Finance & Economics | [US Treasury Fiscal Data](#us-treasury-fiscal-data) | 🟢 no key | — | [`treasury_fiscal.py`](scripts/finance_economics/treasury_fiscal.py) |
| 37 | 💵 Finance & Economics | [World Bank Indicators API](#world-bank-indicators-api) | 🟢 no key | — | [`world_bank.py`](scripts/finance_economics/world_bank.py) |
| 38 | 👥 Census & Demographics | [IPUMS API (microdata)](#ipums-api-microdata) | 🔑 key | `IPUMS_API_KEY` | [`ipums.py`](scripts/census_demographics/ipums.py) |
| 39 | 👥 Census & Demographics | [UN Data Portal — World Population Prospects](#un-data-portal--world-population-prospects) | 🔑 key | `UN_POP_TOKEN` | [`un_population.py`](scripts/census_demographics/un_population.py) |
| 40 | 👥 Census & Demographics | [US Census Bureau Data API](#us-census-bureau-data-api) | 🔑 key | `CENSUS_API_KEY` | [`us_census.py`](scripts/census_demographics/us_census.py) |
| 41 | 🏠 Housing & Real Estate | [CFPB Consumer Complaint Database](#cfpb-consumer-complaint-database) | 🟢 no key | — | [`cfpb_complaints.py`](scripts/housing/cfpb_complaints.py) |
| 42 | 🏠 Housing & Real Estate | [FHFA House Price Index](#fhfa-house-price-index) | 🟢 no key | — | [`fhfa_hpi.py`](scripts/housing/fhfa_hpi.py) |
| 43 | 🏠 Housing & Real Estate | [HMDA Data Browser (mortgage applications)](#hmda-data-browser-mortgage-applications) | 🟢 no key | — | [`hmda.py`](scripts/housing/hmda.py) |
| 44 | 🏠 Housing & Real Estate | [HUD User API (Fair Market Rents, Income Limits, ZIP crosswalk)](#hud-user-api-fair-market-rents-income-limits-zip-crosswalk) | 🔑 key | `HUD_API_TOKEN` | [`hud_user.py`](scripts/housing/hud_user.py) |
| 45 | 🏠 Housing & Real Estate | [OpenFEMA — Flood Insurance Claims & Disaster Declarations](#openfema--flood-insurance-claims--disaster-declarations) | 🟢 no key | — | [`openfema_nfip.py`](scripts/housing/openfema_nfip.py) |
| 46 | 🏠 Housing & Real Estate | [Redfin Data Center (housing market tracker)](#redfin-data-center-housing-market-tracker) | 🟢 no key | — | [`redfin_data_center.py`](scripts/housing/redfin_data_center.py) |
| 47 | 🏠 Housing & Real Estate | [RentCast (property records, rent & value estimates)](#rentcast-property-records-rent--value-estimates) | 🔑 key | `RENTCAST_API_KEY` | [`rentcast.py`](scripts/housing/rentcast.py) |
| 48 | 🏠 Housing & Real Estate | [Zillow Research Data (ZHVI home values, ZORI rents)](#zillow-research-data-zhvi-home-values-zori-rents) | 🟢 no key | — | [`zillow_research.py`](scripts/housing/zillow_research.py) |
| 49 | 🏛️ Political Science & Government | [ACLED (Armed Conflict Location & Event Data)](#acled-armed-conflict-location--event-data) | 🔑 key | `ACLED_EMAIL, ACLED_PASSWORD` | [`acled.py`](scripts/politics_government/acled.py) |
| 50 | 🏛️ Political Science & Government | [Congress.gov API](#congressgov-api) | 🟢 no key | `DATA_GOV_API_KEY` | [`congress_gov.py`](scripts/politics_government/congress_gov.py) |
| 51 | 🏛️ Political Science & Government | [European Parliament Open Data API](#european-parliament-open-data-api) | 🟢 no key | — | [`european_parliament.py`](scripts/politics_government/european_parliament.py) |
| 52 | 🏛️ Political Science & Government | [Federal Register API](#federal-register-api) | 🟢 no key | — | [`federal_register.py`](scripts/politics_government/federal_register.py) |
| 53 | 🏛️ Political Science & Government | [GovInfo API (U.S. Government Publishing Office)](#govinfo-api-us-government-publishing-office) | 🟢 no key | `DATA_GOV_API_KEY` | [`govinfo.py`](scripts/politics_government/govinfo.py) |
| 54 | 🏛️ Political Science & Government | [Grants.gov Search API](#grantsgov-search-api) | 🟢 no key | — | [`grants_gov.py`](scripts/politics_government/grants_gov.py) |
| 55 | 🏛️ Political Science & Government | [Harvard Dataverse (MIT Election Lab & replication data)](#harvard-dataverse-mit-election-lab--replication-data) | 🟢 no key | `DATAVERSE_API_TOKEN (optional)` | [`harvard_dataverse.py`](scripts/politics_government/harvard_dataverse.py) |
| 56 | 🏛️ Political Science & Government | [Open States / Plural (US state legislatures)](#open-states--plural-us-state-legislatures) | 🔑 key | `OPENSTATES_API_KEY` | [`open_states.py`](scripts/politics_government/open_states.py) |
| 57 | 🏛️ Political Science & Government | [OpenFEC (Federal Election Commission)](#openfec-federal-election-commission) | 🟢 no key | `DATA_GOV_API_KEY` | [`openfec.py`](scripts/politics_government/openfec.py) |
| 58 | 🏛️ Political Science & Government | [Regulations.gov API v4](#regulationsgov-api-v4) | 🟢 no key | `DATA_GOV_API_KEY` | [`regulations_gov.py`](scripts/politics_government/regulations_gov.py) |
| 59 | 🏛️ Political Science & Government | [Lobbying Disclosure Act (LDA) API](#lobbying-disclosure-act-lda-api) | 🟢 no key | `LDA_API_KEY (optional)` | [`senate_lda.py`](scripts/politics_government/senate_lda.py) |
| 60 | 🏛️ Political Science & Government | [UK Parliament APIs](#uk-parliament-apis) | 🟢 no key | — | [`uk_parliament.py`](scripts/politics_government/uk_parliament.py) |
| 61 | 🏛️ Political Science & Government | [USAJOBS Search API (federal jobs)](#usajobs-search-api-federal-jobs) | 🔑 key | `USAJOBS_API_KEY, USAJOBS_EMAIL` | [`usajobs.py`](scripts/politics_government/usajobs.py) |
| 62 | 🏛️ Political Science & Government | [USAspending.gov](#usaspendinggov) | 🟢 no key | — | [`usaspending.py`](scripts/politics_government/usaspending.py) |
| 63 | ⚖️ Law & Justice | [CourtListener (Free Law Project)](#courtlistener-free-law-project) | 🟢 no key | `COURTLISTENER_TOKEN (optional)` | [`courtlistener.py`](scripts/law/courtlistener.py) |
| 64 | ⚖️ Law & Justice | [U.S. Department of Justice News API](#us-department-of-justice-news-api) | 🟢 no key | — | [`doj_press_releases.py`](scripts/law/doj_press_releases.py) |
| 65 | ⚖️ Law & Justice | [eCFR API (Code of Federal Regulations)](#ecfr-api-code-of-federal-regulations) | 🟢 no key | — | [`ecfr.py`](scripts/law/ecfr.py) |
| 66 | ⚖️ Law & Justice | [FBI Crime Data Explorer API](#fbi-crime-data-explorer-api) | 🟢 no key | `DATA_GOV_API_KEY` | [`fbi_crime_data.py`](scripts/law/fbi_crime_data.py) |
| 67 | ⚖️ Law & Justice | [Oyez (U.S. Supreme Court cases)](#oyez-us-supreme-court-cases) | 🟢 no key | — | [`oyez.py`](scripts/law/oyez.py) |
| 68 | ⚖️ Law & Justice | [Supreme Court Database (SCDB)](#supreme-court-database-scdb) | 🟢 no key | — | [`supreme_court_database.py`](scripts/law/supreme_court_database.py) |
| 69 | 🩺 Health & Social Science | [CDC Open Data (Socrata SODA)](#cdc-open-data-socrata-soda) | 🟢 no key | `SOCRATA_APP_TOKEN (optional)` | [`cdc_socrata.py`](scripts/health/cdc_socrata.py) |
| 70 | 🩺 Health & Social Science | [ChEMBL (drug discovery bioactivity)](#chembl-drug-discovery-bioactivity) | 🟢 no key | — | [`chembl.py`](scripts/health/chembl.py) |
| 71 | 🩺 Health & Social Science | [ClinicalTrials.gov API v2](#clinicaltrialsgov-api-v2) | 🟢 no key | — | [`clinicaltrials.py`](scripts/health/clinicaltrials.py) |
| 72 | 🩺 Health & Social Science | [CMS Medicare Part D Prescribers (data.cms.gov)](#cms-medicare-part-d-prescribers-datacmsgov) | 🟢 no key | — | [`cms_part_d.py`](scripts/health/cms_part_d.py) |
| 73 | 🩺 Health & Social Science | [CMS Provider Data Catalog (Care Compare)](#cms-provider-data-catalog-care-compare) | 🟢 no key | — | [`cms_provider_data.py`](scripts/health/cms_provider_data.py) |
| 74 | 🩺 Health & Social Science | [DailyMed (NLM drug labels)](#dailymed-nlm-drug-labels) | 🟢 no key | — | [`dailymed.py`](scripts/health/dailymed.py) |
| 75 | 🩺 Health & Social Science | [Delphi Epidata (CMU disease surveillance)](#delphi-epidata-cmu-disease-surveillance) | 🟢 no key | `DELPHI_API_KEY (optional)` | [`delphi_epidata.py`](scripts/health/delphi_epidata.py) |
| 76 | 🩺 Health & Social Science | [Ensembl REST API (genomics)](#ensembl-rest-api-genomics) | 🟢 no key | — | [`ensembl.py`](scripts/health/ensembl.py) |
| 77 | 🩺 Health & Social Science | [Europe PMC](#europe-pmc) | 🟢 no key | — | [`europe_pmc.py`](scripts/health/europe_pmc.py) |
| 78 | 🩺 Health & Social Science | [HAPI FHIR public test server (synthetic EHR data)](#hapi-fhir-public-test-server-synthetic-ehr-data) | 🟢 no key | — | [`hapi_fhir.py`](scripts/health/hapi_fhir.py) |
| 79 | 🩺 Health & Social Science | [MedlinePlus Connect (patient education)](#medlineplus-connect-patient-education) | 🟢 no key | — | [`medlineplus_connect.py`](scripts/health/medlineplus_connect.py) |
| 80 | 🩺 Health & Social Science | [NLM Clinical Table Search Service (ICD-10-CM, LOINC, conditions)](#nlm-clinical-table-search-service-icd-10-cm-loinc-conditions) | 🟢 no key | — | [`nlm_clinical_tables.py`](scripts/health/nlm_clinical_tables.py) |
| 81 | 🩺 Health & Social Science | [NPPES NPI Registry (healthcare providers)](#nppes-npi-registry-healthcare-providers) | 🟢 no key | — | [`nppes_npi.py`](scripts/health/nppes_npi.py) |
| 82 | 🩺 Health & Social Science | [Open Targets Platform (GraphQL)](#open-targets-platform-graphql) | 🟢 no key | — | [`open_targets.py`](scripts/health/open_targets.py) |
| 83 | 🩺 Health & Social Science | [openFDA](#openfda) | 🟢 no key | `OPENFDA_API_KEY (optional)` | [`openfda.py`](scripts/health/openfda.py) |
| 84 | 🩺 Health & Social Science | [Our World in Data (Grapher API)](#our-world-in-data-grapher-api) | 🟢 no key | — | [`our_world_in_data.py`](scripts/health/our_world_in_data.py) |
| 85 | 🩺 Health & Social Science | [PubMed (NCBI E-utilities)](#pubmed-ncbi-e-utilities) | 🟢 no key | `NCBI_API_KEY (optional)` | [`pubmed.py`](scripts/health/pubmed.py) |
| 86 | 🩺 Health & Social Science | [RxNorm / RxNav (NLM drug terminology)](#rxnorm--rxnav-nlm-drug-terminology) | 🟢 no key | — | [`rxnorm.py`](scripts/health/rxnorm.py) |
| 87 | 🩺 Health & Social Science | [UniProt (proteins)](#uniprot-proteins) | 🟢 no key | — | [`uniprot.py`](scripts/health/uniprot.py) |
| 88 | 🩺 Health & Social Science | [WHO Global Health Observatory (GHO OData)](#who-global-health-observatory-gho-odata) | 🟢 no key | — | [`who_gho.py`](scripts/health/who_gho.py) |
| 89 | 🩺 Health & Social Science | [WHO ICD-11 API](#who-icd-11-api) | 🔑 key | `ICD_CLIENT_ID, ICD_CLIENT_SECRET` | [`who_icd11.py`](scripts/health/who_icd11.py) |
| 90 | 🌾 Agriculture & Food | [USDA AMS Market News (MyMarketNews API)](#usda-ams-market-news-mymarketnews-api) | 🔑 key | `AMS_API_KEY` | [`ams_market_news.py`](scripts/agriculture/ams_market_news.py) |
| 91 | 🌾 Agriculture & Food | [USDA NASS CropScape (Cropland Data Layer)](#usda-nass-cropscape-cropland-data-layer) | 🟢 no key | — | [`cropscape.py`](scripts/agriculture/cropscape.py) |
| 92 | 🌾 Agriculture & Food | [USDA ERS — ARMS Farm Financial Data](#usda-ers--arms-farm-financial-data) | 🟢 no key | `DATA_GOV_API_KEY` | [`ers_arms.py`](scripts/agriculture/ers_arms.py) |
| 93 | 🌾 Agriculture & Food | [USDA FAS — Production, Supply & Distribution (PSD) and trade](#usda-fas--production-supply--distribution-psd-and-trade) | 🟢 no key | `DATA_GOV_API_KEY` | [`fas_psd.py`](scripts/agriculture/fas_psd.py) |
| 94 | 🌾 Agriculture & Food | [USDA FoodData Central](#usda-fooddata-central) | 🟢 no key | `DATA_GOV_API_KEY` | [`fooddata_central.py`](scripts/agriculture/fooddata_central.py) |
| 95 | 🌾 Agriculture & Food | [USDA NASS Quick Stats](#usda-nass-quick-stats) | 🔑 key | `NASS_API_KEY` | [`nass_quickstats.py`](scripts/agriculture/nass_quickstats.py) |
| 96 | 🌾 Agriculture & Food | [Open Food Facts](#open-food-facts) | 🟢 no key | — | [`open_food_facts.py`](scripts/agriculture/open_food_facts.py) |
| 97 | 🌾 Agriculture & Food | [USDA NRCS Soil Data Access (SSURGO)](#usda-nrcs-soil-data-access-ssurgo) | 🟢 no key | — | [`soil_data_access.py`](scripts/agriculture/soil_data_access.py) |
| 98 | 🌾 Agriculture & Food | [ISRIC SoilGrids (global soil properties)](#isric-soilgrids-global-soil-properties) | 🟢 no key | — | [`soilgrids.py`](scripts/agriculture/soilgrids.py) |
| 99 | 🌾 Agriculture & Food | [TheMealDB (recipes)](#themealdb-recipes) | 🔑 key | `THEMEALDB_KEY (optional)` | [`themealdb.py`](scripts/agriculture/themealdb.py) |
| 100 | 🎓 Education | [College Scorecard (US Dept. of Education)](#college-scorecard-us-dept-of-education) | 🟢 no key | `DATA_GOV_API_KEY` | [`college_scorecard.py`](scripts/education/college_scorecard.py) |
| 101 | 🎓 Education | [Coursera Catalog API](#coursera-catalog-api) | 🟢 no key | — | [`coursera_catalog.py`](scripts/education/coursera_catalog.py) |
| 102 | 🎓 Education | [Datamuse (word-finding API)](#datamuse-word-finding-api) | 🟢 no key | — | [`datamuse.py`](scripts/education/datamuse.py) |
| 103 | 🎓 Education | [ERIC (Education Resources Information Center)](#eric-education-resources-information-center) | 🟢 no key | — | [`eric.py`](scripts/education/eric.py) |
| 104 | 🎓 Education | [Hipolabs Universities List](#hipolabs-universities-list) | 🟢 no key | — | [`hipolabs_universities.py`](scripts/education/hipolabs_universities.py) |
| 105 | 🎓 Education | [NAEP — The Nation's Report Card](#naep--the-nations-report-card) | 🟢 no key | — | [`naep.py`](scripts/education/naep.py) |
| 106 | 🎓 Education | [OECD Data API — Education at a Glance](#oecd-data-api--education-at-a-glance) | 🟢 no key | — | [`oecd_education.py`](scripts/education/oecd_education.py) |
| 107 | 🎓 Education | [Open Trivia Database](#open-trivia-database) | 🟢 no key | — | [`open_trivia_db.py`](scripts/education/open_trivia_db.py) |
| 108 | 🎓 Education | [UNESCO Institute for Statistics (UIS) Data API](#unesco-institute-for-statistics-uis-data-api) | 🟢 no key | — | [`unesco_uis.py`](scripts/education/unesco_uis.py) |
| 109 | 🎓 Education | [Urban Institute Education Data Portal](#urban-institute-education-data-portal) | 🟢 no key | — | [`urban_education_data.py`](scripts/education/urban_education_data.py) |
| 110 | 🎓 Education | [World Bank EdStats (education statistics)](#world-bank-edstats-education-statistics) | 🟢 no key | — | [`world_bank_edstats.py`](scripts/education/world_bank_edstats.py) |
| 111 | 💼 Jobs & Labor | [Adzuna Jobs API](#adzuna-jobs-api) | 🔑 key | `ADZUNA_APP_ID, ADZUNA_APP_KEY` | [`adzuna.py`](scripts/jobs/adzuna.py) |
| 112 | 💼 Jobs & Labor | [Ashby Job Postings API (startup career pages)](#ashby-job-postings-api-startup-career-pages) | 🟢 no key | — | [`ashby_job_board.py`](scripts/jobs/ashby_job_board.py) |
| 113 | 💼 Jobs & Labor | [CareerOneStop Web API (U.S. Department of Labor)](#careeronestop-web-api-us-department-of-labor) | 🔑 key | `CAREERONESTOP_USER_ID, CAREERONESTOP_TOKEN` | [`careeronestop.py`](scripts/jobs/careeronestop.py) |
| 114 | 💼 Jobs & Labor | [ESCO API (European skills & occupations taxonomy)](#esco-api-european-skills--occupations-taxonomy) | 🟢 no key | — | [`esco.py`](scripts/jobs/esco.py) |
| 115 | 💼 Jobs & Labor | [Greenhouse Job Board API (company career pages)](#greenhouse-job-board-api-company-career-pages) | 🟢 no key | — | [`greenhouse_job_boards.py`](scripts/jobs/greenhouse_job_boards.py) |
| 116 | 💼 Jobs & Labor | [Hacker News 'Who is hiring?' (via Algolia HN Search)](#hacker-news-who-is-hiring-via-algolia-hn-search) | 🟢 no key | — | [`hn_whos_hiring.py`](scripts/jobs/hn_whos_hiring.py) |
| 117 | 💼 Jobs & Labor | [ILOSTAT (International Labour Organization)](#ilostat-international-labour-organization) | 🟢 no key | — | [`ilostat.py`](scripts/jobs/ilostat.py) |
| 118 | 💼 Jobs & Labor | [Census LEHD LODES (jobs by census block)](#census-lehd-lodes-jobs-by-census-block) | 🟢 no key | — | [`lehd_lodes.py`](scripts/jobs/lehd_lodes.py) |
| 119 | 💼 Jobs & Labor | [Lever Postings API (company career pages)](#lever-postings-api-company-career-pages) | 🟢 no key | — | [`lever_postings.py`](scripts/jobs/lever_postings.py) |
| 120 | 💼 Jobs & Labor | [O*NET Web Services (occupations & skills)](#onet-web-services-occupations--skills) | 🔑 key | `ONET_API_KEY` | [`onet.py`](scripts/jobs/onet.py) |
| 121 | 💼 Jobs & Labor | [Remotive (remote jobs)](#remotive-remote-jobs) | 🟢 no key | — | [`remotive.py`](scripts/jobs/remotive.py) |
| 122 | 💼 Jobs & Labor | [The Muse Public API (jobs)](#the-muse-public-api-jobs) | 🟢 no key | `THEMUSE_API_KEY (optional)` | [`the_muse.py`](scripts/jobs/the_muse.py) |
| 123 | 🔬 Science, Space & Research | [arXiv API](#arxiv-api) | 🟢 no key | — | [`arxiv.py`](scripts/science_research/arxiv.py) |
| 124 | 🔬 Science, Space & Research | [CelesTrak (satellite orbital elements)](#celestrak-satellite-orbital-elements) | 🟢 no key | — | [`celestrak.py`](scripts/science_research/celestrak.py) |
| 125 | 🔬 Science, Space & Research | [CERN Open Data Portal](#cern-open-data-portal) | 🟢 no key | — | [`cern_open_data.py`](scripts/science_research/cern_open_data.py) |
| 126 | 🔬 Science, Space & Research | [Crossref REST API](#crossref-rest-api) | 🟢 no key | — | [`crossref.py`](scripts/science_research/crossref.py) |
| 127 | 🔬 Science, Space & Research | [JPL Small-Body Close Approach & SBDB APIs](#jpl-small-body-close-approach--sbdb-apis) | 🟢 no key | — | [`jpl_close_approach.py`](scripts/science_research/jpl_close_approach.py) |
| 128 | 🔬 Science, Space & Research | [JPL Fireball (bolide) API](#jpl-fireball-bolide-api) | 🟢 no key | — | [`jpl_fireball.py`](scripts/science_research/jpl_fireball.py) |
| 129 | 🔬 Science, Space & Research | [JPL Horizons (ephemerides)](#jpl-horizons-ephemerides) | 🟢 no key | — | [`jpl_horizons.py`](scripts/science_research/jpl_horizons.py) |
| 130 | 🔬 Science, Space & Research | [Launch Library 2 (The Space Devs)](#launch-library-2-the-space-devs) | 🟢 no key | — | [`launch_library.py`](scripts/science_research/launch_library.py) |
| 131 | 🔬 Science, Space & Research | [Materials Project API](#materials-project-api) | 🔑 key | `MP_API_KEY` | [`materials_project.py`](scripts/science_research/materials_project.py) |
| 132 | 🔬 Science, Space & Research | [NASA Open APIs (APOD, NeoWs, DONKI, EONET...)](#nasa-open-apis-apod-neows-donki-eonet) | 🟢 no key | `DATA_GOV_API_KEY` | [`nasa.py`](scripts/science_research/nasa.py) |
| 133 | 🔬 Science, Space & Research | [NASA Exoplanet Archive (TAP)](#nasa-exoplanet-archive-tap) | 🟢 no key | — | [`nasa_exoplanet_archive.py`](scripts/science_research/nasa_exoplanet_archive.py) |
| 134 | 🔬 Science, Space & Research | [NASA Image and Video Library](#nasa-image-and-video-library) | 🟢 no key | — | [`nasa_image_library.py`](scripts/science_research/nasa_image_library.py) |
| 135 | 🔬 Science, Space & Research | [NIH RePORTER API](#nih-reporter-api) | 🟢 no key | — | [`nih_reporter.py`](scripts/science_research/nih_reporter.py) |
| 136 | 🔬 Science, Space & Research | [NOAA Space Weather Prediction Center (SWPC) data](#noaa-space-weather-prediction-center-swpc-data) | 🟢 no key | — | [`noaa_swpc.py`](scripts/science_research/noaa_swpc.py) |
| 137 | 🔬 Science, Space & Research | [NSF Award Search API](#nsf-award-search-api) | 🟢 no key | — | [`nsf_awards.py`](scripts/science_research/nsf_awards.py) |
| 138 | 🔬 Science, Space & Research | [OpenAlex (scholarly works)](#openalex-scholarly-works) | 🟢 no key | — | [`openalex.py`](scripts/science_research/openalex.py) |
| 139 | 🔬 Science, Space & Research | [PubChem PUG-REST](#pubchem-pug-rest) | 🟢 no key | — | [`pubchem.py`](scripts/science_research/pubchem.py) |
| 140 | 🔬 Science, Space & Research | [RCSB Protein Data Bank (search + data APIs)](#rcsb-protein-data-bank-search--data-apis) | 🟢 no key | — | [`rcsb_pdb.py`](scripts/science_research/rcsb_pdb.py) |
| 141 | 🔬 Science, Space & Research | [ROR (Research Organization Registry)](#ror-research-organization-registry) | 🟢 no key | — | [`ror.py`](scripts/science_research/ror.py) |
| 142 | 🔬 Science, Space & Research | [SDSS SkyServer (Sloan Digital Sky Survey)](#sdss-skyserver-sloan-digital-sky-survey) | 🟢 no key | — | [`sdss_skyserver.py`](scripts/science_research/sdss_skyserver.py) |
| 143 | 🔬 Science, Space & Research | [Semantic Scholar Academic Graph](#semantic-scholar-academic-graph) | 🟢 no key | `S2_API_KEY (optional)` | [`semantic_scholar.py`](scripts/science_research/semantic_scholar.py) |
| 144 | 🔬 Science, Space & Research | [Spaceflight News API (SNAPI)](#spaceflight-news-api-snapi) | 🟢 no key | — | [`spaceflight_news.py`](scripts/science_research/spaceflight_news.py) |
| 145 | 🔬 Science, Space & Research | [Zenodo (research data & software repository)](#zenodo-research-data--software-repository) | 🟢 no key | — | [`zenodo.py`](scripts/science_research/zenodo.py) |
| 146 | 🤖 AI & Machine Learning | [Anthropic Claude API](#anthropic-claude-api) | 🔑 key | `ANTHROPIC_API_KEY` | [`anthropic_claude.py`](scripts/ai_ml/anthropic_claude.py) |
| 147 | 🤖 AI & Machine Learning | [Epoch AI — Data on AI Models](#epoch-ai--data-on-ai-models) | 🟢 no key | — | [`epoch_ai.py`](scripts/ai_ml/epoch_ai.py) |
| 148 | 🤖 AI & Machine Learning | [Google Gemini API](#google-gemini-api) | 🔑 key | `GEMINI_API_KEY` | [`google_gemini.py`](scripts/ai_ml/google_gemini.py) |
| 149 | 🤖 AI & Machine Learning | [Hugging Face Daily Papers](#hugging-face-daily-papers) | 🟢 no key | — | [`hf_daily_papers.py`](scripts/ai_ml/hf_daily_papers.py) |
| 150 | 🤖 AI & Machine Learning | [Hugging Face Inference Providers](#hugging-face-inference-providers) | 🟢 no key | `HF_TOKEN (needed for chat completions)` | [`hf_inference_providers.py`](scripts/ai_ml/hf_inference_providers.py) |
| 151 | 🤖 AI & Machine Learning | [Kaggle API](#kaggle-api) | 🔑 key | `KAGGLE_USERNAME, KAGGLE_KEY` | [`kaggle.py`](scripts/ai_ml/kaggle.py) |
| 152 | 🤖 AI & Machine Learning | [Ollama (local LLMs)](#ollama-local-llms) | 🟢 no key | `OLLAMA_MODEL (optional, default llama3.2)` | [`ollama.py`](scripts/ai_ml/ollama.py) |
| 153 | 🤖 AI & Machine Learning | [OpenAI API](#openai-api) | 🔑 key | `OPENAI_API_KEY` | [`openai.py`](scripts/ai_ml/openai.py) |
| 154 | 🤖 AI & Machine Learning | [OpenML](#openml) | 🟢 no key | `OPENML_API_KEY (optional)` | [`openml.py`](scripts/ai_ml/openml.py) |
| 155 | 🤖 AI & Machine Learning | [OpenRouter](#openrouter) | 🟢 no key | `OPENROUTER_API_KEY (needed for chat completions)` | [`openrouter.py`](scripts/ai_ml/openrouter.py) |
| 156 | 🤖 AI & Machine Learning | [UCI Machine Learning Repository](#uci-machine-learning-repository) | 🟢 no key | — | [`uci_ml_repo.py`](scripts/ai_ml/uci_ml_repo.py) |
| 157 | ⚙️ Engineering, Energy & Tech | [US Energy Information Administration (EIA) v2](#us-energy-information-administration-eia-v2) | 🔑 key | `EIA_API_KEY` | [`eia.py`](scripts/engineering_energy/eia.py) |
| 158 | ⚙️ Engineering, Energy & Tech | [Electricity Maps API](#electricity-maps-api) | 🔑 key | `ELECTRICITYMAPS_TOKEN` | [`electricity_maps.py`](scripts/engineering_energy/electricity_maps.py) |
| 159 | ⚙️ Engineering, Energy & Tech | [GitHub REST API](#github-rest-api) | 🟢 no key | `GITHUB_TOKEN (optional)` | [`github.py`](scripts/engineering_energy/github.py) |
| 160 | ⚙️ Engineering, Energy & Tech | [Hugging Face Hub API](#hugging-face-hub-api) | 🟢 no key | `HF_TOKEN (optional)` | [`huggingface.py`](scripts/engineering_energy/huggingface.py) |
| 161 | ⚙️ Engineering, Energy & Tech | [NASA POWER (solar & meteorological data)](#nasa-power-solar--meteorological-data) | 🟢 no key | — | [`nasa_power.py`](scripts/engineering_energy/nasa_power.py) |
| 162 | ⚙️ Engineering, Energy & Tech | [NLR (formerly NREL) Developer Network — solar, EV charging](#nlr-formerly-nrel-developer-network--solar-ev-charging) | 🟢 no key | `DATA_GOV_API_KEY` | [`nrel.py`](scripts/engineering_energy/nrel.py) |
| 163 | ⚙️ Engineering, Energy & Tech | [NIST National Vulnerability Database (CVE API 2.0)](#nist-national-vulnerability-database-cve-api-20) | 🟢 no key | `NVD_API_KEY (optional)` | [`nvd_cve.py`](scripts/engineering_energy/nvd_cve.py) |
| 164 | ⚙️ Engineering, Energy & Tech | [Our World in Data — Energy dataset](#our-world-in-data--energy-dataset) | 🟢 no key | — | [`owid_energy.py`](scripts/engineering_energy/owid_energy.py) |
| 165 | ⚙️ Engineering, Energy & Tech | [PVGIS (EU JRC photovoltaic estimates)](#pvgis-eu-jrc-photovoltaic-estimates) | 🟢 no key | — | [`pvgis.py`](scripts/engineering_energy/pvgis.py) |
| 166 | ⚙️ Engineering, Energy & Tech | [UK Carbon Intensity API (National Grid ESO)](#uk-carbon-intensity-api-national-grid-eso) | 🟢 no key | — | [`uk_carbon_intensity.py`](scripts/engineering_energy/uk_carbon_intensity.py) |
| 167 | ⚙️ Engineering, Energy & Tech | [USGS Water Services (streamflow)](#usgs-water-services-streamflow) | 🟢 no key | — | [`usgs_water.py`](scripts/engineering_energy/usgs_water.py) |
| 168 | 🗺️ Transportation & Geospatial | [Bureau of Transportation Statistics (BTS) Open Data](#bureau-of-transportation-statistics-bts-open-data) | 🟢 no key | `SOCRATA_APP_TOKEN (optional)` | [`bts_open_data.py`](scripts/transportation_geo/bts_open_data.py) |
| 169 | 🗺️ Transportation & Geospatial | [US Census Geocoder](#us-census-geocoder) | 🟢 no key | — | [`census_geocoder.py`](scripts/transportation_geo/census_geocoder.py) |
| 170 | 🗺️ Transportation & Geospatial | [Chicago Taxi & Rideshare Trips](#chicago-taxi--rideshare-trips) | 🟢 no key | `SOCRATA_APP_TOKEN (optional)` | [`chicago_taxi.py`](scripts/transportation_geo/chicago_taxi.py) |
| 171 | 🗺️ Transportation & Geospatial | [GBFS Bike-share Feeds (Capital Bikeshare and 1,000+ systems)](#gbfs-bike-share-feeds-capital-bikeshare-and-1000-systems) | 🟢 no key | — | [`gbfs_bikeshare.py`](scripts/transportation_geo/gbfs_bikeshare.py) |
| 172 | 🗺️ Transportation & Geospatial | [MBTA V3 API (Boston transit)](#mbta-v3-api-boston-transit) | 🟢 no key | `MBTA_API_KEY (optional)` | [`mbta.py`](scripts/transportation_geo/mbta.py) |
| 173 | 🗺️ Transportation & Geospatial | [NHTSA Recalls, Complaints & Safety Ratings](#nhtsa-recalls-complaints--safety-ratings) | 🟢 no key | — | [`nhtsa_safety.py`](scripts/transportation_geo/nhtsa_safety.py) |
| 174 | 🗺️ Transportation & Geospatial | [NHTSA vPIC (vehicle VIN decoder)](#nhtsa-vpic-vehicle-vin-decoder) | 🟢 no key | — | [`nhtsa_vpic.py`](scripts/transportation_geo/nhtsa_vpic.py) |
| 175 | 🗺️ Transportation & Geospatial | [OpenStreetMap Nominatim (geocoding)](#openstreetmap-nominatim-geocoding) | 🟢 no key | — | [`nominatim.py`](scripts/transportation_geo/nominatim.py) |
| 176 | 🗺️ Transportation & Geospatial | [OpenSky Network (live flights)](#opensky-network-live-flights) | 🟢 no key | — | [`opensky.py`](scripts/transportation_geo/opensky.py) |
| 177 | 🗺️ Transportation & Geospatial | [OSRM (routing on OpenStreetMap)](#osrm-routing-on-openstreetmap) | 🟢 no key | — | [`osrm.py`](scripts/transportation_geo/osrm.py) |
| 178 | 🗺️ Transportation & Geospatial | [OpenStreetMap Overpass API](#openstreetmap-overpass-api) | 🟢 no key | — | [`overpass.py`](scripts/transportation_geo/overpass.py) |
| 179 | 🗺️ Transportation & Geospatial | [Transitland (GTFS transit data worldwide)](#transitland-gtfs-transit-data-worldwide) | 🔑 key | `TRANSITLAND_API_KEY` | [`transitland.py`](scripts/transportation_geo/transitland.py) |
| 180 | 🗺️ Transportation & Geospatial | [WMATA (DC Metro) API](#wmata-dc-metro-api) | 🔑 key | `WMATA_API_KEY` | [`wmata.py`](scripts/transportation_geo/wmata.py) |
| 181 | 🏅 Sports | [BALLDONTLIE (NBA, NFL, MLB, ...)](#balldontlie-nba-nfl-mlb-) | 🔑 key | `BALLDONTLIE_API_KEY` | [`balldontlie.py`](scripts/sports/balldontlie.py) |
| 182 | 🏅 Sports | [ESPN Site API (unofficial)](#espn-site-api-unofficial) | 🟢 no key | — | [`espn.py`](scripts/sports/espn.py) |
| 183 | 🏅 Sports | [football-data.org (soccer)](#football-dataorg-soccer) | 🔑 key | `FOOTBALL_DATA_TOKEN` | [`football_data.py`](scripts/sports/football_data.py) |
| 184 | 🏅 Sports | [Jolpica-F1 (Ergast-compatible F1 history)](#jolpica-f1-ergast-compatible-f1-history) | 🟢 no key | — | [`jolpica_f1.py`](scripts/sports/jolpica_f1.py) |
| 185 | 🏅 Sports | [Lichess API (chess)](#lichess-api-chess) | 🟢 no key | — | [`lichess.py`](scripts/sports/lichess.py) |
| 186 | 🏅 Sports | [MLB Stats API](#mlb-stats-api) | 🟢 no key | — | [`mlb_stats.py`](scripts/sports/mlb_stats.py) |
| 187 | 🏅 Sports | [NHL Web API (unofficial)](#nhl-web-api-unofficial) | 🟢 no key | — | [`nhl.py`](scripts/sports/nhl.py) |
| 188 | 🏅 Sports | [OpenF1 (Formula 1 telemetry)](#openf1-formula-1-telemetry) | 🟢 no key | — | [`openf1.py`](scripts/sports/openf1.py) |
| 189 | 🏅 Sports | [TheSportsDB](#thesportsdb) | 🔑 key | `THESPORTSDB_KEY (optional)` | [`thesportsdb.py`](scripts/sports/thesportsdb.py) |
| 190 | 🎬 Culture, Media & Entertainment | [Art Institute of Chicago API](#art-institute-of-chicago-api) | 🟢 no key | — | [`art_institute_chicago.py`](scripts/culture_misc/art_institute_chicago.py) |
| 191 | 🎬 Culture, Media & Entertainment | [Deezer API (music charts & catalog)](#deezer-api-music-charts--catalog) | 🟢 no key | — | [`deezer.py`](scripts/culture_misc/deezer.py) |
| 192 | 🎬 Culture, Media & Entertainment | [Discogs API (music releases & marketplace)](#discogs-api-music-releases--marketplace) | 🟢 no key | `DISCOGS_TOKEN (optional)` | [`discogs.py`](scripts/culture_misc/discogs.py) |
| 193 | 🎬 Culture, Media & Entertainment | [Google Books API](#google-books-api) | 🔑 key | `GOOGLE_BOOKS_API_KEY` | [`google_books.py`](scripts/culture_misc/google_books.py) |
| 194 | 🎬 Culture, Media & Entertainment | [Gutendex (Project Gutenberg books)](#gutendex-project-gutenberg-books) | 🟢 no key | — | [`gutendex.py`](scripts/culture_misc/gutendex.py) |
| 195 | 🎬 Culture, Media & Entertainment | [Internet Archive (search & metadata)](#internet-archive-search--metadata) | 🟢 no key | — | [`internet_archive.py`](scripts/culture_misc/internet_archive.py) |
| 196 | 🎬 Culture, Media & Entertainment | [iTunes Search API (Apple music, podcasts, apps, books)](#itunes-search-api-apple-music-podcasts-apps-books) | 🟢 no key | — | [`itunes_search.py`](scripts/culture_misc/itunes_search.py) |
| 197 | 🎬 Culture, Media & Entertainment | [Last.fm API (listening data & tags)](#lastfm-api-listening-data--tags) | 🔑 key | `LASTFM_API_KEY` | [`lastfm.py`](scripts/culture_misc/lastfm.py) |
| 198 | 🎬 Culture, Media & Entertainment | [The Met Museum Collection API](#the-met-museum-collection-api) | 🟢 no key | — | [`met_museum.py`](scripts/culture_misc/met_museum.py) |
| 199 | 🎬 Culture, Media & Entertainment | [MusicBrainz (open music encyclopedia)](#musicbrainz-open-music-encyclopedia) | 🟢 no key | — | [`musicbrainz.py`](scripts/culture_misc/musicbrainz.py) |
| 200 | 🎬 Culture, Media & Entertainment | [Open Library (books)](#open-library-books) | 🟢 no key | — | [`open_library.py`](scripts/culture_misc/open_library.py) |
| 201 | 🎬 Culture, Media & Entertainment | [RAWG Video Games Database API](#rawg-video-games-database-api) | 🔑 key | `RAWG_API_KEY` | [`rawg.py`](scripts/culture_misc/rawg.py) |
| 202 | 🎬 Culture, Media & Entertainment | [Spotify Web API](#spotify-web-api) | 🔑 key | `SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET` | [`spotify.py`](scripts/culture_misc/spotify.py) |
| 203 | 🎬 Culture, Media & Entertainment | [SteamSpy (Steam game statistics)](#steamspy-steam-game-statistics) | 🟢 no key | — | [`steamspy.py`](scripts/culture_misc/steamspy.py) |
| 204 | 🎬 Culture, Media & Entertainment | [TMDB (The Movie Database)](#tmdb-the-movie-database) | 🔑 key | `TMDB_TOKEN` | [`tmdb.py`](scripts/culture_misc/tmdb.py) |
| 205 | 🎬 Culture, Media & Entertainment | [TVmaze API (TV shows & schedules)](#tvmaze-api-tv-shows--schedules) | 🟢 no key | — | [`tvmaze.py`](scripts/culture_misc/tvmaze.py) |
| 206 | 🎬 Culture, Media & Entertainment | [Wikidata Query Service (SPARQL)](#wikidata-query-service-sparql) | 🟢 no key | — | [`wikidata_sparql.py`](scripts/culture_misc/wikidata_sparql.py) |

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

### EPA AirNow API (real-time air quality)

| | |
|:--|:--|
| **Docs** | https://docs.airnowapi.org/webservices |
| **Auth** | API key (query `API_KEY`) |
| **Register / token** | Request an account at https://docs.airnowapi.org/account/request/, then copy the key from the Web Services page. Real-time data are preliminary (use EPA AQS for verified history). |
| **Env var** | `AIRNOW_API_KEY` |
| **Script** | [`scripts/weather_environment/airnow.py`](scripts/weather_environment/airnow.py) |

**What you can get:**

- Current AQI by ZIP code or lat/lon for ozone, PM2.5 and PM10, with category (Good ... Hazardous)
- Daily AQI forecasts by reporting area
- Historical observations by date; monitoring-site data in a bounding box

**Capstone ideas:** Wildfire-smoke exposure, AQI alerts vs. outdoor activity or hospital visits, forecast accuracy.

**Endpoint:** `GET https://www.airnowapi.org/aq/{observation \| forecast}/{zipCode \| latLong}/{current \| historical}/`

| Parameter | Description | Example |
|:--|:--|:--|
| `zipCode / latitude & longitude` | Location | `20052` |
| `distance` | Search radius in miles if no monitor at the location | `25` |
| `date (forecast, historical)` | YYYY-MM-DD (historical: YYYY-MM-DDT00-0000) | `2026-10-12` |
| `format` | application/json \| text/csv \| application/xml | `application/json` |
| `API_KEY` | Your key | `<AIRNOW_API_KEY>` |

```python
from _common import get, key, show


def main():
    d = get("https://www.airnowapi.org/aq/observation/zipCode/current/", params={
        "format": "application/json", "zipCode": "20052", "distance": 25, "API_KEY": key("AIRNOW_API_KEY")})
    show([{k: o[k] for k in ("DateObserved", "HourObserved", "ReportingArea", "ParameterName", "AQI")} |
          {"Category": o["Category"]["Name"]} for o in d])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"DateObserved": "2026-10-12", "HourObserved": 9, "ReportingArea": "Metropolitan Washington", "ParameterName": "O3", "AQI": 31, "Category": "Good"},
  {"DateObserved": "2026-10-12", "HourObserved": 9, "ReportingArea": "Metropolitan Washington", "ParameterName": "PM2.5", "AQI": 46, "Category": "Good"}
]
```
</details>

---

### Climate TRACE (global emissions inventory)

| | |
|:--|:--|
| **Docs** | https://api.climatetrace.org/v7/docs/index.html |
| **Auth** | None |
| **Register / token** | No registration. Independent, satellite- and AI-based estimates of greenhouse-gas emissions for every country and hundreds of thousands of individual sources (power plants, oil fields, steel mills, ...). |
| **Script** | [`scripts/weather_environment/climate_trace.py`](scripts/weather_environment/climate_trace.py) |

**What you can get:**

- Country rankings and totals by gas (CO2, CH4, N2O, CO2e) and sector, with per-capita figures and % change
- Individual emitting assets: location, sector, subsector, asset type, yearly emissions
- Sector/subsector definitions

**Capstone ideas:** Verifying national climate pledges, top emitters by sector, mapping emissions sources against communities.

**Endpoint:** `GET https://api.climatetrace.org/v7/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `rankings/countries?gas=&start=&end=` | Country ranking for a period; add sectors= to restrict | `gas=co2e_100yr&start=2024&end=2024` |
| `sources?gadmId=&sectors=&subsectors=&year=` | Individual sources; filter country with gadmId (country/countries are ignored) | `gadmId=USA&sectors=power` |
| `limit / offset` | Paging | `5` |
| `gas` | co2 \| ch4 \| n2o \| co2e_100yr \| co2e_20yr | `co2e_100yr` |
| `definitions/sectors` | Sector and subsector names | `—` |

```python
from _common import get, show


def main():
    r = get("https://api.climatetrace.org/v7/rankings/countries",
            params={"gas": "co2e_100yr", "start": 2024, "end": 2024, "limit": 5})
    plants = get("https://api.climatetrace.org/v7/sources",
                 params={"gadmId": "USA", "sectors": "power", "year": 2024, "limit": 3})
    show({"global_total_Gt_CO2e_2024": round(r["totals"]["emissionsQuantity"] / 1e9, 1), "top_countries": [
        {"country": c["name"], "Gt_CO2e": round(c["emissionsQuantity"] / 1e9, 2), "share_pct": round(c["percentage"], 1),
         "t_per_capita": round(c["emissionsPerCapita"], 1)} for c in r["rankings"]],
          "largest_US_power_sources": [{"name": s["name"], "type": s["assetType"],
                                        "Mt_CO2e": round(s["emissionsQuantity"] / 1e6, 1)} for s in plants]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:40</summary>

```json
{
  "global_total_Gt_CO2e_2024": 59.9,
  "top_countries": [
    {
      "country": "China",
      "Gt_CO2e": 16.55,
      "share_pct": 27.6,
      "t_per_capita": 11.7
    },
    {
      "country": "United States of America",
      "Gt_CO2e": 5.99,
      "share_pct": 10.0,
      "t_per_capita": 18.1
    },
    {
      "country": "India",
      "Gt_CO2e": 4.61,
      "share_pct": 7.7,
      "t_per_capita": 3.3
    },
    {
      "country": "Russian Federation",
      "Gt_CO2e": 2.83,
      "share_pct": 4.7,
      "t_per_capita": 19.6
    },
    {
      "country": "Indonesia",
      "Gt_CO2e": 1.64,
      "share_pct": 2.7,
      "t_per_capita": 6.0
    },
    "... (247 more)"
  ],
  "largest_US_power_sources": [
    {
      "name": "James H Miller Jr",
      "type": "coal",
      "Mt_CO2e": 18.5
    },
    {
      "name": "Labadie",
      "type": "coal",
      "Mt_CO2e": 14.4
    },
    {
      "name": "Monroe (MI)",
      "type": "coal, oil",
      "Mt_CO2e": 12.8
    }
  ]
}
```
</details>

---

### U.S. Drought Monitor

| | |
|:--|:--|
| **Docs** | https://droughtmonitor.unl.edu/DmData/DataDownload/WebServiceInfo.aspx |
| **Auth** | None |
| **Register / token** | No registration. Weekly maps since 2000 from the National Drought Mitigation Center (UNL), USDA and NOAA. Send Accept: application/json (otherwise CSV/XML). |
| **Script** | [`scripts/weather_environment/drought_monitor.py`](scripts/weather_environment/drought_monitor.py) |

**What you can get:**

- Weekly % of area (or population) in each drought category: D0 abnormally dry ... D4 exceptional
- National, state, county, HUC watershed, climate-region levels
- Shapefiles/GeoJSON of the weekly maps

**Capstone ideas:** Drought impacts on agriculture, wildfire, migration or electricity (hydropower); county panel studies.

**Endpoint:** `GET https://usdmdataservices.unl.edu/api/{level}/{statistic}`

| Parameter | Description | Example |
|:--|:--|:--|
| `level` | USStatistics \| StateStatistics \| CountyStatistics \| HUCStatistics \| ... | `StateStatistics` |
| `statistic` | GetDroughtSeverityStatisticsByAreaPercent \| ...ByArea \| ...ByPopulationPercent \| GetDSCI | `GetDroughtSeverityStatisticsByAreaPercent` |
| `aoi` | Area: state FIPS (two digits, not the abbreviation), county FIPS (five digits), or US | `51 (Virginia)` |
| `startdate / enddate` | M/D/YYYY | `9/1/2026 / 10/1/2026` |
| `statisticsType` | 1 = cumulative (D0-D4 includes worse), 2 = categorical | `1` |
| `Accept (header)` | application/json for JSON | `application/json` |

```python
from _common import get, show


def main():
    d = get("https://usdmdataservices.unl.edu/api/StateStatistics/GetDroughtSeverityStatisticsByAreaPercent",
            headers={"Accept": "application/json"},
            params={"aoi": "51", "startdate": "8/1/2026", "enddate": "10/12/2026", "statisticsType": 1})
    show([{"week": w["mapDate"][:10], **{c: w[c] for c in ("none", "d0", "d1", "d2", "d3", "d4")}}
          for w in sorted(d, key=lambda w: w["mapDate"])[-5:]])
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:40</summary>

```json
[
  {
    "week": "2026-09-08",
    "none": 24.41,
    "d0": 75.59,
    "d1": 35.14,
    "d2": 14.41,
    "d3": 3.79,
    "d4": 0.0
  },
  {
    "week": "2026-09-15",
    "none": 27.65,
    "d0": 72.35,
    "d1": 34.5,
    "d2": 15.25,
    "d3": 3.79,
    "d4": 0.0
  },
  {
    "week": "2026-09-22",
    "none": 28.88,
    "d0": 71.12,
    "d1": 37.17,
    "d2": 15.25,
    "d3": 3.79,
    "d4": 0.0
  },
  {
    "week": "2026-09-29",
    "none": 38.91,
    "d0": 61.09,
    "d1": 32.15,
    "d2": 14.71,
    "d3": 3.79,
    "d4": 0.0
  },
  {
    "week": "2026-10-06",
    "none": 42.41,
    "d0": 57.59,
    "d1": 34.48,
    "d2": 14.71,
    "d3": 3.79,
    "d4": 0.0
  }
]
```
</details>

---

### EPA Air Quality System (AQS) API

| | |
|:--|:--|
| **Docs** | https://aqs.epa.gov/aqsweb/documents/data_api.html |
| **Auth** | Email + key (query `email`, `key`) |
| **Register / token** | Sign up by calling https://aqs.epa.gov/data/api/signup?email=you@gwu.edu - the key is emailed to you. Limits: 10 requests/minute and 1 year of data per request; pause 5 seconds between calls. |
| **Env var** | `AQS_EMAIL, AQS_KEY` |
| **Script** | [`scripts/weather_environment/epa_aqs.py`](scripts/weather_environment/epa_aqs.py) |

**What you can get:**

- Verified historical monitor data (the regulatory record) for criteria pollutants and many air toxics
- Sample, daily and annual summaries by site, county, state, CBSA or bounding box
- Monitor metadata and quality-assurance data

**Capstone ideas:** Long-run pollution trends, health-outcome studies needing verified data, Clean Air Act policy evaluation.

**Endpoint:** `GET https://aqs.epa.gov/data/api/{service}/{geography}`

| Parameter | Description | Example |
|:--|:--|:--|
| `service` | sampleData \| dailyData \| annualData \| monitors \| list | `dailyData` |
| `geography` | bySite \| byCounty \| byState \| byBox \| byCBSA | `byState` |
| `param` | Parameter code(s): 88101 PM2.5, 44201 ozone, 42602 NO2 (up to 5) | `88101` |
| `bdate / edate` | YYYYMMDD, within the same year | `20250101 / 20251231` |
| `state / county / site` | FIPS codes | `state=11` |
| `email / key` | Your credentials | `—` |
| `list/states, list/parametersByClass` | Lookup lists for codes | `list/parametersByClass?pc=CRITERIA` |

```python
from _common import get, key, show


def main():
    d = get("https://aqs.epa.gov/data/api/dailyData/byState", params={
        "email": key("AQS_EMAIL"), "key": key("AQS_KEY"), "param": "88101",
        "bdate": "20250704", "edate": "20250705", "state": "11"})
    show([{k: r.get(k) for k in ("date_local", "site_number", "local_site_name", "parameter", "arithmetic_mean", "aqi")}
          for r in d["Data"]])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"date_local": "2025-07-04", "site_number": "0043", "local_site_name": "McMillan Reservoir", "parameter": "PM2.5 - Local Conditions", "arithmetic_mean": 18.4, "aqi": 64},
  {"date_local": "2025-07-05", "site_number": "0043", "local_site_name": "McMillan Reservoir", "parameter": "PM2.5 - Local Conditions", "arithmetic_mean": 9.1, "aqi": 50}
]
```
</details>

---

### EPA ECHO (Enforcement & Compliance History)

| | |
|:--|:--|
| **Docs** | https://echo.epa.gov/tools/web-services |
| **Auth** | None |
| **Register / token** | No registration. Two-step pattern: get_facilities returns a QueryID, then get_qid pages through the facilities. |
| **Script** | [`scripts/weather_environment/epa_echo.py`](scripts/weather_environment/epa_echo.py) |

**What you can get:**

- 800k+ regulated facilities: Clean Air Act, Clean Water Act, hazardous waste (RCRA), drinking water
- Compliance status, quarters in non-compliance, inspections, formal enforcement actions, penalties
- Facility location, NAICS industry, permits

**Capstone ideas:** Which facilities violate repeatedly, enforcement intensity by neighborhood demographics, penalty deterrence.

**Endpoint:** `GET https://echodata.epa.gov/echo/echo_rest_services.get_facilities  ->  get_qid`

| Parameter | Description | Example |
|:--|:--|:--|
| `p_st / p_zip / p_ct` | State, ZIP or city filter | `p_st=DC` |
| `p_act` | Y -> active facilities only | `Y` |
| `p_med` | Program: A air (CAA), W water (CWA), R hazardous waste (RCRA), S drinking water | `W` |
| `output` | JSON \| XML | `JSON` |
| `get_qid?qid=&pagesize=&pageno=` | Second call: facility rows for the QueryID | `qid=687&pagesize=5` |
| `Results summary fields` | QueryRows, INSPRows, FEARows (formal actions), TotalPenalties | `—` |

```python
from _common import get, show

BASE = "https://echodata.epa.gov/echo/echo_rest_services"


def main():
    summary = get(f"{BASE}.get_facilities", params={"output": "JSON", "p_st": "DC", "p_act": "Y"})["Results"]
    page = get(f"{BASE}.get_qid", params={"output": "JSON", "qid": summary["QueryID"], "pagesize": 5})["Results"]
    show({"active_DC_facilities": summary["QueryRows"], "inspections": summary["INSPRows"],
          "formal_enforcement_actions": summary["FEARows"], "total_penalties": summary["TotalPenalties"],
          "examples": [{k: f.get(k) for k in ("FacName", "FacZip", "FacComplianceStatus", "FacQtrsWithNC")}
                       for f in page["Facilities"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:40</summary>

```json
{
  "active_DC_facilities": "1891",
  "inspections": "654",
  "formal_enforcement_actions": "84",
  "total_penalties": "$1,068,199",
  "examples": [
    {
      "FacName": "25TH PLACE",
      "FacZip": "20018",
      "FacComplianceStatus": "No Violation Identified",
      "FacQtrsWithNC": "0"
    },
    {
      "FacName": "(NPR HEADQUARTERS) NATIONAL PUBLIC RADIO HEADQUARTERS",
      "FacZip": "20002",
      "FacComplianceStatus": "No Violation Identified",
      "FacQtrsWithNC": "6"
    },
    {
      "FacName": "1 K STREET SW",
      "FacZip": "20024",
      "FacComplianceStatus": "No Violation Identified",
      "FacQtrsWithNC": "0"
    },
    {
      "FacName": "10 G ST NE",
      "FacZip": "20002",
      "FacComplianceStatus": "No Violation Identified",
      "FacQtrsWithNC": "0"
    },
    {
      "FacName": "100 POTOMAC AVE SW",
      "FacZip": "20024",
      "FacComplianceStatus": "No Violation Identified",
      "FacQtrsWithNC": "0"
    },
    "... (1886 more)"
  ]
}
```
</details>

---

### EPA Envirofacts (Toxics Release Inventory and more)

| | |
|:--|:--|
| **Docs** | https://www.epa.gov/enviro/envirofacts-data-service-api |
| **Auth** | None |
| **Register / token** | No registration. One URL grammar over dozens of EPA databases: TRI (toxic releases), GHGRP (facility greenhouse-gas emissions), SDWIS (drinking water), RCRA (hazardous waste), and more. |
| **Script** | [`scripts/weather_environment/epa_envirofacts.py`](scripts/weather_environment/epa_envirofacts.py) |

**What you can get:**

- TRI: facilities reporting toxic chemical releases (name, address, county FIPS, industry, coordinates)
- GHGRP: annual greenhouse-gas emissions of large facilities
- SDWIS: drinking-water systems and violations

**Capstone ideas:** Environmental justice (toxic releases vs. neighborhood demographics), polluter mapping, compliance studies.

**Endpoint:** `GET https://data.epa.gov/efservice/{table}/{column}/{value}/rows/{first}:{last}/JSON`

| Parameter | Description | Example |
|:--|:--|:--|
| `table` | Table name, e.g. tri_facility, tri_reporting_form, pub_dim_facility (GHGRP), sdw_contaminant ... | `tri_facility` |
| `{column}/{value}` | Filter pairs (repeatable; also /{column}/<, >, BEGINNING, CONTAINING) | `state_abbr/DC` |
| `rows/{first}:{last}` | Row range (0-based) | `rows/0:9` |
| `format` | JSON \| CSV \| XML \| EXCEL (last path segment) | `JSON` |
| `count` | Append /COUNT to get the number of matching rows | `/tri_facility/state_abbr/VA/COUNT` |

```python
from _common import get, show


def main():
    count = get("https://data.epa.gov/efservice/tri_facility/state_abbr/VA/COUNT/JSON")[0]["TOTALQUERYRESULTS"]
    rows = get("https://data.epa.gov/efservice/tri_facility/state_abbr/VA/fac_closed_ind/0/rows/0:4/JSON")
    show({"VA_TRI_facilities": count, "examples": [
        {k: r.get(k) for k in ("facility_name", "city_name", "county_name", "zip_code")} for r in rows]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:40</summary>

```json
{
  "VA_TRI_facilities": 1247,
  "examples": [
    {
      "facility_name": "HANSON PIPE & PRODUCTS INC MANASSAS PLANT",
      "city_name": "MANASSAS",
      "county_name": "PRINCE WILLIAM",
      "zip_code": "20109"
    },
    {
      "facility_name": "QUARLES PETROLEUM INC. MANASSAS BULK PLANT",
      "city_name": "MANASSAS",
      "county_name": "PRINCE WILLIAM",
      "zip_code": "20109"
    },
    {
      "facility_name": "MANASSAS BROAD RUN READY MIX CONCRETE",
      "city_name": "MANASSAS",
      "county_name": "PRINCE WILLIAM",
      "zip_code": "20109"
    },
    {
      "facility_name": "BENCHMARK ELECTRONICS MANASSAS DIV",
      "city_name": "MANASSAS",
      "county_name": "MANASSAS (CITY)",
      "zip_code": "201103410"
    },
    {
      "facility_name": "BAE SYSTEMS INFORMATION  & ELECTRONIC SYSTEMS INTEGRATION I",
      "city_name": "MANASSAS",
      "county_name": "PRINCE WILLIAM",
      "zip_code": "20110"
    }
  ]
}
```
</details>

---

### GBIF (Global Biodiversity Information Facility)

| | |
|:--|:--|
| **Docs** | https://techdocs.gbif.org/en/openapi/ |
| **Auth** | None for search (account for bulk downloads) |
| **Register / token** | No registration to search (max 100,000 records via paging). For large extracts, create an account and use the asynchronous download API, which mints a citable DOI. |
| **Script** | [`scripts/weather_environment/gbif.py`](scripts/weather_environment/gbif.py) |

**What you can get:**

- 2.5B+ species occurrence records: species, coordinates, date, country, basis of record, dataset
- Species taxonomy and name matching
- Faceted counts (by year, country, species) without downloading records

**Capstone ideas:** Species distribution modelling, climate-driven range shifts, biodiversity trends by region.

**Endpoint:** `GET https://api.gbif.org/v1/occurrence/search`

| Parameter | Description | Example |
|:--|:--|:--|
| `taxonKey / scientificName` | Species or higher taxon (find keys via /v1/species/match?name=) | `scientificName=Haliaeetus leucocephalus` |
| `country / stateProvince / geometry` | ISO country, province name, or WKT polygon | `US` |
| `year / eventDate` | Year or range | `2015,2025` |
| `hasCoordinate / basisOfRecord` | true; HUMAN_OBSERVATION \| PRESERVED_SPECIMEN \| ... | `true` |
| `facet + facetLimit` | Counts by a field (set limit=0 for counts only) | `facet=year` |
| `limit / offset` | Max 300 per page | `0` |
| `/v1/species/match?name=` | Sibling endpoint: resolve a name to a taxonKey | `name=bald eagle` |

```python
from _common import get, show


def main():
    d = get("https://api.gbif.org/v1/occurrence/search", params={
        "scientificName": "Haliaeetus leucocephalus", "country": "US", "year": "2016,2025",
        "hasCoordinate": "true", "limit": 0, "facet": "year", "facetLimit": 10})
    # The latest year looks small until large datasets (e.g. eBird) publish their annual updates
    years = sorted(d["facets"][0]["counts"], key=lambda c: c["name"])
    show({"species": "Bald eagle (Haliaeetus leucocephalus), US", "occurrences_2016_2025": d["count"],
          "by_year": {c["name"]: c["count"] for c in years}}, max_items=10)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:40</summary>

```json
{
  "species": "Bald eagle (Haliaeetus leucocephalus), US",
  "occurrences_2016_2025": 5155400,
  "by_year": {
    "2016": 284183,
    "2017": 343333,
    "2018": 411571,
    "2019": 476494,
    "2020": 582109,
    "2021": 695303,
    "2022": 724066,
    "2023": 780582,
    "2024": 833749,
    "2025": 24010
  }
}
```
</details>

---

### iNaturalist API

| | |
|:--|:--|
| **Docs** | https://api.inaturalist.org/v1/docs/ |
| **Auth** | None for reading (OAuth to post) |
| **Register / token** | No registration to read. Keep to ~1 request/second and ~10,000 requests/day. Research-grade observations also flow into GBIF. |
| **Script** | [`scripts/weather_environment/inaturalist.py`](scripts/weather_environment/inaturalist.py) |

**What you can get:**

- Citizen-science observations with photos: species, location, date, observer, quality grade
- Species counts for any place, taxon search, identification agreements
- Place and project lookups

**Capstone ideas:** Urban biodiversity, phenology (first-bloom dates over time), invasive-species spread, image datasets.

**Endpoint:** `GET https://api.inaturalist.org/v1/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `observations / observations/species_counts` | Observations, or species ranked by observation count | `observations/species_counts` |
| `place_id / lat & lng & radius` | Place (find IDs via /v1/places/autocomplete?q=) or circle in km | `lat=38.9&lng=-77.04&radius=10` |
| `taxon_id / taxon_name / iconic_taxa` | Taxon filters (iconic_taxa: Aves, Plantae, Insecta, ...) | `iconic_taxa=Aves` |
| `quality_grade` | research \| needs_id \| casual | `research` |
| `d1 / d2` | Observed-on date range YYYY-MM-DD | `2026-01-01` |
| `per_page / page / order_by` | Max 200 per page; order_by observed_on \| created_at \| votes | `5` |

```python
from _common import get, show


def main():
    d = get("https://api.inaturalist.org/v1/observations/species_counts", params={
        "lat": 38.9, "lng": -77.04, "radius": 10, "quality_grade": "research", "iconic_taxa": "Aves",
        "d1": "2026-01-01", "per_page": 5})
    show({"bird_species_observed_near_DC_2026": d["total_results"], "most_observed": [
        {"species": r["taxon"]["name"], "common_name": r["taxon"].get("preferred_common_name"), "observations": r["count"]}
        for r in d["results"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:40</summary>

```json
{
  "bird_species_observed_near_DC_2026": 195,
  "most_observed": [
    {
      "species": "Anas platyrhynchos",
      "common_name": "Mallard",
      "observations": 685
    },
    {
      "species": "Turdus migratorius",
      "common_name": "American Robin",
      "observations": 659
    },
    {
      "species": "Passer domesticus",
      "common_name": "House Sparrow",
      "observations": 614
    },
    {
      "species": "Sturnus vulgaris",
      "common_name": "European Starling",
      "observations": 562
    },
    {
      "species": "Mimus polyglottos",
      "common_name": "Northern Mockingbird",
      "observations": 423
    }
  ]
}
```
</details>

---

### NASA FIRMS (active fire detections)

| | |
|:--|:--|
| **Docs** | https://firms.modaps.eosdis.nasa.gov/api/ |
| **Auth** | MAP_KEY in the URL path |
| **Register / token** | Get a free MAP_KEY at https://firms.modaps.eosdis.nasa.gov/api/map_key/ (email). Limit: 5,000 transactions per 10 minutes. |
| **Env var** | `FIRMS_MAP_KEY` |
| **Script** | [`scripts/weather_environment/nasa_firms.py`](scripts/weather_environment/nasa_firms.py) |

**What you can get:**

- Satellite fire/thermal-anomaly detections (VIIRS 375 m, MODIS 1 km) within ~3 hours of overpass
- Latitude/longitude, brightness, fire radiative power (FRP), confidence, day/night, acquisition time
- Archive back to 2000 (MODIS) / 2012 (VIIRS)

**Capstone ideas:** Wildfire spread analysis, smoke-exposure modelling (with AirNow/OpenAQ), agricultural burning detection.

**Endpoint:** `GET https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{source}/{area}/{day_range}[/{date}]`

| Parameter | Description | Example |
|:--|:--|:--|
| `source` | VIIRS_SNPP_NRT \| VIIRS_NOAA20_NRT \| VIIRS_NOAA21_NRT \| MODIS_NRT (and _SP for standard/archive) | `VIIRS_SNPP_NRT` |
| `area` | world, or a bounding box west,south,east,north | `-125,24,-66,50` |
| `day_range` | 1-10 days | `1` |
| `date` | Optional start date YYYY-MM-DD (default: most recent) | `2026-08-15` |
| `/api/country/csv/{MAP_KEY}/{source}/{ISO3}/{days}` | Sibling endpoint: detections in one country | `/USA/1` |

```python
import csv
import io

from _common import get, key, show


def main():
    text = get(f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key("FIRMS_MAP_KEY")}/VIIRS_SNPP_NRT/-125,24,-66,50/1',
               as_json=False)
    rows = list(csv.DictReader(io.StringIO(text)))
    top = sorted(rows, key=lambda r: -float(r["frp"]))[:5]
    show({"detections_last_24h_contiguous_US": len(rows), "highest_FRP": [
        {k: r[k] for k in ("latitude", "longitude", "acq_date", "acq_time", "frp", "confidence", "daynight")} for r in top]})
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "detections_last_24h_contiguous_US": 1843,
  "highest_FRP": [
    {"latitude": 41.21, "longitude": -122.53, "acq_date": "2026-10-11", "acq_time": "2042", "frp": 412.7, "confidence": "h", "daynight": "D"}
  ]
}
```
</details>

---

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

### NOAA Climate at a Glance

| | |
|:--|:--|
| **Docs** | https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/ |
| **Auth** | None |
| **Register / token** | No registration. Build a query on the website, then swap the page URL's ending for /data.json or /data.csv. |
| **Script** | [`scripts/weather_environment/noaa_climate_at_a_glance.py`](scripts/weather_environment/noaa_climate_at_a_glance.py) |

**What you can get:**

- Temperature, precipitation, drought indices (PDSI) and degree days since 1895
- National (contiguous US), regional, state, climate-division, county and city series; global anomalies
- Any time scale: 1-month to 60-month periods

**Capstone ideas:** Local warming trends for any US county, linking climate anomalies to crop yields, energy demand or health.

**Endpoint:** `GET https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/{scope}/time-series/{location}/{parameter}/{timescale}/{month}/{start}-{end}/data.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `scope / location` | national/110 (contiguous US), statewide/{state code}, county/{state-county}, global/globe/land_ocean | `national/110` |
| `parameter` | tavg \| tmax \| tmin \| pcp (precipitation) \| pdsi \| cdd \| hdd | `tavg` |
| `timescale` | Months per period: 1, 3, 12, ... | `12` |
| `month` | Ending month (0 = all months) | `0` |
| `start-end` | Year range | `1895-2025` |
| `data.json / data.csv` | Output format | `data.json` |

```python
from _common import get, show


def main():
    d = get("https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/tavg/12/12/1895-2025/data.json")
    years = {k[:4]: v["value"] for k, v in d["data"].items()}  # 12-month periods ending in December = calendar years
    warmest = sorted(years.items(), key=lambda kv: -kv[1])[:5]
    show({"series": d["description"]["title"] + f' ({d["description"]["units"]})',
          "warmest_years": [{"year": y, "avg_temp_F": t} for y, t in warmest],
          "first_decade_avg": round(sum(list(years.values())[:10]) / 10, 2),
          "last_decade_avg": round(sum(list(years.values())[-10:]) / 10, 2)})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:40</summary>

```json
{
  "series": "Contiguous U.S. January-December Average Temperature (Degrees Fahrenheit)",
  "warmest_years": [
    {
      "year": "2024",
      "avg_temp_F": 55.48
    },
    {
      "year": "2012",
      "avg_temp_F": 55.27
    },
    {
      "year": "2016",
      "avg_temp_F": 54.9
    },
    {
      "year": "2025",
      "avg_temp_F": 54.62
    },
    {
      "year": "2017",
      "avg_temp_F": 54.54
    }
  ],
  "first_decade_avg": 51.42,
  "last_decade_avg": 54.23
}
```
</details>

---

### NOAA Global Monitoring Lab — Mauna Loa CO2

| | |
|:--|:--|
| **Docs** | https://gml.noaa.gov/ccgg/trends/data.html |
| **Auth** | None (CSV/text files) |
| **Register / token** | No registration; no REST API - stable file URLs updated monthly. Also global-average CO2, methane (CH4), N2O and SF6 files on the same page. |
| **Script** | [`scripts/weather_environment/noaa_co2.py`](scripts/weather_environment/noaa_co2.py) |

**What you can get:**

- Monthly mean atmospheric CO2 at Mauna Loa since March 1958 (the Keeling Curve), plus deseasonalized trend
- Daily, weekly and annual files; global marine-surface averages; annual growth rates

**Capstone ideas:** Time-series decomposition and forecasting, teaching seasonality and trend.

**Endpoint:** `GET https://gml.noaa.gov/webdata/ccgg/trends/{gas}/{file}.csv`

| Parameter | Description | Example |
|:--|:--|:--|
| `co2/co2_mm_mlo.csv` | Monthly Mauna Loa: year, month, decimal date, average, deseasonalized, ... | `co2_mm_mlo.csv` |
| `co2/co2_annmean_mlo.csv` | Annual means | `co2_annmean_mlo.csv` |
| `co2/co2_mm_gl.csv` | Global monthly mean | `co2_mm_gl.csv` |
| `ch4/ch4_mm_gl.csv / n2o/n2o_mm_gl.csv` | Global methane / nitrous oxide | `ch4_mm_gl.csv` |
| `comment lines` | Header lines start with # - skip them | `—` |

```python
import csv
import io

from _common import get, show


def main():
    text = get("https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv", as_json=False)
    rows = list(csv.DictReader(io.StringIO("\n".join(l for l in text.splitlines() if not l.startswith("#")))))
    latest, ten_years = rows[-1], rows[-121]
    show({"latest": [{"month": f'{r["year"]}-{int(r["month"]):02d}', "co2_ppm": float(r["average"]),
                      "deseasonalized_ppm": float(r["deseasonalized"])} for r in rows[-3:]],
          "first_record": f'{rows[0]["year"]}-{int(rows[0]["month"]):02d}: {rows[0]["average"]} ppm',
          "change_over_10_years_ppm": round(float(latest["deseasonalized"]) - float(ten_years["deseasonalized"]), 1)})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:40</summary>

```json
{
  "latest": [
    {
      "month": "2026-07",
      "co2_ppm": 429.13,
      "deseasonalized_ppm": 428.74
    },
    {
      "month": "2026-08",
      "co2_ppm": 427.55,
      "deseasonalized_ppm": 429.48
    },
    {
      "month": "2026-09",
      "co2_ppm": 426.07,
      "deseasonalized_ppm": 429.55
    }
  ],
  "first_record": "1958-03: 315.71 ppm",
  "change_over_10_years_ppm": 24.7
}
```
</details>

---

### NOAA Tides & Currents (CO-OPS)

| | |
|:--|:--|
| **Docs** | https://api.tidesandcurrents.noaa.gov/api/prod/ |
| **Auth** | None |
| **Register / token** | No registration. Station IDs: https://tidesandcurrents.noaa.gov/map/ (8594900 = Washington, DC). |
| **Script** | [`scripts/weather_environment/noaa_tides.py`](scripts/weather_environment/noaa_tides.py) |

**What you can get:**

- Observed water levels (6-minute), tide predictions, high/low tides at ~200 coastal stations
- Meteorological data at stations (air/water temperature, wind, pressure)
- Long-term sea level data (monthly means)

**Capstone ideas:** Sea-level-rise trends, nuisance (high-tide) flooding frequency, storm surge vs. property values.

**Endpoint:** `GET https://api.tidesandcurrents.noaa.gov/api/prod/datagetter`

| Parameter | Description | Example |
|:--|:--|:--|
| `station` | Station ID | `8594900` |
| `product` | water_level \| predictions \| hourly_height \| high_low \| monthly_mean \| water_temperature \| wind \| ... | `monthly_mean` |
| `date / begin_date & end_date` | latest \| today \| recent, or a YYYYMMDD range (limits vary by product) | `begin_date=20160101&end_date=20251231` |
| `datum` | Vertical reference: MLLW \| MSL \| NAVD \| STND ... | `MSL` |
| `units / time_zone` | metric \| english; gmt \| lst \| lst_ldt | `metric / gmt` |
| `interval (predictions)` | h hourly, hilo highs/lows, or minutes | `hilo` |
| `format` | json \| csv \| xml | `json` |

```python
from _common import get, show

URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


def main():
    common = {"station": "8594900", "units": "metric", "time_zone": "gmt", "format": "json"}
    latest = get(URL, params={**common, "product": "water_level", "datum": "MLLW", "date": "latest"})
    monthly = get(URL, params={**common, "product": "monthly_mean", "datum": "MSL",
                               "begin_date": "20150101", "end_date": "20251231"})["data"]
    by_year = {}
    for m in monthly:
        if m["MSL"]:
            by_year.setdefault(m["year"], []).append(float(m["MSL"]))
    show({"station": latest["metadata"]["name"], "latest_water_level_m_MLLW": latest["data"][0],
          "annual_mean_sea_level_m": {y: round(sum(v) / len(v), 3) for y, v in by_year.items() if len(v) == 12}},
         max_items=12)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:40</summary>

```json
{
  "station": "Washington",
  "latest_water_level_m_MLLW": {
    "t": "2026-10-12 11:30",
    "v": "0.956",
    "s": "0.003",
    "f": "0,0,0,0",
    "q": "p"
  },
  "annual_mean_sea_level_m": {
    "2015": 0.052,
    "2017": 0.097,
    "2018": 0.191,
    "2019": 0.188,
    "2020": 0.157,
    "2021": 0.13,
    "2022": 0.118,
    "2023": 0.176,
    "2024": 0.22,
    "2025": 0.126
  }
}
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

<details><summary><b>Sample output</b> — live run 2026-10-12 07:40</summary>

```json
{
  "office": "LWX",
  "forecast": [
    {
      "name": "Columbus Day",
      "temperature": 74,
      "temperatureUnit": "F",
      "shortForecast": "Rain Showers Likely then Mostly Cloudy"
    },
    {
      "name": "Tonight",
      "temperature": 57,
      "temperatureUnit": "F",
      "shortForecast": "Mostly Clear"
    },
    {
      "name": "Tuesday",
      "temperature": 81,
      "temperatureUnit": "F",
      "shortForecast": "Sunny"
    }
  ],
  "active_alerts_VA": []
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

<details><summary><b>Sample output</b> — live run 2026-10-12 07:40</summary>

```json
{
  "current": {
    "time": "2026-10-12T07:30",
    "interval": 900,
    "temperature_2m": 14.0,
    "relative_humidity_2m": 90,
    "wind_speed_10m": 10.5
  },
  "daily": {
    "time": [
      "2026-10-12",
      "2026-10-13",
      "2026-10-14"
    ],
    "temperature_2m_max": [
      22.1,
      24.7,
      31.2
    ],
    "temperature_2m_min": [
      14.2,
      12.8,
      19.0
    ],
    "precipitation_sum": [
      1.8,
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

<details><summary><b>Sample output</b> — live run 2026-10-12 07:40</summary>

```json
[
  {
    "mag": 5.5,
    "place": "122 km W of Neiafu, Tonga",
    "time_ms": 1791779892859,
    "depth_km": 195.212,
    "tsunami": 0
  },
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

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

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

### ChEMBL (drug discovery bioactivity)

| | |
|:--|:--|
| **Docs** | https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services |
| **Auth** | None |
| **Register / token** | No registration. Python client: `pip install chembl_webresource_client`. |
| **Script** | [`scripts/health/chembl.py`](scripts/health/chembl.py) |

**What you can get:**

- 2.4M+ bioactive molecules with structures and properties (MW, logP, Lipinski, QED)
- 20M+ bioactivity measurements (IC50, Ki, ...) against protein targets
- Approved drugs, mechanisms of action, indications

**Capstone ideas:** QSAR / molecular property prediction, drug repurposing, target-ligand interaction models.

**Endpoint:** `GET https://www.ebi.ac.uk/chembl/api/data/{resource}.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `molecule/{chembl_id}` | One molecule | `molecule/CHEMBL25 (aspirin)` |
| `molecule?pref_name__iexact= / max_phase=` | Filter molecules (Django-style lookups: __iexact, __icontains, __gte) | `max_phase=4` |
| `activity?target_chembl_id=&standard_type=` | Bioactivity measurements | `target_chembl_id=CHEMBL203&standard_type=IC50` |
| `mechanism?molecule_chembl_id=` | Mechanism of action (often stored on the salt form, e.g. CHEMBL1703 not CHEMBL1431) | `molecule_chembl_id=CHEMBL1703` |
| `drug_indication?molecule_chembl_id=` | Indications (MeSH/EFO) | `—` |
| `limit / offset` | Paging (max 1,000) | `5` |

```python
from _common import get, show

BASE = "https://www.ebi.ac.uk/chembl/api/data"


def main():
    # Mechanisms are recorded on the approved salt form (CHEMBL1703 = metformin hydrochloride), not the parent CHEMBL1431
    mol = get(f"{BASE}/molecule/CHEMBL1703.json")
    mech = get(f"{BASE}/mechanism.json", params={"molecule_chembl_id": "CHEMBL1703"})["mechanisms"]
    props = mol["molecule_properties"]
    show({"molecule": mol["pref_name"], "max_phase": mol["max_phase"],
          "properties": {k: props[k] for k in ("full_mwt", "alogp", "hba", "hbd", "num_ro5_violations", "qed_weighted")},
          "mechanisms": [{"action": m["action_type"], "mechanism": m["mechanism_of_action"]} for m in mech]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

```json
{
  "molecule": "METFORMIN HYDROCHLORIDE",
  "max_phase": "4.0",
  "properties": {
    "full_mwt": "165.63",
    "alogp": "-1.03",
    "hba": 2,
    "hbd": 4,
    "num_ro5_violations": 0,
    "qed_weighted": "0.25"
  },
  "mechanisms": [
    {
      "action": "INHIBITOR",
      "mechanism": "Mitochondrial complex I (NADH dehydrogenase) inhibitor"
    },
    {
      "action": "INHIBITOR",
      "mechanism": "Mitochondrial glycerol-3-phosphate dehydrogenase inhibitor"
    }
  ]
}
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

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

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

### CMS Medicare Part D Prescribers (data.cms.gov)

| | |
|:--|:--|
| **Docs** | https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers |
| **Auth** | None |
| **Register / token** | No registration. Each yearly release has its own dataset UUID (listed in https://data.cms.gov/data.json). The same API serves other data.cms.gov datasets (physician services, hospital costs, ...). |
| **Script** | [`scripts/health/cms_part_d.py`](scripts/health/cms_part_d.py) |

**What you can get:**

- Every prescriber's Medicare Part D prescriptions by drug: claims, 30-day fills, total cost, beneficiaries
- Prescriber specialty and location; brand vs. generic name
- One row per prescriber x drug (25M+ rows per year)

**Capstone ideas:** Opioid and antibiotic prescribing patterns, brand-name prescribing and pharma payments (Open Payments), cost outliers.

**Endpoint:** `GET https://data.cms.gov/data-api/v1/dataset/{uuid}/data`

| Parameter | Description | Example |
|:--|:--|:--|
| `uuid` | Dataset release ID from data.json (2024 release shown) | `9552739e-3d05-4c1b-8eff-ecabf391e2e5` |
| `filter[column]` | Exact-match filter on any column | `filter[Prscrbr_State_Abrvtn]=DC` |
| `keyword` | Full-text keyword across columns | `semaglutide` |
| `column` | Comma list of columns to return | `Prscrbr_Last_Org_Name,Gnrc_Name,Tot_Clms,Tot_Drug_Cst` |
| `size / offset` | Max 5,000 per call | `5` |
| `/data-api/v1/dataset/{uuid}/data/stats` | Row count for a query | `—` |

```python
from collections import defaultdict

from _common import get, show

URL = "https://data.cms.gov/data-api/v1/dataset/9552739e-3d05-4c1b-8eff-ecabf391e2e5/data"


def main():
    rows = get(URL, params={"filter[Prscrbr_State_Abrvtn]": "DC", "filter[Gnrc_Name]": "Semaglutide", "size": 5000})
    by_type = defaultdict(lambda: {"prescribers": 0, "claims": 0})
    for r in rows:
        t = by_type[r["Prscrbr_Type"]]
        t["prescribers"] += 1
        t["claims"] += int(r["Tot_Clms"])
    show({"drug": "semaglutide, DC prescribers", "prescriber_rows": len(rows),
          "total_cost_usd": round(sum(float(r["Tot_Drug_Cst"]) for r in rows)),
          "by_prescriber_type": sorted(({"type": k, **v} for k, v in by_type.items()), key=lambda x: -x["claims"])})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

```json
{
  "drug": "semaglutide, DC prescribers",
  "prescriber_rows": 399,
  "total_cost_usd": 20052702,
  "by_prescriber_type": [
    {
      "type": "Internal Medicine",
      "prescribers": 181,
      "claims": 6583
    },
    {
      "type": "Endocrinology",
      "prescribers": 39,
      "claims": 3283
    },
    {
      "type": "Family Practice",
      "prescribers": 63,
      "claims": 1854
    },
    {
      "type": "Nurse Practitioner",
      "prescribers": 49,
      "claims": 1264
    },
    {
      "type": "Physician Assistant",
      "prescribers": 19,
      "claims": 751
    },
    "... (18 more)"
  ]
}
```
</details>

---

### CMS Provider Data Catalog (Care Compare)

| | |
|:--|:--|
| **Docs** | https://data.cms.gov/provider-data/docs |
| **Auth** | None |
| **Register / token** | No registration. The data behind Medicare's Care Compare sites: hospitals, nursing homes, dialysis, home health, physicians. Dataset IDs are on each dataset page. |
| **Script** | [`scripts/health/cms_provider_data.py`](scripts/health/cms_provider_data.py) |

**What you can get:**

- Hospital general information and overall star ratings, readmissions, mortality, patient-experience (HCAHPS)
- Nursing-home ratings, staffing and penalties; dialysis and home-health quality
- SQL-like filtering, sorting and paging through the DKAN datastore API

**Capstone ideas:** Hospital quality vs. ownership or neighborhood, readmission prediction, nursing-home staffing and outcomes.

**Endpoint:** `GET https://data.cms.gov/provider-data/api/1/datastore/query/{datasetId}/0`

| Parameter | Description | Example |
|:--|:--|:--|
| `datasetId` | Dataset ID: xubh-q36u Hospital General Information, ... | `xubh-q36u` |
| `conditions[0][property] / [value] / [operator]` | Filters (=, <>, >, <, LIKE, IN) | `state = DC` |
| `sorts[0][property] / [order]` | Sort | `hospital_overall_rating / desc` |
| `limit / offset` | Max 1,500 per call | `5` |
| `count / results` | true\|false: include total count / rows | `true` |
| `/api/1/metastore/schemas/dataset/items` | Sibling endpoint: catalog of all datasets | `—` |

```python
from collections import Counter

from _common import get, show


def main():
    d = get("https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0", params={
        "conditions[0][property]": "state", "conditions[0][value]": "MD", "conditions[0][operator]": "=", "limit": 200})
    rows = d["results"]
    show({"maryland_hospitals": d["count"],
          "overall_star_ratings": dict(sorted(Counter(r["hospital_overall_rating"] for r in rows).items())),
          "five_star": [{k: r[k] for k in ("facility_name", "citytown", "hospital_type", "hospital_ownership")}
                        for r in rows if r["hospital_overall_rating"] == "5"][:3]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

```json
{
  "maryland_hospitals": 56,
  "overall_star_ratings": {
    "1": 2,
    "2": 6,
    "3": 18,
    "4": 9,
    "5": 7,
    "Not Available": 14
  },
  "five_star": [
    {
      "facility_name": "JOHNS HOPKINS HOSPITAL, THE",
      "citytown": "BALTIMORE",
      "hospital_type": "Acute Care Hospitals",
      "hospital_ownership": "Voluntary non-profit - Private"
    },
    {
      "facility_name": "MEDSTAR UNION MEMORIAL HOSPITAL",
      "citytown": "BALTIMORE",
      "hospital_type": "Acute Care Hospitals",
      "hospital_ownership": "Voluntary non-profit - Other"
    },
    {
      "facility_name": "UNIVERSITY OF MD SHORE MEDICAL CTR AT CHESTERTOWN",
      "citytown": "CHESTERTOWN",
      "hospital_type": "Acute Care Hospitals",
      "hospital_ownership": "Voluntary non-profit - Private"
    }
  ]
}
```
</details>

---

### DailyMed (NLM drug labels)

| | |
|:--|:--|
| **Docs** | https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm |
| **Auth** | None |
| **Register / token** | No registration. The official FDA-submitted labels (package inserts) as Structured Product Labeling (SPL). |
| **Script** | [`scripts/health/dailymed.py`](scripts/health/dailymed.py) |

**What you can get:**

- Current and historical drug labels: indications, dosage, warnings, boxed warnings, adverse reactions
- Search by drug name, NDC, RxCUI, manufacturer, pharmacologic class
- Label versions over time and label images

**Capstone ideas:** NLP on boxed warnings and adverse-reaction sections, tracking label changes after safety signals.

**Endpoint:** `GET https://dailymed.nlm.nih.gov/dailymed/services/v2/{resource}.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `spls.json?drug_name= / ?rxcui= / ?ndc=` | Labels (set IDs) matching a drug | `drug_name=metformin` |
| `labeler / published_date_comparison` | Filter by manufacturer / publication date | `labeler=Teva` |
| `pagesize / page` | Paging (max 100) | `5` |
| `spls/{setid}/history.json` | Versions of one label | `—` |
| `spls/{setid}.xml` | Full label XML (sections by LOINC code) | `—` |
| `drugnames.json?drug_name=` | Normalized drug names | `drug_name=lipitor` |

```python
from _common import get, show


def main():
    d = get("https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json", params={"drug_name": "semaglutide",
                                                                                   "pagesize": 5})
    show({"labels": d["metadata"]["total_elements"],
          "examples": [{"title": s["title"], "published": s["published_date"], "setid": s["setid"]} for s in d["data"]]},
         max_str=140)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

```json
{
  "labels": 9,
  "examples": [
    {
      "title": "OZEMPIC (ORAL SEMAGLUTIDE) TABLET RYBELSUS (ORAL SEMAGLUTIDE) TABLET [NOVO NORDISK PHARMACEUTICAL INDUSTRIES, LP]",
      "published": "Aug 19, 2026",
      "setid": "27f15fac-7d98-4114-a2ec-92494a91da98"
    },
    {
      "title": "OZEMPIC (SEMAGLUTIDE) INJECTION, SOLUTION [A-S MEDICATION SOLUTIONS]",
      "published": "Jul 31, 2026",
      "setid": "42bdd912-2393-44c4-b7e0-47672ca28991"
    },
    {
      "title": "WEGOVY (SEMAGLUTIDE) INJECTION, SOLUTION WEGOVY (SEMAGLUTIDE) TABLET [NOVO NORDISK PHARMACEUTICAL INDUSTRIES, LP]",
      "published": "Jun 30, 2026",
      "setid": "ee06186f-2aa3-4990-a760-757579d8f77b"
    },
    {
      "title": "OZEMPIC (SEMAGLUTIDE) INJECTION, SOLUTION [NOVO NORDISK PHARMACEUTICAL INDUSTRIES, LP]",
      "published": "Jun 10, 2026",
      "setid": "adec4fd2-6858-4c99-91d4-531f5f2a2d79"
    },
    {
      "title": "WEGOVY (SEMAGLUTIDE) INJECTION, SOLUTION [A-S MEDICATION SOLUTIONS]",
      "published": "Apr 25, 2024",
      "setid": "f5e548d0-cc79-4c34-a3f5-e20a5b8b6564"
    }
  ]
}
```
</details>

---

### Delphi Epidata (CMU disease surveillance)

| | |
|:--|:--|
| **Docs** | https://cmu-delphi.github.io/delphi-epidata/ |
| **Auth** | None (free key raises limits) |
| **Register / token** | Works keyless at low volume. Register at https://api.delphi.cmu.edu/epidata/admin/registration_form for a key. Python client: `pip install epidatpy`. |
| **Env var** | `DELPHI_API_KEY (optional)` |
| **Script** | [`scripts/health/delphi_epidata.py`](scripts/health/delphi_epidata.py) |

**What you can get:**

- CDC FluView: weekly influenza-like illness (ILI) rates nationally, by HHS region and state since 1997
- COVIDcast signals (cases, hospital admissions, search trends, wastewater) by county/state
- Nowcasts and forecasting archives with data revisions ('as of' versions)

**Capstone ideas:** Flu forecasting (the CDC FluSight challenge), nowcasting with revisions, signal-combination models.

**Endpoint:** `GET https://api.delphi.cmu.edu/epidata/{source}/`

| Parameter | Description | Example |
|:--|:--|:--|
| `source` | fluview \| fluview_clinical \| covidcast \| nidss_flu \| ... | `fluview` |
| `regions` | nat \| hhs1..hhs10 \| state codes (fluview) | `nat` |
| `epiweeks` | YYYYWW, list, or range YYYYWW-YYYYWW | `202530-202540` |
| `issues / lag` | Get data as it was reported at an issue week, or with a given reporting lag | `issues=202541` |
| `covidcast: data_source, signal, geo_type, geo_values, time_values` | COVIDcast query fields | `geo_type=state&geo_values=dc` |
| `api_key` | Optional key | `<DELPHI_API_KEY>` |

```python
from _common import get, show


def main():
    d = get("https://api.delphi.cmu.edu/epidata/fluview/", params={"regions": "nat", "epiweeks": "202501-202552"})
    rows = sorted(d["epidata"], key=lambda r: r["epiweek"])
    peak = max(rows, key=lambda r: r["wili"])
    show({"series": "US weighted influenza-like illness (% of visits), 2025", "weeks": len(rows),
          "peak": {"epiweek": peak["epiweek"], "wILI_pct": peak["wili"]},
          "last_4_weeks": [{"epiweek": r["epiweek"], "wILI_pct": r["wili"], "visits": r["num_patients"]} for r in rows[-4:]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

```json
{
  "series": "US weighted influenza-like illness (% of visits), 2025",
  "weeks": 52,
  "peak": {
    "epiweek": 202552,
    "wILI_pct": 8.29853
  },
  "last_4_weeks": [
    {
      "epiweek": 202549,
      "wILI_pct": 3.28462,
      "visits": 2647927
    },
    {
      "epiweek": 202550,
      "wILI_pct": 4.11906,
      "visits": 2678221
    },
    {
      "epiweek": 202551,
      "wILI_pct": 6.00523,
      "visits": 2771975
    },
    {
      "epiweek": 202552,
      "wILI_pct": 8.29853,
      "visits": 2540023
    }
  ]
}
```
</details>

---

### Ensembl REST API (genomics)

| | |
|:--|:--|
| **Docs** | https://rest.ensembl.org/ |
| **Auth** | None |
| **Register / token** | No registration. Max 15 requests/second (55,000/hour). Send content-type=application/json. |
| **Script** | [`scripts/health/ensembl.py`](scripts/health/ensembl.py) |

**What you can get:**

- Genes, transcripts and their genomic coordinates for human and 300+ species
- Genetic variants with consequences (VEP), population allele frequencies, phenotype associations
- Sequences, homologues across species, regulatory features

**Capstone ideas:** Variant-effect analysis pipelines, genomic feature engineering, cross-species comparison.

**Endpoint:** `GET https://rest.ensembl.org/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `lookup/symbol/{species}/{symbol}` | Gene by symbol (expand=1 adds transcripts) | `lookup/symbol/homo_sapiens/BRCA1` |
| `variation/{species}/{rsid}` | A variant: alleles, location, consequence, clinical significance; pops=1 adds population frequencies | `variation/human/rs429358?pops=1` |
| `vep/{species}/hgvs/{notation}` | Variant Effect Predictor | `vep/human/hgvs/ENST00000366667:c.803C>T` |
| `sequence/id/{id}?type=` | Genomic, cDNA or protein sequence | `type=protein` |
| `homology/symbol/{species}/{symbol}` | Orthologues in other species | `—` |
| `content-type` | application/json (query parameter or Accept header) | `application/json` |

```python
from _common import get, show

BASE = "https://rest.ensembl.org"
JSON = {"content-type": "application/json"}


def main():
    gene = get(f"{BASE}/lookup/symbol/homo_sapiens/APOE", params=JSON)
    var = get(f"{BASE}/variation/human/rs429358", params={**JSON, "pops": 1})  # pops=1 adds population frequencies
    show({"gene": {k: gene[k] for k in ("id", "display_name", "biotype", "seq_region_name", "start", "end", "description")},
          "variant": {"id": var["name"], "location": var["mappings"][0]["location"],
                      "alleles": var["mappings"][0]["allele_string"], "consequence": var.get("most_severe_consequence"),
                      "gnomAD_allele_freq": {p["allele"]: p["frequency"] for p in var.get("populations", [])
                                             if p["population"] == "gnomADg:ALL"},
                      "clinical_significance": var.get("clinical_significance")}})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

```json
{
  "gene": {
    "id": "ENSG00000130203",
    "display_name": "APOE",
    "biotype": "protein_coding",
    "seq_region_name": "19",
    "start": 44903787,
    "end": 44909396,
    "description": "apolipoprotein E [Source:HGNC Symbol;Acc:HGNC:613]"
  },
  "variant": {
    "id": "rs429358",
    "location": "19:44908684-44908684",
    "alleles": "T/C",
    "consequence": "missense_variant",
    "gnomAD_allele_freq": {
      "T": 0.8426,
      "C": 0.1574
    },
    "clinical_significance": [
      "uncertain significance",
      "not provided",
      "benign",
      "likely pathogenic",
      "pathogenic",
      "... (6 more)"
    ]
  }
}
```
</details>

---

### Europe PMC

| | |
|:--|:--|
| **Docs** | https://europepmc.org/RestfulWebService |
| **Auth** | None |
| **Register / token** | No registration. Includes PubMed plus preprints, patents, guidelines and 6M+ open-access full texts. |
| **Script** | [`scripts/health/europe_pmc.py`](scripts/health/europe_pmc.py) |

**What you can get:**

- Search 45M+ life-science abstracts and full texts; citation counts; open-access and preprint flags
- Full-text XML for open-access articles
- Text-mined annotations: genes, diseases, chemicals, organisms

**Capstone ideas:** Full-text mining (methods, datasets used), preprint-to-publication tracking, citation analysis.

**Endpoint:** `GET https://www.ebi.ac.uk/europepmc/webservices/rest/search`

| Parameter | Description | Example |
|:--|:--|:--|
| `query` | Search syntax with fields: TITLE:, ABSTRACT:, AUTH:, PUB_YEAR:, OPEN_ACCESS:y, SRC:PPR (preprints) | `"long covid" AND PUB_YEAR:2025 AND OPEN_ACCESS:y` |
| `format` | json \| xml \| dc | `json` |
| `resultType` | idlist \| lite \| core (core adds abstract, MeSH, full-text links) | `lite` |
| `pageSize / cursorMark` | Max 1,000; cursorMark=* then nextCursorMark | `5 / *` |
| `sort` | Field and order, e.g. CITED desc \| P_PDATE_D desc | `CITED desc` |
| `/{source}/{id}/fullTextXML` | Sibling endpoint: open-access full text | `/PMC9000000/fullTextXML` |

```python
from _common import get, show


def main():
    d = get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={
        "query": '"long covid" AND PUB_YEAR:2024', "format": "json", "resultType": "lite",
        "pageSize": 5, "sort": "CITED desc"})
    show({"hits": d["hitCount"], "most_cited": [
        {"title": r["title"], "journal": r.get("journalTitle"), "year": r.get("pubYear"),
         "cited_by": r.get("citedByCount"), "open_access": r.get("isOpenAccess")} for r in d["resultList"]["result"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

```json
{
  "hits": 4925,
  "most_cited": [
    {
      "title": "Global incidence, prevalence, years lived with disability (YLDs), disability-adjusted life-years (DALYs), and healthy life expectancy (HALE) for 371 diseases an...",
      "journal": "Lancet",
      "year": "2024",
      "cited_by": 4469,
      "open_access": "Y"
    },
    {
      "title": "2024 Heart Disease and Stroke Statistics: A Report of US and Global Data From the American Heart Association.",
      "journal": "Circulation",
      "year": "2024",
      "cited_by": 2294,
      "open_access": "N"
    },
    {
      "title": "Long COVID science, research and policy.",
      "journal": "Nat Med",
      "year": "2024",
      "cited_by": 435,
      "open_access": "N"
    },
    {
      "title": "The Human Phenotype Ontology in 2024: phenotypes around the world.",
      "journal": "Nucleic Acids Res",
      "year": "2024",
      "cited_by": 384,
      "open_access": "Y"
    },
    {
      "title": "Long COVID: a clinical update.",
      "journal": "Lancet",
      "year": "2024",
      "cited_by": 353,
      "open_access": "N"
    }
  ]
}
```
</details>

---

### HAPI FHIR public test server (synthetic EHR data)

| | |
|:--|:--|
| **Docs** | https://hapi.fhir.org/ |
| **Auth** | None |
| **Register / token** | No registration. A public sandbox of HL7 FHIR R4 - the standard API every certified US EHR exposes. Data are synthetic/test records posted by developers: never put real patient data here. |
| **Script** | [`scripts/health/hapi_fhir.py`](scripts/health/hapi_fhir.py) |

**What you can get:**

- FHIR resources: Patient, Observation (labs, vitals), Condition, MedicationRequest, Encounter, Procedure ...
- Standard FHIR search parameters, _include/_revinclude, paging bundles
- Practice for SMART-on-FHIR apps that later run against real EHRs (Epic, Cerner)

**Capstone ideas:** Learning FHIR before working with real EHR exports; building ETL from FHIR to analysis tables.

**Endpoint:** `GET https://hapi.fhir.org/baseR4/{ResourceType}`

| Parameter | Description | Example |
|:--|:--|:--|
| `ResourceType` | Patient \| Observation \| Condition \| MedicationRequest \| Encounter \| ... | `Observation` |
| `code` | Coded filter (LOINC/SNOMED), system\|code | `http://loinc.org\|8867-4 (heart rate)` |
| `patient / subject` | Resources for one patient | `patient=Patient/123` |
| `date / _lastUpdated` | Date filters with prefixes ge, le, gt, lt | `_lastUpdated=ge2026-01-01` |
| `_count / _sort / _summary` | Page size, sort field, count-only | `_count=5 / _sort=-_lastUpdated` |
| `_include` | Pull referenced resources into the bundle | `Observation:patient` |

```python
from _common import get, show


def main():
    d = get("https://hapi.fhir.org/baseR4/Observation", params={
        "code": "http://loinc.org|8867-4", "_count": 5, "_sort": "-_lastUpdated"})  # heart rate
    out = []
    for e in d.get("entry", []):
        o = e["resource"]
        q = o.get("valueQuantity", {})
        out.append({"id": o["id"], "code": o["code"]["coding"][0].get("display"), "value": q.get("value"),
                    "unit": q.get("unit"), "subject": (o.get("subject") or {}).get("reference"),
                    "effective": o.get("effectiveDateTime")})
    show({"resourceType": d["resourceType"], "type": d["type"], "heart_rate_observations": out})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

```json
{
  "resourceType": "Bundle",
  "type": "searchset",
  "heart_rate_observations": [
    {
      "id": "101367",
      "code": "Heart rate",
      "value": 59,
      "unit": "beats/minute",
      "subject": "Patient/101365",
      "effective": "2026-10-12T18:30:17+05:30"
    },
    {
      "id": "101053",
      "code": "Heart Rate",
      "value": 58,
      "unit": "/min",
      "subject": "Patient/100987",
      "effective": "2026-10-12T11:30:00+02:00"
    },
    {
      "id": "101014",
      "code": null,
      "value": 55,
      "unit": "/min",
      "subject": "Patient/100987",
      "effective": "2026-10-12T11:15:00+02:00"
    },
    {
      "id": "100991",
      "code": "Heart Rate",
      "value": 60,
      "unit": "/min",
      "subject": "Patient/100987",
      "effective": "2025-10-12T11:00:00+02:00"
    },
    {
      "id": "100936",
      "code": "Heart rate",
      "value": 72,
      "unit": "beats/minute",
      "subject": null,
      "effective": "2026-10-12T15:00:00+08:00"
    }
  ]
}
```
</details>

---

### MedlinePlus Connect (patient education)

| | |
|:--|:--|
| **Docs** | https://medlineplus.gov/medlineplus-connect/web-service/ |
| **Auth** | None |
| **Register / token** | No registration. Max 100 requests/minute per IP; cache responses for 12-24 hours. |
| **Script** | [`scripts/health/medlineplus_connect.py`](scripts/health/medlineplus_connect.py) |

**What you can get:**

- Consumer-friendly health topic pages (English/Spanish) matched to ICD-10-CM, SNOMED CT, RxNorm, NDC and LOINC codes
- Summaries and links for diagnoses, medications and lab tests

**Capstone ideas:** Linking coded EHR/claims data to plain-language explanations; health-literacy and readability studies.

**Endpoint:** `GET https://connect.medlineplus.gov/service`

| Parameter | Description | Example |
|:--|:--|:--|
| `mainSearchCriteria.v.cs` | Code system OID: ICD-10-CM 2.16.840.1.113883.6.90, RxNorm ...6.88, LOINC ...6.1, SNOMED ...6.96 | `2.16.840.1.113883.6.90` |
| `mainSearchCriteria.v.c` | The code | `E11.9` |
| `mainSearchCriteria.v.dn` | Optional display name | `Type 2 diabetes` |
| `informationRecipient.languageCode.c` | en \| es | `en` |
| `knowledgeResponseType` | application/json \| text/xml \| text/html | `application/json` |

```python
import html
import re

from _common import get, show


def main():
    d = get("https://connect.medlineplus.gov/service", params={
        "mainSearchCriteria.v.cs": "2.16.840.1.113883.6.90", "mainSearchCriteria.v.c": "E11.9",
        "knowledgeResponseType": "application/json"})["feed"]
    show([{"title": e["title"]["_value"], "url": e["link"][0]["href"],
           "summary": html.unescape(re.sub("<[^>]+>", " ", e["summary"]["_value"])).split(".")[0].strip() + "."}
          for e in d["entry"]], max_str=220)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

```json
[
  {
    "title": "Diabetes Type 2",
    "url": "https://medlineplus.gov/diabetestype2.html?utm_source=mplusconnect&utm_medium=service",
    "summary": "What is type 2 diabetes?  \n Type 2  diabetes  is a disease in which your blood glucose, or  blood sugar , levels are too high."
  },
  {
    "title": "Type 2 diabetes",
    "url": "https://medlineplus.gov/genetics/condition/type-2-diabetes?utm_source=mplusconnect&utm_medium=service",
    "summary": "Type 2 diabetes is a disorder characterized by abnormally high levels of blood glucose, also called blood sugar."
  },
  {
    "title": "Type 2 Diabetes",
    "url": "https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes/type-2-diabetes?utm_medium=ehr+service&utm_source=medlineplus+connect",
    "summary": "Learn about the symptoms of type 2 diabetes, what causes the disease, how it’s diagnosed, and steps you can take to help prevent or delay type 2 diabetes."
  }
]
```
</details>

---

### NLM Clinical Table Search Service (ICD-10-CM, LOINC, conditions)

| | |
|:--|:--|
| **Docs** | https://clinicaltables.nlm.nih.gov/ |
| **Auth** | None |
| **Register / token** | No registration. Built for autocomplete, so it is fast and returns compact arrays. |
| **Script** | [`scripts/health/nlm_clinical_tables.py`](scripts/health/nlm_clinical_tables.py) |

**What you can get:**

- ICD-10-CM diagnosis codes, ICD-11, LOINC lab tests, SNOMED-based conditions, RxTerms drugs, HCPCS, NPI
- Search by words or code prefix; choose returned fields

**Capstone ideas:** Mapping free-text diagnoses to ICD codes, building code lookups for claims or EHR data.

**Endpoint:** `GET https://clinicaltables.nlm.nih.gov/api/{table}/v3/search`

| Parameter | Description | Example |
|:--|:--|:--|
| `table` | icd10cm \| icd11_codes \| loinc_items \| conditions \| rxterms \| hcpcs \| npi_idv \| ... | `icd10cm` |
| `terms` | Search text (words or code prefix) | `type 2 diabetes` |
| `sf` | Fields to search | `code,name` |
| `df / ef` | Fields to display / extra fields to return | `df=code,name` |
| `maxList / count / offset` | Max results (up to 500), paging | `maxList=7` |
| `response shape` | [total, codes, extra_fields, display_rows] | `—` |

```python
from _common import get, show


def main():
    total, codes, _, rows = get("https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search",
                                params={"sf": "code,name", "terms": "type 2 diabetes", "maxList": 7})
    show({"matching_ICD10CM_codes": total, "examples": [{"code": c, "name": r[1]} for c, r in zip(codes, rows)]},
         max_items=7)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

```json
{
  "matching_ICD10CM_codes": 94,
  "examples": [
    {
      "code": "E11.65",
      "name": "Type 2 diabetes mellitus with hyperglycemia"
    },
    {
      "code": "E11.9",
      "name": "Type 2 diabetes mellitus without complications"
    },
    {
      "code": "E10.A2",
      "name": "Type 1 diabetes mellitus, presymptomatic, Stage 2"
    },
    {
      "code": "E11.21",
      "name": "Type 2 diabetes mellitus with diabetic nephropathy"
    },
    {
      "code": "E11.36",
      "name": "Type 2 diabetes mellitus with diabetic cataract"
    },
    {
      "code": "E11.41",
      "name": "Type 2 diabetes mellitus with diabetic mononeuropathy"
    },
    {
      "code": "E11.42",
      "name": "Type 2 diabetes mellitus with diabetic polyneuropathy"
    }
  ]
}
```
</details>

---

### NPPES NPI Registry (healthcare providers)

| | |
|:--|:--|
| **Docs** | https://npiregistry.cms.hhs.gov/api-page |
| **Auth** | None |
| **Register / token** | No registration. Every US healthcare provider and organization that bills insurance has a National Provider Identifier (NPI). Full monthly file available for bulk use. |
| **Script** | [`scripts/health/nppes_npi.py`](scripts/health/nppes_npi.py) |

**What you can get:**

- Provider name, credential, NPI type (individual/organization), taxonomy (specialty), practice addresses
- Search by name, specialty, city, state, ZIP

**Capstone ideas:** Provider-supply maps (specialists per capita, health-care deserts), linking NPIs to Medicare claims data.

**Endpoint:** `GET https://npiregistry.cms.hhs.gov/api/`

| Parameter | Description | Example |
|:--|:--|:--|
| `version` | API version (required) | `2.1` |
| `number` | Look up one NPI | `1234567893` |
| `first_name / last_name / organization_name` | Name search (* wildcard after 2 letters) | `last_name=Smith` |
| `taxonomy_description` | Specialty text | `cardiology` |
| `city / state / postal_code` | Location | `Washington / DC` |
| `enumeration_type` | NPI-1 (individual) \| NPI-2 (organization) | `NPI-1` |
| `limit / skip` | Max 200 per call; skip up to 1,000 | `5` |

```python
from collections import Counter

from _common import get, show


def main():
    d = get("https://npiregistry.cms.hhs.gov/api/", params={
        "version": "2.1", "taxonomy_description": "pediatrics", "city": "Washington", "state": "DC",
        "enumeration_type": "NPI-1", "limit": 200})
    specialties = Counter(t["desc"] for r in d["results"] for t in r["taxonomies"] if t["primary"])
    show({"pediatrics_providers_returned": d["result_count"], "primary_specialties": specialties.most_common(5),
          "examples": [{"npi": r["number"], "name": f'{r["basic"].get("first_name", "")} {r["basic"].get("last_name", "")}',
                        "credential": r["basic"].get("credential")} for r in d["results"][:3]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

```json
{
  "pediatrics_providers_returned": 200,
  "primary_specialties": [
    [
      "Nurse Practitioner, Pediatrics",
      114
    ],
    [
      "Pediatrics, Pediatric Emergency Medicine",
      28
    ],
    [
      "Pediatrics, Pediatric Infectious Diseases",
      18
    ],
    [
      "Pediatrics, Adolescent Medicine",
      10
    ],
    [
      "Pediatrics",
      5
    ]
  ],
  "examples": [
    {
      "npi": "1326402744",
      "name": "ANNA ABRAMS",
      "credential": "MD"
    },
    {
      "npi": "1538620182",
      "name": "CHIDIMMA ACHOLONU",
      "credential": "MD MPH"
    },
    {
      "npi": "1154759090",
      "name": "OLIVIA ACKERMAN",
      "credential": "APRN"
    }
  ]
}
```
</details>

---

### Open Targets Platform (GraphQL)

| | |
|:--|:--|
| **Docs** | https://platform-docs.opentargets.org/data-access/graphql-api |
| **Auth** | None |
| **Register / token** | No registration. GraphQL: POST a query, get exactly the fields you ask for. Try queries in the browser at https://api.platform.opentargets.org/api/v4/graphql/browser. Diseases use EFO/MONDO IDs. |
| **Script** | [`scripts/health/open_targets.py`](scripts/health/open_targets.py) |

**What you can get:**

- Target-disease association scores combining genetics, drugs, literature, animal models
- Known drugs per disease/target with clinical phase
- Target tractability and safety; disease ontology

**Capstone ideas:** Drug-target prioritization, knowledge-graph link prediction, genetic evidence for drug repurposing.

**Endpoint:** `POST https://api.platform.opentargets.org/api/v4/graphql`

| Parameter | Description | Example |
|:--|:--|:--|
| `search(queryString:, entityNames:)` | Find IDs for diseases, targets or drugs | `queryString: "type 2 diabetes", entityNames: ["disease"]` |
| `disease(efoId:)` | A disease by ID (MONDO_/EFO_), with associatedTargets, knownDrugs, ... | `MONDO_0005148` |
| `target(ensemblId:)` | A gene/protein target, with associatedDiseases, tractability, safety | `ENSG00000148795` |
| `drug(chemblId:)` | A drug with mechanisms and indications | `CHEMBL1431` |
| `page: {index, size}` | Paging inside list fields | `{index: 0, size: 5}` |

```python
import requests

from _common import USER_AGENT, show

URL = "https://api.platform.opentargets.org/api/v4/graphql"


def gql(query):
    r = requests.post(URL, json={"query": query}, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    return r.json()["data"]


def main():
    disease_id = gql('{ search(queryString: "type 2 diabetes", entityNames: ["disease"], page: {index: 0, size: 1})'
                     ' { hits { id } } }')["search"]["hits"][0]["id"]
    d = gql(f'{{ disease(efoId: "{disease_id}") {{ name associatedTargets(page: {{index: 0, size: 5}}) '
            f'{{ count rows {{ target {{ approvedSymbol approvedName }} score }} }} }} }}')["disease"]
    show({"disease": f'{d["name"]} ({disease_id})', "associated_targets": d["associatedTargets"]["count"],
          "top_targets": [{"symbol": r["target"]["approvedSymbol"], "name": r["target"]["approvedName"],
                           "score": round(r["score"], 3)} for r in d["associatedTargets"]["rows"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

```json
{
  "disease": "type 2 diabetes mellitus (MONDO_0005148)",
  "associated_targets": 10206,
  "top_targets": [
    {
      "symbol": "KCNJ11",
      "name": "potassium inwardly rectifying channel subfamily J member 11",
      "score": 0.877
    },
    {
      "symbol": "ABCC8",
      "name": "ATP binding cassette subfamily C member 8",
      "score": 0.876
    },
    {
      "symbol": "GCK",
      "name": "glucokinase",
      "score": 0.868
    },
    {
      "symbol": "PPARG",
      "name": "peroxisome proliferator activated receptor gamma",
      "score": 0.861
    },
    {
      "symbol": "HNF1A",
      "name": "HNF1 homeobox A",
      "score": 0.85
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

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

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

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

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

### PubMed (NCBI E-utilities)

| | |
|:--|:--|
| **Docs** | https://www.ncbi.nlm.nih.gov/books/NBK25501/ |
| **Auth** | None (optional free key raises the limit) |
| **Register / token** | Works keyless at 3 requests/second. Create an NCBI account and generate a key under Account Settings for 10 req/s. Add tool= and email= parameters to identify yourself. |
| **Env var** | `NCBI_API_KEY (optional)` |
| **Script** | [`scripts/health/pubmed.py`](scripts/health/pubmed.py) |

**What you can get:**

- 39M+ biomedical citations: title, authors, journal, date, MeSH terms, abstracts (efetch)
- Search with the full PubMed query language (field tags, MeSH, date ranges, publication types)
- The same E-utilities reach Gene, Protein, Nucleotide, ClinVar, SRA and 30+ other NCBI databases

**Capstone ideas:** Biomedical NLP on abstracts, research-trend analysis by MeSH topic, literature-based discovery.

**Endpoint:** `GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{esearch \| esummary \| efetch \| elink}.fcgi`

| Parameter | Description | Example |
|:--|:--|:--|
| `db` | pubmed \| pmc \| gene \| protein \| clinvar \| ... | `pubmed` |
| `term (esearch)` | Query with field tags: [mh] MeSH, [tiab] title/abstract, [dp] date, [pt] publication type | `"long covid"[tiab] AND 2025[dp]` |
| `retmax / retstart / sort` | Page size (max 10,000), offset; sort=pub_date \| relevance | `5 / 0 / pub_date` |
| `id (esummary, efetch)` | Comma-separated PMIDs from esearch | `39811234,39811230` |
| `rettype / retmode (efetch)` | abstract + text, or xml for full records | `abstract / text` |
| `usehistory` | y -> keep results server-side for large batch downloads (WebEnv + query_key) | `y` |
| `retmode / api_key / tool / email` | json \| xml; optional key; identify your tool | `json` |

```python
import os

from _common import get, show

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def main():
    common = {"db": "pubmed", "retmode": "json", "tool": "gwu-capstone-catalog"}
    if os.getenv("NCBI_API_KEY"):
        common["api_key"] = os.environ["NCBI_API_KEY"]
    s = get(f"{BASE}/esearch.fcgi", params={**common, "term": '"long covid"[tiab] AND 2025[dp]', "retmax": 5,
                                            "sort": "pub_date"})["esearchresult"]
    summ = get(f"{BASE}/esummary.fcgi", params={**common, "id": ",".join(s["idlist"])})["result"]
    show({"articles_2025": s["count"], "latest": [
        {"pmid": i, "title": summ[i]["title"], "journal": summ[i]["source"], "pubdate": summ[i]["pubdate"]}
        for i in summ["uids"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

```json
{
  "articles_2025": "2142",
  "latest": [
    {
      "pmid": "40462722",
      "title": "What Are the Barriers and Supports to a Return to Health From Long COVID? A Qualitative Study Designed, Developed, and Conducted by Individuals With Lived Exper...",
      "journal": "Qual Health Res",
      "pubdate": "2026 Sep"
    },
    {
      "pmid": "40084630",
      "title": "Therapeutic Potential of Psilocybin for Treating Neuropsychiatric Long COVID Symptoms: A Reddit Investigation.",
      "journal": "J Psychoactive Drugs",
      "pubdate": "2026 Sep-Oct"
    },
    {
      "pmid": "42491406",
      "title": "COVID-19 Is Airborne AIDS: Provocative Oversimplification, Emerging Science, or Something in Between?",
      "journal": "AJPM Focus",
      "pubdate": "2026 Aug"
    },
    {
      "pmid": "41177912",
      "title": "\"Everything Looks Normal\": Patient Narratives of Contested Legitimacy in Long COVID Medical Encounters.",
      "journal": "Health Commun",
      "pubdate": "2026 Aug"
    },
    {
      "pmid": "42411649",
      "title": "Clinical neuropsychological evaluation of patients referred from a post-COVID subspeciality clinic.",
      "journal": "Clin Neuropsychol",
      "pubdate": "2026 Jul"
    }
  ]
}
```
</details>

---

### RxNorm / RxNav (NLM drug terminology)

| | |
|:--|:--|
| **Docs** | https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html |
| **Auth** | None |
| **Register / token** | No registration. Max 20 requests/second per IP. |
| **Script** | [`scripts/health/rxnorm.py`](scripts/health/rxnorm.py) |

**What you can get:**

- Normalized names and RxCUI codes for every US clinical drug (ingredient, strength, dose form, brand)
- Relationships: brand <-> generic, ingredients, drug classes (via RxClass), NDC codes
- Approximate matching of messy drug names (e.g. from prescriptions or notes)

**Capstone ideas:** Cleaning and linking medication data across datasets (Medicare Part D, FAERS, EHR extracts).

**Endpoint:** `GET https://rxnav.nlm.nih.gov/REST/{resource}.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `drugs.json?name=` | All clinical and branded drugs for a name | `name=metformin` |
| `rxcui.json?name=` | RxCUI for an exact name | `name=Lipitor` |
| `approximateTerm.json?term=&maxEntries=` | Fuzzy match for misspelled names | `term=metfromin` |
| `rxcui/{rxcui}/related.json?tty=` | Related concepts by term type (IN ingredient, BN brand, SCD clinical drug, ...) | `rxcui/153165/related.json?tty=IN` |
| `rxcui/{rxcui}/ndcs.json` | NDC package codes | `—` |
| `/REST/rxclass/class/byDrugName.json?drugName=` | RxClass: therapeutic/pharmacologic classes | `drugName=metformin` |

```python
from _common import get, show


def main():
    groups = get("https://rxnav.nlm.nih.gov/REST/drugs.json", params={"name": "metformin"})["drugGroup"]["conceptGroup"]
    fuzzy = get("https://rxnav.nlm.nih.gov/REST/approximateTerm.json", params={"term": "metfromin", "maxEntries": 1})
    classes = get("https://rxnav.nlm.nih.gov/REST/rxclass/class/byDrugName.json", params={"drugName": "metformin",
                                                                                          "relaSource": "ATC"})
    by_type = {g["tty"]: [p["name"] for p in g.get("conceptProperties", []) if "/" not in p["name"]]  # skip combos
               for g in groups}
    show({"clinical_drugs (SCD)": by_type.get("SCD", [])[:4], "branded_drugs (SBD)": by_type.get("SBD", [])[:3],
          "fuzzy_match_for_'metfromin'": fuzzy["approximateGroup"]["candidate"][0]["rxcui"],
          "ATC_classes": sorted({c["rxclassMinConceptItem"]["className"]
                                 for c in classes["rxclassDrugInfoList"]["rxclassDrugInfo"]})})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

```json
{
  "clinical_drugs (SCD)": [
    "Modified 24 HR metformin hydrochloride 1000 MG Extended Release Oral Tablet",
    "Osmotic 24 HR metformin hydrochloride 1000 MG Extended Release Oral Tablet",
    "Modified 24 HR metformin hydrochloride 500 MG Extended Release Oral Tablet",
    "Osmotic 24 HR metformin hydrochloride 500 MG Extended Release Oral Tablet"
  ],
  "branded_drugs (SBD)": [
    "24 HR metformin hydrochloride 500 MG Extended Release Oral Tablet [Glucophage]",
    "24 HR metformin hydrochloride 750 MG Extended Release Oral Tablet [Glucophage]",
    "metformin hydrochloride 1000 MG Oral Tablet [Glucophage]"
  ],
  "fuzzy_match_for_'metfromin'": "6762",
  "ATC_classes": [
    "Biguanides",
    "Combinations of oral blood glucose lowering drugs"
  ]
}
```
</details>

---

### UniProt (proteins)

| | |
|:--|:--|
| **Docs** | https://www.uniprot.org/help/api |
| **Auth** | None |
| **Register / token** | No registration. For large result sets use the /stream endpoint or cursor paging (Link header). |
| **Script** | [`scripts/health/uniprot.py`](scripts/health/uniprot.py) |

**What you can get:**

- 250M+ protein sequences; 570k+ expert-reviewed (Swiss-Prot) entries with function, disease involvement, variants
- Gene names, organism, subcellular location, structures (PDB/AlphaFold), GO terms, cross-references

**Capstone ideas:** Protein-function prediction, disease-variant analysis, sequence datasets for ML (embeddings).

**Endpoint:** `GET https://rest.uniprot.org/uniprotkb/{search \| stream \| {accession}}`

| Parameter | Description | Example |
|:--|:--|:--|
| `query` | Query syntax: gene:, organism_id:, reviewed:true, disease:, keyword: | `gene:BRCA1 AND organism_id:9606 AND reviewed:true` |
| `fields` | Columns to return | `accession,gene_names,protein_name,length,cc_disease` |
| `format` | json \| tsv \| fasta \| xml | `json` |
| `size` | Page size (max 500); follow the Link header for more | `5` |
| `/uniprotkb/{accession}` | One entry | `/uniprotkb/P38398` |
| `/uniref, /uniparc, /proteomes` | Other UniProt databases with the same pattern | `—` |

```python
from _common import get, show


def main():
    d = get("https://rest.uniprot.org/uniprotkb/search", params={
        "query": "gene:BRCA1 AND organism_id:9606 AND reviewed:true", "format": "json",
        "fields": "accession,gene_names,protein_name,length,cc_disease", "size": 1})["results"][0]
    show({"accession": d["primaryAccession"], "gene": d["genes"][0]["geneName"]["value"],
          "protein": d["proteinDescription"]["recommendedName"]["fullName"]["value"], "length_aa": d["sequence"]["length"],
          "diseases": [c["disease"]["diseaseId"] for c in d.get("comments", []) if c.get("disease")]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

```json
{
  "accession": "P38398",
  "gene": "BRCA1",
  "protein": "Breast cancer type 1 susceptibility protein",
  "length_aa": 1863,
  "diseases": [
    "Breast cancer",
    "Breast-ovarian cancer, familial, 1",
    "Ovarian cancer",
    "Pancreatic cancer 4",
    "Fanconi anemia, complementation group S"
  ]
}
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

<details><summary><b>Sample output</b> — live run 2026-10-12 09:02</summary>

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

### WHO ICD-11 API

| | |
|:--|:--|
| **Docs** | https://icd.who.int/docs/icd-api/APIDoc-Version2/ |
| **Auth** | OAuth2 client credentials (free) |
| **Register / token** | Register at https://icd.who.int/icdapi -> View API access key(s) for a client ID and secret. Exchange them for a 1-hour token at https://icdaccessmanagement.who.int/connect/token (scope icdapi_access). A Docker image is available to run the API locally without limits. |
| **Env var** | `ICD_CLIENT_ID, ICD_CLIENT_SECRET` |
| **Script** | [`scripts/health/who_icd11.py`](scripts/health/who_icd11.py) |

**What you can get:**

- ICD-11 (the current WHO classification, in use since 2022) and ICD-10: codes, titles, definitions, synonyms
- Search and autocoding of free text to ICD-11 codes
- Hierarchy navigation, linearizations (MMS for mortality/morbidity)

**Capstone ideas:** Automated coding of clinical text or death certificates, mapping ICD-10 to ICD-11.

**Endpoint:** `GET https://id.who.int/icd/release/11/{release}/mms/{search \| autocode \| codeinfo/{code}}`

| Parameter | Description | Example |
|:--|:--|:--|
| `release` | Release ID | `2024-01` |
| `q (search)` | Search text | `type 2 diabetes` |
| `searchText (autocode)` | Free text to map to the best code | `heart attack` |
| `flatResults / highlightingEnabled` | Flat list; turn off <em> highlighting | `true / false` |
| `headers` | Authorization: Bearer <token>, Accept: application/json, Accept-Language: en, API-Version: v2 | `—` |

```python
import re

import requests

from _common import get, key, show


def main():
    token = requests.post("https://icdaccessmanagement.who.int/connect/token", timeout=30, data={
        "client_id": key("ICD_CLIENT_ID"), "client_secret": key("ICD_CLIENT_SECRET"),
        "scope": "icdapi_access", "grant_type": "client_credentials"}).json()["access_token"]
    d = get("https://id.who.int/icd/release/11/2024-01/mms/search", params={
        "q": "type 2 diabetes", "flatResults": "true", "highlightingEnabled": "false"}, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/json", "Accept-Language": "en", "API-Version": "v2"})
    show({"query": "type 2 diabetes", "matches": [
        {"code": e["theCode"], "title": re.sub("<[^>]+>", "", e["title"])} for e in d["destinationEntities"][:5]]})
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "query": "type 2 diabetes",
  "matches": [
    {"code": "5A11", "title": "Type 2 diabetes mellitus"},
    {"code": "JA63.2", "title": "Type 2 diabetes mellitus in pregnancy"}
  ]
}
```
</details>

---

## 🌾 Agriculture & Food

### USDA AMS Market News (MyMarketNews API)

| | |
|:--|:--|
| **Docs** | https://mymarketnews.ams.usda.gov/mymarketnews-api |
| **Auth** | API key as the Basic-auth username (empty password) |
| **Register / token** | Create a free account at https://mymarketnews.ams.usda.gov, then generate a key under My Profile -> API Key. |
| **Env var** | `AMS_API_KEY` |
| **Script** | [`scripts/agriculture/ams_market_news.py`](scripts/agriculture/ams_market_news.py) |

**What you can get:**

- Daily/weekly wholesale and terminal-market prices for livestock, grain, dairy, cotton, fruits and vegetables
- Cattle and hog auction reports, boxed-beef cutout values, retail feature prices
- 1,000+ report series, most with history back to the 2000s

**Capstone ideas:** Price transmission from farm to wholesale to retail, seasonality of produce prices, market-shock detection.

**Endpoint:** `GET https://marsapi.ams.usda.gov/services/v1.2/reports[/{slug_id}]`

| Parameter | Description | Example |
|:--|:--|:--|
| `/reports` | Catalog of report series with slug_id, title, market type | `—` |
| `/reports/{slug_id}` | Data for one report (e.g. 2466 = national daily boxed beef cutout) | `2466` |
| `q` | Filter expression field=value;field=value, incl. date ranges with report_begin_date | `report_begin_date=01/01/2026:10/01/2026` |
| `allSections=true` | Return every section of multi-part reports | `true` |
| `auth` | HTTP Basic: key as username, blank password | `(AMS_API_KEY, '')` |

```python
import requests

from _common import USER_AGENT, key, show


def main():
    r = requests.get("https://marsapi.ams.usda.gov/services/v1.2/reports/2466", auth=(key("AMS_API_KEY"), ""),
                     headers={"User-Agent": USER_AGENT}, params={"q": "report_begin_date=10/01/2026:10/12/2026"}, timeout=60)
    r.raise_for_status()
    show(r.json()["results"][:3])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"report_date": "10/09/2026", "slug_name": "AMS_2466", "report_title": "National Daily Boxed Beef Cutout & Boxed Beef Cuts",
   "current_volume": "112", "choice_600_900_current": "362.18", "select_600_900_current": "339.40"}
]
```
</details>

---

### USDA NASS CropScape (Cropland Data Layer)

| | |
|:--|:--|
| **Docs** | https://nassgeodata.gmu.edu/CropScape/devhelp/help.html |
| **Auth** | None |
| **Register / token** | No registration. The Cropland Data Layer is a 30 m satellite-derived map of what is planted on every field in the US each year (2008-present nationally). |
| **Script** | [`scripts/agriculture/cropscape.py`](scripts/agriculture/cropscape.py) |

**What you can get:**

- Acreage by crop type for any county, state, ASD or custom area (from the 30 m crop map)
- Raster extracts (GeoTIFF) of the crop map for an area and year
- Pixel value at a point; change between years

**Capstone ideas:** Crop rotation and land-use change, irrigation/water demand estimates, pollinator-habitat studies.

**Endpoint:** `GET https://nassgeodata.gmu.edu/axis2/services/CDLService/{GetCDLStat \| GetCDLFile \| GetCDLValue}`

| Parameter | Description | Example |
|:--|:--|:--|
| `year` | Crop year | `2023` |
| `fips` | State (2-digit) or county (5-digit) FIPS | `19169 (Story County, Iowa)` |
| `format` | json \| csv (GetCDLStat) - returns an XML wrapper with a returnURL to the file | `json` |
| `bbox / crs` | Custom area for GetCDLFile / GetCDLStat (Albers coordinates) | `—` |
| `GetCDLValue?year=&x=&y=` | Crop at a point (Albers x/y) | `—` |

```python
import re

from _common import get, show

ROW = re.compile(r'\{value:(\d+), count:(\d+), category:"([^"]+)", color:"[^"]*", acreage:([\d.]+)\}')


def main():
    wrapper = get("https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLStat",
                  params={"year": 2023, "fips": "19169", "format": "json"}, as_json=False)
    file_url = re.search(r"<returnURL>(.*?)</returnURL>", wrapper).group(1)
    text = get(file_url, as_json=False)  # JavaScript-style object (unquoted keys), so parse with a regex
    rows = sorted(((cat, float(acres)) for _, _, cat, acres in ROW.findall(text)), key=lambda r: -r[1])
    show({"county": "Story County, Iowa (2023)", "top_land_cover_acres": [{"category": c, "acres": a} for c, a in rows[:6]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 08:15</summary>

```json
{
  "county": "Story County, Iowa (2023)",
  "top_land_cover_acres": [
    {
      "category": "Corn",
      "acres": 149713.6
    },
    {
      "category": "Soybeans",
      "acres": 119131.3
    },
    {
      "category": "Grass/Pasture",
      "acres": 40169.8
    },
    {
      "category": "Developed/Open Space",
      "acres": 14282.6
    },
    {
      "category": "Deciduous Forest",
      "acres": 11815.4
    },
    "... (1 more)"
  ]
}
```
</details>

---

### USDA ERS — ARMS Farm Financial Data

| | |
|:--|:--|
| **Docs** | https://www.ers.usda.gov/developer/data-apis/arms-data-api |
| **Auth** | api.data.gov key (`DEMO_KEY` works at low volume) |
| **Register / token** | Use your api.data.gov key (https://api.data.gov/signup/). The Agricultural Resource Management Survey (ARMS) is USDA's main source on farm finances and household income. |
| **Env var** | `DATA_GOV_API_KEY` |
| **Script** | [`scripts/agriculture/ers_arms.py`](scripts/agriculture/ers_arms.py) |

**What you can get:**

- Farm income statements and balance sheets (revenue, expenses, net farm income, assets, debt)
- Breakdowns by farm typology, size, commodity specialty, region and state
- Farm household income and operator characteristics

**Capstone ideas:** Farm financial stress, small vs. large farm economics, policy (subsidy) incidence across farm types.

**Endpoint:** `GET https://api.ers.usda.gov/data/arms/{surveydata \| year \| state \| report \| variable \| farmtype \| category}`

| Parameter | Description | Example |
|:--|:--|:--|
| `year` | Survey year(s) | `2023` |
| `report` | Report name: income statement, balance sheet, farm business debt, operator household income, ... | `income statement` |
| `variable` | Variable ID(s) (list via /data/arms/variable) | `kount` |
| `farmtype / category / category_value` | Farm type and breakdown category (list via /farmtype, /category) | `category=Farm Typology` |
| `state` | State name or 'all' | `all` |
| `api_key` | api.data.gov key | `DEMO_KEY` |

```python
import os

from _common import get, show


def main():
    d = get("https://api.ers.usda.gov/data/arms/surveydata", params={
        "api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY"), "year": 2023, "report": "income statement",
        "variable": "kount", "category": "Farm Typology"})
    show({"report": "Farm Business Income Statement (2023), estimated number of farms", "by_farm_type": [
        {"farm_type": r["categoryValue"], "farms": r["estimate"], "rse_pct": r["rse"]} for r in d["data"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 08:02</summary>

```json
{
  "report": "Farm Business Income Statement (2023), estimated number of farms",
  "by_farm_type": [
    {
      "farm_type": "Retirement farms (2011 to present)",
      "farms": 209812,
      "rse_pct": 0
    },
    {
      "farm_type": "Large farms (2011 to present)",
      "farms": 75110,
      "rse_pct": 0
    },
    {
      "farm_type": "Intermediate farms",
      "farms": 674874,
      "rse_pct": 0
    },
    {
      "farm_type": "Commercial farms",
      "farms": 263192,
      "rse_pct": 0
    },
    {
      "farm_type": "Off-farm occupation farms (2011 to present)",
      "farms": 741922,
      "rse_pct": 0
    },
    "... (6 more)"
  ]
}
```
</details>

---

### USDA FAS — Production, Supply & Distribution (PSD) and trade

| | |
|:--|:--|
| **Docs** | https://apps.fas.usda.gov/opendatawebv2/#/home |
| **Auth** | api.data.gov key (`DEMO_KEY` works at low volume) |
| **Register / token** | Use your api.data.gov key (https://api.data.gov/signup/). The old apps.fas.usda.gov/OpenData endpoints now return 403 - use api.fas.usda.gov as below. |
| **Env var** | `DATA_GOV_API_KEY` |
| **Script** | [`scripts/agriculture/fas_psd.py`](scripts/agriculture/fas_psd.py) |

**What you can get:**

- PSD: official USDA estimates of production, consumption, trade and stocks for ~60 commodities in every country
- Export Sales Reporting (ESR): weekly US export sales by commodity and destination
- GATS: US agricultural trade from Census (exports/imports by partner and product)

**Capstone ideas:** Global grain-market balance and price forecasting, food-security exposure to export shocks, trade-war effects.

**Endpoint:** `GET https://api.fas.usda.gov/api/psd/commodity/{commodityCode}/{world \| country/{code} \| country/all}/year/{marketYear}`

| Parameter | Description | Example |
|:--|:--|:--|
| `commodityCode` | From /api/psd/commodities (e.g. 0440000 corn, 0410000 wheat, 2222000 soybeans) | `0440000` |
| `world / country/{code} / country/all` | World total, one country, or every country | `world` |
| `marketYear` | Marketing year | `2024` |
| `/api/psd/commodityAttributes · /unitsOfMeasure · /countries` | Lookups for attributeId, unitId, countryCode | `—` |
| `/api/esr/exports/commodityCode/{c}/allCountries/marketYear/{y}` | Weekly export sales | `—` |
| `/api/gats/censusExports/partnerCode/{p}/year/{y}/month/{m}` | Monthly US exports to a partner | `partnerCode/CH/year/2024/month/1` |
| `api_key` | api.data.gov key | `DEMO_KEY` |

```python
import os

from _common import get, show

BASE = "https://api.fas.usda.gov/api/psd"


def main():
    k = {"api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY")}
    attrs = {a["attributeId"]: a["attributeName"].strip() for a in get(f"{BASE}/commodityAttributes", params=k)}
    units = {u["unitId"]: u["unitDescription"].strip() for u in get(f"{BASE}/unitsOfMeasure", params=k)}
    rows = get(f"{BASE}/commodity/0440000/world/year/2024", params=k)  # corn, world total
    show({"commodity": "Corn, world, market year 2024", "balance": [
        {"attribute": attrs.get(r["attributeId"], r["attributeId"]), "value": r["value"], "unit": units.get(r["unitId"])}
        for r in rows]}, max_items=12)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:18</summary>

```json
{
  "commodity": "Corn, world, market year 2024",
  "balance": [
    {
      "attribute": "Area Harvested",
      "value": 206660.0,
      "unit": "(1000 HA)"
    },
    {
      "attribute": "Beginning Stocks",
      "value": 313822.0,
      "unit": "(1000 MT)"
    },
    {
      "attribute": "Production",
      "value": 1234983.0,
      "unit": "(1000 MT)"
    },
    {
      "attribute": "Imports",
      "value": 186553.0,
      "unit": "(1000 MT)"
    },
    {
      "attribute": "TY Imports",
      "value": 187461.0,
      "unit": "(1000 MT)"
    },
    {
      "attribute": "TY Imp. from U.S.",
      "value": 75124.0,
      "unit": "(1000 MT)"
    },
    {
      "attribute": "Total Supply",
      "value": 1735358.0,
      "unit": "(1000 MT)"
    },
    {
      "attribute": "Exports",
      "value": 187583.0,
      "unit": "(1000 MT)"
    },
    {
      "attribute": "TY Exports",
      "value": 191438.0,
      "unit": "(1000 MT)"
    },
    {
      "attribute": "Feed Dom. Consumption",
      "value": 785752.0,
      "unit": "(1000 MT)"
    },
    {
      "attribute": "FSI Consumption",
      "value": 466382.0,
      "unit": "(1000 MT)"
    },
    {
      "attribute": "Domestic Consumption",
      "value": 1252134.0,
      "unit": "(1000 MT)"
    },
    "... (3 more)"
  ]
}
```
</details>

---

### USDA FoodData Central

| | |
|:--|:--|
| **Docs** | https://fdc.nal.usda.gov/api-guide/ |
| **Auth** | api.data.gov key (`DEMO_KEY` works at low volume) |
| **Register / token** | Use your api.data.gov key (https://api.data.gov/signup/ or https://fdc.nal.usda.gov/api-key-signup). 1,000 req/hour. Data are public domain (CC0). |
| **Env var** | `DATA_GOV_API_KEY` |
| **Script** | [`scripts/agriculture/fooddata_central.py`](scripts/agriculture/fooddata_central.py) |

**What you can get:**

- Nutrient profiles (energy, protein, fat, vitamins, minerals - 150+ nutrients) for 400k+ foods
- Data types: Foundation (lab-analyzed), SR Legacy, Survey (FNDDS, what Americans eat), Branded (labels with UPC)
- Ingredients and serving sizes for branded products

**Capstone ideas:** Nutrition scoring and diet optimization, ultra-processed food classification, recipe nutrient estimation.

**Endpoint:** `GET https://api.nal.usda.gov/fdc/v1/foods/search  ·  GET /fdc/v1/food/{fdcId}`

| Parameter | Description | Example |
|:--|:--|:--|
| `query` | Search text (supports +must -mustnot "phrases") | `cheddar cheese` |
| `dataType` | Foundation \| SR Legacy \| Survey (FNDDS) \| Branded (repeatable) | `Foundation` |
| `pageSize / pageNumber` | Max 200 per page | `5` |
| `sortBy / sortOrder` | dataType.keyword \| lowercaseDescription.keyword \| fdcId \| publishedDate; asc \| desc | `fdcId / asc` |
| `brandOwner` | Filter branded foods by owner | `Kraft Heinz` |
| `/food/{fdcId}?nutrients=` | One food; limit to nutrient numbers | `nutrients=203,204,205` |
| `api_key` | api.data.gov key | `DEMO_KEY` |

```python
import os

from _common import get, show

NUTRIENTS = {"Energy": "kcal", "Protein": "g", "Total lipid (fat)": "g", "Carbohydrate, by difference": "g"}


def main():
    d = get("https://api.nal.usda.gov/fdc/v1/foods/search", params={
        "query": "cheddar cheese", "dataType": "Foundation", "pageSize": 3,
        "api_key": os.getenv("DATA_GOV_API_KEY", "DEMO_KEY")})
    show({"matches": d["totalHits"], "foods": [
        {"fdcId": f["fdcId"], "description": f["description"], "category": f.get("foodCategory"),
         "per_100g": {n["nutrientName"]: f'{n["value"]} {n["unitName"].lower()}' for n in f["foodNutrients"]
                      if n["nutrientName"] in NUTRIENTS and n["unitName"].lower() == NUTRIENTS[n["nutrientName"]]}}
        for f in d["foods"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 08:02</summary>

```json
{
  "matches": 19,
  "foods": [
    {
      "fdcId": 328637,
      "description": "Cheese, cheddar",
      "category": "Dairy and Egg Products",
      "per_100g": {
        "Carbohydrate, by difference": "2.44 g",
        "Energy": "408 kcal",
        "Protein": "23.3 g",
        "Total lipid (fat)": "34.0 g"
      }
    },
    {
      "fdcId": 746767,
      "description": "Cheese, swiss",
      "category": "Dairy and Egg Products",
      "per_100g": {
        "Total lipid (fat)": "31.0 g",
        "Carbohydrate, by difference": "1.44 g",
        "Energy": "393 kcal",
        "Protein": "27.0 g"
      }
    },
    {
      "fdcId": 747429,
      "description": "Cheese, American, restaurant",
      "category": "Dairy and Egg Products",
      "per_100g": {
        "Total lipid (fat)": "31.1 g",
        "Protein": "17.5 g",
        "Carbohydrate, by difference": "6.35 g",
        "Energy": "375 kcal"
      }
    }
  ]
}
```
</details>

---

### USDA NASS Quick Stats

| | |
|:--|:--|
| **Docs** | https://quickstats.nass.usda.gov/api |
| **Auth** | API key (query `key`) |
| **Register / token** | Request a free key at https://quickstats.nass.usda.gov/api (enter email; key arrives instantly). Max 50,000 records per call - use /api/get_counts/ first to size a query. |
| **Env var** | `NASS_API_KEY` |
| **Script** | [`scripts/agriculture/nass_quickstats.py`](scripts/agriculture/nass_quickstats.py) |

**What you can get:**

- Official US agricultural statistics since the 1800s: crop acreage, yield, production, prices received
- Livestock inventories, dairy, poultry; Census of Agriculture (farms, land, operators, sales) every 5 years
- National, state, county and agricultural-district levels; weekly crop progress and condition

**Capstone ideas:** Yield forecasting with weather/soil data, farm consolidation trends, climate impacts on county crop yields.

**Endpoint:** `GET https://quickstats.nass.usda.gov/api/api_GET/`

| Parameter | Description | Example |
|:--|:--|:--|
| `commodity_desc` | Commodity | `CORN` |
| `statisticcat_desc` | AREA PLANTED \| AREA HARVESTED \| YIELD \| PRODUCTION \| PRICE RECEIVED \| INVENTORY \| ... | `YIELD` |
| `agg_level_desc` | NATIONAL \| STATE \| COUNTY \| AGRICULTURAL DISTRICT | `STATE` |
| `state_alpha / county_name` | Location filters | `IA` |
| `year / year__GE / year__LE` | Year, or a range with __GE / __LE suffixes | `year__GE=2015` |
| `source_desc` | SURVEY \| CENSUS | `SURVEY` |
| `short_desc` | Exact data-item name (most precise filter) | `CORN, GRAIN - YIELD, MEASURED IN BU / ACRE` |
| `format / key` | JSON \| CSV \| XML; your key | `JSON` |
| `/api/get_counts/ · /api/get_param_values/?param=` | Sibling endpoints: record count for a query; valid values of a field | `param=commodity_desc` |

```python
from _common import get, key, show


def main():
    d = get("https://quickstats.nass.usda.gov/api/api_GET/", params={
        "key": key("NASS_API_KEY"), "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
        "agg_level_desc": "STATE", "state_alpha": "IA", "source_desc": "SURVEY", "freq_desc": "ANNUAL",
        "reference_period_desc": "YEAR", "year__GE": 2021, "format": "JSON"})["data"]
    show([{k: r[k] for k in ("year", "state_name", "short_desc", "Value")} for r in sorted(d, key=lambda r: -r["year"])])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"year": 2025, "state_name": "IOWA", "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE", "Value": "210.0"},
  {"year": 2024, "state_name": "IOWA", "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE", "Value": "211.0"},
  {"year": 2023, "state_name": "IOWA", "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE", "Value": "201.0"}
]
```
</details>

---

### Open Food Facts

| | |
|:--|:--|
| **Docs** | https://openfoodfacts.github.io/openfoodfacts-server/api/ |
| **Auth** | None for reading |
| **Register / token** | No registration. Send a descriptive User-Agent. Limits: 100 product reads/min, 10 searches/min - for bulk work download the full database (CSV/JSONL/Parquet) instead. ODbL licence. |
| **Script** | [`scripts/agriculture/open_food_facts.py`](scripts/agriculture/open_food_facts.py) |

**What you can get:**

- 3M+ packaged food products worldwide (crowd-sourced): ingredients, nutrition facts, allergens, labels
- Nutri-Score, NOVA processing group (1-4), Eco-Score, additives, packaging
- Look up by barcode or search by category, brand, country

**Capstone ideas:** Ultra-processed food analysis, nutrition-label vs. Nutri-Score studies, food-product image classification.

**Endpoint:** `GET https://world.openfoodfacts.org/api/v2/{product/{barcode} \| search}`

| Parameter | Description | Example |
|:--|:--|:--|
| `product/{barcode}` | One product by EAN/UPC barcode | `product/737628064502` |
| `categories_tags_en / brands_tags / countries_tags_en` | Search filters (English tag names) | `breakfast-cereals` |
| `nutrition_grades_tags / nova_groups_tags` | Filter by Nutri-Score (a-e) or NOVA group (1-4) | `nova_groups_tags=4` |
| `fields` | Comma list of fields to return | `product_name,brands,nutriscore_grade,nova_group` |
| `page / page_size` | Paging | `1 / 5` |
| `sort_by` | popularity_key (other sort keys exist but returned 503 when tested) | `popularity_key` |

```python
from collections import Counter

from _common import get, show


def main():
    d = get("https://world.openfoodfacts.org/api/v2/search", params={
        "categories_tags_en": "breakfast-cereals", "countries_tags_en": "united-states", "sort_by": "popularity_key",
        "fields": "product_name,brands,nutriscore_grade,nova_group", "page_size": 100})
    products = d["products"]
    show({"US_breakfast_cereals": d["count"],
          "nutriscore_of_100_most_popular": dict(sorted(Counter(p.get("nutriscore_grade") for p in products).items(), key=str)),
          "nova_group_of_100_most_popular": dict(sorted(Counter(p.get("nova_group") for p in products).items(), key=str)),
          "examples": [{k: p.get(k) for k in ("product_name", "brands", "nutriscore_grade", "nova_group")} for p in products[:3]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 08:15</summary>

```json
{
  "US_breakfast_cereals": 3851,
  "nutriscore_of_100_most_popular": {
    "a": 27,
    "b": 6,
    "c": 35,
    "d": 21,
    "e": 7,
    "unknown": 4
  },
  "nova_group_of_100_most_popular": {
    "1": 16,
    "3": 36,
    "4": 47,
    "null": 1
  },
  "examples": [
    {
      "product_name": "Pumpkin Seed + Flax Granola",
      "brands": "Nature's Path Organic",
      "nutriscore_grade": "c",
      "nova_group": 3
    },
    {
      "product_name": "Oats 'N Honey Granola Bars",
      "brands": "Nature Valley",
      "nutriscore_grade": "d",
      "nova_group": 4
    },
    {
      "product_name": "grape-nuts cereal",
      "brands": "Post",
      "nutriscore_grade": "c",
      "nova_group": 3
    }
  ]
}
```
</details>

---

### USDA NRCS Soil Data Access (SSURGO)

| | |
|:--|:--|
| **Docs** | https://sdmdataaccess.nrcs.usda.gov/WebServiceHelp.aspx |
| **Auth** | None |
| **Register / token** | No registration. Send SQL (T-SQL) queries against the national SSURGO soil database; max 100,000 rows or ~32 MB per request. |
| **Script** | [`scripts/agriculture/soil_data_access.py`](scripts/agriculture/soil_data_access.py) |

**What you can get:**

- Detailed soil survey for the US: map units, components, horizons
- Properties: texture, organic matter, pH, drainage class, available water capacity, farmland classification
- Spatial functions to query by point or polygon (WKT)

**Capstone ideas:** Soil covariates for yield models, prime-farmland loss to development, land-suitability analysis.

**Endpoint:** `POST https://sdmdataaccess.sc.egov.usda.gov/tabular/post.rest`

| Parameter | Description | Example |
|:--|:--|:--|
| `query (JSON body)` | T-SQL against SSURGO tables: sacatalog, legend, mapunit, component, chorizon, ... | `SELECT TOP 5 areasymbol, areaname FROM sacatalog` |
| `format (JSON body)` | JSON \| JSON+COLUMNNAME \| JSON+COLUMNNAME+METADATA \| XML | `JSON+COLUMNNAME` |
| `SDA_Get_Mukey_from_intersection_with_WktWgs84('point(lon lat)')` | Table function: map units at a point | `point(-93.6 42.0)` |
| `farmlndcl (mapunit)` | Farmland classification, e.g. All areas are prime farmland | `—` |

```python
import requests

from _common import USER_AGENT, show

SQL = """
SELECT TOP 5 mu.musym, mu.muname, mu.farmlndcl, c.compname, c.comppct_r, c.drainagecl
FROM mapunit mu
JOIN component c ON c.mukey = mu.mukey
WHERE mu.mukey IN (SELECT * FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('point(-93.6 42.0)'))
ORDER BY c.comppct_r DESC
"""


def main():
    r = requests.post("https://sdmdataaccess.sc.egov.usda.gov/tabular/post.rest", timeout=60,
                      headers={"User-Agent": USER_AGENT}, json={"query": SQL, "format": "JSON+COLUMNNAME"})
    r.raise_for_status()
    header, *rows = r.json()["Table"]
    show({"point": "-93.6, 42.0 (central Iowa)", "soil_components": [dict(zip(header, row)) for row in rows]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 08:15</summary>

```json
{
  "point": "-93.6, 42.0 (central Iowa)",
  "soil_components": [
    {
      "musym": "1314",
      "muname": "Hanlon-Spillville complex, channeled, 0 to 2 percent slopes",
      "farmlndcl": "Farmland of statewide importance",
      "compname": "Hanlon",
      "comppct_r": "60",
      "drainagecl": "Moderately well drained"
    },
    {
      "musym": "1314",
      "muname": "Hanlon-Spillville complex, channeled, 0 to 2 percent slopes",
      "farmlndcl": "Farmland of statewide importance",
      "compname": "Spillville",
      "comppct_r": "30",
      "drainagecl": "Moderately well drained"
    },
    {
      "musym": "1314",
      "muname": "Hanlon-Spillville complex, channeled, 0 to 2 percent slopes",
      "farmlndcl": "Farmland of statewide importance",
      "compname": "Spillville",
      "comppct_r": "5",
      "drainagecl": "Moderately well drained"
    },
    {
      "musym": "1314",
      "muname": "Hanlon-Spillville complex, channeled, 0 to 2 percent slopes",
      "farmlndcl": "Farmland of statewide importance",
      "compname": "Coland",
      "comppct_r": "5",
      "drainagecl": "Poorly drained"
    }
  ]
}
```
</details>

---

### ISRIC SoilGrids (global soil properties)

| | |
|:--|:--|
| **Docs** | https://rest.isric.org/soilgrids/v2.0/docs |
| **Auth** | None |
| **Register / token** | No registration. Fair use: ~5 calls/minute (the service is a beta). Global 250 m predictions with uncertainty. |
| **Script** | [`scripts/agriculture/soilgrids.py`](scripts/agriculture/soilgrids.py) |

**What you can get:**

- Soil properties anywhere on Earth at 6 depths (0-200 cm): organic carbon, pH, clay/sand/silt, nitrogen, bulk density, CEC
- Mean, median and 5/95% quantiles (uncertainty)
- WRB soil classification at a point

**Capstone ideas:** Soil covariates for global crop or carbon models, soil-carbon sequestration potential, land suitability.

**Endpoint:** `GET https://rest.isric.org/soilgrids/v2.0/properties/query`

| Parameter | Description | Example |
|:--|:--|:--|
| `lon / lat` | Point (WGS84) | `-93.6 / 42.0` |
| `property` | soc \| phh2o \| clay \| sand \| silt \| nitrogen \| bdod \| cec \| cfvo \| ocd \| ocs (repeatable) | `soc` |
| `depth` | 0-5cm \| 5-15cm \| 15-30cm \| 30-60cm \| 60-100cm \| 100-200cm (repeatable) | `0-5cm` |
| `value` | mean \| Q0.05 \| Q0.5 \| Q0.95 \| uncertainty (repeatable) | `mean` |
| `unit conversion` | Divide by unit_measure.d_factor to get target units (e.g. pH*10 -> pH) | `—` |
| `/classification/query?lon=&lat=` | Sibling endpoint: most probable WRB soil class | `—` |

```python
from _common import get, show


def main():
    d = get("https://rest.isric.org/soilgrids/v2.0/properties/query", params=[
        ("lon", -93.6), ("lat", 42.0), ("property", "soc"), ("property", "phh2o"), ("property", "clay"),
        ("depth", "0-5cm"), ("depth", "30-60cm"), ("value", "mean")])
    out = {}
    for layer in d["properties"]["layers"]:
        f, unit = layer["unit_measure"]["d_factor"], layer["unit_measure"]["target_units"]
        out[f'{layer["name"]} ({unit})'] = {dep["label"]: dep["values"]["mean"] / f for dep in layer["depths"]}
    show({"point": "-93.6, 42.0 (central Iowa)", "properties": out})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 08:15</summary>

```json
{
  "point": "-93.6, 42.0 (central Iowa)",
  "properties": {
    "clay (%)": {
      "0-5cm": 29.1,
      "30-60cm": 30.8
    },
    "phh2o (-)": {
      "0-5cm": 6.5,
      "30-60cm": 6.6
    },
    "soc (g/kg)": {
      "0-5cm": 31.8,
      "30-60cm": 7.6
    }
  }
}
```
</details>

---

### TheMealDB (recipes)

| | |
|:--|:--|
| **Docs** | https://www.themealdb.com/api.php |
| **Auth** | Test key `1` in the URL path (free for development) |
| **Register / token** | The test key 1 works for development and education. Supporters get a production key with extra endpoints (multiple filters, latest meals, full list). |
| **Env var** | `THEMEALDB_KEY (optional)` |
| **Script** | [`scripts/agriculture/themealdb.py`](scripts/agriculture/themealdb.py) |

**What you can get:**

- ~600 recipes: name, category, cuisine (area), instructions, ingredients with measures, image, YouTube link
- Filter by main ingredient, category or cuisine; random meal; ingredient and category lists

**Capstone ideas:** Cuisine similarity networks from shared ingredients, recipe recommendation, ingredient-image datasets.

**Endpoint:** `GET https://www.themealdb.com/api/json/v1/{key}/{endpoint}.php`

| Parameter | Description | Example |
|:--|:--|:--|
| `search.php?s= / ?f=` | Search by name, or by first letter | `s=Arrabiata` |
| `lookup.php?i=` | Full recipe by ID | `i=52772` |
| `filter.php?i= / ?c= / ?a=` | Meals by main ingredient, category or area (cuisine) | `a=Italian` |
| `random.php` | One random meal | `—` |
| `list.php?c=list / ?a=list / ?i=list` | All categories / areas / ingredients | `a=list` |

```python
import os

from _common import get, show


def main():
    base = f'https://www.themealdb.com/api/json/v1/{os.getenv("THEMEALDB_KEY", "1")}'
    areas = [a["strArea"] for a in get(f"{base}/list.php", params={"a": "list"})["meals"]]
    italian = get(f"{base}/filter.php", params={"a": "Italian"})["meals"]
    meal = get(f"{base}/lookup.php", params={"i": italian[0]["idMeal"]})["meals"][0]
    ingredients = [f'{meal[f"strMeasure{i}"].strip()} {meal[f"strIngredient{i}"]}'.strip()
                   for i in range(1, 21) if (meal.get(f"strIngredient{i}") or "").strip()]
    show({"cuisines": len(areas), "italian_recipes": len(italian),
          "example": {"name": meal["strMeal"], "category": meal["strCategory"], "ingredients": ingredients,
                      "instructions": meal["strInstructions"]}}, max_items=6, max_str=200)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 08:15</summary>

```json
{
  "cuisines": 195,
  "italian_recipes": 21,
  "example": {
    "name": "Budino Di Ricotta",
    "category": "Dessert",
    "ingredients": [
      "500g Ricotta",
      "4 large Eggs",
      "3 tbs Flour",
      "250g Sugar",
      "1 tsp Cinnamon",
      "Grated Zest of 2 Lemons",
      "... (2 more)"
    ],
    "instructions": "Mash the ricotta and beat well with the egg yolks, stir in the flour, sugar, cinnamon, grated lemon rind and the rum and mix well. You can do this in a food processor. Beat the egg whites until stiff,..."
  }
}
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

### Coursera Catalog API

| | |
|:--|:--|
| **Docs** | https://api.coursera.org/api/courses.v1 (the old docs site build.coursera.org no longer resolves) |
| **Auth** | None |
| **Register / token** | No registration for the public catalog. Keyword search (q=search) is no longer accepted (HTTP 405), so page through the catalog and filter locally. |
| **Script** | [`scripts/education/coursera_catalog.py`](scripts/education/coursera_catalog.py) |

**What you can get:**

- Every Coursera course: name, slug, description, workload, languages, domain/subdomain tags
- Partner (university/company) and instructor IDs, specializations

**Capstone ideas:** Mapping the online-course supply by skill domain, comparing course topics with job-posting skills (see Jobs).

**Endpoint:** `GET https://api.coursera.org/api/courses.v1`

| Parameter | Description | Example |
|:--|:--|:--|
| `fields` | Extra fields: description, workload, primaryLanguages, domainTypes, partnerIds, instructorIds | `name,domainTypes,workload` |
| `start / limit` | Offset and page size (paging.next gives the next start) | `0 / 100` |
| `ids` | Look up specific course IDs | `—` |
| `/api/partners.v1 · /api/instructors.v1` | Sibling endpoints: partner institutions and instructors | `fields=name,location` |

```python
from collections import Counter

from _common import get, show


def main():
    d = get("https://api.coursera.org/api/courses.v1", params={"fields": "name,domainTypes,workload", "limit": 100})
    domains = Counter(t["domainId"] for c in d["elements"] for t in c.get("domainTypes", []))
    show({"total_courses": d["paging"].get("total"), "domains_in_first_100": domains.most_common(6),
          "examples": [{"name": c["name"], "workload": c.get("workload"),
                        "subdomains": [t["subdomainId"] for t in c.get("domainTypes", [])]} for c in d["elements"][:3]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:53</summary>

```json
{
  "total_courses": 24842,
  "domains_in_first_100": [
    [
      "business",
      45
    ],
    [
      "information-technology",
      38
    ],
    [
      "computer-science",
      37
    ],
    [
      "data-science",
      21
    ],
    [
      "physical-science-and-engineering",
      11
    ],
    "... (1 more)"
  ],
  "examples": [
    {
      "name": "Getting started with the Vertex AI Gemini 1.5 Pro Model",
      "workload": "1 hour 30 minutes",
      "subdomains": [
        "cloud-computing",
        "machine-learning"
      ]
    },
    {
      "name": "SPSS: Apply & Interpret Logistic Regression Models",
      "workload": "",
      "subdomains": [
        "data-analysis"
      ]
    },
    {
      "name": "Securing Compute Engine Applications and Resources using Chrome Enterprise Premium",
      "workload": "1 hour 30 minutes",
      "subdomains": [
        "cloud-computing",
        "computer-security-and-networks"
      ]
    }
  ]
}
```
</details>

---

### Datamuse (word-finding API)

| | |
|:--|:--|
| **Docs** | https://www.datamuse.com/api/ |
| **Auth** | None |
| **Register / token** | No registration. Up to 100,000 requests/day. |
| **Script** | [`scripts/education/datamuse.py`](scripts/education/datamuse.py) |

**What you can get:**

- Words by meaning (reverse dictionary), sound, spelling pattern, rhyme, or topic
- Word associations, frequency, syllable counts, parts of speech
- Autocomplete suggestions (/sug)

**Capstone ideas:** Vocabulary and reading-level tools, query expansion for search, word-game or language-learning apps.

**Endpoint:** `GET https://api.datamuse.com/words`

| Parameter | Description | Example |
|:--|:--|:--|
| `ml` | Means like (reverse dictionary) | `a person who studies data` |
| `sl / sp` | Sounds like / spelled like (wildcards * and ?) | `sp=stat*` |
| `rel_trg / rel_syn / rel_rhy` | Triggers (associated words) / synonyms / rhymes | `rel_trg=regression` |
| `topics` | Bias results toward a topic | `statistics` |
| `md` | Metadata flags: d definitions, p parts of speech, s syllables, f frequency | `dpf` |
| `max` | Max results (up to 1,000) | `5` |
| `/sug?s=` | Sibling endpoint: autocomplete | `s=statis` |

```python
from _common import get, show


def main():
    reverse = get("https://api.datamuse.com/words", params={"ml": "a person who studies data", "md": "p", "max": 5})
    assoc = get("https://api.datamuse.com/words", params={"rel_trg": "regression", "max": 8})
    show({"means_like_'a person who studies data'": [w["word"] for w in reverse],
          "words_associated_with_'regression'": [w["word"] for w in assoc]}, max_items=8)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:53</summary>

```json
{
  "means_like_'a person who studies data'": [
    "analyst",
    "statistician",
    "economist",
    "history",
    "epistemologist"
  ],
  "words_associated_with_'regression'": [
    "probit",
    "logit",
    "quantile",
    "anova",
    "regularization",
    "predictor",
    "logistic",
    "multivariate"
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

### OECD Data API — Education at a Glance

| | |
|:--|:--|
| **Docs** | https://gitlab.algobank.oecd.org/public-documentation/dotstat-migration/-/raw/main/OECD_Data_API_documentation.pdf |
| **Auth** | None |
| **Register / token** | No registration. SDMX REST API over all OECD datasets (~1,500 dataflows). Find dataflow IDs and dimension codes in the OECD Data Explorer (https://data-explorer.oecd.org) -> Developer API. Limit: 60 data queries/hour. |
| **Script** | [`scripts/education/oecd_education.py`](scripts/education/oecd_education.py) |

**What you can get:**

- Education at a Glance: attainment, enrollment, spending per student, teachers' salaries, class sizes
- Labour-market outcomes by education: earnings premiums, employment rates, NEET rates, by field of study
- The same API serves every OECD topic (economy, health, environment, PISA-related indicators)

**Capstone ideas:** Returns to education across countries, education spending vs. outcomes, gender gaps in earnings by degree.

**Endpoint:** `GET https://sdmx.oecd.org/public/rest/data/{agency},{dataflow},{version}/{key}`

| Parameter | Description | Example |
|:--|:--|:--|
| `agency,dataflow,version` | Dataflow identifier from the Data Explorer | `OECD.EDU.IMEP,DSD_EAG_LSO_EA@DF_LSO_EARN_REL_UPPER,1.0` |
| `key` | One slot per dimension, separated by dots; + for several values, empty = all. Every slot must be present (a short key returns 403) | `USA+FIN._T.Y25T64.ISCED11A_6...........` |
| `startPeriod / endPeriod` | Year range | `2021` |
| `format` | csvfilewithlabels \| csvfile \| jsondata \| genericdata | `csvfilewithlabels` |
| `dimensionAtObservation` | AllDimensions for flat JSON observations | `AllDimensions` |
| `/public/rest/dataflow/all` | Catalog of all dataflows (send Accept: application/vnd.sdmx.structure+json) | `—` |

```python
import csv
import io

from _common import get, show

DIMS = ["REF_AREA", "SEX", "AGE", "ATTAINMENT_LEV", "EDUCATION_FIELD", "MEASURE", "INCOME", "BIRTH_PLACE", "MIGRATION_AGE",
        "EDU_STATUS", "LABOUR_FORCE_STATUS", "DURATION_UNEMP", "UNIT_MEASURE", "STATISTICAL_OPERATION",
        "WORK_TIME_ARNGMNT", "QUESTIONNAIRE", "FREQ"]


def main():
    want = {"REF_AREA": "USA+DEU+FIN+KOR+GBR+CAN", "SEX": "_T", "AGE": "Y25T64",
            "ATTAINMENT_LEV": "ISCED11A_6+ISCED11A_7_8", "WORK_TIME_ARNGMNT": "FT"}
    key = ".".join(want.get(d, "") for d in DIMS)  # every dimension needs a slot, even if empty
    text = get(f"https://sdmx.oecd.org/public/rest/data/OECD.EDU.IMEP,DSD_EAG_LSO_EA@DF_LSO_EARN_REL_UPPER,1.0/{key}",
               params={"startPeriod": 2021, "format": "csvfilewithlabels"}, as_json=False)
    latest = {}
    for r in csv.DictReader(io.StringIO(text.lstrip("﻿"))):
        k = (r["Reference area"], r["Educational attainment level"])
        if k not in latest or r["TIME_PERIOD"] > latest[k]["TIME_PERIOD"]:
            latest[k] = r
    show({"measure": "Earnings of full-time workers aged 25-64, % of upper-secondary graduates' earnings", "data": [
        {"country": c, "level": lvl, "year": r["TIME_PERIOD"], "relative_earnings_pct": round(float(r["OBS_VALUE"]))}
        for (c, lvl), r in sorted(latest.items())]}, max_items=12)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:53</summary>

```json
{
  "measure": "Earnings of full-time workers aged 25-64, % of upper-secondary graduates' earnings",
  "data": [
    {
      "country": "Canada",
      "level": "Bachelor's or equivalent level",
      "year": "2023",
      "relative_earnings_pct": 148
    },
    {
      "country": "Canada",
      "level": "Master's, doctoral or equivalent level",
      "year": "2023",
      "relative_earnings_pct": 172
    },
    {
      "country": "Finland",
      "level": "Bachelor's or equivalent level",
      "year": "2023",
      "relative_earnings_pct": 122
    },
    {
      "country": "Finland",
      "level": "Master's, doctoral or equivalent level",
      "year": "2023",
      "relative_earnings_pct": 161
    },
    {
      "country": "Germany",
      "level": "Bachelor's or equivalent level",
      "year": "2024",
      "relative_earnings_pct": 148
    },
    {
      "country": "Germany",
      "level": "Master's, doctoral or equivalent level",
      "year": "2024",
      "relative_earnings_pct": 175
    },
    {
      "country": "Korea",
      "level": "Bachelor's or equivalent level",
      "year": "2024",
      "relative_earnings_pct": 132
    },
    {
      "country": "Korea",
      "level": "Master's, doctoral or equivalent level",
      "year": "2024",
      "relative_earnings_pct": 173
    },
    {
      "country": "United Kingdom",
      "level": "Bachelor's or equivalent level",
      "year": "2024",
      "relative_earnings_pct": 133
    },
    {
      "country": "United Kingdom",
      "level": "Master's, doctoral or equivalent level",
      "year": "2024",
      "relative_earnings_pct": 149
    },
    {
      "country": "United States",
      "level": "Bachelor's or equivalent level",
      "year": "2024",
      "relative_earnings_pct": 165
    },
    {
      "country": "United States",
      "level": "Master's, doctoral or equivalent level",
      "year": "2024",
      "relative_earnings_pct": 214
    }
  ]
}
```
</details>

---

### Open Trivia Database

| | |
|:--|:--|
| **Docs** | https://opentdb.com/api_config.php |
| **Auth** | None |
| **Register / token** | No registration. One request per 5 seconds per IP. CC BY-SA 4.0. Use a session token to avoid repeat questions. |
| **Script** | [`scripts/education/open_trivia_db.py`](scripts/education/open_trivia_db.py) |

**What you can get:**

- 4,000+ verified multiple-choice and true/false questions in 24 categories with difficulty levels
- Category list and per-category question counts

**Capstone ideas:** Quiz/edtech app prototypes, evaluating LLMs on trivia by difficulty, question-difficulty prediction.

**Endpoint:** `GET https://opentdb.com/api.php`

| Parameter | Description | Example |
|:--|:--|:--|
| `amount` | Number of questions (max 50) | `5` |
| `category` | Category ID (list via /api_category.php), e.g. 17 Science & Nature, 18 Computers, 19 Mathematics | `18` |
| `difficulty` | easy \| medium \| hard | `medium` |
| `type` | multiple \| boolean | `multiple` |
| `encode` | url3986 \| base64 (default is HTML entities) | `url3986` |
| `token` | Session token from /api_token.php?command=request | `—` |

```python
import html

from _common import get, show


def main():
    cats = get("https://opentdb.com/api_category.php")["trivia_categories"]
    d = get("https://opentdb.com/api.php", params={"amount": 3, "category": 18, "difficulty": "medium", "type": "multiple"})
    show({"categories": len(cats), "computer_science_questions": [
        {"question": html.unescape(q["question"]), "answer": html.unescape(q["correct_answer"]),
         "distractors": [html.unescape(a) for a in q["incorrect_answers"]]} for q in d["results"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:53</summary>

```json
{
  "categories": 24,
  "computer_science_questions": [
    {
      "question": "What does the 'S' in the RSA encryption algorithm stand for?",
      "answer": "Shamir",
      "distractors": [
        "Secure",
        "Schottky",
        "Stable"
      ]
    },
    {
      "question": "Generally, which component of a computer draws the most power?",
      "answer": "Video Card",
      "distractors": [
        "Hard Drive",
        "Processor",
        "Power Supply"
      ]
    },
    {
      "question": "In the programming language \"Python\", which of these statements would display the string \"Hello World\" correctly?",
      "answer": "print(\"Hello World\")",
      "distractors": [
        "console.log(\"Hello World\")",
        "echo \"Hello World\"",
        "printf(\"Hello World\")"
      ]
    }
  ]
}
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

### World Bank EdStats (education statistics)

| | |
|:--|:--|
| **Docs** | https://datatopics.worldbank.org/education/ |
| **Auth** | None |
| **Register / token** | No registration. Same API as the World Bank Indicators entry, but source=12 selects the Education Statistics database (~4,000 indicators, including learning outcomes and projections). Last refreshed June 2024. |
| **Script** | [`scripts/education/world_bank_edstats.py`](scripts/education/world_bank_edstats.py) |

**What you can get:**

- Enrollment, completion, out-of-school rates, pupil-teacher ratios, education spending for every country
- Learning outcomes from international assessments (PISA, TIMSS, PIRLS) and learning poverty
- Attainment projections to 2050

**Capstone ideas:** Cross-country education panels, learning poverty vs. income, school spending efficiency.

**Endpoint:** `GET https://api.worldbank.org/v2/country/{codes}/indicator/{indicator}?source=12`

| Parameter | Description | Example |
|:--|:--|:--|
| `source` | 12 = Education Statistics (EdStats) | `12` |
| `indicator` | EdStats indicator code (browse via /v2/sources/12/indicators) | `SE.TER.ENRR (tertiary gross enrollment)` |
| `country codes` | ISO3 codes separated by ;, or all | `USA;FIN;KOR;BRA;IND` |
| `date / mrv` | Year range, or most recent N values (EdStats is sparse - a range plus local filtering works best) | `2010:2023` |
| `format / per_page` | json \| xml; page size | `json / 100` |

```python
from _common import get, show

INDICATORS = {"SE.TER.ENRR": "tertiary_gross_enrollment_pct", "SE.SEC.ENRR": "secondary_gross_enrollment_pct",
              "SE.XPD.TOTL.GD.ZS": "education_spending_pct_GDP"}


def main():
    out = {}
    for code, label in INDICATORS.items():
        _, rows = get(f"https://api.worldbank.org/v2/country/USA;FIN;KOR;BRA;IND/indicator/{code}",
                      params={"source": 12, "format": "json", "date": "2010:2023", "per_page": 500})
        # EdStats is sparse: walk oldest -> newest and keep each country's latest non-empty value
        for r in sorted(rows or [], key=lambda r: r["date"]):
            if r["value"] is not None:
                out.setdefault(r["country"]["value"], {})[label] = f'{round(r["value"], 1)} ({r["date"]})'
    show(out)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:53</summary>

```json
{
  "Finland": {
    "tertiary_gross_enrollment_pct": "90.3 (2018)",
    "secondary_gross_enrollment_pct": "154.8 (2018)",
    "education_spending_pct_GDP": "6.4 (2017)"
  },
  "India": {
    "tertiary_gross_enrollment_pct": "28.6 (2019)",
    "secondary_gross_enrollment_pct": "73.8 (2019)",
    "education_spending_pct_GDP": "3.8 (2013)"
  },
  "Korea, Rep.": {
    "tertiary_gross_enrollment_pct": "95.9 (2018)",
    "secondary_gross_enrollment_pct": "98.5 (2018)",
    "education_spending_pct_GDP": "4.3 (2016)"
  },
  "United States": {
    "tertiary_gross_enrollment_pct": "88.3 (2018)",
    "secondary_gross_enrollment_pct": "99.3 (2018)",
    "education_spending_pct_GDP": "5.0 (2014)"
  },
  "Brazil": {
    "tertiary_gross_enrollment_pct": "43.5 (2011)",
    "secondary_gross_enrollment_pct": "95.3 (2011)",
    "education_spending_pct_GDP": "6.3 (2017)"
  }
}
```
</details>

---

## 💼 Jobs & Labor

### Adzuna Jobs API

| | |
|:--|:--|
| **Docs** | https://developer.adzuna.com/docs/search |
| **Auth** | app_id + app_key (query parameters) |
| **Register / token** | Register at https://developer.adzuna.com/signup for an app ID and key. Free tier ~250 requests/day; attribution to Adzuna required. |
| **Env var** | `ADZUNA_APP_ID, ADZUNA_APP_KEY` |
| **Script** | [`scripts/jobs/adzuna.py`](scripts/jobs/adzuna.py) |

**What you can get:**

- Millions of live job ads in 19 countries aggregated from many boards: title, company, location, salary, contract type
- Salary histograms and averages by keyword/location, top companies, regional job counts, salary history

**Capstone ideas:** Real-time labor-demand indices, salary prediction from job text, remote-work share by occupation.

**Endpoint:** `GET https://api.adzuna.com/v1/api/jobs/{country}/{search/{page} \| histogram \| top_companies \| history \| geodata}`

| Parameter | Description | Example |
|:--|:--|:--|
| `country` | us \| gb \| ca \| de \| fr \| au \| in \| ... | `us` |
| `what / what_exclude / title_only` | Keywords | `data scientist` |
| `where / distance` | Location text, radius in km | `Washington DC / 25` |
| `salary_min / full_time / permanent` | Filters | `100000` |
| `sort_by` | relevance \| date \| salary | `date` |
| `results_per_page` | Max 50 | `5` |
| `app_id / app_key` | Your credentials | `—` |

```python
from _common import get, key, show


def main():
    d = get("https://api.adzuna.com/v1/api/jobs/us/search/1", params={
        "app_id": key("ADZUNA_APP_ID"), "app_key": key("ADZUNA_APP_KEY"), "what": "data scientist",
        "where": "Washington DC", "results_per_page": 5, "sort_by": "date"})
    show({"count": d["count"], "mean_salary": d.get("mean"), "jobs": [
        {"title": j["title"], "company": j["company"]["display_name"], "location": j["location"]["display_name"],
         "salary_min": j.get("salary_min"), "salary_max": j.get("salary_max"), "created": j["created"]} for j in d["results"]]})
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "count": 2147,
  "mean_salary": 141250.3,
  "jobs": [
    {"title": "Senior Data Scientist", "company": "Booz Allen Hamilton", "location": "Washington, DC",
     "salary_min": 120000, "salary_max": 170000, "created": "2026-10-10T14:22:31Z"}
  ]
}
```
</details>

---

### Ashby Job Postings API (startup career pages)

| | |
|:--|:--|
| **Docs** | https://developers.ashbyhq.com/docs/public-job-posting-api |
| **Auth** | None |
| **Register / token** | No registration. Many startups (Ramp, Notion, Linear, ...) host careers pages on Ashby at jobs.ashbyhq.com/{org}. Compensation data is unusually complete because many postings include salary ranges. |
| **Script** | [`scripts/jobs/ashby_job_board.py`](scripts/jobs/ashby_job_board.py) |

**What you can get:**

- Open jobs with department, team, location(s), employment type, remote flag, publish date
- Compensation summaries (salary ranges, equity) when includeCompensation=true
- Full HTML/plain descriptions

**Capstone ideas:** Startup salary benchmarking, remote vs. on-site pay gaps, equity compensation patterns.

**Endpoint:** `GET https://api.ashbyhq.com/posting-api/job-board/{organization}`

| Parameter | Description | Example |
|:--|:--|:--|
| `organization` | Job-board name from jobs.ashbyhq.com/{organization} | `ramp` |
| `includeCompensation` | true -> add compensation tiers and summary | `true` |
| `response fields` | title, department, team, location, isRemote, employmentType, publishedAt, compensation | `—` |

```python
from collections import Counter

from _common import get, show


def main():
    jobs = get("https://api.ashbyhq.com/posting-api/job-board/ramp", params={"includeCompensation": "true"})["jobs"]
    show({"company": "Ramp", "open_jobs": len(jobs),
          "by_department": Counter(j.get("department") for j in jobs).most_common(5),
          "remote_share": round(sum(bool(j.get("isRemote")) for j in jobs) / len(jobs), 2),
          "with_salary": [{"title": j["title"].strip(), "location": j.get("location"),
                           "compensation": (j.get("compensation") or {}).get("compensationTierSummary")}
                          for j in jobs if (j.get("compensation") or {}).get("compensationTierSummary")][:4]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:53</summary>

```json
{
  "company": "Ramp",
  "open_jobs": 166,
  "by_department": [
    [
      "Sales",
      67
    ],
    [
      "Engineering",
      38
    ],
    [
      "People & Talent",
      14
    ],
    [
      "Product",
      9
    ],
    [
      "Marketing",
      7
    ]
  ],
  "remote_share": 0.9,
  "with_salary": [
    {
      "title": "Security Engineer, Cloud",
      "location": "New York, NY (HQ)",
      "compensation": "$211.4K – $290.6K • Offers Equity"
    },
    {
      "title": "Mobile Engineer, Android",
      "location": "New York, NY (HQ)",
      "compensation": "$189K – $330K • Offers Equity"
    },
    {
      "title": "Software Engineer, Frontend",
      "location": "New York, NY (HQ)",
      "compensation": "$143.2K – $284K • Offers Equity"
    },
    {
      "title": "Credit Risk Associate",
      "location": "New York, NY (HQ)",
      "compensation": "$108K – $200K • Offers Equity"
    }
  ]
}
```
</details>

---

### CareerOneStop Web API (U.S. Department of Labor)

| | |
|:--|:--|
| **Docs** | https://www.careeronestop.org/Developers/WebAPI/web-api.aspx |
| **Auth** | User ID in the path + Bearer token |
| **Register / token** | Register at https://www.careeronestop.org/Developers/WebAPI/registration.aspx; you receive a user ID and a token by email. Free. |
| **Env var** | `CAREERONESTOP_USER_ID, CAREERONESTOP_TOKEN` |
| **Script** | [`scripts/jobs/careeronestop.py`](scripts/jobs/careeronestop.py) |

**What you can get:**

- Occupation profiles with local wages (percentiles by state/metro), employment outlook, education required
- Training programs, certifications and licenses by occupation and location
- American Job Center locations, apprenticeship sponsors, skills-matcher

**Capstone ideas:** Local wage comparisons, training ROI by occupation, workforce-development program mapping.

**Endpoint:** `GET https://api.careeronestop.org/v1/{service}/{userId}/...`

| Parameter | Description | Example |
|:--|:--|:--|
| `occupation/{userId}/{keyword}/{dataLevelOnly}/{start}/{limit}` | Search occupations by keyword (dataLevelOnly N/Y) | `data scientist/N/0/5` |
| `occupation/{userId}/{onetCode}/{location}?wages=true&projectedEmployment=true` | Occupation details for a location | `15-2051.00/DC` |
| `training/{userId}/{keyword}/{location}/{radius}/...` | Training programs near a location | `—` |
| `Authorization (header)` | Bearer token | `Bearer <CAREERONESTOP_TOKEN>` |

```python
from urllib.parse import quote

from _common import get, key, show


def main():
    user, token = key("CAREERONESTOP_USER_ID"), key("CAREERONESTOP_TOKEN")
    d = get(f"https://api.careeronestop.org/v1/occupation/{user}/{quote('data scientist')}/N/0/5",
            headers={"Authorization": f"Bearer {token}"})
    show([{k: o.get(k) for k in ("OnetCode", "OnetTitle", "BrightOutlook")} for o in d["OccupationList"]])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"OnetCode": "15-2051.00", "OnetTitle": "Data Scientists", "BrightOutlook": "Bright"},
  {"OnetCode": "15-2041.00", "OnetTitle": "Statisticians", "BrightOutlook": "Bright"}
]
```
</details>

---

### ESCO API (European skills & occupations taxonomy)

| | |
|:--|:--|
| **Docs** | https://esco.ec.europa.eu/en/use-esco/use-esco-services-api |
| **Auth** | None |
| **Register / token** | No registration. The EU's multilingual classification (28 languages) of ~3,000 occupations and ~14,000 skills, mapped to ISCO-08. Complements O*NET for international work. |
| **Script** | [`scripts/jobs/esco.py`](scripts/jobs/esco.py) |

**What you can get:**

- Occupations with descriptions, ISCO codes, alternative labels in 28 languages
- Essential and optional skills/knowledge for each occupation, skill hierarchies and relations
- Search across occupations and skills

**Capstone ideas:** Skill extraction from job ads (map phrases to ESCO skills), cross-country occupation matching, skill-gap analysis.

**Endpoint:** `GET https://ec.europa.eu/esco/api/{search \| resource/occupation \| resource/skill}`

| Parameter | Description | Example |
|:--|:--|:--|
| `text` | Search text | `data scientist` |
| `type` | occupation \| skill \| concept | `occupation` |
| `language` | Label language (en, de, fr, es, ...) | `en` |
| `limit / offset` | Paging | `5` |
| `resource/occupation?uri=` | Full record incl. hasEssentialSkill / hasOptionalSkill links | `uri=http://data.europa.eu/esco/occupation/...` |

```python
from _common import get, show

BASE = "https://ec.europa.eu/esco/api"


def main():
    hits = get(f"{BASE}/search", params={"text": "data scientist", "type": "occupation", "language": "en", "limit": 3})
    top = hits["_embedded"]["results"][0]
    occ = get(f"{BASE}/resource/occupation", params={"uri": top["uri"], "language": "en"})
    skills = occ["_links"].get("hasEssentialSkill", [])
    show({"matches": hits["total"], "top_match": occ["preferredLabel"]["en"], "isco_code": occ.get("code"),
          "essential_skills": len(skills), "examples": [s["title"] for s in skills[:8]]}, max_items=8)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:53</summary>

```json
{
  "matches": 188,
  "top_match": "data scientist",
  "isco_code": "2511.3",
  "essential_skills": 23,
  "examples": [
    "perform data cleansing",
    "collect ICT data",
    "handle data samples",
    "data mining",
    "information extraction",
    "data models",
    "resource description framework query language",
    "visual presentation techniques"
  ]
}
```
</details>

---

### Greenhouse Job Board API (company career pages)

| | |
|:--|:--|
| **Docs** | https://docs.greenhouse.io/job-board.html |
| **Auth** | None |
| **Register / token** | No registration. Thousands of companies (Airbnb, Stripe, Pinterest, ...) host their careers page on Greenhouse; the board token is the company slug in their careers URL (boards.greenhouse.io/{token}). |
| **Script** | [`scripts/jobs/greenhouse_job_boards.py`](scripts/jobs/greenhouse_job_boards.py) |

**What you can get:**

- Every open job at a company: title, location, department, office, last updated, full description (content=true)
- Departments and offices with their jobs
- Custom metadata fields some companies publish (e.g. remote eligibility)

**Capstone ideas:** Tracking hiring by company/department over time, skill extraction from descriptions, hiring freezes as a signal.

**Endpoint:** `GET https://boards-api.greenhouse.io/v1/boards/{board_token}/{jobs \| departments \| offices}`

| Parameter | Description | Example |
|:--|:--|:--|
| `board_token` | Company slug | `airbnb, stripe, pinterest` |
| `jobs?content=true` | Include the HTML job description | `content=true` |
| `jobs/{id}?questions=true` | One job, with application questions | `—` |
| `departments / offices` | Jobs grouped by department or office | `departments` |

```python
from collections import Counter

from _common import get, show


def main():
    jobs = get("https://boards-api.greenhouse.io/v1/boards/airbnb/jobs")["jobs"]
    depts = get("https://boards-api.greenhouse.io/v1/boards/airbnb/departments")["departments"]
    show({"company": "Airbnb", "open_jobs": len(jobs),
          "jobs_by_department": sorted(((d["name"], len(d["jobs"])) for d in depts if d["jobs"]), key=lambda x: -x[1])[:5],
          "top_locations": Counter(j["location"]["name"] for j in jobs).most_common(4),
          "latest": [{"title": j["title"], "location": j["location"]["name"], "updated": j["updated_at"][:10]}
                     for j in sorted(jobs, key=lambda j: j["updated_at"], reverse=True)[:3]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:53</summary>

```json
{
  "company": "Airbnb",
  "open_jobs": 159,
  "jobs_by_department": [
    [
      "Software Engineering",
      41
    ],
    [
      "Community Support",
      18
    ],
    [
      "Business Development",
      14
    ],
    [
      "Product Management",
      14
    ],
    [
      "Business Operations/Analysis",
      8
    ]
  ],
  "top_locations": [
    [
      "United States",
      36
    ],
    [
      "United States ",
      22
    ],
    [
      "Canada",
      12
    ],
    [
      "Remote - USA",
      8
    ]
  ],
  "latest": [
    {
      "title": "Market Manager, UK",
      "location": "London, United Kingdom",
      "updated": "2026-10-12"
    },
    {
      "title": "Support Specialist, R1",
      "location": "Gurugram, India",
      "updated": "2026-10-11"
    },
    {
      "title": "Staff Software Engineer, Streaming Infrastructure",
      "location": "United States",
      "updated": "2026-10-09"
    }
  ]
}
```
</details>

---

### Hacker News 'Who is hiring?' (via Algolia HN Search)

| | |
|:--|:--|
| **Docs** | https://hn.algolia.com/api |
| **Auth** | None |
| **Register / token** | No registration. A monthly thread since 2011 where tech companies post jobs as comments - a long-running, free dataset of tech hiring. |
| **Script** | [`scripts/jobs/hn_whos_hiring.py`](scripts/jobs/hn_whos_hiring.py) |

**What you can get:**

- Every 'Ask HN: Who is hiring?' thread (posted by user whoishiring) with its date
- All comments (job posts) per thread: company, role, location, REMOTE/ONSITE, salary text, tech stack
- Full-text search across HN stories and comments with date filters

**Capstone ideas:** Tech-hiring cycle index (posts per month), remote share over time, programming-language demand trends.

**Endpoint:** `GET https://hn.algolia.com/api/v1/{search \| search_by_date \| items/{id}}`

| Parameter | Description | Example |
|:--|:--|:--|
| `query` | Search text | `"Who is hiring"` |
| `tags` | story \| comment \| author_{user} \| story_{id} (comma = AND) | `story,author_whoishiring` |
| `numericFilters` | created_at_i>..., points>... | `created_at_i>1735689600` |
| `hitsPerPage / page` | Max 1,000 per page | `12` |
| `tags=comment,story_{id}` | All top-level job posts in one thread | `comment,story_45000000` |
| `/items/{id}` | A thread with its full comment tree | `—` |

```python
import datetime as dt

from _common import get, show

BASE = "https://hn.algolia.com/api/v1"


def main():
    threads = get(f"{BASE}/search_by_date", params={"query": '"Who is hiring"', "tags": "story,author_whoishiring",
                                                     "hitsPerPage": 12})["hits"]
    threads = [t for t in threads if t["title"].startswith("Ask HN: Who is hiring?")][:6]
    latest = threads[0]
    posts = get(f"{BASE}/search", params={"tags": f"comment,story_{latest['objectID']}", "hitsPerPage": 1000})["hits"]
    top = [p for p in posts if p.get("parent_id") == int(latest["objectID"])]  # top-level comments = job posts
    show({"recent_threads": [{"title": t["title"], "comments": t["num_comments"]} for t in threads],
          "latest_thread_job_posts": len(top),
          "share_mentioning_REMOTE": round(sum("remote" in (p.get("comment_text") or "").lower() for p in top) / max(len(top), 1), 2),
          "posted": dt.datetime.fromtimestamp(latest["created_at_i"]).date().isoformat()})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:53</summary>

```json
{
  "recent_threads": [
    {
      "title": "Ask HN: Who is hiring? (October 2026)",
      "comments": 375
    },
    {
      "title": "Ask HN: Who is hiring? (September 2026)",
      "comments": 393
    },
    {
      "title": "Ask HN: Who is hiring? (August 2026)",
      "comments": 376
    },
    {
      "title": "Ask HN: Who is hiring? (July 2026)",
      "comments": 419
    },
    {
      "title": "Ask HN: Who is hiring? (June 2026)",
      "comments": 492
    },
    "... (1 more)"
  ],
  "latest_thread_job_posts": 254,
  "share_mentioning_REMOTE": 0.56,
  "posted": "2026-10-01"
}
```
</details>

---

### ILOSTAT (International Labour Organization)

| | |
|:--|:--|
| **Docs** | https://ilostat.ilo.org/data/api/ |
| **Auth** | None |
| **Register / token** | No registration. The ILO's global labor statistics: harmonized estimates for ~190 countries. Two APIs: an SDMX endpoint (sdmx.ilo.org) and a simpler R/plumber JSON endpoint (used here). |
| **Script** | [`scripts/jobs/ilostat.py`](scripts/jobs/ilostat.py) |

**What you can get:**

- Unemployment, employment, labor-force participation by sex, age, education
- Wages, working hours, informality, child labour, occupational injuries, labor income share
- Modelled estimates that fill gaps across countries and years

**Capstone ideas:** Cross-country labor-market comparisons, youth unemployment drivers, informality vs. development.

**Endpoint:** `GET https://rplumber.ilo.org/data/indicator/`

| Parameter | Description | Example |
|:--|:--|:--|
| `id` | Indicator + frequency, e.g. UNE_DEAP_SEX_AGE_RT_A (unemployment rate, annual) | `UNE_DEAP_SEX_AGE_RT_A` |
| `ref_area` | ISO3 country code(s), + separated | `USA+DEU+ZAF+IND` |
| `sex / classif1` | SEX_T \| SEX_M \| SEX_F; age/education breakdowns e.g. AGE_YTHADULT_Y15-24 | `SEX_T` |
| `timefrom / timeto` | Year range | `2015 / 2025` |
| `format` | .json \| .csv | `.json` |
| `sdmx.ilo.org/rest/data/ILO,{dataflow}/{key}` | SDMX alternative | `ILO,DF_UNE_DEAP_SEX_AGE_RT/USA.A..SEX_T.AGE_YTHADULT_YGE15` |

```python
from _common import get, show


def main():
    d = get("https://rplumber.ilo.org/data/indicator/", params={
        "id": "UNE_DEAP_SEX_AGE_RT_A", "ref_area": "USA+DEU+ZAF+IND+BRA", "sex": "SEX_T", "timefrom": 2024,
        "format": ".json"})
    rows = [r for r in d if r["classif1"] in ("AGE_YTHADULT_YGE15", "AGE_YTHADULT_Y15-24")]
    out = {}
    for r in sorted(rows, key=lambda r: r["time"]):
        label = "youth_15_24" if r["classif1"].endswith("Y15-24") else "all_15plus"
        out.setdefault(r["ref_area"], {})[label] = f'{round(r["obs_value"], 1)}% ({r["time"]})'
    show({"indicator": "Unemployment rate", "by_country": out})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:53</summary>

```json
{
  "indicator": "Unemployment rate",
  "by_country": {
    "BRA": {
      "all_15plus": "5.8% (2025)",
      "youth_15_24": "13.6% (2025)"
    },
    "DEU": {
      "all_15plus": "3.8% (2025)",
      "youth_15_24": "7.1% (2025)"
    },
    "IND": {
      "all_15plus": "4.5% (2025)",
      "youth_15_24": "17.7% (2025)"
    },
    "USA": {
      "all_15plus": "4.3% (2025)",
      "youth_15_24": "10.0% (2025)"
    },
    "ZAF": {
      "all_15plus": "29.6% (2025)",
      "youth_15_24": "53.2% (2025)"
    }
  }
}
```
</details>

---

### Census LEHD LODES (jobs by census block)

| | |
|:--|:--|
| **Docs** | https://lehd.ces.census.gov/data/lodes/LODES8/LODESTechDoc8.3.pdf |
| **Auth** | None (CSV downloads) |
| **Register / token** | No registration; no REST API - gzipped CSVs per state, year and file type. Block GEOIDs join directly to Census tracts (first 11 digits) and the Census Geocoder output. |
| **Script** | [`scripts/jobs/lehd_lodes.py`](scripts/jobs/lehd_lodes.py) |

**What you can get:**

- WAC: jobs by workplace census block, split by age, earnings band, industry (NAICS sector), race, education
- RAC: the same counts by where workers live
- OD: origin-destination commuting flows between home and work blocks

**Capstone ideas:** Job accessibility (with OSRM travel times), commuting networks, spatial mismatch between homes and jobs.

**Endpoint:** `GET https://lehd.ces.census.gov/data/lodes/LODES8/{state}/{wac \| rac \| od}/{file}.csv.gz`

| Parameter | Description | Example |
|:--|:--|:--|
| `state` | Two-letter state code, lower case | `dc` |
| `file name` | {state}_{wac\|rac}_{segment}_{jobtype}_{year}.csv.gz | `dc_wac_S000_JT00_2022.csv.gz` |
| `segment` | S000 all workers, SA01-03 age bands, SE01-03 earnings bands, SI01-03 industry groups | `S000` |
| `job type` | JT00 all jobs, JT01 primary jobs, JT02 private, ... | `JT00` |
| `columns` | w_geocode, C000 total jobs, CNS01-CNS20 jobs by NAICS sector, CE01-03 earnings bands, ... | `—` |
| `OD files` | {state}_od_{main\|aux}_{jobtype}_{year}.csv.gz: w_geocode, h_geocode, S000 | `dc_od_main_JT00_2022.csv.gz` |

```python
import csv
import gzip
import io

import requests

from _common import USER_AGENT, show

URL = "https://lehd.ces.census.gov/data/lodes/LODES8/dc/wac/dc_wac_S000_JT00_2022.csv.gz"
SECTORS = {"CNS12": "Professional/scientific/technical", "CNS20": "Public administration", "CNS16": "Health care",
           "CNS15": "Educational services", "CNS18": "Accommodation & food"}


def main():
    raw = requests.get(URL, headers={"User-Agent": USER_AGENT}, timeout=120).content
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(raw).decode())))
    total = sum(int(r["C000"]) for r in rows)
    tracts = {}
    for r in rows:  # block GEOID -> tract GEOID = first 11 digits
        tracts[r["w_geocode"][:11]] = tracts.get(r["w_geocode"][:11], 0) + int(r["C000"])
    show({"DC_jobs_2022": total, "blocks_with_jobs": len(rows),
          "jobs_by_sector": {name: sum(int(r[c]) for r in rows) for c, name in SECTORS.items()},
          "top_tracts_by_jobs": sorted(tracts.items(), key=lambda kv: -kv[1])[:3]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:53</summary>

```json
{
  "DC_jobs_2022": 656985,
  "blocks_with_jobs": 3991,
  "jobs_by_sector": {
    "Professional/scientific/technical": 123488,
    "Public administration": 119176,
    "Health care": 71852,
    "Educational services": 51368,
    "Accommodation & food": 55310
  },
  "top_tracts_by_jobs": [
    [
      "11001010700",
      81880
    ],
    [
      "11001980000",
      58911
    ],
    [
      "11001005802",
      58565
    ]
  ]
}
```
</details>

---

### Lever Postings API (company career pages)

| | |
|:--|:--|
| **Docs** | https://github.com/lever/postings-api |
| **Auth** | None |
| **Register / token** | No registration. Companies using Lever (Palantir, Netflix, ...) publish postings at jobs.lever.co/{company}; the same slug works in the API. |
| **Script** | [`scripts/jobs/lever_postings.py`](scripts/jobs/lever_postings.py) |

**What you can get:**

- Open postings: title, team, department, location, commitment (full-time/intern), workplace type
- Plain-text and HTML descriptions, requirement lists, salary ranges where published
- Apply and hosted URLs

**Capstone ideas:** Comparing job requirements across companies, salary-transparency analysis, internship pipelines.

**Endpoint:** `GET https://api.lever.co/v0/postings/{company}`

| Parameter | Description | Example |
|:--|:--|:--|
| `mode` | json (default html) | `json` |
| `team / department / location / commitment` | Filters (exact match) | `commitment=Internship` |
| `group` | Group results by team \| department \| location \| commitment | `team` |
| `skip / limit` | Paging | `0 / 50` |
| `/v0/postings/{company}/{postingId}` | One posting | `—` |

```python
from collections import Counter

from _common import get, show


def main():
    posts = get("https://api.lever.co/v0/postings/palantir", params={"mode": "json"})
    show({"company": "Palantir", "open_postings": len(posts),
          "by_team": Counter(p["categories"].get("team") for p in posts).most_common(5),
          "by_commitment": Counter(p["categories"].get("commitment") for p in posts).most_common(),
          "by_workplace": Counter(p.get("workplaceType") for p in posts).most_common(),
          "examples": [{"title": p["text"], "location": p["categories"].get("location")} for p in posts[:3]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:53</summary>

```json
{
  "company": "Palantir",
  "open_postings": 309,
  "by_team": [
    [
      "Dev",
      85
    ],
    [
      "Delta",
      65
    ],
    [
      "Echo",
      32
    ],
    [
      "Information Security",
      19
    ],
    [
      "Legal",
      15
    ]
  ],
  "by_commitment": [
    [
      "Full-time",
      261
    ],
    [
      "Internship",
      44
    ],
    [
      "Fixed-Term",
      4
    ]
  ],
  "by_workplace": [
    [
      "hybrid",
      199
    ],
    [
      "onsite",
      110
    ]
  ],
  "examples": [
    {
      "title": "Administrative Business Partner",
      "location": "Singapore, Singapore"
    },
    {
      "title": "Administrative Business Partner",
      "location": "London, United Kingdom"
    },
    {
      "title": "Administrative Business Partner - ShipOS",
      "location": "Washington, D.C."
    }
  ]
}
```
</details>

---

### O*NET Web Services (occupations & skills)

| | |
|:--|:--|
| **Docs** | https://services.onetcenter.org/reference/ |
| **Auth** | API key (header `X-API-Key`) |
| **Register / token** | Register at https://services.onetcenter.org/developer/signup (free; attribution required). The current API takes an X-API-Key header on GET requests; the older v1.9 Basic-auth API is listed under previous versions. |
| **Env var** | `ONET_API_KEY` |
| **Script** | [`scripts/jobs/onet.py`](scripts/jobs/onet.py) |

**What you can get:**

- ~1,000 US occupations (O*NET-SOC codes): tasks, skills, knowledge, abilities, work activities, technology used
- Keyword search from job titles to occupations; related occupations; education and training requirements
- Bright Outlook flags, wages and employment outlook links (My Next Move / CareerOneStop)

**Capstone ideas:** Skill-based job similarity, automation-exposure scores, matching job ads (Jobs section) to standard occupations.

**Endpoint:** `GET https://api-v2.onetcenter.org/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `online/search?keyword=` | Find occupations from a title or code | `keyword=data scientist` |
| `online/occupations/{code}/summary/skills` | Skills for an occupation (also knowledge, abilities, technology_skills, tasks) | `15-2051.00/summary/skills` |
| `online/occupations/{code}/details/...` | Full ratings (importance/level) instead of summaries | `—` |
| `start / end` | Result range for paged lists | `1 / 20` |
| `X-API-Key (header)` | Your key (query-string keys are rejected) | `<ONET_API_KEY>` |

```python
from _common import get, key, show


def main():
    d = get("https://api-v2.onetcenter.org/online/search", headers={"X-API-Key": key("ONET_API_KEY")},
            params={"keyword": "data scientist"})
    show({"keyword": "data scientist", "matches": [{"code": o["code"], "title": o["title"]} for o in d["occupation"][:5]]})
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "keyword": "data scientist",
  "matches": [
    {"code": "15-2051.00", "title": "Data Scientists"},
    {"code": "15-2041.00", "title": "Statisticians"},
    {"code": "15-1221.00", "title": "Computer and Information Research Scientists"}
  ]
}
```
</details>

---

### Remotive (remote jobs)

| | |
|:--|:--|
| **Docs** | https://github.com/remotive-com/remote-jobs-api |
| **Auth** | None |
| **Register / token** | No registration. Max ~4 requests/day recommended (jobs update daily). Terms: link back to Remotive and do not republish jobs on other job boards. |
| **Script** | [`scripts/jobs/remotive.py`](scripts/jobs/remotive.py) |

**What you can get:**

- Remote job listings: title, company, category, job type, candidate location restrictions, salary text, tags
- Full HTML descriptions, publication date

**Capstone ideas:** Remote-work demand by category, location restrictions in 'remote' jobs, salary disclosure rates.

**Endpoint:** `GET https://remotive.com/api/remote-jobs`

| Parameter | Description | Example |
|:--|:--|:--|
| `category` | Category slug (list via /api/remote-jobs/categories): software-dev, data, design, marketing, ... | `data` |
| `search` | Keyword in title/description | `python` |
| `company_name` | Filter by company | `—` |
| `limit` | Max jobs returned | `50` |

```python
from collections import Counter

from _common import get, show


def main():
    d = get("https://remotive.com/api/remote-jobs", params={"category": "data", "limit": 100})
    jobs = d["jobs"]
    show({"remote_data_jobs": d.get("job-count", len(jobs)),
          "candidate_location": Counter(j["candidate_required_location"] for j in jobs).most_common(4),
          "with_salary_pct": round(100 * sum(bool(j["salary"]) for j in jobs) / max(len(jobs), 1)),
          "examples": [{k: j[k] for k in ("title", "company_name", "job_type", "salary", "publication_date")} for j in jobs[:3]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:53</summary>

```json
{
  "remote_data_jobs": 15,
  "candidate_location": [
    [
      "Worldwide",
      5
    ],
    [
      "Northern America, LATAM, Europe, APAC",
      3
    ],
    [
      "Americas, Europe, Israel",
      2
    ],
    [
      "USA, UK, India, Australia, Ireland, New Zealand, Philippines, Pakistan, Singapore, Mexico",
      1
    ]
  ],
  "with_salary_pct": 53,
  "examples": [
    {
      "title": "Inside Sales Contractor",
      "company_name": "Credit Wellness, LLC",
      "job_type": "full_time",
      "salary": "OTE $25k - $35k",
      "publication_date": "2026-10-08T21:47:32"
    },
    {
      "title": "Tier III Service Desk Engineer",
      "company_name": "Unio Digital",
      "job_type": "full_time",
      "salary": "",
      "publication_date": "2026-10-07T01:11:09"
    },
    {
      "title": "Software Engineer / AI Code Trainer (Python, Web, Full Stack)",
      "company_name": "CodeForAI",
      "job_type": "contract",
      "salary": "$45-$120/Hour",
      "publication_date": "2026-10-05T05:15:43"
    }
  ]
}
```
</details>

---

### The Muse Public API (jobs)

| | |
|:--|:--|
| **Docs** | https://www.themuse.com/developers/api/v2 |
| **Auth** | None (optional key raises the limit) |
| **Register / token** | Works without a key (500 requests/hour). Register at https://www.themuse.com/developers/api/v2/apps for a key (3,600/hour). |
| **Env var** | `THEMUSE_API_KEY (optional)` |
| **Script** | [`scripts/jobs/the_muse.py`](scripts/jobs/the_muse.py) |

**What you can get:**

- Job postings with category, experience level (entry/mid/senior/internship), locations, company, publication date
- Company profiles with size, industries, locations
- Full HTML descriptions

**Capstone ideas:** Entry-level vs. senior demand by field, internship availability, company-culture text analysis.

**Endpoint:** `GET https://www.themuse.com/api/public/{jobs \| companies}`

| Parameter | Description | Example |
|:--|:--|:--|
| `page` | Page number (required, 20 results per page) | `1` |
| `category` | Job category (repeatable): Data and Analytics, Software Engineering, ... | `Data and Analytics` |
| `level` | Entry Level \| Mid Level \| Senior Level \| Management \| Internship (repeatable) | `Entry Level` |
| `location` | City, State (repeatable) | `Washington, DC` |
| `company` | Company name (repeatable) | `—` |
| `descending` | true -> newest first | `true` |
| `api_key` | Optional key | `<THEMUSE_API_KEY>` |

```python
import os
from collections import Counter

from _common import get, show


def main():
    params = [("page", 1), ("category", "Data and Analytics"), ("level", "Entry Level"), ("descending", "true")]
    if os.getenv("THEMUSE_API_KEY"):
        params.append(("api_key", os.environ["THEMUSE_API_KEY"]))
    d = get("https://www.themuse.com/api/public/jobs", params=params)
    show({"entry_level_data_jobs": d["total"],
          "top_locations_on_page": Counter(l["name"] for j in d["results"] for l in j["locations"]).most_common(4),
          "examples": [{"title": j["name"], "company": j["company"]["name"],
                        "locations": [l["name"] for l in j["locations"]][:2], "published": j["publication_date"][:10]}
                       for j in d["results"][:3]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 09:53</summary>

```json
{
  "entry_level_data_jobs": 103,
  "top_locations_on_page": [
    [
      "Albuquerque, NM",
      10
    ],
    [
      "Los Lunas, NM",
      2
    ],
    [
      "Rio Rancho, NM",
      2
    ],
    [
      "Shelton, CT",
      1
    ]
  ],
  "examples": [
    {
      "title": "Cake Decorator",
      "company": "Walmart",
      "locations": [
        "Albuquerque, NM"
      ],
      "published": "2025-08-25"
    },
    {
      "title": "Meat Cutter and Wrapper",
      "company": "Walmart",
      "locations": [
        "Albuquerque, NM"
      ],
      "published": "2025-08-13"
    },
    {
      "title": "Cake Decorator",
      "company": "Walmart",
      "locations": [
        "Albuquerque, NM"
      ],
      "published": "2025-08-25"
    }
  ]
}
```
</details>

---

## 🔬 Science, Space & Research

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

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
[
  {
    "title": "Tracing and Taming Hallucinated Terms via Subsequence-Association Filtering and Breadth-Aware Selection for LLM Query Expansion",
    "published": "2026-10-09T16:22:16Z",
    "authors": [
      "Amit Kumar Jaiswal",
      "Sagar Uprety",
      "Prayag Tiwari"
    ],
    "abstract": "Large language models (LLMs) are now a standard source of query expansion terms but they hallucinate. They inject non existent entities and off-topic terms that add ranking noise and can hurt strong r..."
  },
  {
    "title": "Adaptive DHiSparse: Speculative Decoding over a Load-Adaptive KV Hierarchy",
    "published": "2026-10-09T15:02:35Z",
    "authors": [
      "Naji Alazhar",
      "Wassel Alazhar"
    ],
    "abstract": "Long-context serving runs out of KV memory before the hardware reaches its useful throughput. HiSparse stores history in host memory and fetches only the selected entries, but it uses one cache budget..."
  },
  {
    "title": "Verifiable Retrieval for Legal Reasoning through Structure-driven Knowledge Graph",
    "published": "2026-10-09T14:56:31Z",
    "authors": [
      "Ibrahim Siddig",
      "Sviatoslav Tugeev",
      "Munir Georges"
    ],
    "abstract": "Any automation within reliability-critical contexts such as medicine or law has to be transparent and verifiable. Despite their advantages, current LLM-based systems do not meet these requirements due..."
  }
]
```
</details>

---

### CelesTrak (satellite orbital elements)

| | |
|:--|:--|
| **Docs** | https://celestrak.org/NORAD/documentation/gp-data-formats.php |
| **Auth** | None |
| **Register / token** | No registration. Download each group at most once every 2 hours - a repeat request returns HTTP 403 until the data updates. Elements use the OMM format; propagate positions with the sgp4 Python package. |
| **Script** | [`scripts/science_research/celestrak.py`](scripts/science_research/celestrak.py) |

**What you can get:**

- Current orbital elements (General Perturbations) for ~30,000 tracked objects: stations, Starlink, GPS, weather sats, debris
- Epoch, inclination, eccentricity, mean motion, NORAD ID, international designator
- SATCAT catalog: owner country, launch date, decay date, object type

**Capstone ideas:** Satellite-constellation growth (Starlink), orbital-debris analysis, pass prediction with sgp4.

**Endpoint:** `GET https://celestrak.org/NORAD/elements/gp.php`

| Parameter | Description | Example |
|:--|:--|:--|
| `GROUP` | Named set: stations \| starlink \| gps-ops \| weather \| active \| last-30-days \| ... | `starlink` |
| `CATNR / INTDES / NAME` | Single object by NORAD number, international designator, or name | `CATNR=25544 (ISS)` |
| `FORMAT` | json \| csv \| xml \| tle \| 3le | `json` |
| `satcat/records.php?GROUP=` | Sibling endpoint: catalog records (owner, launch, decay) | `—` |

```python
from _common import get, show


def main():
    starlink = get("https://celestrak.org/NORAD/elements/gp.php", params={"GROUP": "starlink", "FORMAT": "json"})
    iss = get("https://celestrak.org/NORAD/elements/gp.php", params={"CATNR": 25544, "FORMAT": "json"})[0]
    shells = {}
    for s in starlink:
        shells[round(s["INCLINATION"])] = shells.get(round(s["INCLINATION"]), 0) + 1
    show({"starlink_satellites_tracked": len(starlink),
          "starlink_by_inclination_deg": dict(sorted(shells.items(), key=lambda kv: -kv[1])[:5]),
          "iss": {k: iss[k] for k in ("OBJECT_NAME", "EPOCH", "INCLINATION", "MEAN_MOTION", "ECCENTRICITY")} |
                 {"orbital_period_min": round(1440 / iss["MEAN_MOTION"], 1)}})
```

<details><summary><b>Sample output</b> — live run 2026-10-12</summary>

```json
{
  "starlink_satellites_tracked": 11102,
  "starlink_by_inclination_deg": {
    "53": 5095,
    "43": 3614,
    "97": 1513,
    "70": 870,
    "98": 10
  },
  "iss": {
    "OBJECT_NAME": "ISS (ZARYA)",
    "EPOCH": "2026-10-12T03:27:40.741056",
    "INCLINATION": 51.6314,
    "MEAN_MOTION": 15.48833724,
    "ECCENTRICITY": 0.00067311,
    "orbital_period_min": 93.0
  }
}
```
</details>

---

### CERN Open Data Portal

| | |
|:--|:--|
| **Docs** | https://opendata.cern.ch/docs/about |
| **Auth** | None |
| **Register / token** | No registration. Real LHC collision data (CMS, ATLAS, LHCb, ALICE) plus simulations, software and educational datasets small enough for a laptop. Bulk files via HTTP or the cernopendata-client. |
| **Script** | [`scripts/science_research/cern_open_data.py`](scripts/science_research/cern_open_data.py) |

**What you can get:**

- Record search over datasets, simulated data, software, documentation; experiment, year, type, size
- Derived/educational datasets (e.g. dimuon events for Higgs/Z-boson exercises) as CSV
- File lists with sizes and download URIs

**Capstone ideas:** Particle-classification ML on real collision events, reproducing the Higgs/Z peak from the dimuon spectrum.

**Endpoint:** `GET https://opendata.cern.ch/api/records/`

| Parameter | Description | Example |
|:--|:--|:--|
| `q` | Search text | `higgs` |
| `type` | Dataset \| Software \| Documentation; add a subtype as type=Dataset::Derived (Collision \| Simulated \| Derived) | `Dataset::Derived` |
| `experiment` | CMS \| ATLAS \| LHCb \| ALICE \| ... | `CMS` |
| `size / page / sort` | Paging and sort (mostrecent, bestmatch) | `5` |
| `/api/records/{recid}` | One record with files | `—` |

```python
from _common import get, show


def main():
    # Subtypes are given as type=Dataset::Derived (a separate subtype= parameter is ignored)
    d = get("https://opendata.cern.ch/api/records/", params={"q": "dimuon", "type": "Dataset::Derived", "size": 5})
    show({"matching_educational_datasets": d["hits"]["total"], "examples": [
        {"recid": h["metadata"].get("recid"), "title": h["metadata"]["title"],
         "experiment": h["metadata"].get("experiment"), "year": h["metadata"].get("date_created"),
         "files": len(h["metadata"].get("files", []))} for h in d["hits"]["hits"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "matching_educational_datasets": 5,
  "examples": [
    {
      "recid": "700",
      "title": "Dimuon event information derived from the Run2010B public Mu dataset",
      "experiment": [
        "CMS"
      ],
      "year": [
        "2010"
      ],
      "files": 11
    },
    {
      "recid": "301",
      "title": "J/psi to two muons from 2010",
      "experiment": [
        "CMS"
      ],
      "year": [
        "2010"
      ],
      "files": 22
    },
    {
      "recid": "310",
      "title": "Event files for CMS masterclass exercise 2014",
      "experiment": [
        "CMS"
      ],
      "year": [
        "2010"
      ],
      "files": 77
    },
    {
      "recid": "545",
      "title": "Datasets derived from the Run2011A SingleElectron, SingleMu, DoubleElectron, and DoubleMu primary datasets",
      "experiment": [
        "CMS"
      ],
      "year": [
        "2011"
      ],
      "files": 8
    },
    {
      "recid": "303",
      "title": "Events with two muons from 2010",
      "experiment": [
        "CMS"
      ],
      "year": [
        "2010"
      ],
      "files": 12
    }
  ]
}
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

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
[
  {
    "DOI": "10.1006/meth.2001.1262",
    "title": "Analysis of Relative Gene Expression Data Using Real-Time Quantitative PCR and the 2−ΔΔCT Method",
    "journal": "Methods",
    "year": 2001,
    "cited_by": 166794
  },
  {
    "DOI": "10.1186/s13059-014-0550-8",
    "title": "Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2",
    "journal": "Genome Biology",
    "year": 2014,
    "cited_by": 92840
  },
  {
    "DOI": "10.1016/0927-0256(96)00008-0",
    "title": "Efficiency of ab-initio total energy calculations for metals and semiconductors using a plane-wave basis set",
    "journal": "Computational Materials Science",
    "year": 1996,
    "cited_by": 74881
  }
]
```
</details>

---

### JPL Small-Body Close Approach & SBDB APIs

| | |
|:--|:--|
| **Docs** | https://ssd-api.jpl.nasa.gov/doc/cad.html |
| **Auth** | None |
| **Register / token** | No registration. Part of JPL's Solar System Dynamics APIs (also sbdb.api for orbits/physical data, sentry.api for impact risk, scout.api for new objects). |
| **Script** | [`scripts/science_research/jpl_close_approach.py`](scripts/science_research/jpl_close_approach.py) |

**What you can get:**

- Close approaches of asteroids and comets to Earth (or other bodies): date, distance, velocity, size proxy (H)
- Small-Body Database: orbit elements, diameter, albedo, rotation, discovery
- Sentry impact-risk list

**Capstone ideas:** Near-Earth object population statistics, size-frequency distributions, close-approach trend analysis.

**Endpoint:** `GET https://ssd-api.jpl.nasa.gov/{cad \| sbdb \| sentry}.api`

| Parameter | Description | Example |
|:--|:--|:--|
| `date-min / date-max` | Window (YYYY-MM-DD, or now / +60 days) | `now / +60` |
| `dist-max` | Max distance in au (or LD for lunar distances, e.g. 1LD) | `1LD` |
| `h-max` | Max absolute magnitude (smaller H = larger object) | `22` |
| `body` | Earth (default) \| Moon \| Mars \| ALL ... | `Earth` |
| `sort` | date \| dist \| v-rel \| h (prefix - for descending) | `dist` |
| `sbdb.api?sstr=` | Small-Body Database record for one object (phys-par=1 adds size/albedo) | `sstr=Apophis&phys-par=1` |
| `sentry.api` | Objects with non-zero impact probability | `—` |

```python
from _common import get, show


def main():
    d = get("https://ssd-api.jpl.nasa.gov/cad.api", params={"date-min": "now", "date-max": "+60", "dist-max": "1LD",
                                                             "sort": "dist"})
    rows = [dict(zip(d["fields"], r)) for r in d["data"]]
    apophis = get("https://ssd-api.jpl.nasa.gov/sbdb.api", params={"sstr": "Apophis", "phys-par": 1})
    phys = {p["name"]: p["value"] for p in apophis.get("phys_par", [])}
    show({"approaches_within_1_lunar_distance_next_60_days": d["count"], "closest": [
        {"object": r["des"], "date": r["cd"], "distance_lunar": round(float(r["dist"]) / 0.00257, 2),
         "velocity_km_s": round(float(r["v_rel"]), 1), "H": r["h"]} for r in rows[:5]],
          "apophis": {"name": apophis["object"]["fullname"], "diameter_km": phys.get("diameter"), "albedo": phys.get("albedo")}})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "approaches_within_1_lunar_distance_next_60_days": 4,
  "closest": [
    {
      "object": "2026 TP6",
      "date": "2026-Oct-15 00:01",
      "distance_lunar": 0.57,
      "velocity_km_s": 10.3,
      "H": "27.288"
    },
    {
      "object": "2026 TG9",
      "date": "2026-Oct-14 04:58",
      "distance_lunar": 0.59,
      "velocity_km_s": 11.6,
      "H": "26.99"
    },
    {
      "object": "2025 UK9",
      "date": "2026-Oct-31 02:35",
      "distance_lunar": 0.82,
      "velocity_km_s": 7.8,
      "H": "29.83"
    },
    {
      "object": "2026 TF59",
      "date": "2026-Oct-12 17:36",
      "distance_lunar": 0.96,
      "velocity_km_s": 7.0,
      "H": "28.589"
    }
  ],
  "apophis": {
    "name": "99942 Apophis (2004 MN4)",
    "diameter_km": "0.34",
    "albedo": "0.35"
  }
}
```
</details>

---

### JPL Fireball (bolide) API

| | |
|:--|:--|
| **Docs** | https://ssd-api.jpl.nasa.gov/doc/fireball.html |
| **Auth** | None |
| **Register / token** | No registration. Bright meteors detected by U.S. government sensors, reported by CNEOS since 1988. |
| **Script** | [`scripts/science_research/jpl_fireball.py`](scripts/science_research/jpl_fireball.py) |

**What you can get:**

- Each fireball: date/time, latitude/longitude, altitude, velocity, radiated energy, estimated impact energy (kt TNT)

**Capstone ideas:** Spatial/temporal distribution of atmospheric impacts, energy-frequency power laws.

**Endpoint:** `GET https://ssd-api.jpl.nasa.gov/fireball.api`

| Parameter | Description | Example |
|:--|:--|:--|
| `date-min / date-max` | Date range | `2025-01-01` |
| `energy-min / impact-e-min` | Minimum radiated energy (J x10^10) / impact energy (kt) | `impact-e-min=1` |
| `req-loc / req-alt / req-vel` | true -> only events with location / altitude / velocity | `req-loc=true` |
| `sort` | date \| energy \| impact-e \| alt \| vel (prefix - for descending) | `-impact-e` |
| `limit` | Max rows | `5` |

```python
from _common import get, show


def main():
    d = get("https://ssd-api.jpl.nasa.gov/fireball.api", params={"date-min": "2015-01-01", "req-loc": "true",
                                                                  "sort": "-impact-e", "limit": 5})
    show({"largest_since_2015": [dict(zip(d["fields"], r)) for r in d["data"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "largest_since_2015": [
    {
      "date": "2018-12-18 23:48:18",
      "energy": "3130",
      "impact-e": "49",
      "lat": "56.9",
      "lat-dir": "N",
      "lon": "172.4",
      "lon-dir": "E",
      "alt": "26.0",
      "vel": "13.6"
    },
    {
      "date": "2016-02-06 13:55:09",
      "energy": "685",
      "impact-e": "13",
      "lat": "30.4",
      "lat-dir": "S",
      "lon": "25.5",
      "lon-dir": "W",
      "alt": "31.0",
      "vel": "15.6"
    },
    {
      "date": "2020-12-22 23:23:28",
      "energy": "667",
      "impact-e": "12",
      "lat": "31.9",
      "lat-dir": "N",
      "lon": "96.2",
      "lon-dir": "E",
      "alt": "35.5",
      "vel": "13.6"
    },
    {
      "date": "2017-12-15 13:14:37",
      "energy": "400",
      "impact-e": "7.9",
      "lat": "60.2",
      "lat-dir": "N",
      "lon": "170.0",
      "lon-dir": "E",
      "alt": "20.0",
      "vel": "31.4"
    },
    {
      "date": "2023-05-20 11:22:22",
      "energy": "351",
      "impact-e": "7.1",
      "lat": "17.8",
      "lat-dir": "S",
      "lon": "141.9",
      "lon-dir": "E",
      "alt": "29.0",
      "vel": "27.9"
    }
  ]
}
```
</details>

---

### JPL Horizons (ephemerides)

| | |
|:--|:--|
| **Docs** | https://ssd-api.jpl.nasa.gov/doc/horizons.html |
| **Auth** | None |
| **Register / token** | No registration. The authoritative source for positions of planets, moons, asteroids, comets and spacecraft. The response is a text report inside JSON; parse the block between $$SOE and $$EOE. |
| **Script** | [`scripts/science_research/jpl_horizons.py`](scripts/science_research/jpl_horizons.py) |

**What you can get:**

- Where any solar-system body (or spacecraft) is at any time: RA/Dec, distance, magnitude, phase, rise/set
- State vectors (x, y, z, vx, vy, vz) for orbital mechanics
- Physical data for planets and moons

**Capstone ideas:** Orbit visualization, observation planning, validating orbital-mechanics simulations against real ephemerides.

**Endpoint:** `GET https://ssd.jpl.nasa.gov/api/horizons.api`

| Parameter | Description | Example |
|:--|:--|:--|
| `COMMAND` | Target in quotes: 499 Mars, 301 Moon, -125544 ISS, 'Ceres;' (URL-encode the ; when using curl - requests does it) | `'499'` |
| `EPHEM_TYPE` | OBSERVER \| VECTORS \| ELEMENTS | `OBSERVER` |
| `CENTER` | Observer location: 500@399 Earth geocenter, coord@399 with SITE_COORD, 500@10 Sun | `'500@399'` |
| `START_TIME / STOP_TIME / STEP_SIZE` | Time span and step | `'2026-10-12' / '2026-10-15' / '1d'` |
| `QUANTITIES (observer)` | 1 RA/Dec, 9 magnitude, 20 range, 4 alt/az, ... | `'1,9,20'` |
| `format` | json \| text | `json` |

```python
from _common import get, show


def main():
    d = get("https://ssd.jpl.nasa.gov/api/horizons.api", params={
        "format": "json", "COMMAND": "'499'", "EPHEM_TYPE": "OBSERVER", "CENTER": "'500@399'",
        "START_TIME": "'2026-10-12'", "STOP_TIME": "'2026-10-15'", "STEP_SIZE": "'1d'", "QUANTITIES": "'1,9,20'"})
    text = d["result"]
    table = text.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
    header = [l for l in text.splitlines() if "Date__(UT)__HR:MN" in l][0]
    show({"target": "Mars as seen from Earth's center", "columns": " ".join(header.split()),
          "rows": [" ".join(r.split()) for r in table]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "target": "Mars as seen from Earth's center",
  "columns": "Date__(UT)__HR:MN R.A._____(ICRF)_____DEC APmag S-brt delta deldot",
  "rows": [
    "2026-Oct-12 00:00 08 41 20.08 +19 32 16.7 1.063 4.541 1.58734717893662 -12.7203694",
    "2026-Oct-13 00:00 08 43 38.33 +19 24 40.8 1.045 4.532 1.57997954238183 -12.7942703",
    "2026-Oct-14 00:00 08 45 55.79 +19 17 01.4 1.032 4.528 1.57256953339623 -12.8671127",
    "2026-Oct-15 00:00 08 48 12.43 +19 09 18.9 1.036 4.542 1.56511775628645 -12.9389225"
  ]
}
```
</details>

---

### Launch Library 2 (The Space Devs)

| | |
|:--|:--|
| **Docs** | https://ll.thespacedevs.com/docs/ |
| **Auth** | None (15 requests/hour) / paid tiers |
| **Register / token** | Works anonymously at 15 requests/hour. Use lldev.thespacedevs.com for development (stale data, no limit). Higher limits via Patreon-supported keys. |
| **Script** | [`scripts/science_research/launch_library.py`](scripts/science_research/launch_library.py) |

**What you can get:**

- Every orbital launch since 1957 and upcoming launches: rocket, provider, pad, mission, orbit, status, time window
- Agencies, rockets (configurations), astronauts, spacecraft, space stations, docking events, landings (booster reuse)

**Capstone ideas:** Launch cadence by provider, success-rate modeling, reusability economics, launch-delay prediction.

**Endpoint:** `GET https://ll.thespacedevs.com/2.3.0/{resource}/`

| Parameter | Description | Example |
|:--|:--|:--|
| `resource` | launches/upcoming \| launches/previous \| astronauts \| agencies \| spacecraft \| ... | `launches/previous` |
| `mode` | list \| normal \| detailed (amount of nested detail) | `list` |
| `net__gte / net__lte` | Launch time filters (ISO 8601) | `2026-01-01T00:00:00Z` |
| `lsp__name / location__ids / status__ids` | Provider, location, status filters | `lsp__name=SpaceX` |
| `search` | Text search | `Starlink` |
| `limit / offset` | Paging (max 100) | `10` |

```python
from collections import Counter

from _common import get, show


def main():
    d = get("https://ll.thespacedevs.com/2.3.0/launches/previous/", params={
        "mode": "list", "net__gte": "2026-01-01T00:00:00Z", "limit": 100})
    launches = d["results"]
    providers = Counter(l["name"].split(" | ")[0] for l in launches)
    show({"launches_2026_so_far": d["count"], "rockets_in_latest_100": providers.most_common(5),
          "status_in_latest_100": Counter(l["status"]["abbrev"] for l in launches).most_common(),
          "latest": [{"name": l["name"], "net": l["net"], "status": l["status"]["abbrev"]} for l in launches[:3]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "launches_2026_so_far": 245,
  "rockets_in_latest_100": [
    [
      "Falcon 9 Block 5",
      42
    ],
    [
      "Electron",
      8
    ],
    [
      "Long March 12",
      4
    ],
    [
      "Long March 6A",
      3
    ],
    [
      "Long March 8A",
      3
    ]
  ],
  "status_in_latest_100": [
    [
      "Success",
      99
    ],
    [
      "Failure",
      1
    ]
  ],
  "latest": [
    {
      "name": "Falcon 9 Block 5 | SDA Tranche 1 Transport Layer A",
      "net": "2026-10-10T07:39:44Z",
      "status": "Success"
    },
    {
      "name": "Long March 12 | SatNet LEO Group 27",
      "net": "2026-10-09T19:27:00Z",
      "status": "Success"
    },
    {
      "name": "Nuri | NeonSat-2 to 6",
      "net": "2026-10-07T03:25:00Z",
      "status": "Success"
    }
  ]
}
```
</details>

---

### Materials Project API

| | |
|:--|:--|
| **Docs** | https://docs.materialsproject.org/downloading-data/using-the-api |
| **Auth** | API key (header `X-API-KEY`) |
| **Register / token** | Create a free account at https://next-gen.materialsproject.org and copy the key from your dashboard. Python client: `pip install mp-api` (MPRester). |
| **Env var** | `MP_API_KEY` |
| **Script** | [`scripts/science_research/materials_project.py`](scripts/science_research/materials_project.py) |

**What you can get:**

- 150k+ inorganic compounds computed with DFT: formation energy, stability (energy above hull), band gap, density
- Crystal structures, elastic and magnetic properties, battery electrodes, phonons, X-ray diffraction
- Search by formula, elements, chemical system, or property ranges

**Capstone ideas:** Materials-property prediction (band gap, stability) with ML, battery-material screening.

**Endpoint:** `GET https://api.materialsproject.org/materials/summary/`

| Parameter | Description | Example |
|:--|:--|:--|
| `formula / elements / chemsys` | Composition filters | `chemsys=Li-Fe-O` |
| `band_gap_min / band_gap_max` | Property ranges (also energy_above_hull_max, density_min, ...) | `1 / 3` |
| `is_stable` | true -> on the convex hull | `true` |
| `_fields` | Fields to return | `material_id,formula_pretty,band_gap,energy_above_hull,density` |
| `_limit / _skip` | Paging | `5` |
| `X-API-KEY (header)` | Your key | `<MP_API_KEY>` |

```python
from _common import get, key, show


def main():
    d = get("https://api.materialsproject.org/materials/summary/", headers={"X-API-KEY": key("MP_API_KEY")}, params={
        "chemsys": "Li-Fe-O,Li-Fe-P-O", "is_stable": "true", "_limit": 5,
        "_fields": "material_id,formula_pretty,band_gap,energy_above_hull,density"})
    show(d["data"])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"material_id": "mp-19017", "formula_pretty": "LiFePO4", "band_gap": 3.70, "energy_above_hull": 0.0, "density": 3.52},
  {"material_id": "mp-18900", "formula_pretty": "LiFeO2", "band_gap": 1.84, "energy_above_hull": 0.0, "density": 4.31}
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

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

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

### NASA Exoplanet Archive (TAP)

| | |
|:--|:--|
| **Docs** | https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html |
| **Auth** | None |
| **Register / token** | No registration. Query with ADQL (SQL for astronomy) through the TAP service. Table 'pscomppars' has one row per planet with the best parameters; 'ps' has every published solution. |
| **Script** | [`scripts/science_research/nasa_exoplanet_archive.py`](scripts/science_research/nasa_exoplanet_archive.py) |

**What you can get:**

- 6,000+ confirmed exoplanets: discovery year/method/facility, orbital period, radius, mass, equilibrium temperature
- Host-star properties (temperature, mass, distance), transit and radial-velocity data
- TESS and Kepler candidate tables

**Capstone ideas:** Habitable-zone classification, discovery-method bias analysis, planet-radius distributions (the 'radius valley').

**Endpoint:** `GET https://exoplanetarchive.ipac.caltech.edu/TAP/sync`

| Parameter | Description | Example |
|:--|:--|:--|
| `query` | ADQL query | `select pl_name, disc_year, pl_rade from pscomppars where pl_rade < 1.5` |
| `format` | json \| csv \| tsv \| votable | `json` |
| `tables` | pscomppars (one row per planet) \| ps (all solutions) \| toi (TESS candidates) \| ... | `pscomppars` |
| `key columns` | pl_name, hostname, disc_year, discoverymethod, pl_orbper, pl_rade, pl_bmasse, pl_eqt, sy_dist | `—` |

```python
from _common import get, show

URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def main():
    by_method = get(URL, params={"format": "json", "query":
        "select discoverymethod, count(*) as n from pscomppars group by discoverymethod order by n desc"})
    earthlike = get(URL, params={"format": "json", "query":
        "select top 5 pl_name, disc_year, pl_rade, pl_eqt, sy_dist from pscomppars "
        "where pl_rade between 0.8 and 1.25 and pl_eqt between 200 and 300 order by sy_dist"})
    show({"confirmed_planets": sum(r["n"] for r in by_method), "by_discovery_method": by_method[:5],
          "nearest_earth_size_temperate": earthlike})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "confirmed_planets": 6445,
  "by_discovery_method": [
    {
      "discoverymethod": "Transit",
      "n": 4781
    },
    {
      "discoverymethod": "Radial Velocity",
      "n": 1203
    },
    {
      "discoverymethod": "Microlensing",
      "n": 292
    },
    {
      "discoverymethod": "Imaging",
      "n": 97
    },
    {
      "discoverymethod": "Transit Timing Variations",
      "n": 29
    }
  ],
  "nearest_earth_size_temperate": [
    {
      "pl_name": "GJ 1061 d",
      "disc_year": 2020,
      "pl_rade": 1.16,
      "pl_eqt": 242.03,
      "sy_dist": 3.67278
    },
    {
      "pl_name": "Teegarden's Star c",
      "disc_year": 2019,
      "pl_rade": 1.02,
      "pl_eqt": 209.0,
      "sy_dist": 3.83078
    },
    {
      "pl_name": "GJ 1002 b",
      "disc_year": 2022,
      "pl_rade": 1.03,
      "pl_eqt": 230.9,
      "sy_dist": 4.84867
    },
    {
      "pl_name": "TRAPPIST-1 f",
      "disc_year": 2017,
      "pl_rade": 1.045,
      "pl_eqt": 217.7,
      "sy_dist": 12.42988881
    },
    {
      "pl_name": "TRAPPIST-1 e",
      "disc_year": 2017,
      "pl_rade": 0.92,
      "pl_eqt": 249.7,
      "sy_dist": 12.42988881
    }
  ]
}
```
</details>

---

### NASA Image and Video Library

| | |
|:--|:--|
| **Docs** | https://images.nasa.gov/docs/images.nasa.gov_api_docs.pdf |
| **Auth** | None |
| **Register / token** | No registration (separate from api.nasa.gov - no DEMO_KEY limits). Most NASA media are not copyrighted, but check the usage guidelines for logos and identifiable people. |
| **Script** | [`scripts/science_research/nasa_image_library.py`](scripts/science_research/nasa_image_library.py) |

**What you can get:**

- 140k+ NASA images, videos and audio: title, description, keywords, center, photographer, date
- Asset manifests with original-resolution files and captions
- Search by text, center, year range, media type

**Capstone ideas:** Space-image datasets for computer vision, captioning models, mission-history timelines.

**Endpoint:** `GET https://images-api.nasa.gov/{search \| asset/{nasa_id} \| metadata/{nasa_id}}`

| Parameter | Description | Example |
|:--|:--|:--|
| `q` | Free-text search | `james webb` |
| `media_type` | image \| video \| audio (comma list) | `image` |
| `center / keywords / photographer` | Field filters | `center=GSFC` |
| `year_start / year_end` | Date range | `2022 / 2026` |
| `page / page_size` | Paging (page_size max 100) | `1 / 5` |
| `/asset/{nasa_id}` | All file URLs (orig, large, medium, thumb) for one item | `—` |

```python
from _common import get, show


def main():
    d = get("https://images-api.nasa.gov/search", params={"q": "james webb deep field", "media_type": "image",
                                                          "year_start": 2022, "page_size": 5})["collection"]
    show({"hits": d["metadata"]["total_hits"], "images": [
        {"title": i["data"][0]["title"], "date": i["data"][0]["date_created"][:10], "center": i["data"][0].get("center"),
         "nasa_id": i["data"][0]["nasa_id"], "thumbnail": i["links"][0]["href"]} for i in d["items"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "hits": 8,
  "images": [
    {
      "title": "Beyond the Deep Field: Hubble’s Legacy and the Future of Cosmi",
      "date": "2026-06-05",
      "center": "HQ",
      "nasa_id": "NHQ202606050016",
      "thumbnail": "https://images-assets.nasa.gov/image/NHQ202606050016/NHQ202606050016~large.jpg"
    },
    {
      "title": "JWST’s First Full-Color Images",
      "date": "2022-07-12",
      "center": "HQ",
      "nasa_id": "NHQ202207120017",
      "thumbnail": "https://images-assets.nasa.gov/image/NHQ202207120017/NHQ202207120017~medium.jpg"
    },
    {
      "title": "JWST’s First Full-Color Images",
      "date": "2022-07-12",
      "center": "HQ",
      "nasa_id": "NHQ202207120016",
      "thumbnail": "https://images-assets.nasa.gov/image/NHQ202207120016/NHQ202207120016~medium.jpg"
    },
    {
      "title": "Beyond the Deep Field: Hubble’s Legacy and the Future of Cosmi",
      "date": "2026-06-05",
      "center": "HQ",
      "nasa_id": "NHQ202606050014",
      "thumbnail": "https://images-assets.nasa.gov/image/NHQ202606050014/NHQ202606050014~large.jpg"
    },
    {
      "title": "Beyond the Deep Field: Hubble’s Legacy and the Future of Cosmi",
      "date": "2026-06-05",
      "center": "HQ",
      "nasa_id": "NHQ202606050013",
      "thumbnail": "https://images-assets.nasa.gov/image/NHQ202606050013/NHQ202606050013~large.jpg"
    }
  ]
}
```
</details>

---

### NIH RePORTER API

| | |
|:--|:--|
| **Docs** | https://api.reporter.nih.gov/ |
| **Auth** | None |
| **Register / token** | No registration. POST JSON criteria. Max 1 request/second; max 500 records per request and 15,000 per query (narrow by fiscal year for more). |
| **Script** | [`scripts/science_research/nih_reporter.py`](scripts/science_research/nih_reporter.py) |

**What you can get:**

- Every NIH-funded project: title, abstract, PI, organization, award amount, institute (NCI, NIMH, ...), terms
- Fiscal year, activity code (R01, K99, ...), study section, spending categories
- Linked publications (publications/search endpoint)

**Capstone ideas:** Biomedical funding by disease area, success predictors for grants, linking grants to papers (PubMed).

**Endpoint:** `POST https://api.reporter.nih.gov/v2/projects/search`

| Parameter | Description | Example |
|:--|:--|:--|
| `criteria.fiscal_years` | List of fiscal years | `[2025]` |
| `criteria.org_names / org_states` | Organization names (upper case) / states | `["GEORGE WASHINGTON UNIVERSITY"]` |
| `criteria.advanced_text_search` | {operator, search_field, search_text} | `{"search_field": "projecttitle,terms", "search_text": "machine learning"}` |
| `criteria.agencies / activity_codes` | Institute codes / grant types | `["NIMH"] / ["R01"]` |
| `include_fields` | Fields to return | `["ProjectTitle", "AwardAmount", "Organization"]` |
| `offset / limit / sort_field / sort_order` | Paging and sorting (limit max 500) | `0 / 5 / award_amount / desc` |
| `/v2/publications/search` | Sibling endpoint: publications linked to projects | `—` |

```python
import requests

from _common import USER_AGENT, show


def main():
    body = {"criteria": {"fiscal_years": [2025], "org_names": ["GEORGE WASHINGTON UNIVERSITY"]},
            "include_fields": ["ProjectTitle", "AwardAmount", "AgencyIcAdmin", "ActivityCode", "PrincipalInvestigators"],
            "offset": 0, "limit": 5, "sort_field": "award_amount", "sort_order": "desc"}
    r = requests.post("https://api.reporter.nih.gov/v2/projects/search", json=body, timeout=60,
                      headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    d = r.json()
    show({"gwu_nih_projects_fy2025": d["meta"]["total"], "largest": [
        {"title": p["project_title"], "amount_usd": p["award_amount"], "institute": (p.get("agency_ic_admin") or {}).get("abbreviation"),
         "activity": p["activity_code"], "pi": (p.get("principal_investigators") or [{}])[0].get("full_name")}
        for p in d["results"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "gwu_nih_projects_fy2025": 134,
  "largest": [
    {
      "title": "The DC Cohort: A Longitudinal Population-Based Cohort Study of People Living with HIV in Washington, DC",
      "amount_usd": 6502401,
      "institute": "NIAID",
      "activity": "R24",
      "pi": "Amanda Derryck Castel"
    },
    {
      "title": "NICHD Maternal-Fetal Medicine Units (MFMU) Network: Data CoordinatingCenter",
      "amount_usd": 3979209,
      "institute": "NICHD",
      "activity": "U24",
      "pi": "Rebecca Gersnoviez Clifton"
    },
    {
      "title": "AIDS and Cancer Specimen Resource (ACSR)",
      "amount_usd": 3963281,
      "institute": "NCI",
      "activity": "UM1",
      "pi": "Jeffrey Michael Bethony"
    },
    {
      "title": "The Impact of the 11th Street Bridge Park on the Built Environment and Community Health: Washington, D.C.",
      "amount_usd": 3399035,
      "institute": "NINR",
      "activity": "R01",
      "pi": "LORETTA  DIPIETRO"
    },
    {
      "title": "Adjunctive Azithromycin Prophylaxis for Scheduled/Prelabor Cesarean Delivery",
      "amount_usd": 3341494,
      "institute": "NICHD",
      "activity": "U01",
      "pi": "Kim A Boggess"
    }
  ]
}
```
</details>

---

### NOAA Space Weather Prediction Center (SWPC) data

| | |
|:--|:--|
| **Docs** | https://services.swpc.noaa.gov/ |
| **Auth** | None |
| **Register / token** | No registration. Browsable JSON/text files under /json and /products, updated every minute to every few hours. |
| **Script** | [`scripts/science_research/noaa_swpc.py`](scripts/science_research/noaa_swpc.py) |

**What you can get:**

- Planetary K-index (geomagnetic storms), NOAA space-weather scales (G, S, R), alerts and watches
- Real-time solar wind (speed, density, magnetic field), X-ray flux and solar flares
- Aurora forecasts (OVATION), 27-day outlook, sunspot/solar-cycle progression

**Capstone ideas:** Geomagnetic-storm forecasting, solar-cycle analysis, space-weather impacts on GPS or the power grid.

**Endpoint:** `GET https://services.swpc.noaa.gov/{json \| products}/{file}.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `products/noaa-planetary-k-index.json` | 3-hourly Kp for the past week | `—` |
| `products/noaa-scales.json` | Current and forecast G/S/R scale levels | `—` |
| `products/alerts.json` | Recent alerts, watches and warnings (text) | `—` |
| `json/solar-cycle/observed-solar-cycle-indices.json` | Monthly sunspot numbers since 1749 | `—` |
| `json/rtsw/rtsw_wind_1m.json` | Real-time solar wind, 1-minute: proton speed, density, temperature (the old products/solar-wind files are gone) | `—` |
| `json/goes/primary/xrays-1-day.json` | GOES X-ray flux (flare detection) | `—` |

```python
from _common import get, show

BASE = "https://services.swpc.noaa.gov"


def main():
    kp = get(f"{BASE}/products/noaa-planetary-k-index.json")
    cycle = get(f"{BASE}/json/solar-cycle/observed-solar-cycle-indices.json")
    peak = max(kp, key=lambda r: r["Kp"])
    show({"kp_last_week": {"max": peak["Kp"], "at": peak["time_tag"], "storm": peak["Kp"] >= 5},
          "latest_kp": kp[-3:],
          "monthly_sunspot_number": [{"month": r["time-tag"], "ssn": r["ssn"]} for r in cycle[-4:]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "kp_last_week": {
    "max": 7.0,
    "at": "2026-10-10T15:00:00",
    "storm": true
  },
  "latest_kp": [
    {
      "time_tag": "2026-10-12T03:00:00",
      "Kp": 2.33,
      "a_running": 9,
      "station_count": 8
    },
    {
      "time_tag": "2026-10-12T06:00:00",
      "Kp": 3.33,
      "a_running": 18,
      "station_count": 8
    },
    {
      "time_tag": "2026-10-12T09:00:00",
      "Kp": 1.33,
      "a_running": 5,
      "station_count": 8
    }
  ],
  "monthly_sunspot_number": [
    {
      "month": "2026-06",
      "ssn": 94.4
    },
    {
      "month": "2026-07",
      "ssn": 78.1
    },
    {
      "month": "2026-08",
      "ssn": 76.0
    },
    {
      "month": "2026-09",
      "ssn": 60.3
    }
  ]
}
```
</details>

---

### NSF Award Search API

| | |
|:--|:--|
| **Docs** | https://resources.research.gov/common/webapi/awardapisearch-v1.htm |
| **Auth** | None |
| **Register / token** | No registration. Every National Science Foundation award since ~1960 with abstracts. |
| **Script** | [`scripts/science_research/nsf_awards.py`](scripts/science_research/nsf_awards.py) |

**What you can get:**

- Awards: title, abstract, amount obligated, start/end dates, program, directorate, PI and institution
- Search by keyword, institution, state, PI name, program, date range

**Capstone ideas:** Science-funding landscape (which topics/institutions get funded), abstract topic modeling, funding vs. output (OpenAlex).

**Endpoint:** `GET https://api.nsf.gov/services/v1/awards.json`

| Parameter | Description | Example |
|:--|:--|:--|
| `keyword` | Full-text search in title/abstract (supports quotes) | `"machine learning"` |
| `awardeeName` | Institution - wrap in double quotes, otherwise the filter is silently ignored | `"George Washington University"` |
| `awardeeStateCode / awardeeCity` | Institution location | `DC` |
| `dateStart / dateEnd` | Award date range, MM/DD/YYYY | `01/01/2025` |
| `fundProgramName` | NSF program | `—` |
| `printFields` | Documented field selector, but currently ignored (full records are returned) | `—` |
| `rpp / offset` | Results per page (max 25) and 1-based offset | `25 / 1` |

```python
from _common import get, show


def main():
    d = get("https://api.nsf.gov/services/v1/awards.json", params={
        "awardeeName": '"George Washington University"',  # the quotes are required for this filter to apply
        "dateStart": "01/01/2025", "rpp": 25})
    awards = sorted(d["response"]["award"], key=lambda a: -float(a.get("fundsObligatedAmt") or 0))
    show({"gwu_awards_since_2025 (first page of 25)": len(awards), "awardee": awards[0]["awardee"] if awards else None,
          "total_obligated_usd (first page)": round(sum(float(a.get("fundsObligatedAmt") or 0) for a in awards)),
          "largest": [{"title": a["title"], "pi": f'{a.get("piFirstName", "")} {a.get("piLastName", "")}',
                       "amount_usd": a.get("fundsObligatedAmt"), "program": a.get("fundProgramName"),
                       "start": a.get("startDate")} for a in awards[:4]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "gwu_awards_since_2025 (first page of 25)": 25,
  "awardee": "GEORGE WASHINGTON UNIVERSITY (THE)",
  "total_obligated_usd (first page)": 10642251,
  "largest": [
    {
      "title": "VINES: Track 2: CERTAIN: Connectivity Effectiveness and Reliability Through AI-native Networking",
      "pi": "Tian Lan",
      "amount_usd": "1699107",
      "program": "TIP-CHIPS KTA-6 Communications",
      "start": "06/15/2026"
    },
    {
      "title": "NSF/BIO-UKRI/BBSRC: Tipping the balance of amphibian microbiome-immune interactions towards greater antifungal defences",
      "pi": "Leon Grayfer",
      "amount_usd": "1199311",
      "program": "Cross-BIO Activities",
      "start": "08/01/2026"
    },
    {
      "title": "Investigating Nucleon Structure with Electromagnetic Probes",
      "pi": "Evangeline Downie",
      "amount_usd": "750000",
      "program": "Nuclear Physics - Experiment",
      "start": "07/01/2026"
    },
    {
      "title": "CICI: UCSS: Trustworthy Cyberinfrastructure-Enabled Data-Driven Electronic Design Automation for Collaborative Design of Analog/Mixed-Signal Integrated Circuits",
      "pi": "Weidong Cao",
      "amount_usd": "600000",
      "program": "Cybersecurity Innovation",
      "start": "09/01/2026"
    }
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

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "total": 1590500,
  "top_cited": [
    {
      "title": "Accurate structure prediction of biomolecular interactions with AlphaFold 3",
      "year": 2024,
      "cited_by": 15333,
      "venue": "Nature"
    },
    {
      "title": "Detecting Functionality-Specific Vulnerabilities via Retrieving Individual Functionality-Equivalent APIs in Open-Source Repositories",
      "year": 2025,
      "cited_by": 15249,
      "venue": "Leibniz international proceedings in informatics"
    },
    {
      "title": "The Difference between Emergency Remote Teaching and Online Learning",
      "year": 2024,
      "cited_by": 6250,
      "venue": "BRILL eBooks"
    },
    {
      "title": "Global incidence, prevalence, years lived with disability (YLDs), disability-adjusted life-years (DALYs), and healthy life expectancy (HALE) for 371 diseases an...",
      "year": 2024,
      "cited_by": 5011,
      "venue": "The Lancet"
    },
    {
      "title": "Kaldi Speech Recognition Toolkit",
      "year": 2024,
      "cited_by": 4609,
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

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

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

### RCSB Protein Data Bank (search + data APIs)

| | |
|:--|:--|
| **Docs** | https://search.rcsb.org/ · https://data.rcsb.org/ |
| **Auth** | None |
| **Register / token** | No registration. Two APIs: Search (POST a JSON query, get IDs) and Data (GET details by ID, or GraphQL). |
| **Script** | [`scripts/science_research/rcsb_pdb.py`](scripts/science_research/rcsb_pdb.py) |

**What you can get:**

- 230k+ experimentally determined 3D structures of proteins, DNA, RNA and complexes
- Method (X-ray, cryo-EM, NMR), resolution, release date, organism, ligands, citations
- Full-text, sequence, structure-similarity and chemical searches

**Capstone ideas:** Structural-biology datasets for ML (with AlphaFold), cryo-EM adoption trends, drug-target structure lookup.

**Endpoint:** `POST https://search.rcsb.org/rcsbsearch/v2/query  ·  GET https://data.rcsb.org/rest/v1/core/entry/{id}`

| Parameter | Description | Example |
|:--|:--|:--|
| `query (search body)` | {type: terminal, service: full_text \| text \| sequence \| structure, parameters} | `{"service": "full_text", "parameters": {"value": "spike glycoprotein"}}` |
| `return_type` | entry \| polymer_entity \| assembly \| non_polymer_entity | `entry` |
| `request_options.paginate` | {start, rows} | `{"start": 0, "rows": 5}` |
| `request_options.sort` | [{sort_by, direction}], e.g. rcsb_accession_info.initial_release_date | `desc` |
| `/rest/v1/core/entry/{id}` | Entry details: method, resolution, title, dates | `4HHB` |
| `https://data.rcsb.org/graphql` | GraphQL alternative for batch details | `—` |

```python
import requests

from _common import USER_AGENT, get, show


def main():
    q = {"query": {"type": "terminal", "service": "full_text", "parameters": {"value": "spike glycoprotein"}},
         "return_type": "entry",
         "request_options": {"paginate": {"start": 0, "rows": 4},
                             "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}]}}
    r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    hits = r.json()
    out = []
    for h in hits["result_set"]:
        e = get(f'https://data.rcsb.org/rest/v1/core/entry/{h["identifier"]}')
        out.append({"pdb_id": h["identifier"], "title": e["struct"]["title"],
                    "method": e["exptl"][0]["method"], "resolution_A": (e.get("rcsb_entry_info") or {}).get("resolution_combined"),
                    "released": e["rcsb_accession_info"]["initial_release_date"][:10]})
    show({"matching_entries": hits["total_count"], "latest": out})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "matching_entries": 3384,
  "latest": [
    {
      "pdb_id": "26DA",
      "title": "Cryo-EM structure of spike protein within graphene reservoir sandwich",
      "method": "ELECTRON MICROSCOPY",
      "resolution_A": [
        2.88
      ],
      "released": "2026-10-07"
    },
    {
      "pdb_id": "9TGC",
      "title": "Structure of B10 anti-stem antibody in complex with ZEBOV spike complex",
      "method": "ELECTRON MICROSCOPY",
      "resolution_A": [
        2.5
      ],
      "released": "2026-10-07"
    },
    {
      "pdb_id": "9X23",
      "title": "Ra22QT77 RBD in complex with Bat ACE2",
      "method": "ELECTRON MICROSCOPY",
      "resolution_A": [
        2.51
      ],
      "released": "2026-10-07"
    },
    {
      "pdb_id": "38PM",
      "title": "Australian bat lyssavirus glycoprotein PH domain in complex with broadly neutralizing human antibodies A6 and RVC20",
      "method": "ELECTRON MICROSCOPY",
      "resolution_A": [
        3.6
      ],
      "released": "2026-09-30"
    }
  ]
}
```
</details>

---

### ROR (Research Organization Registry)

| | |
|:--|:--|
| **Docs** | https://ror.readme.io/docs/rest-api |
| **Auth** | None |
| **Register / token** | No registration. Persistent IDs for 110k+ universities, labs, hospitals and funders; CC0. From 2026 heavy users are asked to send a client ID header (free). |
| **Script** | [`scripts/science_research/ror.py`](scripts/science_research/ror.py) |

**What you can get:**

- Organization names (incl. aliases and other languages), type, country, website, established year
- Parent/child relationships, external IDs (GRID, ISNI, Wikidata, Crossref Funder ID)
- Affiliation matching: map messy affiliation strings to an organization

**Capstone ideas:** Disambiguating institutions in publication/grant data (OpenAlex, NSF, NIH), building collaboration networks.

**Endpoint:** `GET https://api.ror.org/v2/organizations`

| Parameter | Description | Example |
|:--|:--|:--|
| `query` | Name search | `george washington university` |
| `affiliation` | Match a full affiliation string (returns chosen match + score); spell names out - abbreviations like GWU can mismatch | `Department of Statistics, George Washington University` |
| `filter` | types:education, country.country_code:US, status:active | `types:education` |
| `page` | 20 results per page | `1` |
| `/v2/organizations/{ror_id}` | One organization | `00y4zzh67` |

```python
from _common import get, show


def main():
    aff = "Department of Statistics, George Washington University, Washington DC"
    m = get("https://api.ror.org/v2/organizations", params={"affiliation": aff})
    best = next(i for i in m["items"] if i.get("chosen")) if any(i.get("chosen") for i in m["items"]) else m["items"][0]
    org = best["organization"]
    show({"affiliation_input": aff, "matched": best.get("chosen"),
          "score": best.get("score"), "ror_id": org["id"],
          "name": next(n["value"] for n in org["names"] if "ror_display" in n["types"]),
          "types": org["types"], "established": org.get("established"),
          "external_ids": {e["type"]: e.get("preferred") or e["all"][0] for e in org["external_ids"]}})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "affiliation_input": "Department of Statistics, George Washington University, Washington DC",
  "matched": true,
  "score": 1.0,
  "ror_id": "https://ror.org/00y4zzh67",
  "name": "George Washington University",
  "types": [
    "education",
    "funder"
  ],
  "established": 1821,
  "external_ids": {
    "fundref": "100007108",
    "grid": "grid.253615.6",
    "isni": "0000 0004 1936 9510",
    "wikidata": "Q432637"
  }
}
```
</details>

---

### SDSS SkyServer (Sloan Digital Sky Survey)

| | |
|:--|:--|
| **Docs** | https://skyserver.sdss.org/dr18/en/help/docs/api.aspx |
| **Auth** | None |
| **Register / token** | No registration. Run SQL against the SDSS catalog (hundreds of millions of objects). Queries time out after ~10 minutes and return at most 500,000 rows; use CasJobs (free account) for larger jobs. |
| **Script** | [`scripts/science_research/sdss_skyserver.py`](scripts/science_research/sdss_skyserver.py) |

**What you can get:**

- Photometric objects (PhotoObj): positions, magnitudes in u,g,r,i,z bands, object type (star/galaxy)
- Spectra (SpecObj): redshift, spectral class (GALAXY, QSO, STAR), velocity dispersion
- Image cutouts by coordinates (ImgCutout service)

**Capstone ideas:** Galaxy/star/quasar classification from colors (classic ML task), photometric redshift regression.

**Endpoint:** `GET https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch`

| Parameter | Description | Example |
|:--|:--|:--|
| `cmd` | SQL (SQL Server dialect; use TOP n) | `select top 10 z, class from SpecObj where zWarning = 0` |
| `format` | json \| csv \| xml \| html | `json` |
| `tables` | PhotoObj, SpecObj, Galaxy, Star, PhotoObjAll, SpecPhoto (joined view) | `SpecObj` |
| `ImgCutout/getjpeg?ra=&dec=&scale=&width=&height=` | Sibling endpoint: JPEG cutout of the sky | `ra=180&dec=0&scale=0.4` |

```python
from _common import get, show

SQL = """select class, count(*) as n, avg(z) as mean_redshift, max(z) as max_redshift
from SpecObj where zWarning = 0 group by class"""  # TOP n without ORDER BY returns a biased (nearby-galaxy) slice


def main():
    d = get("https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch", params={"cmd": SQL, "format": "json"})
    show({"sample": "all SDSS spectra with good redshifts (zWarning = 0)", "by_class": [
        {"class": r["class"], "count": r["n"], "mean_z": round(r["mean_redshift"], 3), "max_z": round(r["max_redshift"], 2)}
        for r in d[0]["Rows"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "sample": "all SDSS spectra with good redshifts (zWarning = 0)",
  "by_class": [
    {
      "class": "GALAXY",
      "count": 2639535,
      "mean_z": 0.405,
      "max_z": 2.0
    },
    {
      "class": "QSO",
      "count": 753228,
      "mean_z": 1.709,
      "max_z": 7.04
    },
    {
      "class": "STAR",
      "count": 890744,
      "mean_z": -0.0,
      "max_z": 0.0
    }
  ]
}
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

### Spaceflight News API (SNAPI)

| | |
|:--|:--|
| **Docs** | https://api.spaceflightnewsapi.net/v4/docs/ |
| **Auth** | None |
| **Register / token** | No registration. Aggregates space news from ~40 outlets (NASA, ESA, SpaceNews, Spaceflight Now, ...). |
| **Script** | [`scripts/science_research/spaceflight_news.py`](scripts/science_research/spaceflight_news.py) |

**What you can get:**

- Articles, blogs and reports: title, summary, source site, publish date, image
- Links to related launches and events (Launch Library 2 IDs)

**Capstone ideas:** Space-industry topic and sentiment tracking, linking news coverage to launch outcomes.

**Endpoint:** `GET https://api.spaceflightnewsapi.net/v4/{articles \| blogs \| reports}/`

| Parameter | Description | Example |
|:--|:--|:--|
| `search / title_contains` | Text search | `artemis` |
| `news_site` | Filter by outlet(s), comma list | `NASA,SpaceNews` |
| `published_at_gte / published_at_lte` | Date range | `2026-01-01` |
| `has_launch` | true -> only articles linked to a launch | `true` |
| `ordering` | -published_at \| published_at | `-published_at` |
| `limit / offset` | Paging | `5` |

```python
from collections import Counter

from _common import get, show


def main():
    d = get("https://api.spaceflightnewsapi.net/v4/articles/", params={
        "search": "artemis", "published_at_gte": "2026-01-01", "ordering": "-published_at", "limit": 100})
    show({"artemis_articles_2026": d["count"],
          "top_sources": Counter(a["news_site"] for a in d["results"]).most_common(4),
          "latest": [{"title": a["title"], "site": a["news_site"], "published": a["published_at"][:10]}
                     for a in d["results"][:3]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "artemis_articles_2026": 451,
  "top_sources": [
    [
      "NASA",
      60
    ],
    [
      "SpaceNews",
      9
    ],
    [
      "ESA",
      8
    ],
    [
      "SpacePolicyOnline.com",
      6
    ]
  ],
  "latest": [
    {
      "title": "Artemis II Crew Moon Photo Annotations",
      "site": "NASA",
      "published": "2026-10-07"
    },
    {
      "title": "NASA progress with SLS vehicles for upcoming Artemis missions",
      "site": "NASASpaceflight",
      "published": "2026-10-06"
    },
    {
      "title": "NASA Astronaut Christina Koch at Eagles vs. Rams",
      "site": "NASA",
      "published": "2026-10-05"
    }
  ]
}
```
</details>

---

### Zenodo (research data & software repository)

| | |
|:--|:--|
| **Docs** | https://developers.zenodo.org/ |
| **Auth** | None for search/download (token to deposit) |
| **Register / token** | No registration to search and download public records. CERN-run; every record has a DOI. Also great for publishing your own capstone dataset or code (GitHub integration). |
| **Script** | [`scripts/science_research/zenodo.py`](scripts/science_research/zenodo.py) |

**What you can get:**

- 3M+ records: datasets, software, papers, posters, presentations, each with DOI, licence, versions
- File lists with direct download links and checksums
- Download/view statistics, communities, related identifiers

**Capstone ideas:** Finding reusable datasets, studying research-data reuse, citing/archiving your own capstone outputs.

**Endpoint:** `GET https://zenodo.org/api/records`

| Parameter | Description | Example |
|:--|:--|:--|
| `q` | Elasticsearch query string; fields like resource_type.type:dataset, creators.name: | `climate AND resource_type.type:dataset` |
| `type` | dataset \| software \| publication \| poster \| ... | `dataset` |
| `sort` | bestmatch \| mostrecent \| -mostrecent | `mostrecent` |
| `communities` | Restrict to a community | `—` |
| `size / page` | Page size (max 25 anonymous) | `5` |
| `/api/records/{id}` | One record with files and stats | `—` |

```python
from _common import get, show


def main():
    d = get("https://zenodo.org/api/records", params={"q": "\"air quality\"", "type": "dataset", "sort": "mostrecent", "size": 5})
    show({"matching_datasets": d["hits"]["total"], "latest": [
        {"title": r["metadata"]["title"], "doi": r.get("doi"), "published": r["metadata"].get("publication_date"),
         "files": len(r.get("files", [])), "license": (r["metadata"].get("license") or {}).get("id"),
         "downloads": r.get("stats", {}).get("downloads")} for r in d["hits"]["hits"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:34</summary>

```json
{
  "matching_datasets": 1635,
  "latest": [
    {
      "title": "HYGIENIC ASSESSMENT OF THE IMPACT OF HOUSEHOLD AIR POLLUTION ON WOMEN'S HEALTH IN MOUNTAIN RURAL HOUSEHOLDS (A CASE STUDY OF THE SOKH DISTRICT)",
      "doi": "10.5281/zenodo.23275943",
      "published": "2026-10-10",
      "files": 1,
      "license": "cc-by-4.0",
      "downloads": 0
    },
    {
      "title": "ATMOSFERA IFLOSLANISHINING TARQALISH JARAYONINI DIFFERENSIAL TENGLAMALAR ASOSIDA MATEMATIK MODELLASHTIRISH",
      "doi": "10.5281/zenodo.23274049",
      "published": "2026-10-10",
      "files": 1,
      "license": "cc-by-4.0",
      "downloads": 2
    },
    {
      "title": "Data and code for: How well does MERRA-2 HAQAST CNN PM2.5 capture county annual levels and year-to-year anomalies in the contiguous United States?",
      "doi": "10.5281/zenodo.23271818",
      "published": "2026-10-10",
      "files": 1,
      "license": "cc-by-4.0",
      "downloads": 1
    },
    {
      "title": "Health Indices Dataset from Clinical Study A - TOLIFE Project",
      "doi": "10.5281/zenodo.23263785",
      "published": "2026-10-09",
      "files": 2,
      "license": "cc-by-4.0",
      "downloads": 199
    },
    {
      "title": "Code, trained PINN models and derived results for: Air Purifier Performance in an Occupied Office: Field CADR and Effectiveness from a Reconstructed Device-Off ...",
      "doi": "10.5281/zenodo.23240775",
      "published": "2026-10-08",
      "files": 10,
      "license": "cc-by-4.0",
      "downloads": 27
    }
  ]
}
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

### Electricity Maps API

| | |
|:--|:--|
| **Docs** | https://app.electricitymaps.com/docs |
| **Auth** | API token (header `auth-token`) |
| **Register / token** | Create a free account at https://portal.electricitymaps.com - the free tier covers live and recent data for a single zone for non-commercial use; historical/multi-zone access is paid (academic discounts exist). |
| **Env var** | `ELECTRICITYMAPS_TOKEN` |
| **Script** | [`scripts/engineering_energy/electricity_maps.py`](scripts/engineering_energy/electricity_maps.py) |

**What you can get:**

- Live and historical carbon intensity (gCO2eq/kWh) for 200+ grid zones worldwide
- Power breakdown by source incl. imports/exports, renewable and fossil-free percentages
- Forecasts (paid tiers)

**Capstone ideas:** Carbon-aware computing, comparing grid decarbonization across regions, marginal vs. average emissions.

**Endpoint:** `GET https://api.electricitymap.org/v3/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `carbon-intensity/latest?zone=` | Latest intensity for a zone | `zone=US-MIDA-PJM` |
| `power-breakdown/latest?zone=` | Latest generation and import mix | `zone=DE` |
| `carbon-intensity/history?zone=` | Past 24 hours | `zone=GB` |
| `lat / lon` | Alternative to zone: resolve the zone from coordinates | `38.9 / -77.04` |
| `/v3/zones` | List of zone codes | `—` |
| `auth-token (header)` | Your token | `<ELECTRICITYMAPS_TOKEN>` |

```python
from _common import get, key, show

BASE = "https://api.electricitymap.org/v3"


def main():
    hdr = {"auth-token": key("ELECTRICITYMAPS_TOKEN")}
    ci = get(f"{BASE}/carbon-intensity/latest", headers=hdr, params={"zone": "US-MIDA-PJM"})
    pb = get(f"{BASE}/power-breakdown/latest", headers=hdr, params={"zone": "US-MIDA-PJM"})
    show({"zone": ci["zone"], "carbonIntensity": ci["carbonIntensity"], "datetime": ci["datetime"],
          "renewablePercentage": pb.get("renewablePercentage"), "fossilFreePercentage": pb.get("fossilFreePercentage")})
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "zone": "US-MIDA-PJM",
  "carbonIntensity": 372,
  "datetime": "2026-10-12T12:00:00.000Z",
  "renewablePercentage": 11,
  "fossilFreePercentage": 47
}
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

<details><summary><b>Sample output</b> — live run 2026-10-12 07:40</summary>

```json
{
  "total": 88318,
  "repos": [
    {
      "repo": "huggingface/transformers",
      "stars": 167561,
      "forks": 34846,
      "open_issues": 2339,
      "license": "Apache-2.0"
    },
    {
      "repo": "pytorch/pytorch",
      "stars": 104713,
      "forks": 32187,
      "open_issues": 17710,
      "license": "NOASSERTION"
    },
    {
      "repo": "d2l-ai/d2l-zh",
      "stars": 81598,
      "forks": 12421,
      "open_issues": 127,
      "license": "Apache-2.0"
    },
    {
      "repo": "openbq-org/OpenBB",
      "stars": 74143,
      "forks": 7663,
      "open_issues": 101,
      "license": "NOASSERTION"
    },
    {
      "repo": "rohitg00/ai-engineering-from-scratch",
      "stars": 67797,
      "forks": 11880,
      "open_issues": 70,
      "license": "MIT"
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

<details><summary><b>Sample output</b> — live run 2026-10-12 07:40</summary>

```json
[
  {
    "id": "BAAI/bge-reranker-v2-m3",
    "downloads": 16000331,
    "likes": 1232,
    "library_name": "sentence-transformers"
  },
  {
    "id": "ProsusAI/finbert",
    "downloads": 4346199,
    "likes": 1266,
    "library_name": "transformers"
  },
  {
    "id": "SamLowe/roberta-base-go_emotions",
    "downloads": 3880580,
    "likes": 698,
    "library_name": "transformers"
  },
  {
    "id": "BAAI/bge-reranker-base",
    "downloads": 3864287,
    "likes": 250,
    "library_name": "sentence-transformers"
  },
  {
    "id": "meta-llama/Prompt-Guard-86M",
    "downloads": 2737390,
    "likes": 422,
    "library_name": "transformers"
  }
]
```
</details>

---

### NASA POWER (solar & meteorological data)

| | |
|:--|:--|
| **Docs** | https://power.larc.nasa.gov/docs/services/api/ |
| **Auth** | None |
| **Register / token** | No registration. Global coverage from satellite and reanalysis data, 1981-present (solar from 1984). |
| **Script** | [`scripts/engineering_energy/nasa_power.py`](scripts/engineering_energy/nasa_power.py) |

**What you can get:**

- Daily/hourly/monthly solar irradiance (GHI, DNI, diffuse), temperature, humidity, wind at 2/10/50 m, precipitation
- Any lat/lon on Earth, or regional grids
- Communities tuned for renewable energy (RE), agriculture (AG), buildings (SB)

**Capstone ideas:** Solar and wind resource assessment, crop-yield models with weather covariates, building energy simulation.

**Endpoint:** `GET https://power.larc.nasa.gov/api/temporal/{daily \| hourly \| monthly \| climatology}/point`

| Parameter | Description | Example |
|:--|:--|:--|
| `parameters` | Comma list: ALLSKY_SFC_SW_DWN (solar, kWh/m²/day), T2M, RH2M, WS10M, WS50M, PRECTOTCORR, ... | `ALLSKY_SFC_SW_DWN,T2M` |
| `community` | RE (renewable energy) \| AG (agroclimatology) \| SB (sustainable buildings) | `RE` |
| `latitude / longitude` | Point location | `38.9 / -77.04` |
| `start / end` | YYYYMMDD (daily/hourly) or YYYY (monthly) | `20260101 / 20260930` |
| `format` | JSON \| CSV \| ASCII \| NETCDF | `JSON` |
| `/api/temporal/daily/regional` | Sibling endpoint: a lat/lon box (latitude-min, latitude-max, ...) | `—` |

```python
from _common import get, show


def main():
    d = get("https://power.larc.nasa.gov/api/temporal/monthly/point", params={
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,WS10M", "community": "RE", "latitude": 38.9, "longitude": -77.04,
        "start": 2025, "end": 2025, "format": "JSON"})["properties"]["parameter"]
    months = [k for k in d["ALLSKY_SFC_SW_DWN"] if not k.endswith("13")]  # month 13 = annual value
    show({"location": "Washington, DC (2025)", "annual_solar_kWh_m2_day": d["ALLSKY_SFC_SW_DWN"]["202513"],
          "monthly": [{"month": m[4:], "solar_kWh_m2_day": d["ALLSKY_SFC_SW_DWN"][m], "temp_C": d["T2M"][m],
                       "wind_10m_m_s": d["WS10M"][m]} for m in months[::3]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:41</summary>

```json
{
  "location": "Washington, DC (2025)",
  "annual_solar_kWh_m2_day": 4.1573,
  "monthly": [
    {
      "month": "01",
      "solar_kWh_m2_day": 2.4898,
      "temp_C": -1.95,
      "wind_10m_m_s": 4.96
    },
    {
      "month": "04",
      "solar_kWh_m2_day": 4.9042,
      "temp_C": 13.58,
      "wind_10m_m_s": 4.89
    },
    {
      "month": "07",
      "solar_kWh_m2_day": 6.1805,
      "temp_C": 27.36,
      "wind_10m_m_s": 2.54
    },
    {
      "month": "10",
      "solar_kWh_m2_day": 3.7102,
      "temp_C": 14.88,
      "wind_10m_m_s": 4.44
    }
  ]
}
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

<details><summary><b>Sample output</b> — live run 2026-10-12 07:41</summary>

```json
{
  "station_counts_near_DC": 629,
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

<details><summary><b>Sample output</b> — live run 2026-10-12 07:41</summary>

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

### Our World in Data — Energy dataset

| | |
|:--|:--|
| **Docs** | https://github.com/owid/energy-data |
| **Auth** | None (CSV on GitHub) |
| **Register / token** | No registration. One tidy country-year CSV (~10 MB) combining the Energy Institute Statistical Review, Ember and EIA data; codebook in the repo. CC-BY. |
| **Script** | [`scripts/engineering_energy/owid_energy.py`](scripts/engineering_energy/owid_energy.py) |

**What you can get:**

- Primary energy and electricity by source (coal, gas, oil, nuclear, hydro, wind, solar, biofuels) for every country
- Shares of renewables/low-carbon, per-capita consumption, carbon intensity of electricity, energy per GDP
- Years 1900-present (coverage varies by variable)

**Capstone ideas:** Energy-transition tracking, renewable adoption drivers, decoupling of energy use from growth.

**Endpoint:** `GET https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv`

| Parameter | Description | Example |
|:--|:--|:--|
| `key columns` | country, year, iso_code, population, gdp | `—` |
| `{source}_share_elec` | Share of electricity from a source (%) | `solar_share_elec, wind_share_elec` |
| `{source}_consumption / {source}_electricity` | Primary energy (TWh) / electricity generation (TWh) | `coal_electricity` |
| `carbon_intensity_elec` | gCO2 per kWh of electricity | `—` |
| `owid-energy-codebook.csv` | Variable definitions and sources (same repo) | `—` |

```python
import csv
import io

from _common import get, show


def main():
    rows = csv.DictReader(io.StringIO(get("https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv",
                                          as_json=False)))
    keep = {"United States", "China", "Germany", "India", "World"}
    latest = {}
    for r in rows:  # latest year with a value for each country
        if r["country"] in keep and r["renewables_share_elec"]:
            latest[r["country"]] = r
    show([{"country": c, "year": r["year"], "renewables_share_elec_pct": round(float(r["renewables_share_elec"]), 1),
           "solar_pct": round(float(r["solar_share_elec"] or 0), 1), "wind_pct": round(float(r["wind_share_elec"] or 0), 1),
           "coal_pct": round(float(r["coal_share_elec"] or 0), 1),
           "carbon_intensity_g_per_kWh": r["carbon_intensity_elec"] and round(float(r["carbon_intensity_elec"]))}
          for c, r in sorted(latest.items())])
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:41</summary>

```json
[
  {
    "country": "China",
    "year": "2025",
    "renewables_share_elec_pct": 37.0,
    "solar_pct": 11.1,
    "wind_pct": 10.7,
    "coal_pct": 54.3,
    "carbon_intensity_g_per_kWh": 525
  },
  {
    "country": "Germany",
    "year": "2025",
    "renewables_share_elec_pct": 59.1,
    "solar_pct": 17.9,
    "wind_pct": 27.2,
    "coal_pct": 20.6,
    "carbon_intensity_g_per_kWh": 330
  },
  {
    "country": "India",
    "year": "2025",
    "renewables_share_elec_pct": 24.1,
    "solar_pct": 9.4,
    "wind_pct": 5.0,
    "coal_pct": 70.8,
    "carbon_intensity_g_per_kWh": 670
  },
  {
    "country": "United States",
    "year": "2025",
    "renewables_share_elec_pct": 25.6,
    "solar_pct": 8.6,
    "wind_pct": 10.3,
    "coal_pct": 16.3,
    "carbon_intensity_g_per_kWh": 384
  },
  {
    "country": "World",
    "year": "2025",
    "renewables_share_elec_pct": 33.8,
    "solar_pct": 8.7,
    "wind_pct": 8.5,
    "coal_pct": 33.0,
    "carbon_intensity_g_per_kWh": 458
  }
]
```
</details>

---

### PVGIS (EU JRC photovoltaic estimates)

| | |
|:--|:--|
| **Docs** | https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en |
| **Auth** | None |
| **Register / token** | No registration. Max ~30 requests/second. Covers Europe, Africa, most of Asia and the Americas. |
| **Script** | [`scripts/engineering_energy/pvgis.py`](scripts/engineering_energy/pvgis.py) |

**What you can get:**

- Expected PV output for a system anywhere: monthly and yearly energy (kWh), irradiation, losses
- Optimal tilt/azimuth, hourly time series, typical meteorological year (TMY), off-grid sizing

**Capstone ideas:** Rooftop-solar economics by location, comparing PVGIS vs. NLR PVWatts estimates.

**Endpoint:** `GET https://re.jrc.ec.europa.eu/api/v5_3/{tool}`

| Parameter | Description | Example |
|:--|:--|:--|
| `tool` | PVcalc (grid-connected) \| seriescalc (hourly) \| tmy \| SHScalc (off-grid) \| MRcalc (monthly radiation) | `PVcalc` |
| `lat / lon` | Location | `38.9 / -77.04` |
| `peakpower / loss` | System size in kWp; system losses in % | `4 / 14` |
| `angle / aspect` | Tilt in degrees; azimuth (0 = south, -90 = east, 90 = west) | `30 / 0` |
| `optimalangles` | 1 -> compute optimal tilt and azimuth | `1` |
| `outputformat` | json \| csv \| basic | `json` |

```python
from _common import get, show


def main():
    d = get("https://re.jrc.ec.europa.eu/api/v5_3/PVcalc", params={
        "lat": 38.9, "lon": -77.04, "peakpower": 4, "loss": 14, "optimalangles": 1, "outputformat": "json"})
    out, mount = d["outputs"], d["inputs"]["mounting_system"]["fixed"]
    show({"system": "4 kWp, 14% losses, Washington DC", "optimal_tilt_deg": mount["slope"]["value"],
          "optimal_azimuth_deg": mount["azimuth"]["value"], "yearly_kWh": out["totals"]["fixed"]["E_y"],
          "monthly_kWh": {m["month"]: m["E_m"] for m in out["monthly"]["fixed"]}}, max_items=12)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:41</summary>

```json
{
  "system": "4 kWp, 14% losses, Washington DC",
  "optimal_tilt_deg": 37,
  "optimal_azimuth_deg": -7,
  "yearly_kWh": 5668.39,
  "monthly_kWh": {
    "1": 424.62,
    "2": 441.33,
    "3": 534.35,
    "4": 530.55,
    "5": 515.87,
    "6": 502.62,
    "7": 518.23,
    "8": 501.87,
    "9": 459.97,
    "10": 444.21,
    "11": 423.4,
    "12": 371.36
  }
}
```
</details>

---

### UK Carbon Intensity API (National Grid ESO)

| | |
|:--|:--|
| **Docs** | https://carbon-intensity.github.io/api-definitions/ |
| **Auth** | None |
| **Register / token** | No registration. Half-hourly data for Great Britain since 2017, with 96-hour forecasts. |
| **Script** | [`scripts/engineering_energy/uk_carbon_intensity.py`](scripts/engineering_energy/uk_carbon_intensity.py) |

**What you can get:**

- Carbon intensity of electricity (gCO2/kWh): actual and forecast, with index (very low ... very high)
- Generation mix by fuel (gas, wind, solar, nuclear, imports, ...)
- Regional intensity for 14 DNO regions and by postcode

**Capstone ideas:** Load-shifting / carbon-aware scheduling, forecasting grid intensity, wind share vs. intensity analysis.

**Endpoint:** `GET https://api.carbonintensity.org.uk/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `intensity` | Current half-hour (also /intensity/date/{YYYY-MM-DD}, /intensity/{from}/{to}) | `intensity` |
| `intensity/{from}/fw48h` | 48-hour forecast from a time | `intensity/2026-10-12T12:00Z/fw48h` |
| `generation` | Current generation mix in % (also /generation/{from}/{to}) | `generation` |
| `regional / regional/postcode/{outcode}` | All regions, or one area by outward postcode (full district, e.g. SW1A not SW1) | `regional/postcode/RG10` |
| `intensity/stats/{from}/{to}` | Max/min/average over a period | `—` |

```python
from _common import get, show

BASE = "https://api.carbonintensity.org.uk"


def main():
    now = get(f"{BASE}/intensity")["data"][0]
    mix = get(f"{BASE}/generation")["data"]
    show({"period": f'{now["from"]} - {now["to"]}', "intensity_gCO2_per_kWh": now["intensity"],
          "generation_mix_pct": {f["fuel"]: f["perc"] for f in sorted(mix["generationmix"], key=lambda f: -f["perc"])}},
         max_items=10)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 07:41</summary>

```json
{
  "period": "2026-10-12T11:00Z - 2026-10-12T11:30Z",
  "intensity_gCO2_per_kWh": {
    "forecast": 91,
    "actual": 93,
    "index": "moderate"
  },
  "generation_mix_pct": {
    "wind": 43,
    "gas": 16.3,
    "imports": 14.9,
    "nuclear": 11.2,
    "biomass": 7.8,
    "solar": 6.6,
    "coal": 0,
    "other": 0,
    "hydro": 0
  }
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

<details><summary><b>Sample output</b> — live run 2026-10-12 07:41</summary>

```json
[
  {
    "site": "POTOMAC RIVER NEAR WASH, DC LITTLE FALLS PUMP STA",
    "variable": "Streamflow, ft&#179;/s",
    "latest": {
      "value": "8160",
      "qualifiers": [
        "P"
      ],
      "dateTime": "2026-10-12T06:50:00.000-04:00"
    }
  },
  {
    "site": "POTOMAC RIVER NEAR WASH, DC LITTLE FALLS PUMP STA",
    "variable": "Gage height, ft",
    "latest": {
      "value": "3.77",
      "qualifiers": [
        "P"
      ],
      "dateTime": "2026-10-12T06:50:00.000-04:00"
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

## 🎬 Culture, Media & Entertainment

### Art Institute of Chicago API

| | |
|:--|:--|
| **Docs** | https://api.artic.edu/docs/ |
| **Auth** | None |
| **Register / token** | No registration. 60 requests/minute; use the fields= parameter to keep responses small. CC0 for public-domain works; high-resolution images via IIIF. |
| **Script** | [`scripts/culture_misc/art_institute_chicago.py`](scripts/culture_misc/art_institute_chicago.py) |

**What you can get:**

- 120k+ artworks: title, artist, date, medium, dimensions, place of origin, department, style, subject terms
- Public-domain flag and IIIF image URLs, color data, exhibition history
- Artists, exhibitions, tours, publications

**Capstone ideas:** Computer vision on museum images, color and style evolution over time, provenance and acquisition patterns.

**Endpoint:** `GET https://api.artic.edu/api/v1/{artworks \| artworks/search \| artists \| exhibitions}`

| Parameter | Description | Example |
|:--|:--|:--|
| `q` | Full-text search | `monet` |
| `query (search)` | Elasticsearch query DSL as JSON for precise filters | `{"term": {"is_public_domain": true}}` |
| `fields` | Fields to return | `id,title,artist_title,date_display,image_id,is_public_domain` |
| `limit / page` | Max 100 | `5` |
| `IIIF image URL` | {config.iiif_url}/{image_id}/full/843,/0/default.jpg | `—` |

```python
from _common import get, show


def main():
    d = get("https://api.artic.edu/api/v1/artworks/search", params={
        "q": "monet", "limit": 5,
        "fields": "id,title,artist_title,date_display,medium_display,is_public_domain,image_id"})
    iiif = d["config"]["iiif_url"]
    show({"matches": d["pagination"]["total"], "artworks": [
        {"title": a["title"], "artist": a["artist_title"], "date": a["date_display"], "public_domain": a["is_public_domain"],
         "image": f'{iiif}/{a["image_id"]}/full/843,/0/default.jpg' if a.get("image_id") else None} for a in d["data"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:09</summary>

```json
{
  "matches": 133165,
  "artworks": [
    {
      "title": "Water Lilies",
      "artist": "Claude Monet",
      "date": "1906",
      "public_domain": true,
      "image": "https://www.artic.edu/iiif/2/3c27b499-af56-f0d5-93b5-a7f2f1ad5813/full/843,/0/default.jpg"
    },
    {
      "title": "Arrival of the Normandy Train, Gare Saint-Lazare",
      "artist": "Claude Monet",
      "date": "1877",
      "public_domain": true,
      "image": "https://www.artic.edu/iiif/2/0f1cc0e0-e42e-be16-3f71-2022da38cb93/full/843,/0/default.jpg"
    },
    {
      "title": "Stacks of Wheat (End of Summer)",
      "artist": "Claude Monet",
      "date": "1890–91",
      "public_domain": true,
      "image": "https://www.artic.edu/iiif/2/a38e2828-ec6f-ece1-a30f-70243449197b/full/843,/0/default.jpg"
    },
    {
      "title": "Cliff Walk at Pourville",
      "artist": "Claude Monet",
      "date": "1882",
      "public_domain": true,
      "image": "https://www.artic.edu/iiif/2/b0effb1c-ff23-bbaa-f809-9fd94e31c1a0/full/843,/0/default.jpg"
    },
    {
      "title": "Water Lily Pond",
      "artist": "Claude Monet",
      "date": "1900",
      "public_domain": true,
      "image": "https://www.artic.edu/iiif/2/8534685d-1102-e1e3-e194-94f6e925e8b0/full/843,/0/default.jpg"
    }
  ]
}
```
</details>

---

### Deezer API (music charts & catalog)

| | |
|:--|:--|
| **Docs** | https://developers.deezer.com/api |
| **Auth** | None for public data |
| **Register / token** | No registration for catalog and chart endpoints (50 requests / 5 seconds). OAuth only for user data. The docs page now asks you to log in with a (free) Deezer account. |
| **Script** | [`scripts/culture_misc/deezer.py`](scripts/culture_misc/deezer.py) |

**What you can get:**

- Global and per-genre charts: top tracks, albums, artists, playlists, podcasts
- Track details: duration, BPM, rank (popularity), explicit flag, 30-second preview MP3
- Artists (fan counts, top tracks, related artists), albums, search

**Capstone ideas:** Chart dynamics, audio analysis on preview clips, related-artist networks, explicit-content trends.

**Endpoint:** `GET https://api.deezer.com/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `chart/{genre_id}/{tracks \| albums \| artists \| playlists}` | Charts; genre 0 = all (list via /genre) | `chart/0/tracks` |
| `search?q=` | Search with advanced syntax: artist:"" track:"" album:"" bpm_min: dur_max: | `q=artist:"daft punk"` |
| `track/{id} · album/{id} · artist/{id}` | Details (track includes bpm, rank) | `track/3135556` |
| `artist/{id}/top · artist/{id}/related` | An artist's top tracks / similar artists | `—` |
| `limit / index` | Paging | `10 / 0` |

```python
from _common import get, show


def main():
    tracks = get("https://api.deezer.com/chart/0/tracks", params={"limit": 5})["data"]
    artists = get("https://api.deezer.com/chart/0/artists", params={"limit": 3})["data"]
    show({"top_tracks": [{"position": t["position"], "title": t["title"], "artist": t["artist"]["name"],
                          "duration_s": t["duration"], "explicit": t["explicit_lyrics"]} for t in tracks],
          "top_artists": [a["name"] for a in artists]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:09</summary>

```json
{
  "top_tracks": [
    {
      "position": 1,
      "title": "Dracula (with JENNIE)",
      "artist": "Tame Impala",
      "duration_s": 209,
      "explicit": true
    },
    {
      "position": 2,
      "title": "Boston",
      "artist": "STELLA LEFTY",
      "duration_s": 170,
      "explicit": false
    },
    {
      "position": 3,
      "title": "Choosin' Texas",
      "artist": "Ella Langley",
      "duration_s": 232,
      "explicit": false
    },
    {
      "position": 4,
      "title": "Be Her",
      "artist": "Ella Langley",
      "duration_s": 217,
      "explicit": false
    },
    {
      "position": 5,
      "title": "Been By Now",
      "artist": "Morgan Wallen",
      "duration_s": 213,
      "explicit": false
    }
  ],
  "top_artists": [
    "Taylor Swift",
    "Drake",
    "Bad Bunny"
  ]
}
```
</details>

---

### Discogs API (music releases & marketplace)

| | |
|:--|:--|
| **Docs** | https://www.discogs.com/developers |
| **Auth** | None for public data (token raises limits and unlocks images) |
| **Register / token** | Public data works without auth at 25 requests/minute; a personal token from https://www.discogs.com/settings/developers gives 60/minute. A descriptive User-Agent is required. |
| **Env var** | `DISCOGS_TOKEN (optional)` |
| **Script** | [`scripts/culture_misc/discogs.py`](scripts/culture_misc/discogs.py) |

**What you can get:**

- 16M+ releases (vinyl, CD, cassette...) with labels, formats, countries, credits, tracklists
- Artists and their discographies; master releases grouping editions
- Community data: how many users have/want a release; marketplace price statistics

**Capstone ideas:** Vinyl revival analysis, collectibility (want/have ratios) vs. price, label and credit networks.

**Endpoint:** `GET https://api.discogs.com/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `database/search?q=&type=` | Search; type release \| master \| artist \| label; filters genre, style, year, format, country | `q=nirvana&type=master` |
| `artists/{id}/releases` | An artist's discography (sort=year, sort_order) | `artists/125246/releases?sort=year` |
| `releases/{id} · masters/{id}` | Release details incl. community have/want | `releases/249504` |
| `marketplace/stats/{release_id}` | Lowest price and number for sale | `—` |
| `per_page / page` | Max 100 | `5` |
| `Authorization (header)` | Discogs token=<token> (optional) | `—` |

```python
from _common import get, show


def main():
    d = get("https://api.discogs.com/artists/125246/releases", params={"sort": "year", "sort_order": "asc",
                                                                      "per_page": 100})  # Nirvana
    masters = [r for r in d["releases"] if r["type"] == "master" and r.get("role") == "Main"]
    rel = get("https://api.discogs.com/masters/13814")  # Nevermind
    show({"artist": "Nirvana", "discography_entries": d["pagination"]["items"],
          "main_masters": [f'{r.get("year")} {r["title"]}' for r in masters][:6],
          "nevermind": {"year": rel["year"], "genres": rel["genres"], "styles": rel["styles"],
                        "tracks": len(rel["tracklist"])}})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:09</summary>

```json
{
  "artist": "Nirvana",
  "discography_entries": 2248,
  "main_masters": [
    "1988 Love Buzz b/w Big Cheese",
    "1988 Nirvana",
    "1989 Bleach",
    "1989 Blew",
    "1990 Lithium",
    "... (1 more)"
  ],
  "nevermind": {
    "year": 1991,
    "genres": [
      "Rock"
    ],
    "styles": [
      "Grunge",
      "Alternative Rock"
    ],
    "tracks": 12
  }
}
```
</details>

---

### Google Books API

| | |
|:--|:--|
| **Docs** | https://developers.google.com/books/docs/v1/using |
| **Auth** | API key (query `key`) - effectively required |
| **Register / token** | In https://console.cloud.google.com create a project, enable 'Books API', and create an API key. Keyless calls share a global quota that is often exhausted (HTTP 429). 1,000 requests/day per key by default. |
| **Env var** | `GOOGLE_BOOKS_API_KEY` |
| **Script** | [`scripts/culture_misc/google_books.py`](scripts/culture_misc/google_books.py) |

**What you can get:**

- Tens of millions of books: title, authors, publisher, published date, ISBNs, page count, categories
- Descriptions, average rating, language, cover thumbnails, preview/ebook availability and price

**Capstone ideas:** Publishing trends by subject, metadata enrichment of book datasets via ISBN, description-based recommendation.

**Endpoint:** `GET https://www.googleapis.com/books/v1/volumes`

| Parameter | Description | Example |
|:--|:--|:--|
| `q` | Search with keywords: intitle:, inauthor:, subject:, isbn:, inpublisher: | `subject:statistics` |
| `orderBy` | relevance \| newest | `newest` |
| `printType / filter` | all \| books \| magazines; partial \| full \| free-ebooks \| paid-ebooks \| ebooks | `books` |
| `langRestrict` | Two-letter language code | `en` |
| `maxResults / startIndex` | Max 40 per page | `5 / 0` |
| `key` | Your API key | `<GOOGLE_BOOKS_API_KEY>` |

```python
from _common import get, key, show


def main():
    d = get("https://www.googleapis.com/books/v1/volumes", params={
        "q": "subject:statistics", "orderBy": "newest", "printType": "books", "maxResults": 5,
        "key": key("GOOGLE_BOOKS_API_KEY")})
    show([{k: b["volumeInfo"].get(k) for k in ("title", "authors", "publishedDate", "pageCount", "categories")}
          for b in d.get("items", [])])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"title": "An Introduction to Statistical Learning", "authors": ["Gareth James", "Daniela Witten", "Trevor Hastie", "Robert Tibshirani"],
   "publishedDate": "2023-07-01", "pageCount": 607, "categories": ["Mathematics"]}
]
```
</details>

---

### Gutendex (Project Gutenberg books)

| | |
|:--|:--|
| **Docs** | https://gutendex.com/ |
| **Auth** | None |
| **Register / token** | No registration (uncached queries can take 1-2 minutes, so use long timeouts). JSON index of Project Gutenberg's 75,000+ free public-domain e-books; full texts download as plain text, EPUB or HTML from the formats links. |
| **Script** | [`scripts/culture_misc/gutendex.py`](scripts/culture_misc/gutendex.py) |

**What you can get:**

- Books: title, authors (with birth/death years), subjects, bookshelves, languages, download counts
- Direct links to full text in multiple formats
- Search, topic and language filters, popularity sort

**Capstone ideas:** Classic-literature NLP corpora, stylometry and authorship attribution, sentiment arcs of novels.

**Endpoint:** `GET https://gutendex.com/books/`

| Parameter | Description | Example |
|:--|:--|:--|
| `search` | Words in title or author name | `austen` |
| `topic` | Words in subjects or bookshelves | `detective` |
| `languages` | Comma-separated codes | `en,fr` |
| `author_year_start / author_year_end` | Authors alive in a year range | `1800 / 1899` |
| `sort` | popular (default) \| ascending \| descending (by ID) | `popular` |
| `ids` | Specific Gutenberg IDs | `1342,84` |
| `page` | 32 results per page | `1` |

```python
from _common import get, show


def main():
    d = get("https://gutendex.com/books/", params={"topic": "detective", "languages": "en"},
            timeout=120)  # uncached queries can take 1-2 minutes on this free server
    show({"matching_books": d["count"], "most_downloaded": [
        {"id": b["id"], "title": b["title"], "author": (b["authors"] or [{}])[0].get("name"),
         "downloads": b["download_count"], "plain_text": b["formats"].get("text/plain; charset=us-ascii")
                                                         or b["formats"].get("text/plain; charset=utf-8")}
        for b in d["results"][:5]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:09</summary>

```json
{
  "matching_books": 1387,
  "most_downloaded": [
    {
      "id": 1661,
      "title": "The Adventures of Sherlock Holmes",
      "author": "Doyle, Arthur Conan",
      "downloads": 96353,
      "plain_text": "https://www.gutenberg.org/files/1661/1661-0.txt"
    },
    {
      "id": 2554,
      "title": "Crime and Punishment",
      "author": "Dostoyevsky, Fyodor",
      "downloads": 76289,
      "plain_text": "https://www.gutenberg.org/ebooks/2554.txt.utf-8"
    },
    {
      "id": 65238,
      "title": "The Secret of Chimneys",
      "author": "Christie, Agatha",
      "downloads": 50307,
      "plain_text": "https://www.gutenberg.org/files/65238/65238-0.txt"
    },
    {
      "id": 28780,
      "title": "Peter the Brazen: A Mystery Story of Modern China",
      "author": "Worts, George F. (George Frank)",
      "downloads": 40745,
      "plain_text": "https://www.gutenberg.org/files/28780/28780-8.txt"
    },
    {
      "id": 38983,
      "title": "The Camp Fire Girls Solve a Mystery; Or, The Christmas Adventure at Carver House",
      "author": "Frey, Hildegard G.",
      "downloads": 37424,
      "plain_text": "https://www.gutenberg.org/files/38983/38983-0.txt"
    }
  ]
}
```
</details>

---

### Internet Archive (search & metadata)

| | |
|:--|:--|
| **Docs** | https://archive.org/developers/ |
| **Auth** | None for reading |
| **Register / token** | No registration for search, metadata and downloads. Python package: `pip install internetarchive`. |
| **Script** | [`scripts/culture_misc/internet_archive.py`](scripts/culture_misc/internet_archive.py) |

**What you can get:**

- 40M+ items: books, films, TV news, audio (Grateful Dead, 78rpm records), software, web collections
- Item metadata: title, creator, date, collection, subjects, licence, download and view counts
- File lists and direct downloads for every item; the Wayback Machine has its own APIs

**Capstone ideas:** Public-domain film and audio datasets, TV-news transcript analysis, digital-preservation studies.

**Endpoint:** `GET https://archive.org/advancedsearch.php  ·  GET https://archive.org/metadata/{identifier}`

| Parameter | Description | Example |
|:--|:--|:--|
| `q` | Lucene query over fields: collection:, mediatype:, subject:, year:, creator: | `collection:feature_films AND year:[1920 TO 1959]` |
| `fl[]` | Fields to return (repeatable) | `identifier, title, year, downloads` |
| `sort[]` | Sort field and direction | `downloads desc` |
| `rows / page` | Results per page | `5` |
| `output` | json \| csv \| xml | `json` |
| `/metadata/{identifier}` | Full metadata and file list for one item | `/metadata/night_of_the_living_dead` |

```python
from _common import get, show


def main():
    d = get("https://archive.org/advancedsearch.php", params=[
        ("q", 'collection:feature_films AND subject:"film noir"'), ("fl[]", "identifier"), ("fl[]", "title"),
        ("fl[]", "year"), ("fl[]", "downloads"), ("sort[]", "downloads desc"), ("rows", 5), ("output", "json")])["response"]
    show({"film_noir_feature_films": d["numFound"], "most_downloaded": [
        {k: x.get(k) for k in ("title", "year", "downloads")} | {"url": f'https://archive.org/details/{x["identifier"]}'}
        for x in d["docs"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:09</summary>

```json
{
  "film_noir_feature_films": 1195,
  "most_downloaded": [
    {
      "title": "The Stranger",
      "year": 1946,
      "downloads": 662151,
      "url": "https://archive.org/details/TheStranger_0"
    },
    {
      "title": "Scarlet Street",
      "year": 1945,
      "downloads": 523755,
      "url": "https://archive.org/details/ScarletStreet"
    },
    {
      "title": "Detour",
      "year": 1945,
      "downloads": 372940,
      "url": "https://archive.org/details/Detour"
    },
    {
      "title": "He Walked By Night",
      "year": 1948,
      "downloads": 258155,
      "url": "https://archive.org/details/He_Walked_By_Night.avi"
    },
    {
      "title": "The Red House",
      "year": 1947,
      "downloads": 256884,
      "url": "https://archive.org/details/TheRedHouse"
    }
  ]
}
```
</details>

---

### iTunes Search API (Apple music, podcasts, apps, books)

| | |
|:--|:--|
| **Docs** | https://performance-partners.apple.com/search-api |
| **Auth** | None |
| **Register / token** | No registration. About 20 calls/minute. Covers the Apple Music, Podcasts, App Store, Books and TV catalogs. |
| **Script** | [`scripts/culture_misc/itunes_search.py`](scripts/culture_misc/itunes_search.py) |

**What you can get:**

- Search songs, albums, artists, podcasts, apps, audiobooks, movies, TV episodes by keyword
- Price, release date, genre, track count, artwork URLs, preview clips; podcast feed URLs
- Lookup by iTunes ID, UPC, ISBN or AMG ID; country-specific storefronts

**Capstone ideas:** Podcast landscape analysis (via feed URLs), app-store category studies, music release timing.

**Endpoint:** `GET https://itunes.apple.com/{search \| lookup}`

| Parameter | Description | Example |
|:--|:--|:--|
| `term` | Search text (+ for spaces) | `true crime` |
| `media / entity` | music \| podcast \| software \| ebook \| movie \| tvShow ...; entity narrows (album, musicArtist, podcast) | `media=podcast` |
| `country` | Two-letter storefront | `us` |
| `attribute` | Field to search: artistTerm, albumTerm, genreIndex, ... | `artistTerm` |
| `limit` | Max 200 | `5` |
| `lookup?id= / ?upc= / ?isbn=` | Look up by ID; add entity=album to list an artist's albums | `lookup?id=909253&entity=album` |

```python
from _common import get, show


def main():
    d = get("https://itunes.apple.com/search", params={"term": "data science", "media": "podcast", "country": "us",
                                                        "limit": 5})
    show({"results": d["resultCount"], "podcasts": [
        {"name": p["collectionName"], "publisher": p["artistName"], "episodes": p.get("trackCount"),
         "genres": p.get("genres", [])[:2], "last_release": (p.get("releaseDate") or "")[:10], "feed": p.get("feedUrl")}
        for p in d["results"]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:09</summary>

```json
{
  "results": 5,
  "podcasts": [
    {
      "name": "The Data Science Education Podcast",
      "publisher": "Berkeley Data Science",
      "episodes": 87,
      "genres": [
        "Science",
        "Podcasts"
      ],
      "last_release": "2026-10-02",
      "feed": "https://api.substack.com/feed/podcast/29031.rss"
    },
    {
      "name": "Super Data Science: ML & AI Podcast with Jon Krohn",
      "publisher": "Jon Krohn",
      "episodes": 1035,
      "genres": [
        "Technology",
        "Podcasts"
      ],
      "last_release": "2026-10-06",
      "feed": "https://feeds.megaphone.fm/SUPERDATASCIENCEPTYLTD9836501887"
    },
    {
      "name": "Harvard Data Science Review Podcast",
      "publisher": "Harvard Data Science Review",
      "episodes": 69,
      "genres": [
        "Tech News",
        "Podcasts"
      ],
      "last_release": "2026-08-31",
      "feed": "https://feed.podbean.com/HDSR/feed.xml"
    },
    {
      "name": "The Data Science Blueprint",
      "publisher": "The Data Science Blueprint",
      "episodes": 6,
      "genres": [
        "Technology",
        "Podcasts"
      ],
      "last_release": "2024-10-15",
      "feed": "https://anchor.fm/s/fb9e7338/podcast/rss"
    },
    {
      "name": "Data Science Leaders",
      "publisher": "Domino Data Lab",
      "episodes": 101,
      "genres": [
        "Management",
        "Podcasts"
      ],
      "last_release": "2026-10-06",
      "feed": "https://feeds.captivate.fm/data-science-leaders/"
    }
  ]
}
```
</details>

---

### Last.fm API (listening data & tags)

| | |
|:--|:--|
| **Docs** | https://www.last.fm/api |
| **Auth** | API key (query `api_key`) |
| **Register / token** | Create an API account at https://www.last.fm/api/account/create (instant). Non-commercial use; keep to about 5 requests/second. |
| **Env var** | `LASTFM_API_KEY` |
| **Script** | [`scripts/culture_misc/lastfm.py`](scripts/culture_misc/lastfm.py) |

**What you can get:**

- Global and country charts (top artists/tracks) with play and listener counts
- Crowd-sourced tags (genres, moods), similar artists/tracks, artist bios
- Users' public listening histories (scrobbles)

**Capstone ideas:** Music recommendation (similarity graphs), mood/genre tagging, listening-behavior analysis.

**Endpoint:** `GET https://ws.audioscrobbler.com/2.0/?method={method}`

| Parameter | Description | Example |
|:--|:--|:--|
| `method` | chart.getTopArtists \| geo.getTopTracks \| artist.getSimilar \| track.getTopTags \| tag.getTopTracks \| user.getRecentTracks | `geo.getTopTracks` |
| `artist / track / tag / country / user` | Method-specific arguments | `country=united states` |
| `limit / page` | Paging | `5` |
| `format` | json \| xml | `json` |
| `api_key` | Your key | `<LASTFM_API_KEY>` |

```python
from _common import get, key, show


def main():
    d = get("https://ws.audioscrobbler.com/2.0/", params={
        "method": "chart.gettopartists", "api_key": key("LASTFM_API_KEY"), "format": "json", "limit": 5})
    show({"top_artists": [{k: a[k] for k in ("name", "listeners", "playcount")} for a in d["artists"]["artist"]]})
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "top_artists": [
    {"name": "Taylor Swift", "listeners": "6418811", "playcount": "1823456712"},
    {"name": "The Weeknd", "listeners": "7012345", "playcount": "1213456789"}
  ]
}
```
</details>

---

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

<details><summary><b>Sample output</b> — live run 2026-10-12 10:09</summary>

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

### MusicBrainz (open music encyclopedia)

| | |
|:--|:--|
| **Docs** | https://musicbrainz.org/doc/MusicBrainz_API |
| **Auth** | None (descriptive User-Agent required) |
| **Register / token** | No registration. Max 1 request/second; send a User-Agent like 'AppName/1.0 (contact@example.com)'. Full database dumps are available for bulk analysis. CC0 core data. |
| **Script** | [`scripts/culture_misc/musicbrainz.py`](scripts/culture_misc/musicbrainz.py) |

**What you can get:**

- Artists, release groups (albums), releases, recordings, works, labels with stable MBIDs
- Relationships (members, producers, covers, samples), tags/genres, release dates and countries
- Links to Wikidata, Discogs, Spotify and cover art (Cover Art Archive)

**Capstone ideas:** Music-history networks (collaborations), genre evolution, joining streaming data via MBIDs.

**Endpoint:** `GET https://musicbrainz.org/ws/2/{entity}/[{mbid}]`

| Parameter | Description | Example |
|:--|:--|:--|
| `entity` | artist \| release-group \| release \| recording \| work \| label \| area \| event | `artist` |
| `query (search)` | Lucene syntax with fields: artist:, country:, tag:, type:, date: | `artist:radiohead` |
| `artist= / release= (browse)` | List entities linked to an MBID | `release-group?artist={mbid}&type=album` |
| `inc (lookup)` | Include related data: aliases, tags, genres, recordings, url-rels, artist-rels | `inc=genres+url-rels` |
| `fmt` | json \| xml | `json` |
| `limit / offset` | Max 100 | `5` |

```python
import time

from _common import get, show

BASE = "https://musicbrainz.org/ws/2"


def main():
    artist = get(f"{BASE}/artist/", params={"query": "artist:radiohead", "fmt": "json", "limit": 1})["artists"][0]
    time.sleep(1.1)  # MusicBrainz allows 1 request/second
    albums = get(f"{BASE}/release-group", params={"artist": artist["id"], "type": "album", "fmt": "json", "limit": 100})
    albums_only = sorted((rg for rg in albums["release-groups"] if not rg["secondary-types"]),  # no live/compilation
                    key=lambda rg: rg.get("first-release-date") or "")
    show({"artist": artist["name"], "mbid": artist["id"], "country": artist.get("country"),
          "tags": [t["name"] for t in sorted(artist.get("tags", []), key=lambda t: -t["count"])][:5],
          "albums": [f'{rg.get("first-release-date", "")[:4]} {rg["title"]}' for rg in albums_only]}, max_items=10)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:09</summary>

```json
{
  "artist": "Radiohead",
  "mbid": "a74b1b7f-71a5-4011-9441-d0b5e4122711",
  "country": "GB",
  "tags": [
    "alternative rock",
    "art rock",
    "rock",
    "electronic",
    "british"
  ],
  "albums": [
    "1993 Pablo Honey",
    "1994 The Bends",
    "1997 Radiotick Tracks 2",
    "1997 OK Computer",
    "2000 Kid A",
    "2001 Amnesiac",
    "2003 Hail to the Thief",
    "2007 In Rainbows",
    "2011 The King of Limbs",
    "2016 A Moon Shaped Pool"
  ]
}
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

<details><summary><b>Sample output</b> — live run 2026-10-12 10:10</summary>

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

### RAWG Video Games Database API

| | |
|:--|:--|
| **Docs** | https://api.rawg.io/docs/ |
| **Auth** | API key (query `key`) |
| **Register / token** | Get a free key at https://rawg.io/apidocs (20,000 requests/month on the free tier). Attribution to RAWG with a link is required. |
| **Env var** | `RAWG_API_KEY` |
| **Script** | [`scripts/culture_misc/rawg.py`](scripts/culture_misc/rawg.py) |

**What you can get:**

- 500k+ games across all platforms: release date, genres, platforms, ESRB rating, Metacritic score, playtime
- User ratings distribution, tags, developers, publishers, stores, screenshots, achievements

**Capstone ideas:** What predicts high ratings, platform exclusivity trends, release-calendar analysis.

**Endpoint:** `GET https://api.rawg.io/api/{games \| games/{id} \| genres \| platforms \| developers}`

| Parameter | Description | Example |
|:--|:--|:--|
| `search` | Text search | `zelda` |
| `dates` | Release-date range | `2025-01-01,2025-12-31` |
| `genres / platforms / tags` | IDs or slugs (comma list) | `genres=indie` |
| `ordering` | name \| released \| added \| created \| rating \| metacritic (prefix - to reverse) | `-metacritic` |
| `metacritic` | Score range | `80,100` |
| `page_size / page` | Max 40 | `5` |
| `key` | Your key | `<RAWG_API_KEY>` |

```python
from _common import get, key, show


def main():
    d = get("https://api.rawg.io/api/games", params={"key": key("RAWG_API_KEY"), "dates": "2025-01-01,2025-12-31",
                                                      "ordering": "-metacritic", "page_size": 5})
    show([{"name": g["name"], "released": g["released"], "metacritic": g["metacritic"], "rating": g["rating"],
           "genres": [x["name"] for x in g["genres"]]} for g in d["results"]])
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
[
  {"name": "Example Game", "released": "2025-05-12", "metacritic": 94, "rating": 4.6, "genres": ["Action", "Adventure"]}
]
```
</details>

---

### Spotify Web API

| | |
|:--|:--|
| **Docs** | https://developer.spotify.com/documentation/web-api |
| **Auth** | OAuth2 client credentials (client ID + secret) |
| **Register / token** | Create an app at https://developer.spotify.com/dashboard to get a client ID and secret. Apps start in development mode with limits. Since Nov 2024 new apps can no longer call audio-features, audio-analysis or recommendations - plan around search, artists, albums, tracks and playlists. |
| **Env var** | `SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET` |
| **Script** | [`scripts/culture_misc/spotify.py`](scripts/culture_misc/spotify.py) |

**What you can get:**

- Search artists, albums, tracks, playlists, shows, episodes, audiobooks
- Artist popularity (0-100), followers, genres; album tracks and release dates; track popularity
- Public playlists and podcast (show) metadata

**Capstone ideas:** Popularity vs. release timing, genre landscapes, podcast catalog analysis (audio-features no longer available).

**Endpoint:** `GET https://api.spotify.com/v1/{resource}  (token: POST https://accounts.spotify.com/api/token)`

| Parameter | Description | Example |
|:--|:--|:--|
| `search?q=&type=` | type: artist, album, track, playlist, show, episode (comma list); field filters year:, genre:, tag:new | `q=genre:jazz year:2025&type=album` |
| `artists/{id} · artists/{id}/albums · artists/{id}/top-tracks?market=` | Artist details, discography, top tracks | `market=US` |
| `albums/{id}/tracks · tracks/{id}` | Album tracks; track details incl. popularity | `—` |
| `limit / offset / market` | Max 50 per page; country market | `10 / 0 / US` |
| `grant_type=client_credentials` | Token request (Basic auth with client ID:secret); token lasts 1 hour | `—` |

```python
import requests

from _common import get, key, show


def main():
    tok = requests.post("https://accounts.spotify.com/api/token", data={"grant_type": "client_credentials"},
                        auth=(key("SPOTIFY_CLIENT_ID"), key("SPOTIFY_CLIENT_SECRET")), timeout=30).json()["access_token"]
    d = get("https://api.spotify.com/v1/search", headers={"Authorization": f"Bearer {tok}"},
            params={"q": "genre:jazz year:2025", "type": "album", "limit": 5, "market": "US"})["albums"]["items"]
    show({"query": "genre:jazz year:2025", "albums": [
        {"name": a["name"], "artist": a["artists"][0]["name"], "release_date": a["release_date"],
         "total_tracks": a["total_tracks"]} for a in d]})
```

<details><summary><b>Sample output</b> — ⚠️ illustrative (not run live: needs a key or was rate-limited; shape follows the official docs, values are examples)</summary>

```json
{
  "query": "genre:jazz year:2025",
  "albums": [
    {"name": "Example Jazz Album", "artist": "Example Trio", "release_date": "2025-03-14", "total_tracks": 9}
  ]
}
```
</details>

---

### SteamSpy (Steam game statistics)

| | |
|:--|:--|
| **Docs** | https://steamspy.com/api.php |
| **Auth** | None |
| **Register / token** | No registration. 1 request/second (the 'all' request: 1 per minute). Unofficial: ownership figures are estimated ranges, not exact counts. For official store data see the Steam Web API (key). |
| **Script** | [`scripts/culture_misc/steamspy.py`](scripts/culture_misc/steamspy.py) |

**What you can get:**

- Per-game: estimated owners range, positive/negative reviews, average and median playtime, price, discounts, CCU
- Top 100 by recent players or all time; games by genre or tag; developer and publisher
- User-applied tags with vote counts and supported languages (appdetails)

**Capstone ideas:** Game-success prediction from tags/price/reviews, playtime distributions, pricing and discount strategies.

**Endpoint:** `GET https://steamspy.com/api.php?request={request}`

| Parameter | Description | Example |
|:--|:--|:--|
| `request` | appdetails \| top100in2weeks \| top100forever \| top100owned \| genre \| tag \| all | `top100in2weeks` |
| `appid` | For appdetails | `570 (Dota 2)` |
| `genre / tag` | For genre / tag requests | `genre=Indie` |
| `page` | For 'all' (1,000 games per page) | `0` |

```python
from _common import get, show


def main():
    top = get("https://steamspy.com/api.php", params={"request": "top100in2weeks"})
    games = sorted(top.values(), key=lambda g: -g["ccu"])[:5]
    show({"top_by_peak_concurrent_players_2_weeks": [
        {"name": g["name"], "developer": g["developer"], "owners": g["owners"], "peak_ccu": g["ccu"],
         "positive_review_pct": round(100 * g["positive"] / max(g["positive"] + g["negative"], 1)),
         "price_usd": int(g["price"] or 0) / 100} for g in games]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:10</summary>

```json
{
  "top_by_peak_concurrent_players_2_weeks": [
    {
      "name": "Counter-Strike: Global Offensive",
      "developer": "Valve",
      "owners": "100,000,000 .. 200,000,000",
      "peak_ccu": 1013936,
      "positive_review_pct": 87,
      "price_usd": 0.0
    },
    {
      "name": "PUBG: BATTLEGROUNDS",
      "developer": "PUBG Corporation",
      "owners": "100,000,000 .. 200,000,000",
      "peak_ccu": 314682,
      "positive_review_pct": 59,
      "price_usd": 0.0
    },
    {
      "name": "Rust",
      "developer": "Facepunch Studios",
      "owners": "20,000,000 .. 50,000,000",
      "peak_ccu": 143870,
      "positive_review_pct": 87,
      "price_usd": 39.99
    },
    {
      "name": "Apex Legends",
      "developer": "Respawn",
      "owners": "100,000,000 .. 200,000,000",
      "peak_ccu": 124262,
      "positive_review_pct": 67,
      "price_usd": 0.0
    },
    {
      "name": "Wallpaper Engine",
      "developer": "Wallpaper Engine Team",
      "owners": "20,000,000 .. 50,000,000",
      "peak_ccu": 91184,
      "positive_review_pct": 98,
      "price_usd": 4.99
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

### TVmaze API (TV shows & schedules)

| | |
|:--|:--|
| **Docs** | https://www.tvmaze.com/api |
| **Auth** | None |
| **Register / token** | No registration. At least 20 calls / 10 seconds. CC BY-SA licence (attribute TVmaze). |
| **Script** | [`scripts/culture_misc/tvmaze.py`](scripts/culture_misc/tvmaze.py) |

**What you can get:**

- TV shows: genres, network/streaming service, premiere/end dates, runtime, rating, status, summary
- Episodes with air dates, cast and crew, full daily schedules by country (incl. streaming 'web' schedule)
- Lookup by IMDb/TheTVDB IDs; changes feed for keeping a local copy up to date

**Capstone ideas:** Streaming vs. broadcast release strategies, show cancellation prediction, ratings by genre.

**Endpoint:** `GET https://api.tvmaze.com/{resource}`

| Parameter | Description | Example |
|:--|:--|:--|
| `search/shows?q=` | Fuzzy show search with relevance score | `q=office` |
| `shows/{id}?embed[]=` | Show details; embed episodes, cast, seasons | `shows/526?embed[]=cast` |
| `schedule?country=&date=` | Everything airing on broadcast TV that day | `country=US&date=2026-10-12` |
| `schedule/web?date=` | Streaming releases (Netflix, Prime, ...) | `date=2026-10-12` |
| `lookup/shows?imdb=` | Find a show by IMDb ID (answers with a 301 redirect to the show record) | `imdb=tt0386676` |
| `shows?page=` | Full index, 250 shows per page | `0` |

```python
from collections import Counter

from _common import get, show


def main():
    web = get("https://api.tvmaze.com/schedule/web", params={"date": "2026-10-10"})
    shows = {e["_embedded"]["show"]["id"]: e["_embedded"]["show"] for e in web if e.get("_embedded")}
    services = Counter((s.get("webChannel") or {}).get("name") for s in shows.values())
    show({"date": "2026-10-10", "streaming_episodes_released": len(web), "distinct_shows": len(shows),
          "by_service": services.most_common(5),
          "examples": [{"show": s["name"], "service": (s.get("webChannel") or {}).get("name"), "genres": s["genres"],
                        "rating": s["rating"]["average"]} for s in list(shows.values())[:3]]})
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:10</summary>

```json
{
  "date": "2026-10-10",
  "streaming_episodes_released": 166,
  "distinct_shows": 52,
  "by_service": [
    [
      "Tencent QQ",
      10
    ],
    [
      "YouTube",
      8
    ],
    [
      "iQIYI",
      7
    ],
    [
      "GagaOOLala",
      4
    ],
    [
      "Youku",
      3
    ]
  ],
  "examples": [
    {
      "show": "Трое из Простоквашино",
      "service": "Okko",
      "genres": [
        "Children",
        "Family"
      ],
      "rating": 7.5
    },
    {
      "show": "КСТАТИ",
      "service": "VK Видео Originals",
      "genres": [
        "Comedy",
        "Music"
      ],
      "rating": null
    },
    {
      "show": "Чудо",
      "service": "КиноПоиск",
      "genres": [
        "Drama",
        "Mystery"
      ],
      "rating": null
    }
  ]
}
```
</details>

---

### Wikidata Query Service (SPARQL)

| | |
|:--|:--|
| **Docs** | https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service |
| **Auth** | None (descriptive User-Agent required) |
| **Register / token** | No registration. 60-second query timeout, one query at a time. Build and test queries interactively at https://query.wikidata.org first. CC0 data. |
| **Script** | [`scripts/culture_misc/wikidata_sparql.py`](scripts/culture_misc/wikidata_sparql.py) |

**What you can get:**

- Structured facts on 110M+ items: films, awards, books, people, places, companies, artworks
- Arbitrary joins: e.g. award winners with release dates, directors, box office, birthplaces
- IDs that link to IMDb, MusicBrainz, Discogs, VIAF, OpenAlex and more

**Capstone ideas:** Linking entertainment datasets via shared IDs, award-winner analyses, gender/diversity studies in film and music.

**Endpoint:** `GET https://query.wikidata.org/sparql`

| Parameter | Description | Example |
|:--|:--|:--|
| `query` | SPARQL query (wd: items, wdt: direct properties, p:/ps:/pq: for qualifiers) | `SELECT ?film WHERE { ?film wdt:P31 wd:Q11424 } LIMIT 5` |
| `format` | json \| xml \| csv (or Accept header) | `json` |
| `common IDs` | Q11424 film, P166 award received, P585 point in time, P57 director, P345 IMDb ID | `—` |
| `SERVICE wikibase:label` | Adds human-readable ?xLabel variables | `bd:serviceParam wikibase:language "en"` |

```python
from _common import get, show

QUERY = """
SELECT ?filmLabel ?year ?directorLabel WHERE {
  ?film wdt:P31 wd:Q11424 ;                      # instance of: film
        p:P166 ?award .
  ?award ps:P166 wd:Q102427 ; pq:P585 ?date .    # Academy Award for Best Picture, with date
  OPTIONAL { ?film wdt:P57 ?director . }
  BIND(YEAR(?date) AS ?year)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
} ORDER BY DESC(?year) LIMIT 8
"""


def main():
    rows = get("https://query.wikidata.org/sparql", params={"query": QUERY, "format": "json"})["results"]["bindings"]
    show({"best_picture_winners": [{"year": r["year"]["value"], "film": r["filmLabel"]["value"],
                                    "director": r.get("directorLabel", {}).get("value")} for r in rows]}, max_items=8)
```

<details><summary><b>Sample output</b> — live run 2026-10-12 10:10</summary>

```json
{
  "best_picture_winners": [
    {
      "year": "2026",
      "film": "One Battle After Another",
      "director": "Paul Thomas Anderson"
    },
    {
      "year": "2025",
      "film": "Anora",
      "director": "Sean Baker"
    },
    {
      "year": "2024",
      "film": "Oppenheimer",
      "director": "Christopher Nolan"
    },
    {
      "year": "2023",
      "film": "Everything Everywhere All at Once",
      "director": "Dan Kwan"
    },
    {
      "year": "2023",
      "film": "Everything Everywhere All at Once",
      "director": "Daniel Scheinert"
    },
    {
      "year": "2022",
      "film": "CODA",
      "director": "Siân Héder"
    },
    {
      "year": "2021",
      "film": "Nomadland",
      "director": "Chloé Zhao"
    },
    {
      "year": "2020",
      "film": "Parasite",
      "director": "Bong Joon-ho"
    }
  ]
}
```
</details>

---
