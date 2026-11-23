"""Example project shown as grey placeholder text in the submission form (Report_Gen style)."""

EXAMPLE = {
    "Proposed by": "e.g. Dr. Jane Smith",
    "Proposed by email": "e.g. jsmith@organization.org",
    "collaborator": "e.g. DC Department of Energy & Environment",
    "funding_opportunity": "e.g. Summer internship for 1 student; AWS credits",
    "github_repo": "e.g. jsmith/heatwatch-dc",
    "project_name": "e.g. HeatWatch-DC: Can Satellite Imagery and Street-Level Sensors Predict "
                    "Neighborhood Heat Risk in Washington, DC?",

    "Objective": """Urban heat islands make some DC neighborhoods up to 15°F hotter than others on the same
summer afternoon, and heat is the leading weather-related cause of death in the US. City agencies
plan cooling centers and tree planting from coarse, outdated maps. The goal of this project is to build
a reproducible machine-learning pipeline that predicts block-level land-surface temperature from
open satellite imagery and street-level sensor data, and to test which features actually drive heat risk.

Key Objectives:
1. Assemble a block-level dataset joining Landsat 8/9 land-surface temperature, Sentinel-2 vegetation
   indices, building footprints, tree canopy, and DC street-sensor readings for summers 2019-2025.
2. Train and compare spatial baselines (linear regression, random forest, XGBoost) against a CNN on
   image patches and a graph neural network over the street network.
3. Test three explicit hypotheses: (H1) tree canopy explains more variance than impervious surface;
   (H2) the CNN beats tabular models by at least 10% RMSE; (H3) models transfer to unseen wards.
4. Explain predictions with SHAP and produce a ward-by-ward heat-risk map with uncertainty.
5. Release the full pipeline as an open-source, config-driven repository and an interactive dashboard.""",

    "Dataset": """All datasets below are public and free to download.

TIER 1 -- SATELLITE IMAGERY:
1. Landsat 8/9 Collection 2 Level-2 land-surface temperature (30 m): https://earthexplorer.usgs.gov/
2. Sentinel-2 L2A multispectral imagery (10 m) for NDVI / NDBI: https://dataspace.copernicus.eu/

TIER 2 -- URBAN FORM:
1. DC building footprints and land use: https://opendata.dc.gov/
2. 2020 DC tree canopy layer (1 m): https://opendata.dc.gov/

TIER 3 -- GROUND TRUTH:
1. DC street-level heat sensor campaign (temperature every 5 minutes, 2021-2024)
2. NOAA hourly weather station records (Reagan National, Dulles): https://www.ncei.noaa.gov/""",

    "Rationale": """Heat vulnerability falls hardest on neighborhoods with less tree cover and more pavement,
which in DC overlap strongly with lower-income wards. Existing heat maps come from a single day of
measurements and cannot be updated each summer. A validated, open pipeline lets the city refresh its
heat-risk map every year, target tree planting where it lowers temperature most, and place cooling
centers where people need them. Students gain experience with geospatial data, deep learning on
imagery, graph models, and explainability on a problem with direct public impact.""",

    "Approach": """Phase 1 -- Data pipeline (Weeks 1-3):
1. Download and cloud-mask Landsat and Sentinel-2 scenes; compute LST, NDVI, NDBI per city block.
2. Join building, canopy, and land-use layers; aggregate street-sensor readings to blocks.

Phase 2 -- Modeling (Weeks 4-8):
1. Fit tabular baselines with spatially blocked cross-validation (no leakage between neighbors).
2. Train a ResNet-18 on 64x64 image patches and a GraphSAGE model over the street graph.
3. Run leave-one-ward-out experiments to test spatial transfer (H3).

Phase 3 -- Analysis and release (Weeks 9-14):
1. SHAP feature attribution to test H1; paired statistical tests across folds for H2.
2. Build a Streamlit dashboard with the ward-level heat-risk map and uncertainty bands.
3. Write the final report and package the code as a pip-installable repository.

Compute: all models train on a single GPU (e.g. 1x NVIDIA T4 or A10G) in under 2 hours each.""",

    "Timeline": """Week 1:    Literature review, data access, environment setup
Week 2:    Satellite download and cloud masking; LST / NDVI computation
Week 3:    Join urban-form layers and street sensors into the block-level dataset
Week 4:    EDA and spatially blocked cross-validation design
Week 5:    Tabular baselines (linear, random forest, XGBoost)
Week 6:    CNN on image patches
Week 7:    Graph neural network on the street network
Week 8:    Leave-one-ward-out transfer experiments
Week 9:    SHAP explainability and hypothesis tests
Week 10:   Uncertainty estimation and error analysis
Week 11:   Streamlit dashboard
Week 12:   Packaging, documentation, reproducibility checks
Week 13:   Final report and paper draft
Week 14:   Presentation and code release""",

    "Research Contributions": """1. The first open, reproducible block-level heat-risk benchmark for Washington, DC.
2. A controlled comparison of tabular, CNN, and graph models for urban temperature prediction.
3. Evidence on which urban features (canopy vs. pavement) most strongly drive local heat.
4. A conference paper submission (e.g. ACM SIGSPATIAL or an AGU session) and an open-source package.""",

    "Possible Issues": """1. Cloud cover reduces usable summer scenes -- mitigate by compositing across several dates.
2. Street-sensor coverage is uneven across wards -- report results per ward and weight the loss.
3. Spatial leakage can inflate accuracy -- use spatially blocked cross-validation throughout.
4. Large raster files -- process tiles lazily with rasterio / xarray and cache block-level features.""",

    "Additional Resources": """1. Hoffman et al. (2020), "The Effects of Historical Housing Policies on Resident Exposure to
   Intra-Urban Heat," Climate.
2. Google Earth Engine tutorials for Landsat LST: https://developers.google.com/earth-engine
3. PyTorch Geometric documentation: https://pytorch-geometric.readthedocs.io/""",
}
