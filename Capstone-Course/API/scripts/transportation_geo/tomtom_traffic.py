META = dict(
    name="TomTom Traffic API",
    domain="Transportation & Geospatial",
    docs="https://docs.tomtom.com/traffic-api/documentation/tomtom-maps/v1/product-information/introduction",
    auth="API key (query `key`)",
    env="TOMTOM_API_KEY",
    signup="Create a free account at https://developer.tomtom.com and copy the default API key. Free tier: 2,500 "
           "non-tile requests/day. Also includes routing, search and map APIs.",
    provides=["Traffic flow for a road segment: current vs. free-flow speed and travel time, confidence, road closures",
              "Traffic incidents (accidents, jams, roadworks) in a bounding box with delays",
              "Flow and incident map tiles"],
    ideas="Congestion indices for a city, incident impact on travel times, traffic vs. weather/events models.",
    endpoint="GET https://api.tomtom.com/traffic/services/{4 | 5}/{flowSegmentData | incidentDetails}/...",
    params=[
        ("flowSegmentData/{style}/{zoom}/json?point=lat,lon", "Flow on the road nearest a point; style absolute | relative", "absolute/10/json?point=38.8977,-77.0365"),
        ("unit", "KMPH | MPH", "MPH"),
        ("/traffic/services/5/incidentDetails?bbox=&fields=", "Incidents in a box (v5)", "bbox=-77.12,38.80,-76.91,38.99"),
        ("key", "Your API key", "<TOMTOM_API_KEY>"),
    ],
    sample="""{
  "point": "38.8977,-77.0365 (Washington, DC)",
  "current_speed_mph": 18,
  "free_flow_speed_mph": 27,
  "congestion_pct": 33,
  "current_travel_time_s": 214,
  "free_flow_travel_time_s": 143,
  "confidence": 0.98,
  "road_closure": false
}""",
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, key, show


def main():
    f = get("https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json",
            params={"point": "38.8977,-77.0365", "unit": "MPH", "key": key("TOMTOM_API_KEY")})["flowSegmentData"]
    show({"point": "38.8977,-77.0365 (Washington, DC)", "current_speed_mph": f["currentSpeed"],
          "free_flow_speed_mph": f["freeFlowSpeed"],
          "congestion_pct": round(100 * (1 - f["currentSpeed"] / f["freeFlowSpeed"])),
          "current_travel_time_s": f["currentTravelTime"], "free_flow_travel_time_s": f["freeFlowTravelTime"],
          "confidence": f["confidence"], "road_closure": f["roadClosure"]})


if __name__ == "__main__":
    main()
