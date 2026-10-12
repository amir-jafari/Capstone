META = dict(
    name="OpenF1 (Formula 1 telemetry)",
    domain="Sports",
    docs="https://openf1.org/",
    auth="None for historical data (paid tier for live sessions)",
    signup="No registration for historical data since 2023. Rate-limited to a few requests/sec.",
    provides=["Sessions, meetings, drivers, lap times, sector times, pit stops, stints/tyres, race control messages",
              "Car telemetry (~3.7 Hz): speed, RPM, throttle, brake, gear, DRS", "Car position (x, y, z) on track, intervals, weather"],
    ideas="Tyre-degradation models, pit-strategy optimization, driver comparison from telemetry.",
    endpoint="GET https://api.openf1.org/v1/{resource}",
    params=[
        ("resource", "sessions | meetings | drivers | laps | pit | stints | car_data | location | position | intervals | weather | race_control", "laps"),
        ("session_key / meeting_key", "Identify a session (use latest for the most recent)", "session_key=9693"),
        ("year / session_type / country_name", "Filters on sessions/meetings", "year=2025&session_type=Race"),
        ("driver_number", "Car number", "1"),
        ("field>=value / field<=value", "Comparison filters on any numeric/date field", "lap_number>=10&speed>=300"),
        ("csv", "true → CSV instead of JSON", "true"),
    ],
)
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    race = get("https://api.openf1.org/v1/sessions", params={"year": 2025, "session_type": "Race",
                                                            "country_name": "Australia"})[0]
    laps = get("https://api.openf1.org/v1/laps", params={"session_key": race["session_key"], "driver_number": 1})
    show({"session": {k: race[k] for k in ("session_key", "location", "date_start")},
          "laps_driver_1": [{k: l.get(k) for k in ("lap_number", "lap_duration", "duration_sector_1", "st_speed")}
                            for l in laps[1:]]})


if __name__ == "__main__":
    main()
