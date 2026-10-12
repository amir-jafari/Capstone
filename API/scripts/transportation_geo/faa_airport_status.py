META = dict(
    name="FAA NAS Airport Status (delays & ground stops)",
    domain="Transportation & Geospatial",
    docs="https://nasstatus.faa.gov/",
    auth="None",
    signup="No registration. One XML document with every current FAA traffic-management program, refreshed every "
           "minute; poll it and store snapshots to build a history.",
    provides=["Ground stops and ground-delay programs with reasons (weather, volume, equipment) and average delay",
              "Arrival/departure delays, airport closures, airspace flow programs"],
    ideas="Building a delay-event dataset, linking delays to weather (METARs), airport-level disruption risk.",
    endpoint="GET https://nasstatus.faa.gov/api/airport-status-information",
    params=[
        ("(no parameters)", "Returns all current programs as XML", "—"),
        ("Delay_type/Name", "Ground Stop Programs | Ground Delay Programs | General Arrival/Departure Delay Info | Airport Closures", "—"),
        ("Program fields", "ARPT (airport), Reason, End_Time / Avg delay, Max delay", "—"),
    ],
)
import xml.etree.ElementTree as ET

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # find _common.py
from _common import get, show


def main():
    root = ET.fromstring(get("https://nasstatus.faa.gov/api/airport-status-information", as_json=False))
    out = {"updated": root.findtext("Update_Time"), "programs": {}}
    for dt in root.findall("Delay_type"):
        rows = []
        for el in dt.iter():
            if el.find("ARPT") is not None:
                rows.append({c.tag: (c.text or "").strip() for c in el if len(c) == 0 and (c.text or "").strip()})
        out["programs"][dt.findtext("Name")] = rows
    show(out)


if __name__ == "__main__":
    main()
