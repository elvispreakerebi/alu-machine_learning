#!/usr/bin/env python3
"""
Display the number of launches per rocket using the (unofficial) SpaceX API.
Output: one line per rocket, "Rocket Name: count", sorted by count desc then name A-Z.
"""
import requests


def main():
    try:
        r = requests.post(
            "https://api.spacexdata.com/v5/launches/query",
            json={"query": {}, "options": {"pagination": False}},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        docs = data.get("docs", [])
    except (requests.RequestException, ValueError, KeyError):
        return
    # Count launches per rocket ID
    count_by_id = {}
    for launch in docs:
        rid = launch.get("rocket")
        if rid:
            count_by_id[rid] = count_by_id.get(rid, 0) + 1
    # Resolve rocket ID -> name (cache)
    id_to_name = {}
    for rid in count_by_id:
        try:
            rr = requests.get(
                f"https://api.spacexdata.com/v4/rockets/{rid}",
                timeout=10,
            )
            rr.raise_for_status()
            id_to_name[rid] = rr.json().get("name") or rid
        except (requests.RequestException, ValueError, KeyError):
            id_to_name[rid] = rid
    # Count by name (in case same name from different IDs - unlikely)
    count_by_name = {}
    for rid, count in count_by_id.items():
        name = id_to_name.get(rid, rid)
        count_by_name[name] = count_by_name.get(name, 0) + count
    # Sort: by count descending, then by name A-Z
    items = sorted(
        count_by_name.items(),
        key=lambda x: (-x[1], x[0]),
    )
    for name, count in items:
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
