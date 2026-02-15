#!/usr/bin/env python3
"""
Display the upcoming SpaceX launch using the (unofficial) SpaceX API.
Format: <launch name> (<date local>) <rocket name> - <launchpad> (<locality>)
"""
import time
import requests


def main():
    """
    Fetch upcoming launches from SpaceX API; print the soonest one.
    Format: launch name (date_local) rocket name - launchpad name (locality).
    """
    try:
        r = requests.get(
            "https://api.spacexdata.com/v5/launches/upcoming",
            timeout=10,
        )
        r.raise_for_status()
        launches = r.json()
    except (requests.RequestException, ValueError):
        return
    if not launches:
        return
    # Soonest: date_unix >= now, else min; same date = first in API order
    now_ts = int(time.time())
    future = [
        launch for launch in launches
        if (launch.get("date_unix") or 0) >= now_ts
    ]
    if future:
        launch = min(future, key=lambda launch: launch.get("date_unix", 0))
    else:
        launch = min(launches, key=lambda launch: launch.get("date_unix") or 0)
    name = launch.get("name") or ""
    date_local = launch.get("date_local") or ""
    rocket_id = launch.get("rocket")
    launchpad_id = launch.get("launchpad")
    rocket_name = ""
    if rocket_id:
        try:
            rr = requests.get(
                "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id),
                timeout=10,
            )
            rr.raise_for_status()
            rocket_name = rr.json().get("name") or ""
        except (requests.RequestException, ValueError, KeyError):
            pass
    launchpad_name = ""
    locality = ""
    if launchpad_id:
        try:
            pad_url = "https://api.spacexdata.com/v4/launchpads/{}"
            rp = requests.get(pad_url.format(launchpad_id), timeout=10)
            rp.raise_for_status()
            pad = rp.json()
            launchpad_name = pad.get("name") or ""
            locality = pad.get("locality") or ""
        except (requests.RequestException, ValueError, KeyError):
            pass
    print("{} ({}) {} - {} ({})".format(
        name, date_local, rocket_name, launchpad_name, locality))


if __name__ == "__main__":
    main()
