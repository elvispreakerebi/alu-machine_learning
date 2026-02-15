#!/usr/bin/env python3
"""
Return list of starships that can hold at least passengerCount passengers.
Uses SWAPI (https://swapi.dev/api/starships/) with pagination.
"""
import requests


def availableShips(passengerCount):
    """
    Return list of ship names that can hold at least passengerCount passengers.
    Handles pagination; treats unknown/n/a passengers as 0.
    """
    base = "https://swapi.dev/api/starships/"
    out = []

    url = base
    while url:
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError, KeyError):
            return out

        for ship in data.get("results", []):
            name = ship.get("name")
            raw = (ship.get("passengers") or "").strip().lower()
            if raw in ("unknown", "n/a", ""):
                passengers = 0
            else:
                try:
                    passengers = int(raw.replace(",", ""))
                except ValueError:
                    passengers = 0
            if passengers >= passengerCount:
                out.append(name)

        url = data.get("next")

    return out
