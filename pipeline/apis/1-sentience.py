#!/usr/bin/env python3
"""
Return list of home planet names for all sentient species.
Uses SWAPI species and planets endpoints with pagination.
"""
import requests


def _planet_name(url, cache):
    """Resolve planet URL to name; use cache to avoid repeated requests."""
    if url in cache:
        return cache[url]
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        name = r.json().get("name") or "unknown"
    except (requests.RequestException, ValueError, KeyError):
        name = "unknown"
    cache[url] = name
    return name


def sentientPlanets():
    """
    Return list of names of home planets of all sentient species.
    Paginates species; fetches planet name from homeworld URL.
    Null homeworld is returned as "unknown".
    """
    out = []
    seen = set()
    cache = {}
    url = "https://swapi.dev/api/species/"

    while url:
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError, KeyError):
            return out

        for sp in data.get("results", []):
            designation = (sp.get("designation") or "").strip().lower()
            classification = (sp.get("classification") or "").strip().lower()
            if designation != "sentient" and classification != "sentient":
                continue
            homeworld = sp.get("homeworld")
            if not homeworld:
                name = "unknown"
            else:
                name = _planet_name(homeworld, cache)
            if name not in seen:
                seen.add(name)
                out.append(name)

        url = data.get("next")

    return out
