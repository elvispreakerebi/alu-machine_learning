#!/usr/bin/env python3
"""
Print the location of a GitHub user from the API URL.
Usage: ./2-user_location.py https://api.github.com/users/<username>
"""
import sys
import time
import requests


def main():
    """
    Fetch GitHub user URL from argv; print location, "Not found", or rate limit.
    Handles 200 (location), 404 (Not found), 403 (Reset in X min from header).
    """
    if len(sys.argv) < 2:
        return
    url = sys.argv[1]
    try:
        r = requests.get(url, timeout=10)
    except requests.RequestException:
        print("Not found")
        return
    if r.status_code == 403:
        reset = r.headers.get("X-Ratelimit-Reset")
        if reset:
            try:
                reset_ts = int(reset)
                minutes = max(0, (reset_ts - int(time.time())) // 60)
                print("Reset in {} min".format(minutes))
            except (ValueError, TypeError):
                print("Reset in 0 min")
        else:
            print("Reset in 0 min")
        return
    if r.status_code == 404:
        print("Not found")
        return
    if r.status_code != 200:
        print("Not found")
        return
    try:
        data = r.json()
        location = data.get("location")
        print(location if location else "")
    except (ValueError, KeyError):
        print("")


if __name__ == "__main__":
    main()
