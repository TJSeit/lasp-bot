"""
discovery_script.py — One-shot LASP mission discovery helper.

Scrapes the LASP missions portal (lasp.colorado.edu/missions/) and saves a
JSON list of mission names and URLs. The output (lasp_missions_seed.json) can
be used to seed the corpus builder or to explore which missions are available.

Usage:
    cd indexer
    python discovery_script.py
"""

import json

import requests
from bs4 import BeautifulSoup


def get_lasp_mission_list() -> list[dict]:
    """Return a list of {mission, url} dicts from the LASP missions portal."""
    url = "https://lasp.colorado.edu/missions/"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        missions = []
        for link in soup.select('a[href*="/missions/"]'):
            name = link.text.strip()
            href = link.get("href")

            # Filter out generic links and duplicates.
            if name and href and href.startswith("https://lasp.colorado.edu/missions/"):
                if not any(m["url"] == href for m in missions):
                    missions.append({"mission": name, "url": href})

        return missions

    except Exception as e:
        print(f"Error scraping LASP: {e}")
        return []


if __name__ == "__main__":
    mission_data = get_lasp_mission_list()

    # Display the first five missions found.
    for m in mission_data[:5]:
        print(f"Found: {m['mission']} -> {m['url']}")

    # Save to JSON for use in the corpus-building pipeline.
    with open("lasp_missions_seed.json", "w") as f:
        json.dump(mission_data, f, indent=4)

    print(f"\nSaved {len(mission_data)} missions to lasp_missions_seed.json")
