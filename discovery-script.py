import requests
from bs4 import BeautifulSoup
import json

def get_lasp_mission_list():
    url = "https://lasp.colorado.edu/missions/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        missions = []
        # LASP missions are typically listed in a catalog grid or list
        # We look for anchor tags within the mission catalog section
        for link in soup.select('a[href*="/missions/"]'):
            name = link.text.strip()
            href = link.get('href')
            
            # Filter out generic links and duplicates
            if name and href and href.startswith('https://lasp.colorado.edu/missions/'):
                if not any(m['url'] == href for m in missions):
                    missions.append({"mission": name, "url": href})
        
        return missions

    except Exception as e:
        print(f"Error scraping LASP: {e}")
        return []

# Run discovery
mission_data = get_lasp_mission_list()

# Display the first 5 missions found
for m in mission_data[:5]:
    print(f"Found: {m['mission']} -> {m['url']}")

# Save to JSON for your training pipeline
with open('lasp_missions_seed.json', 'w') as f:
    json.dump(mission_data, f, indent=4)