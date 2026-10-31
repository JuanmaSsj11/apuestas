from curl_cffi import requests
import re

session = requests.Session(impersonate="chrome120")
for js in ["/60.2389c5ea.js", "/main.45680fc7.js"]:
    url = "https://www.playdoit.mx" + js
    r = session.get(url)
    print("Searching in", js)
    matches = re.findall(r'.{0,60}(?:altenar|sportsbook|deportes).{0,60}', r.text, re.IGNORECASE)
    print(f"Found {len(matches)} matches")
    for m in matches[:10]:
        print("  ", m.strip())
