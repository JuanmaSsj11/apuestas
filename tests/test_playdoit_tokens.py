from curl_cffi import requests
import re
import json

session = requests.Session(impersonate="chrome120")
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-MX,es;q=0.9"
}

# Check if there is an endpoint like /api/games/getAltenarUrl or similar
test_urls = [
    "https://www.playdoit.mx/api/v1/sportsbook/token",
    "https://www.playdoit.mx/api/sportsbook",
    "https://www.playdoit.mx/billfold/altenarUrl",
    "https://www.playdoit.mx/cms/bia/widget",
    "https://www.playdoit.mx/api/player/altenar"
]

for u in test_urls:
    try:
        r = session.get(u, headers=headers)
        print("URL:", u, "Status:", r.status_code, "Len:", len(r.text))
        if r.status_code == 200:
            print("  Response:", r.text[:300])
    except Exception as e:
        print("Error:", e)
