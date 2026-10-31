from curl_cffi import requests
import re

session = requests.Session(impersonate="chrome120")
r = session.get("https://www.playdoit.mx/main.45680fc7.js")

# Find base URLs or api endpoints
endpoints = re.findall(r'["\'](/api/[a-zA-Z0-9_\-/]+)["\']', r.text)
print("API endpoints found in main.js:", set(endpoints))

games_urls = re.findall(r'["\'](/cms/[a-zA-Z0-9_\-/]+)["\']', r.text)
print("CMS endpoints found:", set(games_urls[:10]))
