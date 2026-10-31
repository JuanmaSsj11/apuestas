from curl_cffi import requests
import re

session = requests.Session(impersonate="chrome120")
r = session.get("https://www.playdoit.mx/main.45680fc7.js")

paths = re.findall(r'["\'](/[^"\']*(?:billfold|sportsbook|altenar)[^"\']*)["\']', r.text, re.IGNORECASE)
print("Paths found:", set(paths))
