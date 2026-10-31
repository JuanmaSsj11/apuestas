from curl_cffi import requests
import re
import json

session = requests.Session(impersonate="chrome120")
r = session.get("https://www.playdoit.mx/deportes")
print("Status:", r.status_code)
scripts = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', r.text)
print("Scripts:")
for s in scripts:
    print(" ", s)

# Check inline scripts for URLs or config
inlines = re.findall(r'<script(?:(?!src).)*?>(.*?)</script>', r.text, re.DOTALL)
print("Inlines count:", len(inlines))
for idx, inl in enumerate(inlines):
    if any(k in inl.lower() for k in ['altenar', 'biahosted', 'sportsbook', 'api', 'url', 'token']):
        print(f"Inline {idx} excerpt:", inl[:300].strip())
