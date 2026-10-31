from curl_cffi import requests
import re
import json

session = requests.Session(impersonate="chrome120")
r = session.get("https://sports.caliente.mx")
print("Caliente status:", r.status_code)

links = re.findall(r'https?://[^\s"\'<>]+(?:api|data|json|feed)[^\s"\'<>]*', r.text)
print("Data links in Caliente:", list(set(links))[:10])

# Look for script sources
scripts = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', r.text)
print("Caliente scripts:", [s for s in scripts if 'caliente' in s or s.startswith('/')][:5])
