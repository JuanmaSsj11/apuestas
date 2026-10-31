from curl_cffi import requests
import re

session = requests.Session(impersonate="chrome120")
r = session.get("https://www.playdoit.mx/main.45680fc7.js")

# Find WebSocket or API host URLs
ws = re.findall(r'wss?://[^\s"\'<>]+', r.text)
print("WebSockets:", set(ws))

# Find host definitions or baseUrl
base_urls = re.findall(r'(?:baseUrl|apiUrl|apiHost|serverUrl)\s*[:=]\s*["\']([^"\']+)["\']', r.text)
print("Base URLs:", set(base_urls))

# Find domain strings with playdoit or backend
backend = re.findall(r'https?://[a-zA-Z0-9\.\-_]*(?:api|backend|core|billfold|altenar)[a-zA-Z0-9\.\-_]*', r.text)
print("Backends:", set(backend))
