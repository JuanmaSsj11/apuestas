from curl_cffi import requests
import re
import json

session = requests.Session(impersonate="chrome120")
r = session.get("https://www.playdoit.mx/deportes")

# Find all JSON endpoints or URLs in the HTML page
matches = re.findall(r'https?://[^\s"\'<>]+(?:altenar|biahosted|playdoit)[^\s"\'<>]*', r.text)
print("Matches in HTML:", set(matches))

# Check /cms/ or api calls
api_calls = re.findall(r'["\'](/api/[^"\']+)["\']', r.text)
print("API calls in HTML:", set(api_calls))

# Let's check if there is an Altenar widget URL
widget_calls = re.findall(r'https?://[^\s"\'<>]+(?:widget|sportsbook|sb2)[^\s"\'<>]*', r.text)
print("Widget calls:", set(widget_calls))
