from curl_cffi import requests
import re

session = requests.Session(impersonate="chrome120")
for js in ["/runtime.1e0ba15f.js", "/267.4bc0a009.js", "/60.2389c5ea.js", "/main.45680fc7.js"]:
    url = "https://www.playdoit.mx" + js
    r = session.get(url)
    print("JS:", js, "Size:", len(r.text))
    # Search for domain names or API endpoints
    domains = re.findall(r'https?://[a-zA-Z0-9\.\-_]+', r.text)
    bia = [d for d in domains if 'biahosted' in d or 'altenar' in d or 'sportsbook' in d or 'api' in d]
    if bia:
        print("  Domains:", set(bia))
    # Look for paths containing Sportsbook
    paths = re.findall(r'["\'](/[a-zA-Z0-9_\-/]*sportsbook[a-zA-Z0-9_\-/]*)["\']', r.text, re.IGNORECASE)
    if paths:
        print("  Sportsbook paths:", set(paths))
