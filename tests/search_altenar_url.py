from curl_cffi import requests
import re

session = requests.Session(impersonate="chrome120")
r = session.get("https://www.playdoit.mx/main.45680fc7.js")
for term in ["altenarUrl", "skinid=pl", "biahosted"]:
    for m in re.finditer(re.escape(term), r.text):
        idx = m.start()
        print(f"Term '{term}':", r.text[max(0, idx-100):min(len(r.text), idx+200)])
