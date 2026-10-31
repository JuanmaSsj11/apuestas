from curl_cffi import requests
import re

session = requests.Session(impersonate="chrome120")
r = session.get("https://www.playdoit.mx/main.45680fc7.js")
pos = r.text.find("loadWidgetBanners")
if pos != -1:
    print(r.text[pos-400:pos+600])
else:
    print("loadWidgetBanners not found")
