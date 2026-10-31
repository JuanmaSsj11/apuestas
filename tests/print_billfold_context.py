from curl_cffi import requests
import re

session = requests.Session(impersonate="chrome120")
r = session.get("https://www.playdoit.mx/main.45680fc7.js")
pos = r.text.find("._api.billfold")
if pos != -1:
    print(r.text[max(0, pos-100):min(len(r.text), pos+300)])
