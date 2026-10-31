from curl_cffi import requests
import json

session = requests.Session(impersonate="chrome120")
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Origin": "https://www.playdoit.mx",
    "Referer": "https://www.playdoit.mx/deportes",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-MX,es;q=0.9"
}

params = {
    "timezoneOffset": -360,
    "langId": 4,
    "skinName": "playdoit",
    "configId": 1,
    "culture": "es-MX",
    "countryCode": "MX",
    "deviceType": "Desktop",
    "sportids": "0",
    "categoryids": "0",
    "champids": "0",
    "group": "AllEvents",
    "period": "periodall",
    "withLive": "false"
}

hosts = [
    "https://sb2frontend-altenar2.biahosted.com/api/Sportsbook/GetEvents",
    "https://sb2frontend-altenar2.biahosted.com/Sportsbook/GetEvents",
    "https://sb1capi-altenar.biahosted.com/Sportsbook/GetEvents"
]

for skin in ["playdoit", "playdoit2", "playdoit2_dark"]:
    params["skinName"] = skin
    for h in hosts:
        try:
            r = session.get(h, params=params, headers=headers)
            print(f"Host: {h.split('/')[2]} | Skin: {skin} | Status: {r.status_code} | Len: {len(r.text)}")
            if r.status_code == 200:
                print("SUCCESS!")
                print(r.text[:500])
                break
        except Exception as e:
            print("Error:", e)
