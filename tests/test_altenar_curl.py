from curl_cffi import requests

session = requests.Session(impersonate="chrome120")
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Origin": "https://www.playdoit.mx",
    "Referer": "https://www.playdoit.mx/deportes",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-MX,es;q=0.9"
}

urls_to_try = [
    "https://sb2frontend-altenar2.biahosted.com/api/Sportsbook/GetSports?skinName=playdoit",
    "https://sb2frontend-altenar2.biahosted.com/api/Sportsbook/GetSports?skinName=playdoit2_dark",
    "https://sb2frontend-altenar2.biahosted.com/api/Sportsbook/GetEvents?sportId=66&skinName=playdoit2_dark",
    "https://sb2frontend-altenar2.biahosted.com/api/Sportsbook/GetLiveOverview?skinName=playdoit2_dark",
    "https://sb2frontend-altenar2.biahosted.com/api/Widget/GetSports?skinName=playdoit2_dark",
    "https://sb2frontend-altenar2.biahosted.com/api/Sportsbook/GetDailyHighlights?skinName=playdoit2_dark"
]

for u in urls_to_try:
    try:
        r = session.get(u, headers=headers)
        print("URL:", u.split("?")[0], "Params:", u.split("?")[1] if "?" in u else "", "Status:", r.status_code, "Length:", len(r.text))
        if r.status_code == 200:
            print("  Response preview:", r.text[:300])
    except Exception as e:
        print("  Error:", e)
