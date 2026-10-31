from curl_cffi import requests

session = requests.Session(impersonate="chrome120")
r = session.get("https://www.playdoit.mx/cms/bia/widget")
print("Status:", r.status_code)
print(r.text)
