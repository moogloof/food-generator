import urllib.request
import io
import json
import requests
import pickle
from bs4 import BeautifulSoup
from PIL import Image

url = "https://www.allrecipes.com/element-api/content-proxy/faceted-searches-load-more?search=sandwiches&page={}"
page = 1

data = []

while True:
	r = requests.get(url.format(page))
	r = r.json()

	rhtml = r["html"]
	rhtml = BeautifulSoup(rhtml, "html.parser")

	for img in rhtml.find_all("img"):
		print(img["src"])
		try:
			loadedimg = io.BytesIO(urllib.request.urlopen(img["src"]).read())
			loadedimg = list(Image.open(loadedimg).convert("L").getdata().resize((250, 250)))
			data.append(loadedimg)
		except:
			pass

	if not r["hasNext"]:
		break

	page += 1

with open("data.bin", "wb") as f:
	pickle.dump(data, f)
