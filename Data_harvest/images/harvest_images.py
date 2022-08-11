from urllib import request
import requests
import urllib3
import pandas as pd
import os

type = input("(0) date or (1) format?\n")
if type == "0":
    type = "date"
elif type == "1":
    type = "format"
else:
    print("Wrong input, terminating...")
    exit(1)

data = pd.read_csv("id_list_" + type + ".csv", encoding="ISO-8859-1")
id_remove = []
idx_remove = []

# Remove files from to be downloaded list that are already downloaded
for filename in os.listdir("data"):
    id = os.path.splitext(filename)[0]
    id_remove.append(id)

print("Removing " + str(len(id_remove)) + " from " + str(len(data.id)))

for i in range(0, len(data.id)):
    if data.id[i] in id_remove:
        idx_remove.append(i)

data.drop(idx_remove, axis=0, inplace=True)

print("Leaving: " + str(len(data.id)))
print("Cleaned up the download list")

url = "https://www.rijksmuseum.nl/api/en/collection/"
key = "?key=yt0FYhrb"

count = 1
list_length = len(data.id)

for obj in data.id:
    try:
        response = requests.get(url + obj + key)

        response_json = response.json()

        image_url = response_json["artObject"]["webImage"]["url"]

        image = requests.get(image_url)

        file = open(type + "_data/" + obj + ".jpeg", "wb")
        file.write(image.content)
        file.close()

        print("Finished: " + str(count) + "/" + str(list_length), end='\r')

        count = count + 1
    except:
        print("Couldn't retrieve image: " + obj)
        count = count + 1

print("Finished succesfully")