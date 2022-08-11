from urllib import request
import requests
import urllib3
from xml.dom import minidom

# URLs used to get data. Resumption token is appended to the resumption_Url
base_Url = "http://www.rijksmuseum.nl/api/oai/yt0FYhrb?verb=ListRecords&set=subject:PublicDomainImages&metadataPrefix=dc"
resumption_Url = "http://www.rijksmuseum.nl/api/oai/yt0FYhrb?verb=ListRecords&set=subject:PublicDomainImages&metadataPrefix=dc&resumptionToken="

# Count counts the number of xml files so far. CompleteListSize is the amount of records in total
count = 1
completeListSize = 0

# Get the first xml
response = requests.get(base_Url)

file = open("data/" + str(count) + ".xml", "wb")
file.write(response.content)
file.close()

print("Finished: " + str(count), end='\r')

# Use the first xml to know how much data is in the total set
parsed = minidom.parse("data/" + str(count) + ".xml")
resumptionList = parsed.getElementsByTagName("resumptionToken")
completeListSize = int(resumptionList[0].getAttribute("completeListSize"))

# Get the remaining xml data
while count*20 < completeListSize:
    # Look for the resumption token in the previous xml
    parsed = minidom.parse("data/" + str(count) + ".xml")
    resumptionList = parsed.getElementsByTagName("resumptionToken")
    token = resumptionList[0].firstChild.nodeValue
    
    # Get the next xml using the resumption token
    response = requests.get(resumption_Url + token)

    # Save the new xml using count+1 as the index
    count = count + 1
    file = open("data/" + str(count) + ".xml", "wb")
    file.write(response.content)
    file.close()

    print("Finished: " + str(count), end='\r')

print("Finished all")
