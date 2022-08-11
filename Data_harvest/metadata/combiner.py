from xml.etree import ElementTree as et

first = et.parse("xml_data/1.xml")
firstRoot = first.getroot()

firstRoot.remove(firstRoot.find("responseDate"))
firstRoot.remove(firstRoot.find("request"))

firstListRecords = firstRoot.find("ListRecords")
firstListRecords.remove(firstListRecords.find("resumptionToken"))

count = 2
numberOfFiles = 3100

while (count <= numberOfFiles):
    fileName = "xml_data/" + str(count) + ".xml"
    second = et.parse(fileName)
    secondRoot = second.getroot()
    
    secondListRecords = secondRoot.find("ListRecords")
    secondListRecords.remove(secondListRecords.find("resumptionToken"))

    for record in secondListRecords:
        firstListRecords.append(record)
    
    count = count + 1

first.write("output.xml",xml_declaration=True)
