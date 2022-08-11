import lxml.etree as et

# LOAD XML AND XSL
xsl = et.parse('removeNamespace.xsl')

# CONFIGURE AND RUN TRANSFORMER
transform = et.XSLT(xsl)

count = 1
numberOfFiles = 3100

while (count <= numberOfFiles):
    fileName = "xml_data/" + str(count) + ".xml"
    doc = et.parse(fileName)
    result = transform(doc)

    # OUTPUT RESULT TREE TO FILE
    with open(fileName, 'wb') as f:
        f.write(result)
    
    count = count + 1