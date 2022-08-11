# import required module
import os
from PIL import Image

# assign directory
directory = "data"

counter = 1
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    try:
        if os.path.isfile(f):
            image = Image.open(f)
            image = image.resize((600,600))
            image.save(f)
    except:
        next
    
    print("Resized: " + str(counter), end='\r')
    counter = counter + 1

print("Finished succesfully")