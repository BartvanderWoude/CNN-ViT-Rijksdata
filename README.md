# Comparing CNN and ViT on their ability to multi-task classify Rijksmuseum art
This GitHub repository contains (almost) all files regarding Bart van der Woude's (s398891) Bachelor thesis Artificial Intelligence at the Rijksuniversiteit Groningen,
supervised by Dr. M. Sabatelli. The thesis can be found using the link: ...

This repository contains an API data harvester for the open-source Rijksdata dataset (https://data.rijksmuseum.nl/) metadata, as well as an image harvester. It 
contains tools for cleaning up the harvested metadata, resizing images, etc.

In order for these models to work straight
out of the box you will need to download the created dataset using the link: https://drive.google.com/file/d/1UzZysureK4M7oTRy9ZPlbR1QnAGZDAXq/view?usp=sharing  This can then be extracted in the Implementation/data/rijksdata/ folder. 

After doing this the training and testing environment can be run from Implementation/ using:
```
python3 environment.py
```

You will need to have installed the python packages:
```
torch
torchvision
pandas
numpy
```

## Data harvest
This project makes use of the Rijksmuseum open-source dataset called Rijksdata (https://data.rijksmuseum.nl/). This database contains metadata and images of present 
and past art historical objects found in the Rijksmuseum. The Rijksdata database makes use of an API to access its data. This folder contains tools for harvesting and cleaning up
this data.

### Metadata
records.py: This script harvests a large number of XML files from Rijksdata, each file containing 20 records that also have an image in the database.

combiner.py: This script combines all XML files into a single large XML file.

removeNamespace.py: Removes namespaces and other unnecessary information from full XML file.

### Images
id_list_date.csv: A .csv file containing the IDs of all records containing an image and a usable date stamp.

id_list_date.csv: A .csv file containing the IDs of all records containing an image and a usable list of materials.

harvest_images.py: This script harvests the images from Rijksdata using the IDs from the ID lists and stores it in the folder data/.

resize.py: This script resizes all images in the resize/ folder to 600x600 pixels.

## Data analysis
The harvested metadata needs to be analyzed in order for it to be useful.

data_analysis_script.R: This R script contains instructions how to analyze the metadata XML file. It also contains instructions on how to create a list of art object IDs
paired with a target century, as well as a list of art object IDs paired with a target list of materials.

## Implementation
This folder contains the heart of the project; the actual implementation of the Convolutional Neural Network and the Vision Transformer. In order for this to work straight
out of the box you will need to download the created dataset using the link: https://drive.google.com/file/d/1UzZysureK4M7oTRy9ZPlbR1QnAGZDAXq/view?usp=sharing  This can then be extracted in the Implementation/data/rijksdata/ folder. 

After doing this the training and testing environment can be run from Implementation/ using:
```
python3 environment.py
```

environment.py: Script handling user input and pushing command to the CNN and ViT handlers.

get_linearlayer_size.py: Script that helps find the output size of the vector that feeds into the MLP heads of the CNN and ViT.

### Library
#### Dataset
This folder contains the DataLoader classes for both the date classification dataset and the material classification dataset.

date_dataset.py: Contains DataLoader class for date dataset.

format_dataset.py: Contains DataLoader class for material dataset.

#### Cnn_pytorch
cnn.py: Contains the class for the CNN architecture.

cnn_handler.py: Contains the class that handles the CNN model, such as training, testing, saving, etc.

#### Local_vit_pytorch
A local version of the ViT library created by GitHub user lucidrains (https://github.com/lucidrains/vit-pytorch).

vit.py: An existing file in lucidrain's ViT library, but there are changes to the ViT class to fit the model architecture to our needs.

vit_handler.py: A new file that contains the class that handles the ViT model, such as training, testing, saving, etc.

### Saved_models
This folder is used to save and load models from.
