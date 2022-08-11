from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image

class FormatDataset(Dataset):
    def __init__(self, dataroot):
        # Path to data
        self.dataroot = dataroot
        # Store picture IDs
        self.samples = []
        # Length of dataset
        self.length = -1
        # Dictionary used to convert material string to unique material index
        self.dict = {
                        "aardewerk" : 0,
                        "brons" : 1,
                        "chine collé" : 2,
                        "dekverf" : 3,
                        "doek" : 4,
                        "eikenhout" : 5,
                        "email" : 6,
                        "faience" : 7,
                        "fluweel" : 8,
                        "fotodrager" : 9,
                        "geprepareerd papier" : 10,
                        "glas" : 11,
                        "glas-in-lood" : 12,
                        "glazuur" : 13,
                        "gouache (waterverf)" : 14,
                        "goud" : 15,
                        "goudkleurig bladmetaal" : 16,
                        "grafiet" : 17,
                        "grisailleverf" : 18,
                        "hout" : 19,
                        "ijzer" : 20,
                        "inkt" : 21,
                        "ivoor" : 22,
                        "kant (materiaal)" : 23,
                        "karton" : 24,
                        "katoen" : 25,
                        "kobalt" : 26,
                        "koper" : 27,
                        "kraakporselein" : 28,
                        "krijt" : 29,
                        "lak" : 30,
                        "leer" : 31,
                        "linnen" : 32,
                        "lood" : 33,
                        "marmer" : 34,
                        "messing" : 35,
                        "metaal" : 36,
                        "notenhout" : 37,
                        "olieverf" : 38,
                        "paneel" : 39,
                        "parelmoer" : 40,
                        "perkament" : 41,
                        "porselein" : 42,
                        "potlood" : 43,
                        "steengoed" : 44,
                        "terracotta" : 45,
                        "tin" : 46,
                        "tinglazuur" : 47,
                        "verf" : 48,
                        "verguldsel" : 49,
                        "waterverf" : 50,
                        "wol" : 51,
                        "zijde" : 52,
                        "zilver" : 53,
                        "zilverdraad" : 54
                        }

        self._init_dataset()
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Load the image based on ID
        image = Image.open(self.dataroot + self.samples[idx][0] + ".jpeg").convert("RGB")
        
        # Perform normalization using dataset average and sd
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transformer(image)
        img_np = np.array(img)

        # Return normalized image as np array and target material vector
        return (img_np, self.samples[idx][1])
    
    def _init_dataset(self):
        # Read the list of picture IDs and their target materials
        data = pd.read_csv(self.dataroot + "id_list_format.csv", encoding="ISO-8859-1")
        # Store length of dataset (number of pictures)
        self.length = len(data.id)

        # For each material string convert it to unique material index vector
        # based on the dictionary
        for i in range(0, len(data.id)):
            id = data.id[i]
            array = np.zeros((55,), dtype=int)
            for column in data.columns:
                if column == "id":
                    continue
                
                if data[column][i] == "NON":
                    break
                
                if data[column][i] in self.dict:
                    idx = self.dict[ data[column][i] ]
                    array[idx] = 1
            
            # Add ID and created target material vector to samples
            self.samples.append((id, array))