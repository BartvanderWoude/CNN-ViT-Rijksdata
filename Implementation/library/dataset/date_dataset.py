from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image

class DateDataset(Dataset):
    def __init__(self, dataroot):
        # Path to data
        self.dataroot = dataroot
        # Store picture IDs
        self.samples = []
        # Length of dataset
        self.length = -1

        self._init_dataset()
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Load image based on ID
        image = Image.open(self.dataroot + self.samples[idx][0] + ".jpeg").convert("RGB")

        # Perform normalization using dataset average and sd
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transformer(image)
        img_np = np.array(img)

        # Return normalized image as numpy array and target date
        return (img_np, self.samples[idx][1] - 1)
    
    def _init_dataset(self):
        # Read the list of picture IDs and their target date
        data = pd.read_csv(self.dataroot + "id_list_date.csv", encoding="ISO-8859-1")
        # Store length of dataset
        self.length = len(data.id)

        # Store IDs in 'self.samples'
        for i in range(0, len(data.id)):
            self.samples.append((data.id[i], data.century[i]))