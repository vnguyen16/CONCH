
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt


# updated dataset class for our private dataset with numpy files
class HistopathologyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Custom PyTorch Dataset for loading histopathology patches from .npy files.
        
        Args:
            csv_file (str): Path to the dataset metadata CSV file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_map = {'FA': 0, 'PT': 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image']
        image = np.load(img_path) 

        if image.shape[-1] == 3:  
            image = np.transpose(image, (2, 0, 1))  

        if image.shape[1] != 224 or image.shape[2] != 224:
            import skimage.transform
            image = skimage.transform.resize(image, (3, 224, 224), anti_aliasing=True)
        
        image = torch.tensor(image, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        class_name = self.data.iloc[idx]['class']
        label = self.label_map[class_name]  # Convert class name to label

        return image, label, img_path
        # return image, label


# Dataset class for BreakHis dataset
class HistopathologyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_map = {
            'B_A': 0,
            'B_F': 1,
            'B_PT': 2,
            'B_TA': 3,
            'M_DC': 4,
            'M_LC': 5,
            'M_MC': 6,
            'M_PC': 7
        } 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image']
        class_name = self.data.iloc[idx]['class']
        subclass_name = self.data.iloc[idx]['subclass']
        label = self.label_map[class_name + '_' + subclass_name]
        image = plt.imread(img_path)
        image = skimage.transform.resize(image, (224, 224))
        image = image.transpose((2, 0, 1))
        if self.transform:
            image = self.transform(image)
        return image, label


# Dataset class for BreaKHis dataset for FA and PT classes
class HistopathologyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_map = {
            'B_F': 0,   # Fibroadenoma → Class 0
            'B_PT': 1   # Phyllodes Tumor → Class 1
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image']
        class_name = self.data.iloc[idx]['class']
        subclass_name = self.data.iloc[idx]['subclass']
        label = self.label_map[class_name + '_' + subclass_name]
        image = plt.imread(img_path)
        image = skimage.transform.resize(image, (224, 224))
        image = image.transpose((2, 0, 1))  
        if self.transform:
            image = self.transform(image)

        return image, label
