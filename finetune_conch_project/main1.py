# %%
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
import torch
from torch import nn
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import glob

import matplotlib.pyplot as plt
import numpy as np

import tqdm

import skimage

from torch.utils.data import DataLoader, Dataset

import os
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download

# show all jupyter output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %%
root = Path('../').resolve()
os.chdir(root)

# %% [markdown]
# CONCH

# %%
class CONCHModelForFinetuning(nn.Module):
    def __init__(self, num_classes=2, config={'hidden_size': 512}): # change number of classes for each dataset(8 for breast, 2 for breast)
        super().__init__()
        self.config = config
        self.model = self.make_conch()
        # self.fc = nn.Linear(self.config['hidden_size'], num_classes) # full finetuning?

        # linear probing
        # Freeze all parameters in the backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Only this linear layer will be trained
        self.fc = nn.Linear(self.config['hidden_size'], num_classes)

    def make_conch(self):
        # Load the model from "create_model_from_pretrained"
        model_cfg = 'conch_ViT-B-16'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # checkpoint_path = 'checkpoints/CONCH/pytorch_model.bin'
        checkpoint_path = 'C:\\Users\\Vivian\\Documents\\CONCH\\checkpoints\\conch\\pytorch_model.bin' # load in checkpoint here
        # checkpoint_path = r'C:\Users\Vivian\Documents\CONCH\_finetune_weights\Fold2_F_PT_model.pth' # loading breakhis finetuned model
        model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, device=device)
        
        return model
        
    def forward(self, x):
        out, h = self.model.visual(x)
        return self.fc(out)

# %%
model = CONCHModelForFinetuning().to('cuda')

# %% [markdown]
# load breakhis (PT + FA) checkpoint

# %%
class CONCHModelForFinetuning(nn.Module):
    def __init__(self, num_classes=2, config={'hidden_size': 512}, checkpoint_path=None):
        super().__init__()
        self.config = config
        self.model = self.make_conch()
        self.fc = nn.Linear(self.config['hidden_size'], num_classes)

        if checkpoint_path is not None:
            print(f"Loading fine-tuned weights from: {checkpoint_path}")
            self.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))

    def make_conch(self):
        # Load the base pretrained model
        model_cfg = 'conch_ViT-B-16'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_checkpoint = 'C:\\Users\\Vivian\\Documents\\CONCH\\checkpoints\\conch\\pytorch_model.bin'
        model, _ = create_model_from_pretrained(model_cfg, base_checkpoint, device=device)
        return model

    def forward(self, x):
        out, h = self.model.visual(x)
        return self.fc(out)


# %%
checkpoint_path = r'C:\Users\Vivian\Documents\CONCH\_finetune_weights\Fold2_F_PT_model.pth'
model = CONCHModelForFinetuning(num_classes=2, checkpoint_path=checkpoint_path).to('cuda')

# %% [markdown]
# UNI 2

# %%


class UNI2ModelForFinetuning(nn.Module):
    def __init__(self, num_classes=2): # change number of classes for each dataset
        super().__init__()
        # self.config = config
        self.model = self.make_uni2()
        # self.fc = nn.Linear(1536, num_classes)  # Match Vision Transformer output # full finetuning

        # Freeze all backbone parameters for linear probing
        for param in self.model.parameters():
            param.requires_grad = False

        # Add a small trainable classification head
        self.fc = nn.Linear(1536, num_classes)  # Match Vision Transformer output

    def make_uni2(self):
        # # Load the model from "create_model_from_pretrained"
        # model_cfg = 'conch_ViT-B-16'
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # # checkpoint_path = 'checkpoints/CONCH/pytorch_model.bin'
        # checkpoint_path = 'C:\\Users\\Vivian\\Documents\\CONCH\\checkpoints\\conch\\pytorch_model.bin' # load in checkpoint here
        # model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, device=device)
        
        # return model

        # local_dir = 'assets\\ckpts\\uni2-h'
        local_dir = 'C:\\Users\\Vivian\\Documents\\UNI2\\UNI\\assets\\ckpts\\uni2-h' 
        os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
        # hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
       
        timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224, 
        'patch_size': 14, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0, 
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
        }
        model = timm.create_model(**timm_kwargs)
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        # transform = transforms.Compose(
        # [
        # transforms.Resize(224),
        # transforms.CenterCrop(224),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # ]
        # )

        return model 
        
    def forward(self, x):
        out = self.model(x)
        # out, h = self.model.visual(x)
        return self.fc(out)

# %%
model = UNI2ModelForFinetuning().to('cuda')

# %% [markdown]
# UNI

# %%
import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download


class UNIModelForFinetuning(nn.Module):
    def __init__(self, num_classes=2,checkpoint_path=None): # change number of classes accordingly 
        ## ************ change for UNI ************
        super().__init__()
        # self.config = config
        self.model = self.make_uni()
        # self.fc = nn.Linear(self.config['hidden_size'], num_classes) # keep commented
        # self.fc = nn.Linear(1024, num_classes)  # Match Vision Transformer output # full finetuning

        #***** Freeze all backbone parameters - linear probing *****
        for param in self.model.parameters():
            param.requires_grad = False

        # Add a trainable classification head
        self.fc = nn.Linear(1024, num_classes)

        # ----- Load checkpoint if needed -----
        if checkpoint_path:
            print(f"Loading checkpoint from: {checkpoint_path}")
            self.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
    
    def make_uni(self):
        # login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

        local_dir = r"C:\Users\Vivian\Documents\CONCH\checkpoints\uni" # load in UNI model
        os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
        
        # hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        
        # transform = transforms.Compose(
        #     [
        #         transforms.Resize(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     ]
        # )
        # model.eval()
        return model 
    
    def forward(self, x):
        out = self.model(x)
        # out, h = self.model.visual(x)
        return self.fc(out)

# %%
model = UNIModelForFinetuning().to('cuda')

# %%
# Loading a checkpoint
model = UNIModelForFinetuning(num_classes=2, checkpoint_path=r"C:\Users\Vivian\Documents\CONCH\_finetune_weights_UNI\linprob_ann_CL_uni.pth").to('cuda')

# %% [markdown]
# ResNet

# %%
import torchvision.models as models
import torch.nn as nn

# resnet = models.resnet18(pretrained=True) # resnet18
resnet = models.resnet50(pretrained=True) # resnet50
resnet.fc = nn.Linear(resnet.fc.in_features, 2)  # 2 output classes for CrossEntropyLoss


# %%
model = resnet.to('cuda')

# %% [markdown]
# <!-- Making metadata -->

# %%
# saves metadata to csv for each fold and mode - use this 
# need to make metadata for new datasets
def make_metadata(fold):
    metadata = pd.DataFrame()
    for mode in ['train', 'test']:
        pathname = f'/Users/Vivian/Documents/CONCH/Folds/Fold {fold}/{mode}/'
        images = os.listdir(pathname)
        for image in images:
            if not image.startswith('SOB'):
                continue
            label = image.split('-')[0].replace('SOB_', '')
            class_name, subclass_name = label.split('_')
            #metadata = metadata.append({'image': pathname+image, 'fold': fold, 'mode': mode, 'class': class_name, 'subclass': subclass_name}, ignore_index=True)
            metadata = pd.concat([metadata, pd.DataFrame({'image': pathname+image, 'fold': fold, 'mode': mode, 'class': class_name, 'subclass': subclass_name}, index=[0])], ignore_index=True)
        metadata.to_csv(f'/Users/Vivian/Documents/CONCH/Folds/Fold {fold}/{mode}/metadata.csv', index=False)
    return metadata

# %%
# making metadata for our private dataset
import os
import pandas as pd

def make_metadata():
    metadata = pd.DataFrame()
    
    # # Define paths
    # base_path = "patches"  # Root where patches are stored
    # metadata_dir = "metadata"  # Directory containing train/test CSVs
    # output_dir = "metadata/fine_tuning"  # Where metadata CSVs will be saved
    # os.makedirs(output_dir, exist_ok=True)

    # ***********Define paths for only test set
    base_path = "patches_annotated"  # Root where patches are stored
    metadata_dir = "metadata"  # Directory containing train/test CSVs
    output_dir = "metadata/fine_tuning"  # Where metadata CSVs will be saved
    os.makedirs(output_dir, exist_ok=True)
    # ************

    # Load train/test slide selections
    # train_slides = pd.read_csv(os.path.join(metadata_dir, "train_patient_split.csv"))
    test_slides = pd.read_csv(os.path.join(metadata_dir, "test_ann_series8.csv"))

    # train_slides = pd.read_csv(os.path.join(metadata_dir, "train_all_slides.csv"))
    # test_slides = pd.read_csv(os.path.join(metadata_dir, "test_all_slides.csv"))


    # Process both train and test sets
    # for mode, slides_df in zip(["train", "test"], [train_slides, test_slides]):
    for mode, slides_df in zip(["test"], [test_slides]): # only creating test set

        entries = []
        
        for _, row in slides_df.iterrows():
            slide_name = row["Filename"]
            class_name = row["Class"]
            magnification = f"{row['Magnification']}x"

            # Define the path where patches are stored for this slide
            slide_patch_dir = os.path.join(base_path, magnification, class_name, slide_name)

            # Ensure slide directory exists and has patches
            if not os.path.exists(slide_patch_dir) or len(os.listdir(slide_patch_dir)) == 0:
                print(f"Skipping {slide_name}: No patches found.")
                continue
            
            # Add all patches in the slide directory to the metadata
            for patch_file in os.listdir(slide_patch_dir):
                if patch_file.endswith(".npy"):  # Ensure we're only adding valid patch files
                    patch_path = os.path.join(slide_patch_dir, patch_file)
                    entries.append({
                        "image": patch_path,
                        "fold": 1,  # Since you're using only one fold
                        "mode": mode,
                        "class": class_name,
                        "magnification": magnification
                    })

        # Convert list to DataFrame
        mode_metadata = pd.DataFrame(entries)

        # output_csv = 

        # Save metadata CSV for the mode
        mode_metadata.to_csv(os.path.join(output_dir, f"{mode}_ann_series8_2.csv"), index=False) # manually change metadata file name here

        print(f"Saved {mode}_ann_series8_2.csv with {len(mode_metadata)} entries.")

    return metadata

# %%
# making metadata for our private dataset - train, val, test
import os
import pandas as pd

def make_metadata():
    # === CONFIGURATION ===
    annotated_base_path = r"C:\Users\Vivian\Documents\CONCH\patches_tiled\patches_10x"
    # fallback_base_path = "patches"
    metadata_dir = "metadata/patient_split_annotate/slide_csv"
    output_dir = "metadata/patient_split_annotate/patch_csv_10x"
    os.makedirs(output_dir, exist_ok=True)

    # Define your CSVs and the filenames for outputs
    slide_sets = {
        "train": {
            "input_csv": os.path.join(metadata_dir, "train_split.csv"),
            "output_csv": "train_patches.csv"
        },
        "val": {
            "input_csv": os.path.join(metadata_dir, "val_split.csv"),
            "output_csv": "val_patches.csv"
        },
        "test": {
            "input_csv": os.path.join(metadata_dir, "test_split.csv"),
            "output_csv": "test_patches.csv"
        }
    }

    # === PROCESS EACH SET ===
    for mode, paths in slide_sets.items():
        slides_df = pd.read_csv(paths["input_csv"])
        entries = []

        for _, row in slides_df.iterrows():
            slide_name = row["Filename"]
            class_name = row["Class"]
            magnification = f"{row['Magnification']}x"

            # Try annotated path first
            slide_patch_dir = os.path.join(annotated_base_path, magnification, class_name, slide_name)
            if not os.path.exists(slide_patch_dir) or len(os.listdir(slide_patch_dir)) == 0:
                print(f"[{mode}] Skipping {slide_name}: No patches found in annotated path.")
                continue

            # # Fallback to regular patches if needed
            # if not os.path.exists(slide_patch_dir) or len(os.listdir(slide_patch_dir)) == 0:
            #     slide_patch_dir = os.path.join(fallback_base_path, magnification, class_name, slide_name)
            #     if not os.path.exists(slide_patch_dir) or len(os.listdir(slide_patch_dir)) == 0:
            #         print(f"[{mode}] Skipping {slide_name}: No patches found in either location.")
            #         continue

            # Collect patch metadata
            for patch_file in os.listdir(slide_patch_dir):
                if patch_file.endswith(".npy"):
                    patch_path = os.path.join(slide_patch_dir, patch_file)
                    entries.append({
                        "image": patch_path,
                        "fold": 1,
                        "mode": mode,
                        "class": class_name,
                        "magnification": magnification
                    })

        # Save metadata to CSV
        mode_metadata = pd.DataFrame(entries)
        output_csv_path = os.path.join(output_dir, paths["output_csv"])
        mode_metadata.to_csv(output_csv_path, index=False)
        print(f"[{mode}] Saved {paths['output_csv']} with {len(mode_metadata)} entries.")


# %%
make_metadata() 

# %%
# Custom Dataset class
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
        }  # Example mapping of subclasses to numerical labels

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



# %%
# using breakhis dataset but only with 2 classes FA and PT 

from torch.utils.data import Dataset
import pandas as pd
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt

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

        # Map "B_F" -> 0, "B_PT" -> 1
        label = self.label_map[class_name + '_' + subclass_name]

        # Load and preprocess image
        image = plt.imread(img_path)
        image = skimage.transform.resize(image, (224, 224))
        image = image.transpose((2, 0, 1))  # Convert to C x H x W for PyTorch
        if self.transform:
            image = self.transform(image)

        return image, label


# %%
# updated dataset class for our private dataset with numpy files

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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

        # Mapping FA -> 0, PT -> 1
        self.label_map = {'FA': 0, 'PT': 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image patch
        img_path = self.data.iloc[idx]['image']
        image = np.load(img_path)  # Load .npy file (already in NumPy format)

        # Ensure image is in (C, H, W) format for PyTorch
        if image.shape[-1] == 3:  # Check if image is in (H, W, C) format
            image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)

        # Resize to 224x224 if needed
        if image.shape[1] != 224 or image.shape[2] != 224:
            import skimage.transform
            image = skimage.transform.resize(image, (3, 224, 224), anti_aliasing=True)
        
        # Normalize pixel values
        image = torch.tensor(image, dtype=torch.float32)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Get label
        class_name = self.data.iloc[idx]['class']
        label = self.label_map[class_name]  # Convert class name to label

        return image, label, img_path
        # return image, label


# %% [markdown]
# right now, we are only using the first fold?

# %%
# train_data = HistopathologyDataset('/Users/Vivian/Documents/CONCH/Folds/Fold 1/train/metadata.csv')
# test_data = HistopathologyDataset('/Users/Vivian/Documents/CONCH/Folds/Fold 1/test/metadata.csv')

# early results --> pick one fold 
# pick a fold to train and test

train_data = HistopathologyDataset('/Users/Vivian/Documents/CONCH/Folds/Fold 2/train/metadata.csv')
test_data = HistopathologyDataset('/Users/Vivian/Documents/CONCH/Folds/Fold 2/test/metadata.csv')

# %%
# make a dataloder for me please 
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=True)

# %% [markdown]
# instantiating train and test set for private data

# %%
# # Example usage
# subset of slides
# train_dataset = HistopathologyDataset("metadata/fine_tuning/train_metadata.csv")
# test_dataset = HistopathologyDataset("metadata/fine_tuning/test_metadata.csv")

# testing with all slides successfully tiled
# train_dataset = HistopathologyDataset("metadata\\fine_tuning\\train_patient_split_metadata.csv")
# test_dataset = HistopathologyDataset("metadata\\fine_tuning\\test_patient_split_metadata.csv")

# adding val set
# train_dataset = HistopathologyDataset("metadata\\patient_split_annotate\\patch_csv\\train_cleaned_CL_patches.csv")
train_dataset = HistopathologyDataset(r"C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv_2.5x\train_patches.csv")
# val_dataset = HistopathologyDataset("metadata\\patient_split_annotate\\patch_csv\\val_cleaned_CL_patches.csv")
val_dataset = HistopathologyDataset(r"C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv_2.5x\val_patches.csv")
test_dataset = HistopathologyDataset(r"C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv_2.5x\test_patches.csv")

# Check dataset sample
# sample_image, sample_label = train_dataset[0]
# print("Image shape:", sample_image.shape)
# print("Label:", sample_label)

# %%
# make a dataloder 
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# %% [markdown]
# UNI - Cleanlab issues patches - run2

# %%
import torch
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Setup
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Only optimize the classification head (linear layer) - for linear probing
optimizer = torch.optim.Adam(model.fc.parameters(), lr=2e-5) # increased learning rate

criterion = torch.nn.CrossEntropyLoss()
device = 'cuda'
num_epochs = 5
patience = 5

# model_save_path = '/Users/Vivian/Documents/CONCH/_finetune_weights_CONCH/with_val_earlystop.pth'
# csv_save_path = "/Users/Vivian/Documents/CONCH/patch_predictions/with_val_earlystop.csv"

# Define model save path 
model_dir = "/Users/Vivian/Documents/CONCH/_finetune_weights_ResNet50/10x"
# Define CSV path for saving patch predictions
csv_dir = "patch_predictions/10x"

# check for model save path directory 
os.makedirs(model_dir, exist_ok=True)
model_save_path = os.path.join(model_dir, "linprob.pth") # change model name here
# check for csv save path directory 
os.makedirs(csv_dir, exist_ok=True)
csv_save_path = os.path.join(csv_dir, "ResNet50_linprob.csv") # change prediction csv name here


best_val_accuracy = 0
epochs_no_improve = 0

# Mixed Precision
scaler = torch.cuda.amp.GradScaler()

# Metrics tracking
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Store per-epoch metrics as a list of dicts
epoch_metrics = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()

    total_train_loss = 0
    correct_train, total_train = 0, 0
    all_train_labels, all_train_preds = [], []

    for images, labels, _ in tqdm.tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()
        _, predicted = torch.max(outputs, 1) # without softmax (auroc)

        # probs = torch.softmax(outputs, dim=1)[:, 1]  # probability for class 1
        # _, predicted = torch.max(outputs, 1)    

        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
        all_train_labels.extend(labels.cpu().numpy())
        all_train_preds.extend(predicted.cpu().numpy())

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # print(f"Train Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.4f}")
    train_precision = precision_score(all_train_labels, all_train_preds, average="binary")
    train_recall = recall_score(all_train_labels, all_train_preds, average="binary")
    train_f1 = f1_score(all_train_labels, all_train_preds, average="binary")

    print(f"Train Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.4f} | "
        f"Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1 Score: {train_f1:.4f}")


    # --- Validation ---
    model.eval()
    total_val_loss = 0
    correct_val, total_val = 0, 0
    all_val_labels, all_val_preds = [], []

    with torch.no_grad():
        for images, labels, _ in tqdm.tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)
            all_val_labels.extend(labels.cpu().numpy())
            all_val_preds.extend(predicted.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    val_precision = precision_score(all_val_labels, all_val_preds, average="binary")
    val_recall = recall_score(all_val_labels, all_val_preds, average="binary")
    val_f1 = f1_score(all_val_labels, all_val_preds, average="binary")

    print(f"Val Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.4f} | "
          f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1 Score: {val_f1:.4f}")


    # Collect metrics for this epoch
    epoch_metrics.append({
        "Epoch": epoch + 1,
        "Train Loss": avg_train_loss,
        "Train Accuracy": train_accuracy,
        "Train Precision": train_precision,
        "Train Recall": train_recall,
        "Train F1 Score": train_f1,
        "Val Loss": avg_val_loss,
        "Val Accuracy": val_accuracy,
        "Val Precision": val_precision,
        "Val Recall": val_recall,
        "Val F1 Score": val_f1
    })


    # --- Early Stopping ---
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ Model saved with improved val accuracy: {val_accuracy:.4f}")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s)")
        if epochs_no_improve >= patience:
            print("⏹️ Early stopping triggered.")
            break

# --- Plotting ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# --- Final Test Evaluation ---
print("\n🔍 Evaluating best model on the test set...")
model.load_state_dict(torch.load(model_save_path))
model.eval()

total_test_loss = 0
correct_test, total_test = 0, 0
all_test_labels, all_test_preds = [], []
test_predictions_list = []

with torch.no_grad():
    for images, labels, file_paths in tqdm.tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)
        all_test_labels.extend(labels.cpu().numpy())
        all_test_preds.extend(predicted.cpu().numpy())

        for i in range(labels.size(0)):
            test_predictions_list.append([file_paths[i], predicted[i].item(), labels[i].item()])

avg_test_loss = total_test_loss / len(test_loader)
test_accuracy = correct_test / total_test
test_precision = precision_score(all_test_labels, all_test_preds, average="binary")
test_recall = recall_score(all_test_labels, all_test_preds, average="binary")
test_f1 = f1_score(all_test_labels, all_test_preds, average="binary")

print(f"\n📊 Final Test Results:\n"
      f"Test Loss: {avg_test_loss:.4f} | Accuracy: {test_accuracy:.4f} | "
      f"Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1 Score: {test_f1:.4f}")

# Save test predictions
test_csv_path = csv_save_path.replace(".csv", "_test.csv")
df_test = pd.DataFrame(test_predictions_list, columns=["Patch Path", "Predicted", "True Label"])
df_test.to_csv(test_csv_path, index=False)
print(f"✅ Test predictions saved to: {test_csv_path}")

# --- Append summary row ---
summary_row = {
    "Epoch": "Best Val",
    "Train Loss": "",
    "Train Accuracy": "",
    "Train Precision": "",
    "Train Recall": "",
    "Train F1 Score": "",
    "Val Loss": "",
    "Val Accuracy": best_val_accuracy,
    "Val Precision": "",
    "Val Recall": "",
    "Val F1 Score": "",
    "Test Loss": avg_test_loss,
    "Test Accuracy": test_accuracy,
    "Test Precision": test_precision,
    "Test Recall": test_recall,
    "Test F1 Score": test_f1
}
epoch_metrics.append(summary_row)


metrics_df = pd.DataFrame(epoch_metrics)
metrics_csv_path = csv_save_path.replace(".csv", "_epoch_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"📊 Epoch metrics saved to: {metrics_csv_path}")

# # Save as CSV (one row)
# summary_path = csv_save_path.replace(".csv", "_metrics_summary.csv")
# pd.DataFrame([metrics]).to_csv(summary_path, index=False)
# print(f"📁 Metrics summary saved to: {summary_path}")

# %%
# --- Final Test Evaluation ---
print("\n🔍 Evaluating best model on the test set...")
model.load_state_dict(torch.load(model_save_path))
model.eval()

total_test_loss = 0
correct_test, total_test = 0, 0
all_test_labels, all_test_preds = [], []
test_predictions_list = []

with torch.no_grad():
    for images, labels, file_paths in tqdm.tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)
        all_test_labels.extend(labels.cpu().numpy())
        all_test_preds.extend(predicted.cpu().numpy())

        for i in range(labels.size(0)):
            test_predictions_list.append([file_paths[i], predicted[i].item(), labels[i].item()])

avg_test_loss = total_test_loss / len(test_loader)
test_accuracy = correct_test / total_test
test_precision = precision_score(all_test_labels, all_test_preds, average="binary")
test_recall = recall_score(all_test_labels, all_test_preds, average="binary")
test_f1 = f1_score(all_test_labels, all_test_preds, average="binary")

print(f"\n📊 Final Test Results:\n"
      f"Test Loss: {avg_test_loss:.4f} | Accuracy: {test_accuracy:.4f} | "
      f"Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1 Score: {test_f1:.4f}")

# Save test predictions
test_csv_path = csv_save_path.replace(".csv", "_test.csv")
df_test = pd.DataFrame(test_predictions_list, columns=["Patch Path", "Predicted", "True Label"])
df_test.to_csv(test_csv_path, index=False)
print(f"✅ Test predictions saved to: {test_csv_path}")

# --- Append summary row ---
summary_row = {
    "Epoch": "Best Val",
    "Train Loss": "",
    "Train Accuracy": "",
    "Train Precision": "",
    "Train Recall": "",
    "Train F1 Score": "",
    "Val Loss": "",
    "Val Accuracy": best_val_accuracy,
    "Val Precision": "",
    "Val Recall": "",
    "Val F1 Score": "",
    "Test Loss": avg_test_loss,
    "Test Accuracy": test_accuracy,
    "Test Precision": test_precision,
    "Test Recall": test_recall,
    "Test F1 Score": test_f1
}
epoch_metrics.append(summary_row)


metrics_df = pd.DataFrame(epoch_metrics)
metrics_csv_path = csv_save_path.replace(".csv", "_epoch_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"📊 Epoch metrics saved to: {metrics_csv_path}")

# # Save as CSV (one row)
# summary_path = csv_save_path.replace(".csv", "_metrics_summary.csv")
# pd.DataFrame([metrics]).to_csv(summary_path, index=False)
# print(f"📁 Metrics summary saved to: {summary_path}")

# %% [markdown]
# Using wandb.ai

# %%
import os
import torch
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Initialize Weights & Biases
wandb.init(project="resnet50_finetuning", name="linprob_2.5x", config={
    "epochs": 3,
    "lr": 2e-5,
    "model": "ResNet50",
    "task": "linear_probing",
    "magnification": "2.5x"
})

# Optimizer setup
optimizer = torch.optim.Adam(model.fc.parameters(), lr=wandb.config.lr)
criterion = torch.nn.CrossEntropyLoss()
device = 'cuda'
num_epochs = wandb.config.epochs
patience = 5

# Paths
model_dir = "/Users/Vivian/Documents/CONCH/_finetune_weights_ResNet50/wandb_2.5x"
csv_dir = "patch_predictions/wandb_2.5x"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

model_save_path = os.path.join(model_dir, "linprob.pth")
csv_save_path = os.path.join(csv_dir, "ResNet50_linprob.csv")

# Mixed Precision
scaler = torch.cuda.amp.GradScaler()

# Metrics tracking
best_val_accuracy = 0
epochs_no_improve = 0
epoch_metrics = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()

    total_train_loss = 0
    correct_train, total_train = 0, 0
    all_train_labels, all_train_preds, all_train_probs = [], [], []

    for images, labels, _ in tqdm.tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1]
        _, predicted = torch.max(outputs, 1)

        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
        all_train_labels.extend(labels.cpu().numpy())
        all_train_preds.extend(predicted.cpu().numpy())
        all_train_probs.extend(probs.detach().cpu().numpy())

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_precision = precision_score(all_train_labels, all_train_preds)
    train_recall = recall_score(all_train_labels, all_train_preds)
    train_f1 = f1_score(all_train_labels, all_train_preds)
    train_auroc = roc_auc_score(all_train_labels, all_train_probs)

    print(f"Train Loss: {avg_train_loss:.4f} | Acc: {train_accuracy:.4f} | "
          f"Prec: {train_precision:.4f} | Rec: {train_recall:.4f} | F1: {train_f1:.4f} | AUROC: {train_auroc:.4f}")

    # --- Validation ---
    model.eval()
    total_val_loss = 0
    correct_val, total_val = 0, 0
    all_val_labels, all_val_preds, all_val_probs = [], [], []

    with torch.no_grad():
        for images, labels, _ in tqdm.tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_val_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)

            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)
            all_val_labels.extend(labels.cpu().numpy())
            all_val_preds.extend(predicted.cpu().numpy())
            all_val_probs.extend(probs.detach().cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    val_precision = precision_score(all_val_labels, all_val_preds)
    val_recall = recall_score(all_val_labels, all_val_preds)
    val_f1 = f1_score(all_val_labels, all_val_preds)
    val_auroc = roc_auc_score(all_val_labels, all_val_probs)

    print(f"Val Loss: {avg_val_loss:.4f} | Acc: {val_accuracy:.4f} | "
          f"Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | F1: {val_f1:.4f} | AUROC: {val_auroc:.4f}")

    # Log to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train/loss": avg_train_loss,
        "train/accuracy": train_accuracy,
        "train/precision": train_precision,
        "train/recall": train_recall,
        "train/f1": train_f1,
        "train/auroc": train_auroc,
        "val/loss": avg_val_loss,
        "val/accuracy": val_accuracy,
        "val/precision": val_precision,
        "val/recall": val_recall,
        "val/f1": val_f1,
        "val/auroc": val_auroc,
    })

    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_save_path)
        print("✅ Saved best model")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("⏹️ Early stopping triggered")
            break

    epoch_metrics.append({
        "Epoch": epoch + 1,
        "Train Loss": avg_train_loss,
        "Train Accuracy": train_accuracy,
        "Train Precision": train_precision,
        "Train Recall": train_recall,
        "Train F1 Score": train_f1,
        "Train AUROC": train_auroc,
        "Val Loss": avg_val_loss,
        "Val Accuracy": val_accuracy,
        "Val Precision": val_precision,
        "Val Recall": val_recall,
        "Val F1 Score": val_f1,
        "Val AUROC": val_auroc
    })

# Save epoch metrics to CSV
metrics_df = pd.DataFrame(epoch_metrics)
metrics_csv_path = csv_save_path.replace(".csv", "_epoch_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"📊 Saved metrics to: {metrics_csv_path}")


# --- Final Test Evaluation ---
print("\n🔍 Evaluating best model on the test set...")
model.load_state_dict(torch.load(model_save_path))
model.eval()

total_test_loss = 0
correct_test, total_test = 0, 0
all_test_labels, all_test_preds = [], []
test_predictions_list = []

with torch.no_grad():
    for images, labels, file_paths in tqdm.tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_test_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1]
        _, predicted = torch.max(outputs, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)
        all_test_labels.extend(labels.cpu().numpy())
        all_test_preds.extend(probs.cpu().numpy())  # For AUROC

        # Append test predictions for CSV
        for i in range(labels.size(0)):
            test_predictions_list.append([
                file_paths[i],
                int(predicted[i].cpu()),
                int(labels[i].cpu())
            ])

avg_test_loss = total_test_loss / len(test_loader)

# Convert to NumPy arrays
y_true = np.array(all_test_labels)
y_pred = np.array([int(p > 0.5) for p in all_test_preds])  # threshold probs at 0.5
y_probs = np.array(all_test_preds)  # for AUROC

# Compute metrics
test_accuracy = (y_pred == y_true).mean()
test_precision = precision_score(y_true, y_pred, average="binary")
test_recall = recall_score(y_true, y_pred, average="binary")
test_f1 = f1_score(y_true, y_pred, average="binary")
test_auroc = roc_auc_score(y_true, y_probs)

print(f"\n📊 Final Test Results:\n"
      f"Test Loss: {avg_test_loss:.4f} | Accuracy: {test_accuracy:.4f} | "
      f"Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1 Score: {test_f1:.4f} | AUROC: {test_auroc:.4f}")

wandb.log({
    "test/loss": avg_test_loss,
    "test/accuracy": test_accuracy,
    "test/precision": test_precision,
    "test/recall": test_recall,
    "test/f1": test_f1,
    "test/auroc": test_auroc
})

# %%
# --- Final Test Evaluation ---
print("\n🔍 Evaluating best model on the test set...")
model.load_state_dict(torch.load(model_save_path))
model.eval()

total_test_loss = 0
correct_test, total_test = 0, 0
all_test_labels, all_test_preds = [], []
test_predictions_list = []

with torch.no_grad():
    for images, labels, file_paths in tqdm.tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_test_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1]
        _, predicted = torch.max(outputs, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)
        all_test_labels.extend(labels.cpu().numpy())
        all_test_preds.extend(probs.cpu().numpy())  # For AUROC

        # Append test predictions for CSV
        for i in range(labels.size(0)):
            test_predictions_list.append([
                file_paths[i],
                int(predicted[i].cpu()),
                int(labels[i].cpu())
            ])

avg_test_loss = total_test_loss / len(test_loader)

# Convert to NumPy arrays
y_true = np.array(all_test_labels)
y_pred = np.array([int(p > 0.5) for p in all_test_preds])  # threshold probs at 0.5
y_probs = np.array(all_test_preds)  # for AUROC

# Compute metrics
test_accuracy = (y_pred == y_true).mean()
test_precision = precision_score(y_true, y_pred, average="binary")
test_recall = recall_score(y_true, y_pred, average="binary")
test_f1 = f1_score(y_true, y_pred, average="binary")
test_auroc = roc_auc_score(y_true, y_probs)

print(f"\n📊 Final Test Results:\n"
      f"Test Loss: {avg_test_loss:.4f} | Accuracy: {test_accuracy:.4f} | "
      f"Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1 Score: {test_f1:.4f} | AUROC: {test_auroc:.4f}")

wandb.log({
    "test/loss": avg_test_loss,
    "test/accuracy": test_accuracy,
    "test/precision": test_precision,
    "test/recall": test_recall,
    "test/f1": test_f1,
    "test/auroc": test_auroc
})


