"""
This script evaluates a fine-tuned model on a test dataset of annotated images.
"""
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import tqdm

from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
import os
from PIL import Image
from pathlib import Path

import os
from torchvision import transforms
import timm
# ---------------------------------------

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import skimage.transform

# -----------------------------
# Dataset Class
# -----------------------------
class HistopathologyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_map = {'FA': 0, 'PT': 1}  # Adjust if using more classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image']
        image = np.load(img_path)
        if image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
        if image.shape[1] != 224 or image.shape[2] != 224:
            image = skimage.transform.resize(image, (3, 224, 224), anti_aliasing=True)
        image = torch.tensor(image, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        label = self.label_map[self.data.iloc[idx]['class']]
        return image, label, img_path

# -----------------------------
# CONCH Model Class
# -----------------------------
# class CONCHModelForFinetuning(nn.Module):
#     def __init__(self, num_classes=2, config={'hidden_size': 512}):
#         super().__init__()
#         self.config = config
#         self.model = self.make_conch()
#         self.fc = nn.Linear(self.config['hidden_size'], num_classes)

#     def make_conch(self):
#         # from your_module import create_model_from_pretrained  # Replace with actual import
#         model_cfg = 'conch_ViT-B-16'
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         checkpoint_path = 'C:\\Users\\Vivian\\Documents\\CONCH\\checkpoints\\conch\\pytorch_model.bin'
#         model, _ = create_model_from_pretrained(model_cfg, checkpoint_path, device=device)
#         return model

#     def forward(self, x):
#         out, h = self.model.visual(x)
#         return self.fc(out)

# -----------------------------
# UNI Model Class
# -----------------------------
class UNIModelForFinetuning(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = self.make_uni()
        self.fc = nn.Linear(1024, num_classes)

    def make_uni(self):
        local_dir = r"C:\Users\Vivian\Documents\CONCH\checkpoints\uni"
        os.makedirs(local_dir, exist_ok=True)
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16,
            init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(
            torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"),
            strict=True
        )
        return model

    def forward(self, x):
        out = self.model(x)
        return self.fc(out)

# -----------------------------
# Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CONCHModelForFinetuning(num_classes=2).to(device) # CONCH model
model = UNIModelForFinetuning(num_classes=2).to(device) # UNI model

# Load fine-tuned checkpoint
# fine_tuned_checkpoint = r'C:\Users\Vivian\Documents\CONCH\_finetune_weights_CONCH\patient_split_70.pth' # conch checkpoint
# fine_tuned_checkpoint = r'C:\Users\Vivian\Documents\CONCH\_finetune_weights_UNI\patient_split_70.pth' # fully finetuned
# fine_tuned_checkpoint = r'C:\Users\Vivian\Documents\CONCH\_finetune_weights_UNI\patient_split_70_linprob.pth' # linear probe
fine_tuned_checkpoint = r'C:\Users\Vivian\Documents\CONCH\_finetune_weights_UNI\all_slides_90_10.pth'
model.load_state_dict(torch.load(fine_tuned_checkpoint, map_location=device), strict=False)
model.eval()

# Load test data
# test_dataset = HistopathologyDataset(r"C:\Users\Vivian\Documents\CONCH\metadata\fine_tuning\test_ann_series8.csv")
test_dataset = HistopathologyDataset(r'C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv\test_patches.csv')
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# # Output directory for slide-wise predictions
# output_dir = r"C:\Users\Vivian\Documents\CONCH\patch_predictions\by_slide"
# os.makedirs(output_dir, exist_ok=True)

# Evaluation loop
criterion = nn.CrossEntropyLoss()
total_test_loss = 0
num_test_batches = 0
correct_test = 0
total_test = 0
all_test_labels = []
all_test_preds = []
predictions_list = []

with torch.no_grad():
    for images, labels, paths in tqdm.tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_test_loss += loss.item()
        num_test_batches += 1

        _, predicted = torch.max(outputs, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)
        all_test_labels.extend(labels.cpu().numpy())
        all_test_preds.extend(predicted.cpu().numpy())

        for i in range(labels.size(0)):
            predictions_list.append([paths[i], predicted[i].item(), labels[i].item()])

# Metrics
accuracy = correct_test / total_test
precision = precision_score(all_test_labels, all_test_preds)
recall = recall_score(all_test_labels, all_test_preds)
f1 = f1_score(all_test_labels, all_test_preds)

print("\n--- Evaluation Results ---")
print(f"Test Loss     : {total_test_loss / num_test_batches:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")

# # Save predictions to CSV
csv_save_path = r"C:\Users\Vivian\Documents\CONCH\patch_predictions\UNI90_test_ann_only.csv"
df = pd.DataFrame(predictions_list, columns=["Patch Path", "Predicted", "True Label"])
df.to_csv(csv_save_path, index=False)
print(f"\nSaved predictions to {csv_save_path}")
