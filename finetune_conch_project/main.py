from config.config import Config
from models.model_conch import CONCHModelForFinetuning
from data.data_loader import HistopathologyDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from utils.train_models import train_and_validate, evaluate_test
from utils.metadata_utils import make_breakhis_metadata, make_metadata_from_split_csv


cfg = Config()
device = torch.device(cfg.device)

# model = load_model(cfg)
model = CONCHModelForFinetuning(
    num_classes=cfg.num_classes,
    hidden_size=cfg.hidden_size,
    checkpoint_path=cfg.finetuned_conch_checkpoint
).to(device)

train_ds = HistopathologyDataset(cfg.train_csv)
val_ds = HistopathologyDataset(cfg.val_csv)
test_ds = HistopathologyDataset(cfg.test_csv)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)   
test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=cfg.learning_rate)
scaler = torch.cuda.amp.GradScaler()

# Train and Validate
model_path, train_val_metrics = train_and_validate(model, train_loader, val_loader, cfg) # pass criterion and optimizer here??***
# train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)

# Test
evaluate_test(model, test_loader, model_path, cfg)

# ======== Metadata Generation ========
# # Create metadata for BreakHis Fold 2
# make_breakhis_metadata(fold=2)

# # Or generate metadata for private dataset
# make_metadata_from_split_csv(
#     annotated_base_path=r"C:\Users\Vivian\Documents\CONCH\patches_tiled\patches_10x",
#     metadata_dir="metadata/patient_split_annotate/slide_csv",
#     output_dir="metadata/patient_split_annotate/patch_csv_10x"
# )
# ======================================