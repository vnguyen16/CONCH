import wandb
import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch
import tqdm


def train_and_validate(model, train_loader, val_loader, test_loader, cfg):
    device = torch.device(cfg.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=cfg.wandb_config["lr"])
    scaler = torch.cuda.amp.GradScaler()
    if cfg.use_wandb:
        num_epochs = cfg.wandb_config["epochs"]
    else:
        num_epochs = cfg.epochs

    # Setup logging directories
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.csv_dir, exist_ok=True)
    model_save_path = os.path.join(cfg.model_dir, cfg.model_filename)
    csv_save_path = os.path.join(cfg.csv_dir, cfg.prediction_filename)

    # Initialize WandB
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=cfg.wandb_config)

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

        if cfg.use_wandb:
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
            if epochs_no_improve >= cfg.patience:
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
    # metrics_df.to_csv(metrics_csv_path, index=False)
    # print(f"📊 Saved metrics to: {metrics_csv_path}")

    return model_save_path, metrics_df


def evaluate_test(model, test_loader, model_save_path, cfg):
    device = torch.device(cfg.device)
    criterion = torch.nn.CrossEntropyLoss()

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

    test_metrics_df = []

    test_metrics_df.append({
        "Test Loss": avg_test_loss,
        "Test Accuracy": test_accuracy,
        "Test Precision": test_precision,   
        "Test Recall": test_recall,
        "Test F1 Score": test_f1,
        "Test AUROC": test_auroc
    })
    
    return test_metrics_df