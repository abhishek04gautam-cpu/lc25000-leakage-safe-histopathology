"""Training and evaluation utilities for LC25000."""
import copy
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from config import IMAGENET_NORMALIZE_MEAN, IMAGENET_NORMALIZE_STD, LC25000_BATCH_SIZE, LC25000_HORIZONTAL_FLIP_P, LC25000_ROTATION_DEGREES, LC25000_UNFREEZE_EPOCH, MODELS_DIR, RANDOM_STATE, RESULTS_DIR
from visualization import save_roc_curve

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        image = self.X[index]
        label = int(self.y[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def get_train_transform(use_augmentation=True, augmentation_policy="standard"):
    steps = [transforms.ToPILImage(), transforms.Resize((224, 224))]
    if use_augmentation:
        steps.extend([transforms.RandomHorizontalFlip(p=LC25000_HORIZONTAL_FLIP_P), transforms.RandomRotation(LC25000_ROTATION_DEGREES)])
        if augmentation_policy == "appearance_jitter":
            steps.append(transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02))
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_NORMALIZE_MEAN, std=IMAGENET_NORMALIZE_STD)])
    return transforms.Compose(steps)

def get_eval_transform():
    return transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_NORMALIZE_MEAN, std=IMAGENET_NORMALIZE_STD)])

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def compute_class_weights(y_train):
    class_counts = Counter(y_train)
    total_samples = sum(class_counts.values())
    num_classes = len(np.unique(y_train))
    weights = np.asarray([total_samples / class_counts[i] if i in class_counts and class_counts[i] > 0 else 0.0 for i in range(num_classes)], dtype=np.float32)
    return weights / weights.sum() if weights.sum() > 0 else weights

def get_selection_metric_name():
    return "macro_f1"

def get_unfreeze_epoch():
    return LC25000_UNFREEZE_EPOCH

def evaluate_on_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    mean_loss = total_loss / max(len(loader), 1)
    macro_f1 = precision_recall_fscore_support(np.asarray(all_labels), np.asarray(all_predictions), average="macro", zero_division=0)[2]
    return mean_loss, macro_f1

def set_transfer_stage(model, stage):
    if not hasattr(model, "model"):
        return
    backbone_family = getattr(model, "backbone_family", "resnet")
    for p in model.model.parameters():
        p.requires_grad = False
    if stage == "head":
        if hasattr(model.model, "classifier"):
            for p in model.model.classifier.parameters(): p.requires_grad = True
        if hasattr(model.model, "fc"):
            for p in model.model.fc.parameters(): p.requires_grad = True
    elif stage == "final_block":
        if backbone_family == "densenet":
            for name, p in model.model.features.named_parameters():
                if name.startswith("denseblock4") or name.startswith("norm5"):
                    p.requires_grad = True
            for p in model.model.classifier.parameters(): p.requires_grad = True
        elif backbone_family == "efficientnet":
            for block_index in [6, 7, 8]:
                if block_index < len(model.model.features):
                    for p in model.model.features[block_index].parameters(): p.requires_grad = True
            for p in model.model.classifier.parameters(): p.requires_grad = True
        else:
            if hasattr(model.model, "layer4"):
                for p in model.model.layer4.parameters(): p.requires_grad = True
            if hasattr(model.model, "fc"):
                for p in model.model.fc.parameters(): p.requires_grad = True

def make_optimizer(model, stage):
    if not hasattr(model, "model"):
        return torch.optim.Adam(model.parameters(), lr=1e-3)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable_parameters, lr=1e-3 if stage == "head" else 1e-4, weight_decay=1e-4)

def unfreeze_full_model_with_discriminative_lr(model):
    for p in model.model.parameters():
        p.requires_grad = True
    backbone_family = getattr(model, "backbone_family", "resnet")
    if backbone_family in ["densenet", "efficientnet"]:
        return torch.optim.AdamW([
            {"params": model.model.features.parameters(), "lr": 1e-5},
            {"params": (model.model.classifier.parameters() if hasattr(model.model, "classifier") else model.model.fc.parameters()), "lr": 1e-4},
        ], weight_decay=1e-4)
    return torch.optim.AdamW([
        {"params": model.model.conv1.parameters(), "lr": 1e-6},
        {"params": model.model.bn1.parameters(), "lr": 1e-6},
        {"params": model.model.layer1.parameters(), "lr": 1e-6},
        {"params": model.model.layer2.parameters(), "lr": 1e-6},
        {"params": model.model.layer3.parameters(), "lr": 1e-5},
        {"params": model.model.layer4.parameters(), "lr": 5e-5},
        {"params": model.model.fc.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)

def save_prediction_outputs(model, dataset_name, y_true, y_pred, y_prob, confidences, class_names):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)
    report_dict = classification_report(y_true, y_pred, labels=list(range(num_classes)), target_names=class_names, zero_division=0, output_dict=True)
    report_path = RESULTS_DIR / f"{dataset_name}_{model.model_name}_classification_report.csv"
    pd.DataFrame(report_dict).transpose().to_csv(report_path)
    predictions_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "confidence": np.round(confidences, 6), "true_label": [class_names[int(l)] for l in y_true], "pred_label": [class_names[int(l)] for l in y_pred]})
    for class_index, class_name in enumerate(class_names):
        predictions_df[f"prob_{class_name}"] = np.round(y_prob[:, class_index], 6)
    predictions_path = RESULTS_DIR / f"{dataset_name}_{model.model_name}_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    misclassified_df = predictions_df[predictions_df["y_true"] != predictions_df["y_pred"]].copy()
    if not misclassified_df.empty:
        misclassified_df.sort_values(by="confidence", ascending=False).to_csv(RESULTS_DIR / f"{dataset_name}_{model.model_name}_misclassifications.csv", index=False)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    rows = []
    for t in range(cm.shape[0]):
        for p in range(cm.shape[1]):
            if t != p and int(cm[t, p]) > 0:
                rows.append({"true_class_index": t, "pred_class_index": p, "true_class": class_names[t], "pred_class": class_names[p], "count": int(cm[t, p])})
    if rows:
        pd.DataFrame(rows).sort_values(by="count", ascending=False).to_csv(RESULTS_DIR / f"{dataset_name}_{model.model_name}_top_confusions.csv", index=False)
    print(f"Classification report saved to: {report_path}")
    print(f"Predictions saved to: {predictions_path}")
    return cm

def train_cnn(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs=50, class_names=None, dataset_name="LC25000", use_augmentation=True, use_class_weights=True, augmentation_policy="standard"):
    torch.manual_seed(RANDOM_STATE)
    device = get_device()
    model = model.to(device)
    print(f"Training {dataset_name}_{model.model_name} on device: {device}")
    train_loader = DataLoader(ImageDataset(X_train, y_train, get_train_transform(use_augmentation, augmentation_policy)), batch_size=LC25000_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ImageDataset(X_val, y_val, get_eval_transform()), batch_size=LC25000_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ImageDataset(X_test, y_test, get_eval_transform()), batch_size=LC25000_BATCH_SIZE, shuffle=False)
    num_classes = len(np.unique(y_train))
    class_weights = compute_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device)) if use_class_weights else nn.CrossEntropyLoss()
    best_val_f1, best_model_state = 0.0, copy.deepcopy(model.state_dict())
    patience, patience_counter = 8, 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "epochs_ran": 0}
    if hasattr(model, "model") and getattr(model, "training_mode", None) == "head_only":
        stages = [("head", epochs)]
    elif hasattr(model, "model"):
        s1 = min(6, max(1, epochs // 3)); stages = [("head", s1), ("final_block", epochs - s1)]
    else:
        stages = [("full", epochs)]
    for stage_name, stage_epochs in stages:
        set_transfer_stage(model, stage_name)
        optimizer = make_optimizer(model, stage_name)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2) if stage_name == "final_block" else None
        for epoch in range(stage_epochs):
            if getattr(model, "training_mode", None) != "head_only" and stage_name == "final_block" and epoch == get_unfreeze_epoch() and hasattr(model, "model"):
                print("Unfreezing full pretrained model for fine-tuning...")
                optimizer = unfreeze_full_model_with_discriminative_lr(model)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
            model.train(); total_train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(); outputs = model(images); loss = criterion(outputs, labels); loss.backward(); optimizer.step()
                total_train_loss += loss.item()
            mean_train_loss = total_train_loss / max(len(train_loader), 1)
            val_loss, val_f1 = evaluate_on_loader(model, val_loader, criterion, device)
            history["train_loss"].append(mean_train_loss); history["val_loss"].append(val_loss); history["val_f1"].append(val_f1); history["epochs_ran"] += 1
            print(f"[{stage_name}] Epoch {epoch + 1}/{stage_epochs} | Train Loss: {mean_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1; best_model_state = copy.deepcopy(model.state_dict()); patience_counter = 0
            else:
                patience_counter += 1
            if scheduler is not None: scheduler.step(val_loss)
            if patience_counter >= patience:
                print("Early stopping triggered."); break
        if patience_counter >= patience: break
    model.load_state_dict(best_model_state)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{dataset_name}_{model.model_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved to: {model_path}")
    model.eval(); all_predictions=[]; all_labels=[]; all_confidences=[]; all_probabilities=[]
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            probabilities = torch.softmax(model(images), dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
            all_predictions.extend(predictions.cpu().numpy()); all_labels.extend(labels.cpu().numpy()); all_confidences.extend(confidences.cpu().numpy()); all_probabilities.extend(probabilities.cpu().numpy())
    y_true = np.asarray(all_labels).astype(int); y_pred = np.asarray(all_predictions).astype(int); y_prob = np.asarray(all_probabilities, dtype=float); confidences = np.asarray(all_confidences, dtype=float)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1 = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, y_prob[:, 1]) if num_classes == 2 else roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")
    if class_names is None: class_names = [str(i) for i in range(num_classes)]
    print("Per-class metrics:\n", classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, zero_division=0))
    save_roc_curve(y_true, y_prob, class_names, dataset_name, model.model_name)
    cm = save_prediction_outputs(model, dataset_name, y_true, y_pred, y_prob, confidences, class_names)
    return accuracy, precision, recall, weighted_f1, macro_f1, balanced_accuracy, roc_auc, cm, history
