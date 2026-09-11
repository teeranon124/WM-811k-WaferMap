import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, f1_score

sys.path.insert(0, ".")
from src.dataset import WaferDataset
from src.model import WaferResNet

print("="*60)
print("TRAINING ADVANCED MODEL: WaferResNet (Residual + SE-Attention)")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# 1. Load Data
data = np.load("data/processed_wafers.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]
classes = data["classes"]

train_dataset = WaferDataset(X_train, y_train)
val_dataset = WaferDataset(X_val, y_val)
test_dataset = WaferDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True)

# 2. Build Model
model = WaferResNet(num_classes=len(classes)).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"WaferResNet Trainable Parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

# 3. Training Loop (12 Epochs)
epochs = 12
best_val_acc = 0.0
t_train_start = time.time()

for epoch in range(1, epochs + 1):
    t_ep = time.time()
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * len(targets)
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == targets).sum().item()
        total_train += len(targets)
        
    scheduler.step()
    train_loss = running_loss / total_train
    train_acc = correct_train / total_train
    
    # Validation
    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * len(targets)
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == targets).sum().item()
            total_val += len(targets)
            
    val_loss = val_loss / total_val
    val_acc = correct_val / total_val
    ep_time = time.time() - t_ep
    
    print(f"Epoch [{epoch:2d}/{epochs:2d}] ({ep_time:4.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/wafer_resnet_best.pth")

total_train_time = time.time() - t_train_start
print(f"\nTraining completed in {total_train_time:.2f}s! Best Val Acc: {best_val_acc*100:.2f}%")

# 4. Evaluation on Test Set (7,104 wafers)
model.load_state_dict(torch.load("models/wafer_resnet_best.pth"))
model.eval()

# Benchmark Latency
sample_tensor = torch.from_numpy(X_test[:1]).float().to(device)
for _ in range(10): _ = model(sample_tensor)
torch.cuda.synchronize()
t_bench_start = time.time()
n_bench = 1000
with torch.no_grad():
    for _ in range(n_bench): _ = model(sample_tensor)
torch.cuda.synchronize()
resnet_latency_ms = ((time.time() - t_bench_start) / n_bench) * 1000
print(f"WaferResNet GPU Latency (RTX 4050): {resnet_latency_ms:.3f} ms/wafer")

# Collect Predictions
all_preds, all_targets = [], []
with torch.no_grad():
    for imgs, targets in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

test_acc = accuracy_score(all_targets, all_preds)
test_f1 = f1_score(all_targets, all_preds, average="macro")

print("\n" + "="*55)
print("WAFER-RESNET TEST SET RESULTS:")
print(f"  Overall Accuracy:    {test_acc*100:.2f}%")
print(f"  Macro F1-Score:      {test_f1*100:.2f}%")
print(f"  GPU Latency:         {resnet_latency_ms:.3f} ms/wafer")
print("="*55)

report = classification_report(all_targets, all_preds, target_names=classes, digits=4)
print("\nDetailed Classification Report:")
print(report)

# Save Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay.from_predictions(
    all_targets, all_preds, display_labels=classes, cmap="Blues", ax=ax, colorbar=True
)
plt.title(f"Confusion Matrix - WaferResNet (Accuracy: {test_acc*100:.2f}%)", fontsize=12, fontweight="bold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("notebooks/resnet_confusion_matrix.png", dpi=150)
print("Saved notebooks/resnet_confusion_matrix.png")

# Save Metrics
metrics = {
    "model": "WaferResNet (Residual + SE-Attention)",
    "accuracy": float(test_acc),
    "macro_f1": float(test_f1),
    "gpu_latency_ms": float(resnet_latency_ms),
    "train_time_sec": float(total_train_time),
    "num_parameters": int(total_params),
    "classification_report": classification_report(all_targets, all_preds, target_names=classes, output_dict=True)
}
with open("models/resnet_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Saved metrics to models/resnet_metrics.json")
print("\nAll Done! WaferResNet training complete.")
