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
from src.model import WaferCNN

print("="*60)
print("PHASE 2: Train WaferCNN on NVIDIA RTX 4050 GPU")
print("="*60)

# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# 1. Load Processed Arrays
print("\n[Step 1/5] Loading preprocessed data from data/processed_wafers.npz...")
data = np.load("data/processed_wafers.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]
classes = data["classes"]

print(f"Train samples: {len(X_train):,}")
print(f"Val samples:   {len(X_val):,}")
print(f"Test samples:  {len(X_test):,}")
print(f"Classes ({len(classes)}): {list(classes)}")

# 2. PyTorch DataLoaders
train_dataset = WaferDataset(X_train, y_train)
val_dataset = WaferDataset(X_val, y_val)
test_dataset = WaferDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True)

# 3. Model, Loss, Optimizer
model = WaferCNN(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model created. Trainable Parameters: {total_params:,}")

# 4. Training Loop (10 Epochs)
print("\n[Step 2/5] Training for 10 Epochs on RTX 4050...")
epochs = 10
best_val_acc = 0.0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

t_train_start = time.time()

for epoch in range(1, epochs + 1):
    t_epoch_start = time.time()
    
    # Train
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
    epoch_time = time.time() - t_epoch_start
    
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    
    print(f"Epoch [{epoch:2d}/{epochs:2d}] ({epoch_time:4.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.2f}%")
    
    # Save Best Model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/wafer_cnn_best.pth")

total_train_time = time.time() - t_train_start
print(f"\nTraining completed in {total_train_time:.2f}s! Best Val Acc: {best_val_acc*100:.2f}%")

# 5. Evaluate on Test Set (Unseen 7,104 wafers)
print("\n[Step 3/5] Evaluating on Test Set...")
model.load_state_dict(torch.load("models/wafer_cnn_best.pth"))
model.eval()

# Benchmark Latency per wafer on GPU
warmup = 10
sample_tensor = torch.from_numpy(X_test[:1]).float().to(device)
for _ in range(warmup):
    _ = model(sample_tensor)

torch.cuda.synchronize()
t_bench_start = time.time()
n_bench = 1000
with torch.no_grad():
    for _ in range(n_bench):
        _ = model(sample_tensor)
torch.cuda.synchronize()
gpu_latency_ms = ((time.time() - t_bench_start) / n_bench) * 1000
print(f"CNN GPU Latency (RTX 4050): {gpu_latency_ms:.3f} ms/wafer")

# Collect all test predictions
all_preds = []
all_targets = []
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

print("\n" + "="*50)
print(f"WAFER-CNN TEST SET RESULTS:")
print(f"  Overall Accuracy:    {test_acc*100:.2f}%")
print(f"  Macro F1-Score:      {test_f1*100:.2f}%")
print(f"  GPU Inference Time:  {gpu_latency_ms:.3f} ms/wafer")
print("="*50)

report = classification_report(all_targets, all_preds, target_names=classes, digits=4)
print("\nDetailed Classification Report:")
print(report)

# 6. Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay.from_predictions(
    all_targets, all_preds, display_labels=classes, cmap="Greens", ax=ax, colorbar=True
)
plt.title("Confusion Matrix - WaferCNN (PyTorch RTX 4050)", fontsize=13, fontweight="bold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("notebooks/cnn_confusion_matrix.png", dpi=150)
print("Saved notebooks/cnn_confusion_matrix.png")

# 7. Plot Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", color="blue")
ax1.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", color="red")
ax1.set_title("Loss Curves", fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("CrossEntropy Loss")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(range(1, epochs + 1), [a*100 for a in history["train_acc"]], label="Train Acc", color="blue")
ax2.plot(range(1, epochs + 1), [a*100 for a in history["val_acc"]], label="Val Acc", color="green")
ax2.set_title("Accuracy Curves", fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("notebooks/cnn_training_curves.png", dpi=150)
print("Saved notebooks/cnn_training_curves.png")

# 8. Save Metrics
metrics = {
    "model": "WaferCNN (3-Conv + GAP)",
    "accuracy": float(test_acc),
    "macro_f1": float(test_f1),
    "gpu_latency_ms": float(gpu_latency_ms),
    "train_time_sec": float(total_train_time),
    "num_parameters": int(total_params),
    "classification_report": classification_report(all_targets, all_preds, target_names=classes, output_dict=True)
}
with open("models/cnn_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Saved metrics to models/cnn_metrics.json")
print("\nAll Done! CNN training and evaluation completed successfully.")
