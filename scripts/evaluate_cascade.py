import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, f1_score

sys.path.insert(0, ".")
from src.features import extract_wafer_features
from src.model import WaferCNN

print("="*60)
print("PHASE 3: Hybrid Cascaded System (RF Gatekeeper + CNN Expert)")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# 1. Load Processed Arrays & Raw Test Wafers
print("\n[Step 1/4] Loading Data...")
data = np.load("data/processed_wafers.npz")
X_test_tensor = data["X_test"]
y_test = data["y_test"]
classes = list(data["classes"])
label2id = {c: i for i, c in enumerate(classes)}
id2label = {i: c for i, c in enumerate(classes)}

# Load Handcrafted Features for RF
feat_df = pd.read_csv("data/handcrafted_features.csv")
X_feat = feat_df.drop(columns=["target"])
y_feat = feat_df["target"].map(label2id).values

# 2. Load Pretrained Models
print("\n[Step 2/4] Loading Pretrained Models...")
# Retrain RF on the training portion quickly (1.1s)
from sklearn.model_selection import train_test_split
indices = np.arange(len(y_feat))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_feat)

rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
rf.fit(X_feat.iloc[train_idx], y_feat[train_idx])
print("Random Forest (Stage 1) loaded successfully!")

# Load WaferCNN
cnn = WaferCNN(num_classes=len(classes)).to(device)
cnn.load_state_dict(torch.load("models/wafer_cnn_best.pth"))
cnn.eval()
print("WaferCNN (Stage 2) loaded successfully!")

# 3. Benchmark Cascade Inference on Test Set (7,104 wafers)
print("\n[Step 3/4] Running Hybrid Cascade Inference on 7,104 Test Wafers...")
CONF_THRESHOLD = 0.85
SCRATCH_ID = label2id["Scratch"]

X_test_feat = X_feat.iloc[test_idx].reset_index(drop=True)
y_test_clean = y_feat[test_idx]

# Stage 1: Random Forest Predict Probabilities
t_start = time.time()
rf_probs = rf.predict_proba(X_test_feat)
rf_preds = np.argmax(rf_probs, axis=1)
rf_confs = np.max(rf_probs, axis=1)

cascade_preds = []
routed_to_cnn_count = 0
rf_handled_count = 0

t_cnn_total = 0.0

for i in range(len(y_test_clean)):
    pred_rf = rf_preds[i]
    conf_rf = rf_confs[i]
    
    # Decision Gate:
    # If RF confidence is high (>= 0.85) AND not Scratch: Accept RF!
    if conf_rf >= CONF_THRESHOLD and pred_rf != SCRATCH_ID:
        cascade_preds.append(pred_rf)
        rf_handled_count += 1
    else:
        # Route to CNN (Stage 2)
        routed_to_cnn_count += 1
        t_c0 = time.time()
        with torch.no_grad():
            img_tensor = torch.from_numpy(X_test_tensor[i:i+1]).float().to(device)
            out = cnn(img_tensor)
            pred_cnn = torch.argmax(out, dim=1).item()
        t_cnn_total += (time.time() - t_c0)
        cascade_preds.append(pred_cnn)

total_time = time.time() - t_start
avg_latency_ms = (total_time / len(y_test_clean)) * 1000

cascade_preds = np.array(cascade_preds)

# 4. Evaluation
acc = accuracy_score(y_test_clean, cascade_preds)
macro_f1 = f1_score(y_test_clean, cascade_preds, average="macro")

print("\n" + "="*55)
print("HYBRID CASCADED SYSTEM BENCHMARK:")
print(f"  Overall Accuracy:           {acc*100:.2f}%")
print(f"  Macro F1-Score:             {macro_f1*100:.2f}%")
print(f"  Average End-to-End Latency: {avg_latency_ms:.3f} ms/wafer")
print(f"  Wafers resolved by RF (CPU):{rf_handled_count:,} ({rf_handled_count/len(y_test_clean)*100:.1f}%)")
print(f"  Wafers escalated to CNN:    {routed_to_cnn_count:,} ({routed_to_cnn_count/len(y_test_clean)*100:.1f}%)")
print("="*55)

report = classification_report(y_test_clean, cascade_preds, target_names=classes, digits=4)
print("\nDetailed Classification Report:")
print(report)

# 5. Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay.from_predictions(
    y_test_clean, cascade_preds, display_labels=classes, cmap="Purples", ax=ax, colorbar=True
)
plt.title(f"Confusion Matrix - Hybrid Cascade (RF Gatekeeper + CNN Expert)\nAccuracy: {acc*100:.2f}% | Latency: {avg_latency_ms:.3f} ms", fontsize=12, fontweight="bold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("notebooks/cascade_confusion_matrix.png", dpi=150)
print("Saved notebooks/cascade_confusion_matrix.png")

# Save Metrics
metrics = {
    "model": "Hybrid Cascade (RF + CNN)",
    "accuracy": float(acc),
    "macro_f1": float(macro_f1),
    "average_latency_ms": float(avg_latency_ms),
    "rf_handled_percent": float(rf_handled_count/len(y_test_clean)*100),
    "cnn_escalated_percent": float(routed_to_cnn_count/len(y_test_clean)*100),
    "classification_report": classification_report(y_test_clean, cascade_preds, target_names=classes, output_dict=True)
}
with open("models/cascade_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Saved metrics to models/cascade_metrics.json")
print("\nAll Done! Cascade evaluation complete.")
