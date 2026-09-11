import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import sys
sys.path.insert(0, ".")
from src.features import extract_wafer_features

print("="*60)
print("PART 1: Hand-crafted Features + Random Forest Baseline")
print("="*60)

# 1. Load Dataset
print("\n[Step 1/5] Loading WM-811K raw dataset...")
t0 = time.time()
df = pd.read_pickle("data/LSWMD.pkl")

def parse_label(val):
    if isinstance(val, np.ndarray) and val.size > 0:
        return val[0][0]
    return "Unlabeled"

df["failureLabel"] = df["failureType"].apply(parse_label)
labeled_df = df[df["failureLabel"] != "Unlabeled"].copy()
print(f"Loaded {len(labeled_df):,} labeled wafers in {time.time()-t0:.2f}s")

# 2. Balanced Sampling
print("\n[Step 2/5] Sampling dataset for training...")
# Keep all defect wafers (25,519) and sample 10,000 'none' wafers
defect_df = labeled_df[labeled_df["failureLabel"] != "none"]
none_df = labeled_df[labeled_df["failureLabel"] == "none"].sample(n=10000, random_state=42)
sample_df = pd.concat([defect_df, none_df]).reset_index(drop=True)

print(f"Total samples for experiment: {len(sample_df):,}")
print("Class breakdown:")
print(sample_df["failureLabel"].value_counts())

# 3. Extract 29 Hand-crafted Features
print("\n[Step 3/5] Extracting 29 hand-crafted features for all wafers...")
t_feat_start = time.time()

feature_list = []
for idx, row in sample_df.iterrows():
    feats = extract_wafer_features(row["waferMap"])
    feature_list.append(feats)

X = pd.DataFrame(feature_list)
y = sample_df["failureLabel"].values

t_feat_total = time.time() - t_feat_start
avg_feat_time_ms = (t_feat_total / len(sample_df)) * 1000
print(f"Extracted {len(X.columns)} features from {len(X):,} wafers in {t_feat_total:.2f}s")
print(f"Average feature extraction time: {avg_feat_time_ms:.3f} ms/wafer")

# Save features for reuse
X_with_label = X.copy()
X_with_label["target"] = y
X_with_label.to_csv("data/handcrafted_features.csv", index=False)
print("Saved features to data/handcrafted_features.csv")

# 4. Stratified Train/Test Split
print("\n[Step 4/5] Splitting into 80% Train / 20% Test (Stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}")

# 5. Train Random Forest
print("\n[Step 5/5] Training Random Forest Classifier (100 estimators)...")
t_train_start = time.time()
rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
train_time = time.time() - t_train_start
print(f"Model trained in {train_time:.2f}s")

# Inference benchmark on Test Set
t_infer_start = time.time()
y_pred = rf.predict(X_test)
t_infer_total = time.time() - t_infer_start
infer_latency_ms = (t_infer_total / len(X_test)) * 1000
total_latency_ms = avg_feat_time_ms + infer_latency_ms

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\n" + "="*50)
print(f"RANDOM FOREST BASELINE RESULTS:")
print(f"  Overall Accuracy:            {acc*100:.2f}%")
print(f"  Macro F1-Score:              {macro_f1*100:.2f}%")
print(f"  Feature Extraction Latency:  {avg_feat_time_ms:.3f} ms/wafer")
print(f"  RF Inference Latency:        {infer_latency_ms:.3f} ms/wafer")
print(f"  Total Latency (End-to-End):  {total_latency_ms:.3f} ms/wafer")
print("="*50)

print("\nDetailed Classification Report:")
classes = np.unique(y)
report = classification_report(y_test, y_pred, target_names=classes, digits=4)
print(report)

# 6. Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, labels=classes, cmap="Blues", ax=ax, colorbar=True
)
plt.title("Confusion Matrix - Hand-crafted Features + Random Forest", fontsize=13, fontweight="bold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("notebooks/rf_confusion_matrix.png", dpi=150)
print("Saved notebooks/rf_confusion_matrix.png")

# 7. Plot Top 15 Feature Importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:15]
top_feats = [X.columns[i] for i in indices]
top_scores = importances[indices]

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_feats)), top_scores[::-1], color="teal")
plt.yticks(range(len(top_feats)), top_feats[::-1])
plt.xlabel("Gini Feature Importance")
plt.title("Top 15 Most Important Hand-crafted Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("notebooks/rf_feature_importance.png", dpi=150)
print("Saved notebooks/rf_feature_importance.png")

# 8. Save Metrics to JSON
metrics = {
    "model": "Hand-crafted Features + Random Forest",
    "accuracy": float(acc),
    "macro_f1": float(macro_f1),
    "feature_latency_ms": float(avg_feat_time_ms),
    "inference_latency_ms": float(infer_latency_ms),
    "total_latency_ms": float(total_latency_ms),
    "train_time_sec": float(train_time),
    "num_features": int(len(X.columns)),
    "classification_report": classification_report(y_test, y_pred, target_names=classes, output_dict=True)
}
with open("models/baseline_rf_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Saved metrics to models/baseline_rf_metrics.json")
print("\nAll Done! Baseline experiment completed successfully.")
