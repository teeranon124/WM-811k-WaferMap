import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
sys.path.insert(0, ".")
from src.dataset import resize_wafer, wafer_to_tensor, augment_wafer

print("="*60)
print("PHASE 2: Data Preprocessing & Circular Geometric Augmentation")
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

# 2. Balanced Sampling (Exact same seed & split as RF baseline)
print("\n[Step 2/5] Sampling dataset...")
defect_df = labeled_df[labeled_df["failureLabel"] != "none"]
none_df = labeled_df[labeled_df["failureLabel"] == "none"].sample(n=10000, random_state=42)
sample_df = pd.concat([defect_df, none_df]).reset_index(drop=True)

# Map string labels to integers
classes = sorted(sample_df["failureLabel"].unique().tolist())
label2id = {cls_name: i for i, cls_name in enumerate(classes)}
id2label = {i: cls_name for i, cls_name in enumerate(classes)}
print(f"Classes ({len(classes)}): {classes}")

raw_images = sample_df["waferMap"].values
raw_labels = np.array([label2id[name] for name in sample_df["failureLabel"].values])

# 3. Stratified Train / Test Split (Identical to RF test set)
print("\n[Step 3/5] Stratified Train (80%) / Test (20%) split...")
indices = np.arange(len(raw_labels))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=raw_labels
)

# Further split Train into Train (90%) and Val (10%)
train_idx, val_idx = train_test_split(
    train_idx, test_size=0.1, random_state=42, stratify=raw_labels[train_idx]
)
print(f"Train count: {len(train_idx):,}, Val count: {len(val_idx):,}, Test count: {len(test_idx):,}")

# 4. Resize and Convert Test & Val Sets (NO augmentation on Test/Val!)
print("\n[Step 4/5] Resizing Test and Validation sets (56x56, cv2.INTER_NEAREST)...")
X_val = np.stack([wafer_to_tensor(resize_wafer(raw_images[i])) for i in val_idx])
y_val = raw_labels[val_idx]

X_test = np.stack([wafer_to_tensor(resize_wafer(raw_images[i])) for i in test_idx])
y_test = raw_labels[test_idx]

# 5. Resize and Augment ONLY the Training Set
print("\n[Step 5/5] Performing Circular Geometric Augmentation on Training Set...")
# Define augmentation multipliers for each class in Train set
# Target: ~4,000 - 7,000 samples per class
train_images_list = []
train_labels_list = []

# Count training samples before augmentation
before_counts = {id2label[c]: 0 for c in range(len(classes))}
for i in train_idx:
    before_counts[id2label[raw_labels[i]]] += 1

print("Train counts BEFORE Augmentation:")
for cls_name, count in before_counts.items():
    print(f"  {cls_name:<12}: {count:,}")

for i in train_idx:
    img_resized = resize_wafer(raw_images[i])
    label_id = raw_labels[i]
    cls_name = id2label[label_id]
    
    # Base sample (always included)
    train_images_list.append(wafer_to_tensor(img_resized))
    train_labels_list.append(label_id)
    
    # Circular Augmentations based on rarity:
    if cls_name == "Near-full": # Extreme minority (~107 samples) -> 7 augmentations
        for rot in [1, 2, 3]:
            train_images_list.append(wafer_to_tensor(augment_wafer(img_resized, rot_k=rot)))
            train_labels_list.append(label_id)
        for flip in ["h", "v"]:
            train_images_list.append(wafer_to_tensor(augment_wafer(img_resized, flip_h=(flip=="h"), flip_v=(flip=="v"))))
            train_labels_list.append(label_id)
        for rot in [1, 2]:
            train_images_list.append(wafer_to_tensor(augment_wafer(img_resized, rot_k=rot, flip_h=True)))
            train_labels_list.append(label_id)
            
    elif cls_name in ["Donut", "Random"]: # Rare (~400-600 samples) -> 5 augmentations
        for rot in [1, 2, 3]:
            train_images_list.append(wafer_to_tensor(augment_wafer(img_resized, rot_k=rot)))
            train_labels_list.append(label_id)
        train_images_list.append(wafer_to_tensor(augment_wafer(img_resized, flip_h=True)))
        train_labels_list.append(label_id)
        train_images_list.append(wafer_to_tensor(augment_wafer(img_resized, flip_v=True)))
        train_labels_list.append(label_id)
        
    elif cls_name == "Scratch": # Scratch is rare (~859 samples) -> 5x multiplier
        for rot in [1, 2, 3]:
            train_images_list.append(wafer_to_tensor(augment_wafer(img_resized, rot_k=rot)))
            train_labels_list.append(label_id)
        train_images_list.append(wafer_to_tensor(augment_wafer(img_resized, flip_h=True)))
        train_labels_list.append(label_id)
        
    elif cls_name in ["Center", "Edge-Loc", "Loc"]: # Moderate (~2500-3700 samples) -> 2x multiplier (1 rotation)
        train_images_list.append(wafer_to_tensor(augment_wafer(img_resized, rot_k=2)))
        train_labels_list.append(label_id)

X_train = np.stack(train_images_list)
y_train = np.array(train_labels_list)

after_counts = {id2label[c]: 0 for c in range(len(classes))}
for l in y_train:
    after_counts[id2label[l]] += 1

print("\nTrain counts AFTER Circular Augmentation:")
for cls_name, count in after_counts.items():
    print(f"  {cls_name:<12}: {count:,} (Multiplier: x{count/before_counts[cls_name]:.1f})")

print(f"\nTotal Train samples after Augmentation: {len(X_train):,}")

# Save to compressed NPZ
np.savez_compressed(
    "data/processed_wafers.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    classes=np.array(classes)
)
print("Saved all arrays into data/processed_wafers.npz successfully!")

# Plot Before vs After Augmentation Bar Chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(classes))
width = 0.35

rects1 = ax.bar(x - width/2, [before_counts[c] for c in classes], width, label="Before Augmentation (Raw)", color="lightcoral")
rects2 = ax.bar(x + width/2, [after_counts[c] for c in classes], width, label="After Circular Augmentation", color="mediumseagreen")

ax.set_ylabel("Number of Samples")
ax.set_title("Training Set Class Balancing via Circular Geometric Augmentation", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=35)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.7)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig("notebooks/data_augmentation_comparison.png", dpi=150)
print("Saved notebooks/data_augmentation_comparison.png successfully!")
