import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, ".")
from src.model import WaferResNet
from src.explainability import GradCAM, overlay_heatmap_on_wafer

print("="*60)
print("GENERATING EXPLAINABLE AI (Grad-CAM) VISUAL DEMOS")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Data & Model
data = np.load("data/processed_wafers.npz")
X_test = data["X_test"]
y_test = data["y_test"]
classes = list(data["classes"])

model = WaferResNet(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("models/wafer_resnet_best.pth"))
model.eval()

gradcam = GradCAM(model)

# 2. Pick representative samples of interesting defects
sample_classes = ["Scratch", "Center", "Donut", "Edge-Ring", "Loc"]
fig, axes = plt.subplots(len(sample_classes), 4, figsize=(16, 4 * len(sample_classes)))

for row_idx, target_name in enumerate(sample_classes):
    target_id = classes.index(target_name)
    # Find test samples matching this class where model predicted correctly
    matching_indices = np.where(y_test == target_id)[0]
    
    # Pick a sample
    sample_idx = None
    for idx in matching_indices:
        t = torch.from_numpy(X_test[idx:idx+1]).float().to(device)
        with torch.no_grad():
            pred = torch.argmax(model(t), dim=1).item()
        if pred == target_id:
            sample_idx = idx
            break
            
    if sample_idx is None:
        sample_idx = matching_indices[0]
        
    input_tensor = torch.from_numpy(X_test[sample_idx:sample_idx+1]).float().to(device)
    cam, pred_class = gradcam.generate_cam(input_tensor, target_class=target_id)
    
    # Reconstruct 2D wafer map from one-hot tensor (channel 0: bg, 1: good, 2: defect)
    t_np = X_test[sample_idx] # (3, 56, 56)
    wmap = np.zeros((56, 56), dtype=np.uint8)
    wmap[t_np[1] > 0.5] = 1
    wmap[t_np[2] > 0.5] = 2
    
    overlay = overlay_heatmap_on_wafer(wmap, cam, alpha=0.6)
    
    # Col 1: Original Raw Wafer Map
    axes[row_idx, 0].imshow(wmap, cmap="viridis")
    axes[row_idx, 0].set_title(f"Target Class: {target_name}\n(Ground Truth)", fontsize=11, fontweight="bold")
    axes[row_idx, 0].axis("off")
    
    # Col 2: Binary Defect Dies
    defect_mask = (wmap == 2).astype(np.uint8)
    axes[row_idx, 1].imshow(defect_mask, cmap="gray")
    axes[row_idx, 1].set_title(f"Defect Die Mask\n(Failed Dies = White)", fontsize=11, fontweight="bold")
    axes[row_idx, 1].axis("off")
    
    # Col 3: Grad-CAM Activation Heatmap
    axes[row_idx, 2].imshow(cam, cmap="jet")
    axes[row_idx, 2].set_title(f"Grad-CAM Heatmap\n(Red = High Attention)", fontsize=11, fontweight="bold")
    axes[row_idx, 2].axis("off")
    
    # Col 4: Overlay on Wafer
    axes[row_idx, 3].imshow(overlay)
    axes[row_idx, 3].set_title(f"AI Decision Overlay\nPred: {classes[pred_class]}", fontsize=11, fontweight="bold", color="darkgreen")
    axes[row_idx, 3].axis("off")

plt.tight_layout()
plt.savefig("assets/gradcam_visualizations.png", dpi=150)
print("Saved assets/gradcam_visualizations.png successfully!")
print("Grad-CAM generation complete.")
