import numpy as np
import cv2
import torch
import torch.nn.functional as F

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    for visual explanations of Wafer defect predictions.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generates Grad-CAM heatmap for input_tensor (1, 3, 56, 56).
        Returns:
          cam_map: (56, 56) float array normalized to [0, 1]
          predicted_class: int
        """
        input_tensor.requires_grad = True
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        # Target score for backpropagation
        score = output[0, target_class]
        score.backward()
        
        # Pull gradients and activations from registered hooks
        gradients = self.model.gradients.detach()   # (1, C, H, W)
        activations = self.model.activations.detach() # (1, C, H, W)
        
        # Global Average Pooling on gradients to get channel weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True) # (1, C, 1, 1)
        
        # Linear combination of weighted activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True) # (1, 1, H, W)
        
        # Apply ReLU to focus only on features that positively contribute to the target class
        cam = F.relu(cam)
        
        # Resize to input dimensions (56, 56)
        cam = F.interpolate(cam, size=(56, 56), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, target_class

def overlay_heatmap_on_wafer(wafer_map, cam, threshold=0.35):
    """
    Creates an industrial-grade, crisp explainability overlay:
    - Background: Black
    - Wafer Body (Good dies): Clean Dark Navy/Slate Gray
    - Defect Dies: Sharp White
    - AI Attention Focus: Glowing Crimson Red highlighting exactly where the model looked!
    """
    h, w = wafer_map.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 1. Base clean styling
    # Good dies = clean professional dark slate (40, 50, 65)
    overlay[wafer_map == 1] = [38, 48, 60]
    
    # 2. Defect dies = Crisp White (230, 230, 230)
    overlay[wafer_map == 2] = [225, 225, 225]
    
    # 3. AI Attention: Where CAM is high (> threshold) within wafer
    # We tint both good and defect dies with a vibrant heat highlight
    active_mask = (cam >= threshold) & (wafer_map > 0)
    
    if np.any(active_mask):
        # Apply Jet colormap only to the high-attention zone
        norm_cam = (cam - threshold) / (1.0 - threshold + 1e-6)
        norm_cam = np.clip(norm_cam, 0, 1)
        
        cam_uint8 = np.uint8(255 * norm_cam)
        heatmap_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_HOT)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # For defect dies inside the attention zone: Glowing Neon Gold/Red
        defect_and_active = (wafer_map == 2) & active_mask
        overlay[defect_and_active] = [255, 60, 20] # Intense Red-Orange
        
        # For wafer area in the attention zone: Soft Warm Glow
        wafer_and_active = (wafer_map == 1) & active_mask
        overlay[wafer_and_active] = np.clip(
            0.4 * overlay[wafer_and_active] + 0.6 * heatmap_colored[wafer_and_active], 0, 255
        ).astype(np.uint8)
        
    return overlay
