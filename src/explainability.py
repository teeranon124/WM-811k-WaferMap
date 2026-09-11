import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

def overlay_heatmap_on_wafer(wafer_map, cam, threshold=0.30, alpha=0.55):
    """
    Overlays AI Attention softly onto the ORIGINAL wafer map (viridis palette):
    - Base: Identical exact colors as Ground Truth (channel 0: purple, 1: teal, 2: yellow)
    - Attention: Only where CAM > threshold, tint softly with a warm reddish highlight
      leaving low-attention areas completely in their authentic original colors.
    """
    # 1. Render the EXACT original wafer map using matplotlib's viridis colormap
    viridis = plt.get_cmap('viridis')
    # wafer_map has values {0, 1, 2} -> normalize to [0, 1] for colormap
    base_rgba = viridis(wafer_map / 2.0)
    base_rgb = (base_rgba[:, :, :3] * 255).astype(np.uint8)
    
    # 2. Highlight only high-attention region
    overlay = base_rgb.copy()
    
    # Mask where AI is paying attention (> threshold) and within the wafer disk
    attention_mask = (cam >= threshold) & (wafer_map > 0)
    
    if np.any(attention_mask):
        # We use a soft crimson/red highlight for attention
        highlight_color = np.array([255, 30, 30], dtype=np.float32)
        
        # Normalized intensity of attention from 0 to 1
        intensity = (cam[attention_mask] - threshold) / (1.0 - threshold + 1e-6)
        intensity = np.clip(intensity, 0, 1)[:, np.newaxis]
        
        # Blend base with highlight based on attention intensity & alpha
        current_pixels = overlay[attention_mask].astype(np.float32)
        blended = (1.0 - alpha * intensity) * current_pixels + (alpha * intensity) * highlight_color
        overlay[attention_mask] = np.clip(blended, 0, 255).astype(np.uint8)
        
    return overlay
