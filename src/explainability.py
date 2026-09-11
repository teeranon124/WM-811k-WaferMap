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

def overlay_heatmap_on_wafer(wafer_map, cam, alpha=0.55):
    """
    Overlays colored Grad-CAM heatmap directly onto the wafer map.
    """
    # Create RGB representation of wafer map:
    # 0: Dark purple / black background
    # 1: Dark teal (good die)
    # 2: Bright yellow (defect die)
    h, w = wafer_map.shape
    base_img = np.zeros((h, w, 3), dtype=np.uint8)
    base_img[wafer_map == 0] = [20, 10, 30]      # Background
    base_img[wafer_map == 1] = [30, 110, 110]    # Good die
    base_img[wafer_map == 2] = [240, 230, 30]    # Defect die
    
    # Heatmap with COLORMAP_JET
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend only within the valid wafer region (wafer_map > 0)
    blended = base_img.copy()
    valid_mask = (wafer_map > 0)
    blended[valid_mask] = np.clip(
        (1 - alpha) * base_img[valid_mask] + alpha * heatmap_colored[valid_mask],
        0, 255
    ).astype(np.uint8)
    
    return blended
