import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def resize_wafer(wafer_map, target_size=(56, 56)):
    """
    Resizes a 2D wafer map to target_size using cv2.INTER_NEAREST
    to strictly preserve discrete values: 0 (background), 1 (pass), 2 (fail).
    """
    return cv2.resize(wafer_map.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)

def wafer_to_tensor(wafer_map):
    """
    Converts a 2D wafer map (56, 56) with values {0, 1, 2} into a 3-channel one-hot tensor (3, 56, 56).
      Channel 0: Background mask (wafer_map == 0)
      Channel 1: Normal die mask (wafer_map == 1)
      Channel 2: Defect die mask  (wafer_map == 2)
    """
    c0 = (wafer_map == 0).astype(np.float32)
    c1 = (wafer_map == 1).astype(np.float32)
    c2 = (wafer_map == 2).astype(np.float32)
    tensor = np.stack([c0, c1, c2], axis=0)
    return tensor

def augment_wafer(wafer_map, rot_k=0, flip_h=False, flip_v=False):
    """
    Applies circular geometric transformations:
      - rot_k: 0 (0 deg), 1 (90 deg), 2 (180 deg), 3 (270 deg)
      - flip_h: Horizontal flip
      - flip_v: Vertical flip
    """
    img = wafer_map.copy()
    if rot_k > 0:
        img = np.rot90(img, rot_k)
    if flip_h:
        img = np.fliplr(img)
    if flip_v:
        img = np.flipud(img)
    return img

class WaferDataset(Dataset):
    """
    PyTorch Dataset for 3-channel Wafer Maps.
    """
    def __init__(self, images, labels):
        # images: (N, 3, 56, 56) float32
        # labels: (N,) int64
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
