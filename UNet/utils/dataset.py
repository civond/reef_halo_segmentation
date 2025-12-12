from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch

class ImageDataset(Dataset):
    def __init__(self, data_dir, train=True, transform = None):
        """
        data_dir: ./data
        This function expects subfolders:
            ./data/img/
            ./data/mask/
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "img")
        self.mask_dir = os.path.join(data_dir, "mask")
        self.transform = transform
        
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".png")])
        self.masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(".png")])
        assert len(self.images) == len(self.masks), "Number of images and masks must match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load image
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load RGB mask
        mask_name = self.masks[index]
        mask_path = os.path.join(self.mask_dir, mask_name)
        #mask_rgb = np.array(Image.open(mask_path))              # Mask is in RGB form. Sand ring = green, coral = blue
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
        
        # Create BW mask
        mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
        mask[(mask_rgb == [0, 255, 0]).all(axis=2)] = 1
        mask[(mask_rgb == [0, 0, 255]).all(axis=2)] = 2
        #print("mask unique:", np.unique(mask))

        # Apply transform
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.long)
        else:
            mask = mask.long()  # already a tensor, just convert dtype
                
        return image, mask, mask_name