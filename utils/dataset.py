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

        # Load mask
        mask_name = self.masks[index]
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = mask // 255

        # Apply transform
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0) # This will transform mask dimmensions from [height, width] --> [1, height, width]

            image = image.float() / 255.0
            mask = mask.float()


        # Calculate bounding box from augmented mask
        pos = torch.where(mask[0] == 1)

        if len(pos[0]) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        else:
            y_min = torch.min(pos[0])
            y_max = torch.max(pos[0])
            x_min = torch.min(pos[1])
            x_max = torch.max(pos[1])

            # Ensure non-zero width/height
            if y_max == y_min:
                y_max += 1
            if x_max == x_min:
                x_max += 1

            boxes = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
            labels = torch.ones((1,), dtype=torch.int64)  # Class = 1 for coral

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": mask,
            }

        return image, target, mask_name