from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, df, train=True, transform = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Index df by row
        row = self.df.iloc[index]
        img_path = row["img_path"]
        mask_path = row["mask_path"]

        # Load image data
        image = np.array(Image.open(img_path).convert("RGB"))
        mask_name = row["filename"]
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = mask // 255

        # Apply transform
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0) # This will transform mask dimmensions from [height, width] --> [1, height, width]


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

        # Create target dict
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": mask,
            }

        return image, target, mask_name