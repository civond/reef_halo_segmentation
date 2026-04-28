from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class ImageDataset(Dataset):
    def __init__(self, df, train=True, transform = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def visualize_components(self, image, mask_np, cc, num_labels):
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        plt.close()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[0].axis("off")

        # Binary mask
        axes[1].imshow(mask_np, cmap="gray")
        axes[1].set_title(f"Binary Mask")
        axes[1].axis("off")

        # Connected components (each label gets a different color)
        colored_cc = np.zeros((*cc.shape, 3), dtype=np.float32)
        colors = cm.tab20(np.linspace(0, 1, max(num_labels, 2)))
        for i in range(1, num_labels):  # skip background (0)
            colored_cc[cc == i] = colors[i % len(colors)][:3]

        axes[2].imshow(colored_cc)
        axes[2].set_title(f"Connected Components: {num_labels - 1} instances")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
        plt.close() 
        print(f"Components found: {num_labels - 1}")
            
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
            #mask = mask.unsqueeze(0) # This will transform mask dimmensions from [height, width] --> [1, height, width]


        if isinstance(mask, torch.Tensor):
            mask_np = mask.numpy().astype(np.uint8)
        else:
            mask_np = mask.astype(np.uint8)

        H, W = mask_np.shape
        num_labels, cc = cv2.connectedComponents(mask_np)
        #self.visualize_components(image, mask_np, cc, num_labels)
        
        # Generate masks for each object
        boxes = []
        labels = []
        masks_out = []
        for i in range(1, num_labels):  # skip background
            component = (cc == i).astype(np.uint8)

            ys, xs = np.where(component)
            if len(xs) == 0:
                continue

            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            if x_max == x_min:
                x_max += 1
            if y_max == y_min:
                y_max += 1

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(1)
            masks_out.append(component)

        # Convert everything safely
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks_out = torch.zeros((0, H, W), dtype=torch.uint8)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks_out = torch.tensor(np.array(masks_out), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks_out,
        }

        return image, target, mask_name