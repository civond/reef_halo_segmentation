import torch
from torchvision import transforms
from torch.optim import AdamW
import argparse
import pandas as pd
from datetime import datetime
import cv2
import numpy as np
import toml
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Custom functions
from utils.load_tif import load_tif
from utils.tile_img import tile_img
from utils.create_fig import create_fig
from utils.get_maskrcnn_model import get_maskrcnn_model
from utils.create_transforms import create_train_transforms, create_val_transforms
from utils.get_loader import get_train_loader, get_val_loader
from utils.train_fn import train_fn
from utils.val_fn import val_fn


class Trainer:
    def __init__(self, config_path: str):
        self.config = toml.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        # Hyperparameters
        hp = self.config["Hyperparameters"]
        self.learning_rate = hp["learning_rate"]
        self.batch_size = hp["batch_size"]
        self.num_epochs = hp["num_epochs"]
        self.num_workers = hp["num_workers"]
        self.image_height = hp["image_height"]
        self.image_width = hp["image_width"]
        self.pin_memory = hp["pin_memory"]
        self.train_flag = hp["train"]
        self.patience = hp["patience"]
        self.min_delta = hp["min_delta"]

        # Paths
        paths = self.config["Paths"]
        self.train_dir = paths["train_data_dir"]
        self.val_dir = paths["val_data_dir"]
        self.save_model_dir = "./model/"
        self.save_csv_dir = "./logs/"
        self.save_fig_dir = "./figures/"

        # Ensure directories exist
        os.makedirs(self.save_model_dir, exist_ok=True)
        os.makedirs(self.save_csv_dir, exist_ok=True)
        os.makedirs(self.save_fig_dir, exist_ok=True)

        # Model
        self.model = get_maskrcnn_model().to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )

        # Metric tracking
        self.train_loss_arr = []
        self.val_loss_arr = []
        self.train_dice_arr = []
        self.val_dice_arr = []
        self.best_val_loss = float("inf")
        self.best_model_weights = None
        self.patience_counter = 0

        # Transforms
        self.train_transform = create_train_transforms(
            self.image_height, 
            self.image_width
        )
        self.val_transform = create_val_transforms(
            self.image_height, 
            self.image_width)
        
        # Dataloaders
        self.train_loader = get_train_loader(
            data_dir=self.train_dir,
            batch_size=self.batch_size,
            transform=self.train_transform,
            num_workers=self.num_workers,
            train=True,
            pin_memory=self.pin_memory
        )

        self.val_loader = get_val_loader(
            data_dir=self.val_dir,
            batch_size=self.batch_size,
            transform=self.val_transform,
            num_workers=self.num_workers,
            train=False,
            pin_memory=self.pin_memory
        )

    def train_loop(self):
        for epoch in range(self.num_epochs):
            print(f"\nEpoch: {epoch}")


            # Train
            train_loss = train_fn(
                self.device, 
                self.train_loader, 
                self.model, 
                self.optimizer
            )

            print(f"Avg. Train Loss: {train_loss}")
            self.train_loss_arr.append(train_loss)


            # Validation
            val_loss = val_fn(
                self.device, 
                self.val_loader, 
                self.model
            
            )
            print(f"Avg. Val Loss: {val_loss}")
            self.val_loss_arr.append(val_loss)

            # Check Patience
            if val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_weights = self.model.state_dict()
                print("\tNew best loss")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print("Early stopping triggered. Exit train loop")
                    break

        self._save_outputs()

    def _save_outputs(self):
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
        save_model_pth = os.path.join(self.save_model_dir, f"{timestamp}.pth")
        save_csv_pth = os.path.join(self.save_csv_dir, f"{timestamp}.csv")
        save_fig_pth = os.path.join(self.save_fig_dir, f"{timestamp}.png")

        # Save model
        try:
            print(f"\tSaving model at: {save_model_pth}")
            torch.save(self.best_model_weights, save_model_pth)
        except Exception as e:
            print(f"Error during saving model: {e}")
        
        # Save CSV
        try:
            print(f"\tSaving csv at: {save_csv_pth}")
            df = pd.DataFrame({"train_loss": self.train_loss_arr, "val_loss": self.val_loss_arr})
            df.to_csv(save_csv_pth, index=False)
        except Exception as e:
            print(f"Error during saving csv: {e}")
        
        # Save figure
        try:
            print(f"\tSaving figure at: {save_fig_pth}")
            create_fig(df, save_fig_pth)
        except Exception as e:
            print(f"Error during saving figure: {e}")

class Inference:
    def __init__(self, config_path: str):
        self.config = toml.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        # Paths
        paths = self.config["Paths"]
        self.model_pth = paths["model_pth"]
        self.tif_pth = paths["tif_pth"]
        self.mask_output_pth = paths["mask_output_pth"]
        self.blended_output_pth = paths["blended_output_pth"]
        os.makedirs("./output/", exist_ok=True)
        
        # Parameters
        params = self.config["Parameters"]
        self.tile_size = params["tile_size"]
        self.score_threshold = params["score_threshold"]

        # Model
        self.model = get_maskrcnn_model().to(self.device)
        self.model.load_state_dict(torch.load(self.model_pth, map_location=self.device))
        self.model.eval()

        # Images
        self.raster = None
        self.raster_mask = None
    
    def process_tif(self):
        [img, profile] = load_tif(tif_pth=self.tif_pth)
        img = np.transpose(img, (1, 2, 0))  # transpose from (C, H, W) ---> (H, W, C)
        for i in range(4):
            img[:, :, i] = cv2.normalize(img[:, :, i], None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype('uint8')
        img = img[:,:,:3]
        img = img[..., ::-1]

        # Save img in self.raster
        self.raster = img

        [tiles, coords, dims] = tile_img(
            img,
            tile_size=self.tile_size
        )

        # Initialize empty array
        n_rows, n_cols = dims
        full_h = n_rows * self.tile_size
        full_w = n_cols * self.tile_size
        classification_raster = np.zeros((full_h, full_w), dtype=np.uint8)

        print(f"\tPerforming inference on {n_rows*n_cols} tiles.")
        for i, tile in enumerate(tiles):

            # Skip over empty tiles
            if np.max(tile) != 0:
                transform = transforms.ToTensor()
                image_tensor = transform(tile).to(self.device)
                images = [image_tensor]

                with torch.no_grad():
                    outputs = self.model(images)

                output = outputs[0]
                boxes = output['boxes']       # [N,4] bounding boxes
                labels = output['labels']     # [N] predicted class labels
                scores = output['scores']     # [N] confidence scores
                masks = output['masks']       # [N,1,H,W] predicted masks


                # Filter predictions by score threshold
                keep = scores > self.score_threshold

                boxes = boxes[keep]
                labels = labels[keep]
                masks = masks[keep]
                scores = scores[keep]


                """print(f"masks: {masks}")
                print(f"boxes: {boxes}")
                print(f"scores: {scores}")"""

                if len(masks) == 0:
                    continue  # no detections

                # Combine all masks into a single tile mask
                tile_mask = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)

                for j in range(len(masks)):
                    mask = masks[j, 0].cpu().numpy()  # [H,W]
                    mask_bin = mask > 0.5             # binary mask
                    class_label = labels[j].item()
                    tile_mask[mask_bin] = class_label  # assign label to masked pixels

                # Place tile_mask into the correct location in the full raster
                row_idx, col_idx = coords[i]
                y = row_idx * self.tile_size
                x = col_idx * self.tile_size
                classification_raster[y:y+self.tile_size, x:x+self.tile_size] = tile_mask

        # Normalize and write to png
        max_label = classification_raster.max()
        mask = (classification_raster / max_label * 255).astype(np.uint8)
        self.raster_mask = mask

        print(f"\tWriting mask output to: {self.mask_output_pth}")
        cv2.imwrite(self.mask_output_pth, mask)


    def overlay_mask(self, alpha=0.3, thickness=-1):

        # Check raster and mask
        if self.raster is None or self.raster_mask is None:
            raise ValueError("Run process_tif() before calling overlay_mask().")

        print("Overlaying mask...")
        binary_mask = (self.raster_mask > 0).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,    # only outer contours
            cv2.CHAIN_APPROX_SIMPLE
        )

        overlay = self.raster.copy()

        # Draw contours in color (blue or something noticeable)
        cv2.drawContours(
            overlay,
            contours,
            -1,                  # draw all contours
            (0, 0, 255),         # red contour (BGR)
            thickness
        )

        # Fuse with original using alpha blend
        blended = cv2.addWeighted(self.raster, 1 - alpha, overlay, alpha, 0)
        print(f"\tWriting overlayed image to {self.blended_output_pth}")
        cv2.imwrite(self.blended_output_pth, blended)

def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "crossval", "inference"],
        help="Please choose from: ['train', 'crossval', 'inference']"
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to TOML configuration file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode
    config_path = args.config_file

    # Train
    if mode.lower() == "train":
        print("Running training loop...")
        trainer = Trainer(config_path)
        trainer.train_loop()

    # Cross Validation
    if mode.lower() == "crossval":
        pass

    # Inference
    if mode.lower() == "inference":
        inf = Inference(config_path)
        inf.process_tif()
        inf.overlay_mask()


if __name__ == "__main__":
    main()