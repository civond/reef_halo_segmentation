import torch
import albumentations as A
import cv2
import numpy as np
import toml
import sys
import segmentation_models_pytorch as smp
import os
import tqdm
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Custom imports
#from utils.load_tif import load_tif
#from utils.tile_img import tile_img
from utils.get_loader import get_loader
from utils.create_transforms import create_transforms


class Inference:
    def __init__(self, config_path: str):
        self.config = toml.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        # Paths
        paths = self.config["Paths"]
        self.model_pth = paths["model_pth"]
        self.img_pth = paths["img_pth"]

        # Assert path
        if not self.img_pth.lower().endswith((".tif", ".png")):
            print(f"Error: Expected .tif or .png file, got: {self.img_pth}")
            sys.exit(1)

        # Ensure output dir is created
        self.output_dir = paths["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Parameters
        params = self.config["Parameters"]
        self.tile_size = params["tile_size"]
        self.mask_threshold = params["mask_threshold"]
        self.use_overlap = params.get("use_overlap", False)

        # Model
        self.model = smp.Unet(
            'efficientnet-b3', 
            encoder_weights='imagenet',
            classes=3,
            activation=None
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.model_pth, map_location=self.device))
        self.model.eval()

        # Images
        self.raster = None
        self.raster_mask = None
    
    def process_img(self):

        # ------ EXPECTS a 3-channel RGB .png ------
        if self.img_pth.endswith(".png"):
            img = cv2.imread(self.img_pth)

        # ------ EXPECTS a uint16, 4-channel .tif file ------
        # Would be ideal to convert to grayscale -> CLAHE in the future
        else:
            [img, profile] = load_tif(tif_pth=self.img_pth)
            img = np.transpose(img, (1, 2, 0))                              # transpose from (C, H, W) ---> (H, W, C)
            
            img = img[:, :, :3]                                             # Clip 4th channel
            img = img.astype(np.float32)
            max_val = np.max(img)
            img = img / max_val                                             # Normalize image between 0-1
            img = img * 255                                                 # Multiply by 255 to restore range
            img = img.astype(np.uint8)                                      # uint8 img
            

        # THESE TWO LINES ARE REQUIRED FOR THE ORIGINAL TEST IMAGE
        #img = img[:,:,:3]
        #img = img[..., ::-1]

        # Save img in self.raster
        self.raster = img
        [h_img, w_img] = img.shape[:2]

        """[tiles, coords, dims] = tile_img(
            img,
            tile_size=self.tile_size,
            overlap=self.use_overlap
        )"""

        # Initialize empty array
        #n_rows, n_cols = dims
        #full_h = n_rows * self.tile_size
        #full_w = n_cols * self.tile_size
        #classification_raster = np.zeros((full_h, full_w), dtype=np.uint8)

        # Create transform object
        transform = create_transforms(
            IMAGE_HEIGHT=256,
            IMAGE_WIDTH=256,
            train=False
        )

        loader = get_loader(data_dir="./UNet/data_test/", 
                                batch_size=64, 
                                transform=transform, 
                                num_workers=8, 
                                train=False, 
                                pin_memory=True)

        loop = tqdm.tqdm(loader, desc="Running inference")
        with torch.no_grad():
            for data, labels, filenames in loop:
                data = data.to(self.device)
                labels = labels.to(self.device)     # for BCE

                # Forward pass
                predictions = self.model(data)
                preds = torch.argmax(predictions, dim=1)  # shape [B, H, W]
                preds = preds.cpu().numpy().astype("uint8")

                #loss = loss_fn(predictions, labels)
                #total_loss += loss.item()
                #dice = dice_score(predictions, labels)
                #total_dice += dice.item()

                for i in range(preds.shape[0]):
                    mask = preds[i]             # shape [H, W], values 0/1/2

                    colors = np.array([
                        [0,   0,   0],     # class 0
                        [0, 255,   0],     # class 1
                        [255,   0, 0],     # class 2
                    ], dtype=np.uint8)

                    mask_rgb = colors[mask]


                    save_path = os.path.join("./UNet/output/", filenames[i])
                    os.makedirs("./Unet/output/", exist_ok=True)
                    print(mask_rgb.shape)
                    print(save_path)

                    cv2.imwrite(save_path, mask_rgb)

                #save_path = os.path.join(save_dir, filenames[i])
                #cv2.imwrite(save_path, mask)

        """print(f"\tPerforming inference on {n_rows*n_cols} tiles.")
        for i, tile in enumerate(tiles):

            # Skip over empty tiles
            if np.max(tile) != 0:
                # This loop does inference one at a time. 
                # Would be useful to put all images in a tensor and classify all directly in future iterations
                transformed = transform(image=tile)
                image_tensor = transformed["image"].to(self.device)
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
                
                # Skip no detections
                if len(masks) == 0:
                    continue

                # Combine all masks into a single tile mask
                tile_mask = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)

                for j in range(len(masks)):
                    mask = masks[j, 0].cpu().numpy()  # [H,W]
                    mask_bin = mask > self.mask_threshold
                    class_label = labels[j].item()
                    tile_mask[mask_bin] = class_label # assign label to masked pixels

                # Place tile_mask into the correct location in the full raster
                # coords contains (y, x) pixel coordinates directly
                y, x = coords[i]
                # Use maximum to combine overlapping regions (relevant when use_overlap=True)
                classification_raster[y:y+self.tile_size, x:x+self.tile_size] = np.maximum(
                    classification_raster[y:y+self.tile_size, x:x+self.tile_size],
                    tile_mask
                )

        # Normalize and write to png
        max_label = classification_raster.max()
        mask = (classification_raster / max_label * 255).astype(np.uint8)
        mask = mask[:h_img, :w_img]

        self.raster_mask = mask

        mask_output_pth = os.path.join(self.output_dir,'output_mask.png')
        print(f"\tWriting mask output to: {mask_output_pth}")
        cv2.imwrite(mask_output_pth, mask)"""


    def overlay_mask(self, alpha=0.3, thickness=-1):

        # Check raster and mask
        if self.raster is None or self.raster_mask is None:
            raise ValueError("Run process_tif() before calling overlay_mask().")
        
        assert self.raster.shape[:2] == self.raster_mask.shape[:2], \
            f"Raster dimensions {self.raster.shape[:2]} and mask dimensions {self.raster_mask.shape[:2]} do not match"

        print(f"Raster shape: {self.raster.shape[:2]}")
        print(f"Mask shape: {self.raster_mask.shape[:2]}")
        print("Overlaying mask...")
        binary_mask = (self.raster_mask > 0).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        overlay = self.raster.copy()

        # Draw contours in color
        cv2.drawContours(
            overlay,
            contours,
            -1,
            (0, 0, 255),
            thickness
        )

        # Fuse with original using alpha blend
        blended = cv2.addWeighted(self.raster, 1 - alpha, overlay, alpha, 0)

        blended_output_pth = os.path.join(self.output_dir,'output_blended.png')
        print(f"\tWriting overlayed image to {blended_output_pth}")
        cv2.imwrite(blended_output_pth, blended)