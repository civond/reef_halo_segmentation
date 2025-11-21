import torch
from torchvision import transforms
import cv2
import numpy as np
import toml
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Custom imports
from utils.load_tif import load_tif
from utils.tile_img import tile_img
from utils.get_maskrcnn_model import get_maskrcnn_model

class Inference:
    def __init__(self, config_path: str):
        self.config = toml.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        # Paths
        paths = self.config["Paths"]
        self.model_pth = paths["model_pth"]
        self.tif_pth = paths["tif_pth"]
        #self.mask_output_pth = paths["mask_output_pth"]
        #self.blended_output_pth = paths["blended_output_pth"]

        # Ensure output dir is created
        self.output_dir = paths["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Parameters
        params = self.config["Parameters"]
        self.tile_size = params["tile_size"]
        self.score_threshold = params["score_threshold"]
        self.mask_threshold = params["mask_threshold"]

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
                    mask_bin = mask > self.mask_threshold            # binary mask
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

        mask_output_pth = os.path.join(self.output_dir,'output_mask.png')
        print(f"\tWriting mask output to: {mask_output_pth}")
        cv2.imwrite(mask_output_pth, mask)


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

        blended_output_pth = os.path.join(self.output_dir,'output_blended.png')
        print(f"\tWriting overlayed image to {blended_output_pth}")
        cv2.imwrite(blended_output_pth, blended)