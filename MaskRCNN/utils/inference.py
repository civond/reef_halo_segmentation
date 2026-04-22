import torch
import cv2
import numpy as np
import toml
from tqdm import tqdm
import os
import time
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Custom imports
from utils.load_tif import load_tif
from utils.percentile_stretch import percentile_stretch
from utils.tile_img import tile_img
from utils.create_transforms import create_transforms
from utils.get_maskrcnn_model import get_maskrcnn_model

class Inference:
    def __init__(self, config_path: str):
        self.config = toml.load(config_path)

        # Filepaths
        paths = self.config["Paths"]
        self.model_pth = paths["model_pth"]
        self.img_pth = paths["img_pth"]

        # Ensure output dir exists
        self.output_dir = paths["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Parameters
        settings = self.config["Settings"]
        self.tile_size = settings["tile_size"]
        self.score_threshold = settings["score_threshold"]
        self.mask_threshold = settings["mask_threshold"]
        self.use_overlap = settings["use_overlap"]
        self.use_cuda = settings["use_cuda"]
        self.rgb_idx = settings["rgb_idx"]
        self.batch_size = settings["batch_size"]

        # Check cuda availability
        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA requested but not available."
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print("Using device:", self.device)

        # Check image type
        assert self.img_pth.lower().endswith((".tif", ".png")), f"Invalid image format: {self.img_pth}. Expected .tif or .png"

        # Model
        self.model = get_maskrcnn_model().to(self.device)
        self.model.load_state_dict(torch.load(self.model_pth, map_location=self.device))
        self.model.eval()

        # Images
        self.img_name = self.img_pth.split('/')[-1]
        self.img_name = self.img_name.split('.')[0]

        # Preallocate for raster and mask
        self.raster = None
        self.raster_mask = None
        self.valid_coords = None
    
    # Load .png or .tif image
    def load_satellite_img(self):
        if self.img_pth.endswith(".png"):
            img = cv2.imread(self.img_pth)
        
        # if .tif
        else:
            [img, _] = load_tif(tif_pth=self.img_pth)
            img = np.transpose(img, (1, 2, 0))          
            
            # Print per channel values/mean
            max_vals = np.max(img, axis=(0, 1))
            print(f"\tChannel max vals: {max_vals}")
            means = np.mean(img, axis=(0, 1))
            print("\tChannel means:", means)


            # Process .tif
            img = img[:, :, self.rgb_idx].astype(np.float32)         # Prevents division by zero
            img = np.where(img == 0, np.nan, img)               # Removes NaN's

            # Perform percentile stretch
            for i in range(3):
                print(f"\tStretching channel {i}")
                img[:, :, i] = percentile_stretch(img[:, :, i])
                

            # Convert invalid values to int and normalize to 255
            img = np.nan_to_num(
                img, 
                nan=0.0, 
                posinf=0.0, 
                neginf=0.0
            )
            img = (img * 255).astype(np.uint8)

        # Save img in self.raster
        self.raster = img

    # Perform inference on satellite image
    def perform_inference(self):
        print(f"Tiling: {self.img_pth}.")

        [img, coords, dims] = tile_img(
            self.raster,
            tile_size=self.tile_size,
            overlap=self.use_overlap
        )

        # Initialize empty array
        n_rows, n_cols = dims
        full_h = n_rows * self.tile_size
        full_w = n_cols * self.tile_size
        self.classification_raster = np.zeros((full_h, full_w), dtype=np.uint8)

        # Create transform object
        transform = create_transforms(mode="inf")

        print(f"Performing inference on {n_rows*n_cols} tiles.")
        
        valid_tiles = []
        valid_coords = []
        
        inf_start_time = time.perf_counter() # Inference start time

        # GPU inference
        if self.use_cuda == True:

            # If image contains nonzero values, flag as valid for inference
            for i, tile in enumerate(img):
                if np.max(tile) != 0:
                    valid_tiles.append(tile)
                    valid_coords.append(coords[i])

            # Pre-transform all tiles
            transformed_tiles = []
            for tile in valid_tiles:
                transformed = transform(image=tile)
                transformed_tiles.append(transformed["image"])


            all_outputs = []
            for batch_num, batch_start in enumerate(tqdm(range(0, len(transformed_tiles), self.batch_size), desc="Inference"), start=1):
                batch = torch.stack(transformed_tiles[batch_start : batch_start + self.batch_size]).to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(batch)  # list of tensors, not stacked

                for output_dict in outputs:
                    cpu_dict = {}

                    for key, value in output_dict.items():
                        if torch.is_tensor(value):
                            cpu_dict[key] = value.cpu()
                        else:
                            cpu_dict[key] = value

                    all_outputs.append(cpu_dict)
            self.raster_mask = all_outputs
            self.valid_coords = valid_coords

        # CPU inference
        if self.use_cuda == False:
            print("\t(this will take a while)")
           # collect valid tiles
            for i, tile in enumerate(img):
                if np.max(tile) != 0:
                    valid_tiles.append(tile)
                    valid_coords.append(coords[i])

            all_outputs = []

            for tile in valid_tiles:

                transformed = transform(image=tile)
                image_tensor = transformed["image"].to(self.device)

                with torch.no_grad():
                    outputs = self.model([image_tensor])[0]  # single image

                cpu_dict = {}

                for key, value in outputs.items():
                    if torch.is_tensor(value):
                        cpu_dict[key] = value.cpu()
                    else:
                        cpu_dict[key] = value

                all_outputs.append(cpu_dict)

            self.raster_mask = all_outputs
            self.valid_coords = valid_coords
    
            """output = outputs[0]
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
            self.classification_raster[y:y+self.tile_size, x:x+self.tile_size] = np.maximum(
                self.classification_raster[y:y+self.tile_size, x:x+self.tile_size],
                tile_mask
            )"""

            """# Normalize and write to png
            max_label = self.classification_raster.max()
            mask = (self.classification_raster / max_label * 255).astype(np.uint8)
            mask = mask[:h_img, :w_img]

            self.raster_mask = mask

            mask_output_pth = os.path.join(self.output_dir,f"{self.img_name}_pred_mask.png")
            print(f"\tWriting mask output to: {mask_output_pth}")
            cv2.imwrite(mask_output_pth, mask)"""

        # Calculate elapsed time
        inf_end_time = time.perf_counter() # Inference end time
        elapsed_time = inf_end_time - inf_start_time
        print(f"\tComplete!")
        print(f"\tElapsed time: {elapsed_time:.4f}s ({(elapsed_time/len(valid_tiles)):.6f}s/sample)")
    
    # Convert tensors to mask    
    def generate_mask(self):
        # Post-processing (unchanged)
            for k, output in enumerate(self.raster_mask):
                boxes = output['boxes']
                labels = output['labels']
                scores = output['scores']
                masks = output['masks']

                keep = scores > self.score_threshold
                masks = masks[keep]
                labels = labels[keep]

                if len(masks) == 0:
                    continue

                tile_mask = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)

                for j in range(len(masks)):
                    mask = masks[j, 0].cpu().numpy()
                    mask_bin = mask > self.mask_threshold
                    class_label = labels[j].item()
                    tile_mask[mask_bin] = class_label

                y, x = self.valid_coords[k]

                self.classification_raster[y:y+self.tile_size, x:x+self.tile_size] = np.maximum(
                    self.classification_raster[y:y+self.tile_size, x:x+self.tile_size],
                    tile_mask
                )
            # Normalize and write to png
            [h_img, w_img] = self.raster.shape[:2]
            max_label = self.classification_raster.max()
            mask = (self.classification_raster / max_label * 255).astype(np.uint8)
            mask = mask[:h_img, :w_img]

            self.raster_mask = mask

            mask_output_pth = os.path.join(self.output_dir,f"{self.img_name}_pred_mask.png")
            print(f"\tWriting mask output to: {mask_output_pth}")
            cv2.imwrite(mask_output_pth, mask)

    # Overlay mask onto original image and save as .png
    def overlay_mask(self, alpha=0.2, thickness=-1):

        # Check raster and mask
        if self.raster is None or self.raster_mask is None:
            raise ValueError("Run process_tif() before calling overlay_mask().")
        
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

        blended_output_pth = os.path.join(self.output_dir,f"{self.img_name}_blended.png")
        print(f"\tWriting overlayed image to {blended_output_pth}")
        cv2.imwrite(blended_output_pth, blended)