import numpy as np
import torch
from torchvision import transforms
import cv2
from load_tif import load_tif
from tile_img import tile_img
from get_maskrcnn_model import get_maskrcnn_model


tif_path = "./test_data/belieze.tif"
tile_size = 400

[img, profile] = load_tif(tif_pth=tif_path)

img = np.transpose(img, (1, 2, 0))  # transpose from (C, H, W) ---> (H, W, C)
for i in range(4):
    img[:, :, i] = cv2.normalize(img[:, :, i], None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype('uint8')
img = img[:,:,:3]
img = img[..., ::-1]

[tiles, coords, dims] = tile_img(
    img,
    tile_size=tile_size
    )

# Initialize empty array
n_rows, n_cols = dims
full_h = n_rows * tile_size
full_w = n_cols * tile_size
classification_raster = np.zeros((full_h, full_w), dtype=np.uint8)

### Load model
state_path = "./model/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_maskrcnn_model()
load_result = model.load_state_dict(
    torch.load(state_path, 
    map_location=device
))  

print(load_result)
model.eval()
model.to(device)

for i, tile in enumerate(tiles):

    # Skip over empty tiles
    if np.max(tile) != 0:

        transform = transforms.ToTensor()
        image_tensor = transform(tile).to(device)
        images = [image_tensor]

        
        with torch.no_grad():
            outputs = model(images)

        output = outputs[0]
        boxes = output['boxes']       # [N,4] bounding boxes
        labels = output['labels']     # [N] predicted class labels
        scores = output['scores']     # [N] confidence scores
        masks = output['masks']       # [N,1,H,W] predicted masks


        # Filter predictions by score threshold
        threshold = 0.1
        keep = scores > threshold

        boxes = boxes[keep]
        labels = labels[keep]
        masks = masks[keep]
        scores = scores[keep]


        print(f"masks: {masks}")
        print(f"boxes: {boxes}")
        print(f"scores: {scores}")

        if len(masks) == 0:
            continue  # no detections

        # Combine all masks into a single tile mask
        tile_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)

        for j in range(len(masks)):
            mask = masks[j, 0].cpu().numpy()  # [H,W]
            mask_bin = mask > 0.5             # binary mask
            class_label = labels[j].item()
            tile_mask[mask_bin] = class_label  # assign label to masked pixels

        # Place tile_mask into the correct location in the full raster
        row_idx, col_idx = coords[i]
        y = row_idx * tile_size
        x = col_idx * tile_size
        classification_raster[y:y+tile_size, x:x+tile_size] = tile_mask

# Normalize and write to png
max_label = classification_raster.max()
vis_raster = (classification_raster / max_label * 255).astype(np.uint8)
cv2.imwrite("test_vis.png", vis_raster)
