# overlay_mask.py
import cv2
import numpy as np

def overlay_mask(img, mask, color=(255, 0, 0), alpha=0.5, output_path=None):
    mask_bin = (mask > 0).astype(np.uint8)
    mask_rgb = np.zeros_like(img)
    mask_rgb[mask_bin == 1] = color

    # Overlay the mask on the original image
    overlay = cv2.addWeighted(img, 1 - alpha, mask_rgb, alpha, 0)

    # Save if requested
    if output_path is not None:
        cv2.imwrite(output_path, overlay)

    return overlay


if __name__ == "__main__":
    # Example usage
    img_path = "./test_data/belieze.png"          # path to your original image
    mask_path = "./test_data/test_vis.png"             # path to your binary mask (0/1)
    output_path = "overlay.png"
    tile_size=400

    # Read images
    img = cv2.imread(img_path)         # BGR format
    [h,w,c] = img.shape

    # Pad the image such that the dimensions match the tile size
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8)

    overlay = overlay_mask(img, mask, color=(255,0,0), alpha=0.5, output_path=output_path)

