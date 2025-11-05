import cv2
import numpy as np
import rasterio
from PIL import Image
import os

def convert_satellite_imgs():
    
    img_dir = "./raw/img/"
    mask_dir = "./raw/label/"
    img_write_dir = "./data/img/"
    mask_write_dir = "./data/mask/"

    for filename in os.listdir(img_dir):
        if filename.endswith(".tif"):
            raster_path = os.path.join(img_dir, filename)
            mask_path = os.path.join(mask_dir, filename)

            name_temp = filename.split(".")[0]
            name_temp = name_temp + ".png"
            img_write_path = os.path.join(img_write_dir, name_temp)
            mask_write_path = os.path.join(mask_write_dir, name_temp)
            
            
            """print(raster_path)
            print(img_write_path)
            print(name_temp)"""

            if (os.path.exists(raster_path) == True) and os.path.exists(mask_path) == True:
                try:
                    # Open raster image and convert to RGB format
                    with rasterio.open(raster_path) as src:
                        img = src.read()
                        profile = src.profile

                    img = np.transpose(img, (1, 2, 0))  # transpose from (C, H, W) ---> (H, W, C)
                    for i in range(4):
                        img[:, :, i] = cv2.normalize(img[:, :, i], None, 0, 255, cv2.NORM_MINMAX)
                    img = img.astype('uint8')
                    img = img[:,:,:3]

                    # Open raster mask and convert to binary mask
                    with rasterio.open(mask_path) as src:
                        label = src.read(1) # Read binary data

                    binary_array = (label != 0)
                    binary_array = binary_array.astype(np.uint8) * 255
                    mask = Image.fromarray(binary_array, mode='L')
                    
                    
                    # Write img and mask as png
                    cv2.imwrite(img_write_path, img)
                    mask.save(mask_write_path)
                    


                    """cv2.imshow("rgb", img)
                    cv2.imshow("mask", binary_array)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()"""

                except Exception as e:
                    print(f"Error loading {raster_path}: {e}")


if __name__ == "convert_satellite_imgs":
    convert_satellite_imgs()