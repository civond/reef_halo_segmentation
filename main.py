import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
model = maskrcnn_resnet50_fpn(weights=weights)
model.eval()


import numpy as np
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"


import cv2
from utils.load_tif import load_tif
from utils.tile_img import tile_img

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    img_dir = "./data/img/"
    mask_dir = "./data/mask/"

    # Training values
    learning_rate = 1e-5
    batch_size = 64
    num_epochs = 10
    num_workers = 10
    image_height = 400
    image_width = 400
    pin_memory = True
    load_model = False

    # Import MaskRCNN
    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=400
        )
    
    print(model)






    """model_path = "./model/Model2.pth"
    tif_path = "./data/belieze.tif"
    
    model = torch.load(model_path, 
                       map_location=torch.device('cpu')
                       )
    #print(model)

    [img, profile] = load_tif(tif_pth=tif_path)
    img = np.transpose(img, (1, 2, 0))  # transpose from (C, H, W) ---> (H, W, C)
    print(img.shape)

    img = tile_img(
        img,
        tile_size=512
        )"""

if __name__ == "__main__":
    main()