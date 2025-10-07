import torch
import numpy as np
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"


import cv2
from utils.load_tif import load_tif

def main():
    model_path = "./model/Model2.pth"
    tif_path = "./data/belieze.tif"
    
    model = torch.load(model_path)
    #print(model)

    [img, profile] = load_tif(tif_pth=tif_path)
    img = np.transpose(img, (1, 2, 0))  # transpose from (C, H, W) ---> (H, W, C)

    print(img.shape)

    num = 10000
    sub_img = img[num:num+500, num:num+500, :3]
    sub_img = cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR)

    # Display the image
    cv2.imshow("500x500 Portion", sub_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()