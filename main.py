import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import cv2
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Custom functions
from utils.load_tif import load_tif
from utils.tile_img import tile_img
from utils.get_maskrcnn_model import get_maskrcnn_model
from utils.create_transforms import create_train_transforms, create_val_transforms
from utils.get_loader import get_train_loader, get_val_loader
from utils.train_fn import train_fn
from utils.val_fn import val_fn



# Main loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dir = "./train/"
    val_dir = "./val/"

    # Training values
    learning_rate = 1e-5
    batch_size = 32
    num_epochs = 10
    num_workers = 10
    image_height = 400
    image_width = 400
    pin_memory = True
    load_model = False
    train = True

    # Import MaskRCNN
    model = get_maskrcnn_model()
    model.to(device)

    # Create transforms object
    train_transforms = create_train_transforms(
        image_height, 
        image_width,
        train
    )

    val_transforms = create_val_transforms(
        image_height, 
        image_width,
        train
    )
    
    # Create train and validation loaders  
    """train_loader = get_train_loader(
        train_dir, 
        batch_size, 
        train_transforms, 
        num_workers, 
        train, 
        pin_memory
    )"""

    train_loader = get_train_loader(
        data_dir=train_dir, 
        batch_size=batch_size, 
        transform=train_transforms, 
        num_workers=num_workers, 
        train=True, 
        pin_memory=pin_memory
    )

    val_loader = get_val_loader(
        data_dir=val_dir, 
        batch_size=batch_size, 
        transform=val_transforms, 
        num_workers=num_workers, 
        train=False, 
        pin_memory=pin_memory
    )

    # Train loop
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch}")

        # Train
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scaler = torch.amp.GradScaler(device="cuda")
        
        train_loss = train_fn(
            device, 
            train_loader, 
            model, 
            optimizer, 
            scaler
        )

        val_loss = val_fn(
            device=device, 
            loader=val_loader, 
            model=model
        )

    # Save model
    torch.save(model.state_dict(), "./model/model.pth")

        
        #training_loss.append(train_loss)
        #training_dice.append(train_dice)




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