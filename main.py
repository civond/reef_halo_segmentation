import torch
from torch.optim import AdamW
import argparse
import pandas as pd
import toml
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


# Arg Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    # First positional argument: mode
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "inference"],
        help="Operation mode: 'train' or 'inference'"
    )
    
    # Second positional argument: config
    parser.add_argument(
        "config_file", 
        type=str,
        help="Path to TOML configuration file"
    )
    return parser.parse_args()


# Main loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    args = parse_args()
    mode = args.mode
    config_path = args.config_file
    config = toml.load(config_path)

    if mode.lower() == "train":
        print("Running training loop...")

        # Access TOML values
        train_dir = config["Paths"]["train_data_dir"]
        val_dir = config["Paths"]["val_data_dir"]
        save_model_pth = config["Paths"]["save_model_pth"]
        save_csv_pth = config["Paths"]["save_csv_pth"]

        learning_rate = config["Hyperparameters"]["learning_rate"]
        batch_size = config["Hyperparameters"]["batch_size"]
        num_epochs = config["Hyperparameters"]["num_epochs"]
        num_workers = config["Hyperparameters"]["num_workers"]
        image_height = config["Hyperparameters"]["image_height"]
        image_width = config["Hyperparameters"]["image_width"]
        pin_memory = config["Hyperparameters"]["pin_memory"]
        train = config["Hyperparameters"]["train"]
        patience = config["Hyperparameters"]["patience"]
        min_delta = config["Hyperparameters"]["min_delta"]


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
    )
    
    # Create train and validation loaders  
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

    # Track loss
    train_loss_arr = []
    val_loss_arr = []

    # Early Stopping
    best_val_loss = float("inf")
    best_model_weights = None
    patience_counter = 0

    # Optimizer and scaler
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4
        )   
    
    scaler = torch.amp.GradScaler(device="cuda")
    
    # Train loop
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch}")

        # Train
        train_loss = train_fn(
            device, 
            train_loader, 
            model, 
            optimizer, 
            scaler
        )
        print(f"Avg. Train Loss: {train_loss}")
        train_loss_arr.append(train_loss)

        val_loss = val_fn(
            device=device, 
            loader=val_loader, 
            model=model
        )
        print(f"Avg. Val Loss: {val_loss}")
        val_loss_arr.append(val_loss)

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = model.state_dict()
            print(f"\tNew best loss")

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered. Exit train loop")
                break



    # Save model and loss history
    print(f"Saving model at: {save_model_pth}")
    torch.save(
        best_model_weights, 
        save_model_pth
        )

    print(f"Saving csv at: {save_csv_pth}")
    df = pd.DataFrame({
        "train_loss": train_loss_arr,
        "val_loss": val_loss_arr
    })

    df.to_csv(
        save_csv_pth, 
        index=False
        )



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