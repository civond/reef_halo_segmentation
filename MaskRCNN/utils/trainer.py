import torch
from torch.optim import AdamW
import argparse
import pandas as pd
from datetime import datetime
import toml
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Custom functions
from utils.create_fig import create_fig
from utils.get_maskrcnn_model import get_maskrcnn_model
from utils.create_transforms import create_transforms
from utils.get_loader import get_loader
from utils.train_fn import train_fn
from utils.val_fn import val_fn

class Trainer:
    def __init__(self, config_path: str):
        self.config = toml.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        # Hyperparameters
        hyperparameters = self.config["Hyperparameters"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.batch_size = hyperparameters["batch_size"]
        self.num_epochs = hyperparameters["num_epochs"]
        self.num_workers = hyperparameters["num_workers"]
        self.image_height = hyperparameters["image_height"]
        self.image_width = hyperparameters["image_width"]
        self.pin_memory = hyperparameters["pin_memory"]
        self.patience = hyperparameters["patience"]
        self.min_delta = hyperparameters["min_delta"]
        self.score_threshold = hyperparameters["score_threshold"]

        # Import dataset
        paths = self.config["Paths"]
        self.dataset = pd.read_csv(paths["dataset_path"])
        df_train = self.dataset[self.dataset["fold"].isin([0, 1, 2, 3])]
        df_valid = self.dataset[self.dataset["fold"] == 4]

        # Save output file locations
        self.save_model_dir = "./MaskRCNN/model/"
        self.save_csv_dir = "./MaskRCNN/logs/"
        self.save_fig_dir = "./MaskRCNN/figures/"

        # Ensure directories exist
        os.makedirs(self.save_model_dir, exist_ok=True)
        os.makedirs(self.save_csv_dir, exist_ok=True)
        os.makedirs(self.save_fig_dir, exist_ok=True)

        # Model
        self.model = get_maskrcnn_model().to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )

        # Metric tracking
        self.train_loss_arr = []
        self.val_loss_arr = []
        self.train_dice_arr = []
        self.val_dice_arr = []
        self.best_val_loss = float("inf")
        self.best_model_weights = None
        self.patience_counter = 0

        # Create transforms objects
        self.train_transform = create_transforms(
            self.image_height, 
            self.image_width,
            train=True
        )
        self.val_transform = create_transforms(
            self.image_height, 
            self.image_width,
            train=False
        )
        
        # Create dataloaders
        self.train_loader = get_loader(
            df=df_train,
            batch_size=self.batch_size,
            transform=self.train_transform,
            num_workers=self.num_workers,
            train=True,
            pin_memory=self.pin_memory
        )

        self.val_loader = get_loader(
            df=df_valid,
            batch_size=self.batch_size,
            transform=self.val_transform,
            num_workers=self.num_workers,
            train=False,
            pin_memory=self.pin_memory
        )

    def train_loop(self):
        for epoch in range(self.num_epochs):
            print(f"\nEpoch: {epoch}")


            # Train
            train_loss = train_fn(
                device=self.device, 
                loader=self.train_loader, 
                model=self.model, 
                optimizer=self.optimizer
            )

            print(f"\tAvg. Train Loss: {train_loss}")
            self.train_loss_arr.append(train_loss)


            # Validation
            val_loss, val_dice = val_fn(
                device=self.device, 
                loader=self.val_loader, 
                model=self.model,
                score_threshold=self.score_threshold
            )

            print(f"\tAvg. Val Loss: {val_loss}")
            print(f"\tAvg. Val Dice: {val_dice}")
            self.val_loss_arr.append(val_loss)
            self.val_dice_arr.append(val_dice)

            # Check Patience
            if val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_weights = self.model.state_dict()
                print("\t\tNew best loss")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print("\tEarly stopping triggered. Exit train loop")
                    break

        self._save_outputs()

    def _save_outputs(self):
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
        save_model_pth = os.path.join(self.save_model_dir, f"{timestamp}.pth")
        save_csv_pth = os.path.join(self.save_csv_dir, f"{timestamp}.csv")
        save_fig_pth = os.path.join(self.save_fig_dir, f"{timestamp}.png")

        # Save model
        try:
            print(f"\tSaving model at: {save_model_pth}")
            torch.save(self.best_model_weights, save_model_pth)
        except Exception as e:
            print(f"Error during saving model: {e}")
        
        # Save CSV
        try:
            print(f"\tSaving csv at: {save_csv_pth}")
            df = pd.DataFrame({
                "train_loss": self.train_loss_arr, 
                "val_loss": self.val_loss_arr,
                "val_dice": self.val_dice_arr
            })

            df.to_csv(save_csv_pth, index=False)
        except Exception as e:
            print(f"Error during saving csv: {e}")
        
        # Save figure
        try:
            print(f"\tSaving figure at: {save_fig_pth}")
            create_fig(
                df=df, 
                save_fig_path=save_fig_pth,
                show_plot=False
            )
            
        except Exception as e:
            print(f"Error during saving figure: {e}")