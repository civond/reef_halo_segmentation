import torch
from torch.optim import AdamW
from torch.utils.data import Subset
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold
import toml
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

# Custom functions
from utils.create_fig import create_fig
from utils.get_maskrcnn_model import get_maskrcnn_model
from utils.create_transforms import create_train_transforms, create_val_transforms
from utils.dataset import ImageDataset
from utils.collate_fn import collate_fn
from utils.train_fn import train_fn
from utils.val_fn import val_fn


class KFoldTrainer:
    def __init__(self, config_path: str):
        self.config = toml.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        # Hyperparameters
        hp = self.config["Hyperparameters"]
        self.learning_rate = hp["learning_rate"]
        self.batch_size = hp["batch_size"]
        self.num_epochs = hp["num_epochs"]
        self.num_workers = hp["num_workers"]
        self.image_height = hp["image_height"]
        self.image_width = hp["image_width"]
        self.pin_memory = hp["pin_memory"]
        self.train_flag = hp["train"]
        self.patience = hp["patience"]
        self.min_delta = hp["min_delta"]
        self.n_splits = hp["n_splits"]
        self.random_seed = hp.get("random_seed", 42)
        self.score_threshold = hp.get("score_threshold", 0.5)

        # Paths
        paths = self.config["Paths"]
        self.data_dir = paths["data_dir"]

        # Output directories (timestamped)
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
        self.save_model_dir = "./model/"
        self.save_csv_dir = "./logs/"
        self.save_fig_dir = "./figures/"

        # Ensure directories exist
        os.makedirs(self.save_model_dir, exist_ok=True)
        os.makedirs(self.save_csv_dir, exist_ok=True)
        os.makedirs(self.save_fig_dir, exist_ok=True)

        # Transforms
        self.train_transform = create_train_transforms(
            self.image_height,
            self.image_width
        )
        self.val_transform = create_val_transforms(
            self.image_height,
            self.image_width
        )

        # Create datasets (same data, different transforms)
        self.train_dataset = ImageDataset(
            data_dir=self.data_dir,
            transform=self.train_transform
        )
        self.val_dataset = ImageDataset(
            data_dir=self.data_dir,
            transform=self.val_transform
        )

        # K-Fold splitter
        self.kfold = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_seed
        )

        # Results storage
        self.fold_results = []

        self._print_config()

    def _print_config(self):
        print(f"\nK-Fold Configuration:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Total samples: {len(self.train_dataset)}")
        print(f"  Number of folds: {self.n_splits}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Image size: {self.image_height}x{self.image_width}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Random seed: {self.random_seed}")

    def _create_fold_loaders(self, train_idx, val_idx):
        """Create train and validation loaders for a specific fold."""
        train_subset = Subset(self.train_dataset, train_idx)
        val_subset = Subset(self.val_dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )

        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )

        return train_loader, val_loader

    def _train_single_fold(self, fold, train_loader, val_loader):
        """Train a single fold and return metrics."""
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{self.n_splits}")
        print(f"{'='*60}")

        # Initialize fresh model for this fold
        model = get_maskrcnn_model()
        model.to(self.device)

        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )

        # Tracking
        train_loss_arr = []
        val_loss_arr = []
        val_dice_arr = []
        best_val_loss = float("inf")
        best_model_weights = None
        patience_counter = 0

        # Training loop
        for epoch in range(self.num_epochs):
            print(f"\nEpoch: {epoch}")

            # Train
            train_loss = train_fn(
                device=self.device,
                loader=train_loader,
                model=model,
                optimizer=optimizer
            )
            print(f"\tAvg. Train Loss: {train_loss}")
            train_loss_arr.append(train_loss)

            # Validation
            val_loss, val_dice = val_fn(
                device=self.device,
                loader=val_loader,
                model=model,
                score_threshold=self.score_threshold
            )
            print(f"\tAvg. Val Loss: {val_loss}")
            print(f"\tAvg. Val Dice: {val_dice}")
            val_loss_arr.append(val_loss)
            val_dice_arr.append(val_dice)

            # Early stopping check
            if val_loss < (best_val_loss - self.min_delta):
                best_val_loss = val_loss
                patience_counter = 0
                best_model_weights = model.state_dict().copy()
                print("\t\tNew best loss")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("\tEarly stopping triggered. Exit train loop")
                    break

        # Save fold outputs
        self._save_fold_outputs(
            fold=fold,
            model_weights=best_model_weights,
            train_loss_arr=train_loss_arr,
            val_loss_arr=val_loss_arr,
            val_dice_arr=val_dice_arr
        )

        print(f"\nFold {fold + 1} complete. Best Val Loss: {best_val_loss}")

        return {
            "fold": fold + 1,
            "best_val_loss": best_val_loss,
            "best_val_dice": max(val_dice_arr) if val_dice_arr else 0.0,
            "final_train_loss": train_loss_arr[-1],
            "final_val_loss": val_loss_arr[-1],
            "final_val_dice": val_dice_arr[-1] if val_dice_arr else 0.0,
            "epochs_trained": len(train_loss_arr)
        }

    def _save_fold_outputs(self, fold, model_weights, train_loss_arr, val_loss_arr, val_dice_arr):
        """Save model, CSV, and figure for a single fold."""
        fold_suffix = f"_fold{fold + 1}"

        # Save model
        model_path = os.path.join(
            self.save_model_dir,
            f"{self.timestamp}{fold_suffix}.pth"
        )
        try:
            print(f"\tSaving model at: {model_path}")
            torch.save(model_weights, model_path)
        except Exception as e:
            print(f"Error saving model: {e}")

        # Save CSV
        csv_path = os.path.join(
            self.save_csv_dir,
            f"{self.timestamp}{fold_suffix}.csv"
        )
        try:
            print(f"\tSaving CSV at: {csv_path}")
            df = pd.DataFrame({
                "train_loss": train_loss_arr,
                "val_loss": val_loss_arr,
                "val_dice": val_dice_arr
            })
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error saving CSV: {e}")

        # Save figure
        fig_path = os.path.join(
            self.save_fig_dir,
            f"{self.timestamp}{fold_suffix}.png"
        )
        try:
            print(f"\tSaving figure at: {fig_path}")
            create_fig(df=df, save_fig_path=fig_path, show_plot=False)
        except Exception as e:
            print(f"Error saving figure: {e}")

    def _save_summary(self):
        """Save aggregated k-fold results."""
        print(f"\n{'='*60}")
        print("K-FOLD CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")

        df_results = pd.DataFrame(self.fold_results)
        print("\n", df_results.to_string(index=False))

        # Statistics
        mean_val_loss = df_results["best_val_loss"].mean()
        std_val_loss = df_results["best_val_loss"].std()
        mean_val_dice = df_results["best_val_dice"].mean()
        std_val_dice = df_results["best_val_dice"].std()

        print(f"\n{'='*60}")
        print(f"Mean Best Val Loss: {mean_val_loss:.4f} +/- {std_val_loss:.4f}")
        print(f"Mean Best Val Dice: {mean_val_dice:.4f} +/- {std_val_dice:.4f}")
        print(f"{'='*60}")

        # Save summary CSV
        summary_path = os.path.join(
            self.save_csv_dir,
            f"{self.timestamp}_kfold_summary.csv"
        )
        df_results.to_csv(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")

    def train_kfold(self):
        """Run k-fold cross-validation training."""
        dataset_indices = range(len(self.train_dataset))

        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(dataset_indices)):
            print(f"\nFold {fold + 1}: Train samples = {len(train_idx)}, Val samples = {len(val_idx)}")

            # Create loaders for this fold
            train_loader, val_loader = self._create_fold_loaders(train_idx, val_idx)

            # Train this fold
            fold_result = self._train_single_fold(fold, train_loader, val_loader)
            self.fold_results.append(fold_result)

        # Save aggregated results
        self._save_summary()
