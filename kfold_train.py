import torch
import torch.optim as optim
import argparse
import pandas as pd
import toml
import os
from sklearn.model_selection import KFold
from torch.utils.data import Subset

os.environ["QT_QPA_PLATFORM"] = "xcb"

# Custom functions
from utils.get_maskrcnn_model import get_maskrcnn_model
from utils.create_transforms import create_train_transforms, create_val_transforms
from utils.dataset import ImageDataset
from utils.collate_fn import collate_fn
from utils.train_fn import train_fn
from utils.val_fn import val_fn


def parse_args():
    parser = argparse.ArgumentParser(description="K-Fold cross-validation training")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to TOML configuration file"
    )
    return parser.parse_args()


def create_fold_loaders(train_dataset, val_dataset, train_idx, val_idx, batch_size, num_workers, pin_memory):
    """Create train and validation loaders for a specific fold"""
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


def train_single_fold(fold, train_loader, val_loader, device, config):
    """Train a single fold and return the best validation loss"""
    print(f"\n{'='*60}")
    print(f"FOLD {fold + 1}")
    print(f"{'='*60}")

    # Initialize fresh model for this fold
    model = get_maskrcnn_model()
    model.to(device)

    # Optimizer and scaler
    learning_rate = config["Hyperparameters"]["learning_rate"]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(device="cuda")

    # Early stopping parameters
    patience = config["Hyperparameters"]["patience"]
    min_delta = config["Hyperparameters"]["min_delta"]
    num_epochs = config["Hyperparameters"]["num_epochs"]

    # Track loss
    train_loss_arr = []
    val_loss_arr = []
    best_val_loss = float("inf")
    patience_counter = 0

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

            print(f"\tNew best loss")

            # Save best model for this fold
            fold_model_path = config["Paths"]["save_model_pth"].replace(".pth", f"_fold{fold + 1}.pth")
            torch.save(model.state_dict(), fold_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered. Exit train loop")
                break

    # Save fold loss history
    fold_csv_path = config["Paths"]["save_csv_pth"].replace(".csv", f"_fold{fold + 1}.csv")
    df_fold = pd.DataFrame({
        "train_loss": train_loss_arr,
        "val_loss": val_loss_arr
    })
    df_fold.to_csv(fold_csv_path, index=False)
    print(f"\nFold {fold + 1} complete. Best Val Loss: {best_val_loss}")

    return best_val_loss, train_loss_arr, val_loss_arr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Parse arguments and load config
    args = parse_args()
    config = toml.load(args.config_file)

    # Load hyperparameters
    data_dir = config["Paths"]["data_dir"]
    batch_size = config["Hyperparameters"]["batch_size"]
    num_workers = config["Hyperparameters"]["num_workers"]
    image_height = config["Hyperparameters"]["image_height"]
    image_width = config["Hyperparameters"]["image_width"]
    pin_memory = config["Hyperparameters"]["pin_memory"]
    n_splits = config["Hyperparameters"]["n_splits"]
    random_seed = config["Hyperparameters"].get("random_seed", 42)

    print(f"\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Number of folds: {n_splits}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_height}x{image_width}")
    print(f"  Random seed: {random_seed}")

    # Load train parameter
    train = config["Hyperparameters"]["train"]

    # Create transforms
    train_transforms = create_train_transforms(
        image_height,
        image_width,
        train
    )

    val_transforms = create_val_transforms(
        image_height,
        image_width,
    )

    # Create datasets with different transforms
    train_dataset = ImageDataset(data_dir=data_dir, transform=train_transforms)
    val_dataset = ImageDataset(data_dir=data_dir, transform=val_transforms)

    print(f"\nTotal samples in dataset: {len(train_dataset)}")

    # K-Fold cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # Store results for each fold
    fold_results = []

    # Iterate over folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(train_dataset)))):
        print(f"\nFold {fold + 1}: Train samples = {len(train_idx)}, Val samples = {len(val_idx)}")

        # Create loaders for this fold
        train_loader, val_loader = create_fold_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        # Train this fold
        best_val_loss, train_losses, val_losses = train_single_fold(
            fold=fold,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config
        )

        fold_results.append({
            "fold": fold + 1,
            "best_val_loss": best_val_loss,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "epochs_trained": len(train_losses)
        })

    # Aggregate results
    print(f"\n{'='*60}")
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")

    df_results = pd.DataFrame(fold_results)
    print("\n", df_results.to_string(index=False))

    # Calculate statistics
    mean_val_loss = df_results["best_val_loss"].mean()
    std_val_loss = df_results["best_val_loss"].std()

    print(f"\n{'='*60}")
    print(f"Mean Best Val Loss: {mean_val_loss} ± {std_val_loss}")
    print(f"{'='*60}")

    # Save aggregated results
    summary_csv_path = config["Paths"]["save_csv_pth"].replace(".csv", "_kfold_summary.csv")
    df_results.to_csv(summary_csv_path, index=False)
    print(f"\nSaved summary to: {summary_csv_path}")


if __name__ == "__main__":
    main()
