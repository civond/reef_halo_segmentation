import argparse
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Custom functions
from utils.trainer import Trainer
from utils.kfold_trainer import KFoldTrainer
from utils.inference import Inference

def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "crossval", "inference"],
        help="Please choose from: ['train', 'crossval', 'inference']"
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to TOML configuration file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode
    config_path = args.config_file

    # Train
    if mode.lower() == "train":
        print("Running training loop...")
        trainer = Trainer(config_path)
        trainer.train_loop()

    # Cross Validation
    if mode.lower() == "crossval":
        print("Running k-fold cross-validation...")
        kfold_trainer = KFoldTrainer(config_path)
        kfold_trainer.train_kfold()

    # Inference
    if mode.lower() == "inference":
        inf = Inference(config_path)
        inf.process_img()
        inf.overlay_mask()


if __name__ == "__main__":
    main()