# =========== Written by Dorian Yeh =============
# To execute, input python main.py [flag] [config path] in the command line

import argparse
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Custom classes
from utils.Model_Trainer import Model_Trainer
from utils.Model_KFoldTrainer import Model_KFoldTrainer
from utils.Model_Inference import Model_Inference
from utils.Model_Evaluation import Model_Evaluation

def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "eval", "crossval", "inference"],
        help="Please choose from: ['train', 'eval', 'crossval', 'inference']"
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
        model_trainer = Model_Trainer(config_path)
        model_trainer.train_loop()

    # Cross Validation
    if mode.lower() == "crossval":
        print("Running k-fold cross-validation...")
        model_kfoldtrainer = Model_KFoldTrainer(config_path)
        model_kfoldtrainer.train_kfold()

    # Model Evaluation
    if mode.lower() == "eval":
        print("Running model eval. using test set...")
        model_eval = Model_Evaluation(config_path)
        model_eval.eval()

    # Model Inference on Unseen Data
    if mode.lower() == "inference":
        model_inference = Model_Inference(config_path)
        model_inference.load_satellite_img()
        model_inference.perform_inference()
        model_inference.generate_mask()
        model_inference.overlay_mask()

if __name__ == "__main__":
    main()