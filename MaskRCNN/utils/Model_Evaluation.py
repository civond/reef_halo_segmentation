import torch
from torchvision.ops import nms
import cv2
import numpy as np
import toml
from tqdm import tqdm
import os
import pandas as pd
import time
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Custom imports
from utils.get_loader import get_loader
from utils.eval_fn import eval_fn
from utils.create_transforms import create_transforms
from utils.get_maskrcnn_model import get_maskrcnn_model

class Model_Evaluation:
    def __init__(self, config_path: str):
        self.config = toml.load(config_path)
        
        # Parameters
        settings = self.config["Settings"]

        self.useFold = settings["useFold"]
        self.score_threshold = settings["score_threshold"]
        self.mask_threshold = settings["mask_threshold"]
        self.use_cuda = settings["use_cuda"]
        self.batch_size = settings["batch_size"]
        self.iou_threshold = settings["iou_threshold"]

        # Filepaths
        paths = self.config["Paths"]
        self.dataset = pd.read_csv(paths["dataset_pth"])
        self.dataset = self.dataset[self.dataset["fold"] == self.useFold]
        self.model_pth = paths["model_pth"]
        self.img_pth = paths["img_pth"]

        # Ensure output dir exists
        self.output_dir = paths["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        # Assert CUDA availability
        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA requested but not available."
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print("Using device:", self.device)

        # Import Model
        self.model = get_maskrcnn_model().to(self.device)
        self.model.load_state_dict(torch.load(self.model_pth, map_location=self.device))
        self.model.eval()

        # Create transform object
        self.eval_transform = create_transforms(
            mode='valid'
        )

        # Create dataloader object
        self.eval_loader = get_loader(
            df=self.dataset,
            batch_size=self.batch_size,
            transform = self.eval_transform,
            num_workers=self.num_workers,
            train=False,
            pin_memory=self.pin_memory
        )

    def eval(self):
        [eval_predictions, eval_dice] = eval_fn(
            device=self.device,
            loader=self.eval_loader,
            model=self.model,
            score_threshold=self.score_threshold
        )
        self._save_outputs(eval_predictions)


    def _save_outputs(self, predictions):
        print("")
