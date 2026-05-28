import torch
from torchvision.ops import nms
import toml
import os
import pandas as pd
import datetime
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Custom imports
from utils.get_loader import get_loader
from utils.eval_fn import eval_fn
from utils.create_transforms import create_transforms
from utils.get_maskrcnn_model import get_maskrcnn_model
from utils.eval_aucroc import eval_aucroc

class Model_Evaluation:
    def __init__(self, config_path: str):
        self.config = toml.load(config_path)
        
        # Parameters
        settings = self.config["Settings"]
        self.save_logits = settings["save_logits"]
        self.num_workers = settings["num_workers"]
        self.pin_memory = settings["pin_memory"]
        self.useFold = settings["useFold"]
        self.score_threshold = settings["score_threshold"]
        self.mask_threshold = settings["mask_threshold"]
        self.use_cuda = settings["use_cuda"]
        self.batch_size = settings["batch_size"]
        self.iou_threshold = settings["iou_threshold"]

        # Filepaths
        paths = self.config["Paths"]
        self.dataset_path = paths["dataset_pth"]
        self.dataset = pd.read_csv(self.dataset_path)
        self.dataset = self.dataset[self.dataset["fold"] == self.useFold]
        self.model_pth = paths["model_pth"]

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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.output_dir, "eval_"+timestamp)
        os.makedirs(save_dir, exist_ok=True)

        # Generate predictions
        [eval_iou, eval_dice] = eval_fn(
            device=self.device,
            loader=self.eval_loader,
            model=self.model,
            score_threshold=self.score_threshold,
            save_dir=save_dir,
            save_logits=self.save_logits
        )

        # Calculate AUC-ROC
        [auc, optimal_threshold] = eval_aucroc(
            save_dir=save_dir
        )

        # Output log file with settings used for reproducibility
        settings_path = os.path.join(save_dir, "log.txt")
        with open(settings_path, "w") as f:
            f.write(f"Mean IOU        : {eval_iou}\n")
            f.write(f"Mean Dice       : {eval_dice}\n")
            f.write(f"AUC             : {auc}\n")
            f.write(f"optimal_thresh  : {optimal_threshold}\n")
            f.write(f"timestamp       : {timestamp}\n")
            f.write(f"dataset_pth     : {self.dataset_path}\n")
            f.write(f"model_pth       : {self.model_pth}\n")
            f.write(f"fold            : {self.useFold}\n")
            f.write(f"score_threshold : {self.score_threshold}\n")
            f.write(f"mask_threshold  : {self.mask_threshold}\n")
            f.write(f"iou_threshold   : {self.iou_threshold}\n")
            f.write(f"batch_size      : {self.batch_size}\n")

