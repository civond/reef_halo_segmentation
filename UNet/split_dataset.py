import os
import random
import shutil
from sklearn.model_selection import train_test_split

# Paths
img_dir = "./UNet/data/img"
msk_dir = "./UNet/data/mask"

out_root = "./UNet/dataset/"

splits = ["train", "val", "test"]

# create output directories
for s in splits:
    os.makedirs(os.path.join(out_root, s, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_root, s, "masks"), exist_ok=True)

# load filenames
files = sorted([f for f in os.listdir(img_dir)
                if f.lower().endswith((".png", ".jpg", ".tif"))])

# 1) split into 60% train, 40% temp
train_files, temp_files = train_test_split(
    files, test_size=0.4, random_state=42, shuffle=True
)

# 2) split temp into 20% val and 20% test
val_files, test_files = train_test_split(
    temp_files, test_size=0.5, random_state=42, shuffle=True
)

def copy_files(file_list, split):
    for fname in file_list:
        shutil.copy(
            os.path.join(img_dir, fname),
            os.path.join(out_root, split, "images", fname)
        )
        shutil.copy(
            os.path.join(msk_dir, fname),
            os.path.join(out_root, split, "masks", fname)
        )

copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("Done!")
print("Train:", len(train_files))
print("Val:  ", len(val_files))
print("Test: ", len(test_files))