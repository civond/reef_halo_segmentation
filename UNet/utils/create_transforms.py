import albumentations as A
from albumentations.pytorch import ToTensorV2

# Create transform object for training and validation/inference
def create_transforms(IMAGE_HEIGHT, IMAGE_WIDTH, train=True):
    if train == True:
        transform = A.Compose([
        A.HorizontalFlip(
            p=0.5                       # 50% probability
        ),
        A.VerticalFlip(
            p=0.5                       # 50% probability
        ),
        A.Rotate(
            limit=35,                   # +/- 35 degree rotation
            p=0.5                       # 50% probability
        ),
        A.Normalize(
            mean=(0, 0, 0),
            std=(1, 1, 1),
            max_pixel_value=255.0
        ),
        ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    else:
        transform = A.Compose([
        A.Normalize(
            mean=(0, 0, 0),
            std=(1, 1, 1),
            max_pixel_value=255.0
        ),
        ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    return transform