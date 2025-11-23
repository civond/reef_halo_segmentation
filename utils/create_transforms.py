import albumentations as A
from albumentations.pytorch import ToTensorV2
#from utils.get_loader import get_loader


# Create transforms object
def create_train_transforms(IMAGE_HEIGHT, IMAGE_WIDTH):
    
    transform = A.Compose([
        A.Resize(
            IMAGE_HEIGHT, 
            IMAGE_WIDTH
        ),
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
        A.RGBShift(
            r_shift_limit=(-10,10),     # Shift red channel up to +/- 10
            g_shift_limit=(-10,10),     # Shift green channel up to +/- 10
            b_shift_limit=(-10,10),     # Shift blue channel up to +/- 10
            p=0.5                       # 50% probability
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,       # +/- 20% brightness shift
            contrast_limit=0.2,         # +/- 20% contrast shift
            p=0.5                       # 50% probability
        ),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})
        
        
    return transform

def create_val_transforms(IMAGE_HEIGHT, IMAGE_WIDTH):
    transform = A.Compose([
    A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    ToTensorV2()
    ], additional_targets={'mask': 'mask'})
    
    return transform