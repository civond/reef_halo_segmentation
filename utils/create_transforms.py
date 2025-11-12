import albumentations as A
from albumentations.pytorch import ToTensorV2
#from utils.get_loader import get_loader


# Create transforms object
def create_train_transforms(IMAGE_HEIGHT, IMAGE_WIDTH, TRAIN=True):
    if TRAIN == True:
        transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=35, p=0.5),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})
        
    if TRAIN == False:
        transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})
        
    return transform

def create_val_transforms(IMAGE_HEIGHT, IMAGE_WIDTH):
    transform = A.Compose([
    A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    ToTensorV2()
    ], additional_targets={'mask': 'mask'})
    
    return transform