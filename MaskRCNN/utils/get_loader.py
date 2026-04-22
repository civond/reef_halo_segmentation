from utils.dataset import ImageDataset
from utils.collate_fn import collate_fn
from torch.utils.data import DataLoader


def get_loader(
        df,
        batch_size,
        transform,
        num_workers=4,
        train=True,
        pin_memory=True
        ):
    
    # Load dataset
    dataset = ImageDataset(
        df=df, 
        train=train,
        transform=transform
        )

    # Create loader object
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=train,
        collate_fn=collate_fn
    )

    return loader