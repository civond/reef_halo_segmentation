import torch
from tqdm import tqdm

def val_fn(device, loader, model):
    model.eval()
    loop = tqdm(loader, desc="Validating", leave=False)

    # Track loss and dice score
    total_loss = 0
    #total_dice = 0

    with torch.no_grad():
        for batch_idx, (images, targets, filenames) in enumerate(loop):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            model.train() # Calculate loss
            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
            model.eval() # Reset to eval

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
    avg_loss = total_loss / len(loader)
    #avg_dice = total_dice / len(loader) 

    # update tqdm loop
    loop.set_postfix(loss=loss.item())

    return avg_loss