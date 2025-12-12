import torch
from tqdm import tqdm

from utils.calculate_dice import calculate_dice

def val_fn(device, loader, model, loss_fn, score_threshold=0.5):
    model.eval()
    loop = tqdm(loader, desc="Validating", leave=False)

    # Track loss and dice score
    total_loss = 0
    total_dice = 0

    with torch.no_grad():
        for batch_idx, (images, labels, filenames) in enumerate(loop):
            images = images.to(device)
            labels = labels.to(device)

            # Compute validation loss via forward pass (requires model.train())
            predictions = model(images)  # <-- extract tensor
            loss = loss_fn(predictions, labels)
            
            # Accumulate loss/dice
            total_loss += loss.item()
            total_dice += float(calculate_dice(predictions.detach(), labels.detach()))

            # Update loop
            loop.set_postfix(loss=loss.item())
        
    avg_loss = total_loss / len(loader) # calculate avg loss
    avg_dice = total_dice / len(loader) 

    # update tqdm loop
    loop.set_postfix(loss=loss.item())

    return avg_loss, avg_dice