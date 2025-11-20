import torch
from tqdm import tqdm
#from utils.dice_score import dice_score

def train_fn(device, loader, model, optimizer, scaler):
    model.train()  # make sure model is in training mode
    loop = tqdm(loader)
    
    # Track loss and dice score
    total_loss = 0
    total_dice = 0

    # Main loop
    for batch_idx, (images, targets, filenames) in enumerate(loop):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        # forward pass
        with torch.amp.autocast('cuda'):
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
    
        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient Clipping
        max_norm = 1.0 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Scale after clipping
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader) 

    return avg_loss#, avg_dice