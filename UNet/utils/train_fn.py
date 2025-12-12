import torch
from tqdm import tqdm
from utils.calculate_dice import calculate_dice

def train_fn(device, loader, model, optimizer, loss_fn):
    model.train()  # make sure model is in training mode
    loop = tqdm(loader)
    
    # Track loss and dice score
    total_loss = 0
    total_dice = 0

    # Main loop
    for batch_idx, (images, labels, filenames) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)


        ## FP32 Full Precision
        # forward pass
        predictions = model(images)  # <-- extract tensor
        loss = loss_fn(predictions, labels)
    
        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        max_norm = 1.0 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Step optimizer
        optimizer.step()
        total_loss += loss.item()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        total_dice += float(calculate_dice(predictions.detach(), labels.detach()))

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader) 

    return avg_loss, avg_dice