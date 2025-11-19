import torch
from utils.get_maskrcnn_model import get_maskrcnn_model
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

state_path = "./model/model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_maskrcnn_model()
load_result = model.load_state_dict(
    torch.load(state_path, 
    map_location=device
))  

print(load_result)
model.eval()
model.to(device)

img_name = "000000030.png"
img_path = "./data/img/" + img_name
mask_path = "./data/mask/" + img_name

# Load image
image = Image.open(img_path).convert("RGB")
maskk = Image.open(mask_path).convert("RGB")

transform = transforms.ToTensor()
image_tensor = transform(image).to(device)

# Mask R-CNN expects a list of images
images = [image_tensor]

with torch.no_grad():
    outputs = model(images)  # list of 

output = outputs[0]
boxes = output['boxes']       # [N,4] bounding boxes
labels = output['labels']     # [N] predicted class labels
scores = output['scores']     # [N] confidence scores
masks = output['masks']       # [N,1,H,W] predicted masks

# Filter predictions by score threshold
threshold = 0.7
keep = scores > threshold

boxes = boxes[keep]
labels = labels[keep]
masks = masks[keep]
scores = scores[keep]

print(f"masks: {masks}")
print(f"boxes: {boxes}")
print(f"scores: {scores}")

# Convert image tensor to numpy for plotting
img_np = image_tensor.cpu().permute(1, 2, 0).numpy()

"""for i in range(masks.shape[0]):
    mask = masks[i, 0].cpu().numpy()      # [H, W]
    mask_binary = mask > 0.5              # threshold to binary
    plt.imshow(mask_binary, cmap='gray')
    plt.title(f"Mask {i+1}")
    plt.axis('off')
    plt.show()"""


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(img_np)

for i in range(len(boxes)):
    # Draw bounding box
    x_min, y_min, x_max, y_max = boxes[i].cpu().numpy()
    rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor='blue',
        facecolor='none'
    )
    ax.add_patch(rect)
    
    # Overlay mask
    mask_i = masks[i, 0].cpu().numpy() > 0.5
    ax.imshow(mask_i, alpha=0.5)
    
    # Add score text
    ax.text(x_min, y_min-5, f"{scores[i]:.2f}", color='red', fontsize=12, weight='bold')
plt.title(img_path)

plt.axis('off')

"""plt.figure(2)
plt.imshow(maskk)
plt.title(img_name)"""
plt.show()