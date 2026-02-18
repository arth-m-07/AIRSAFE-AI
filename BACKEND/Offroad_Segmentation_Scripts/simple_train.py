import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ---------------- Dataset ---------------- #

class OffroadDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0

        mask = cv2.imread(mask_path, 0)

        image = torch.tensor(image).permute(2,0,1).float()
        mask = torch.tensor(mask).long()

        return image, mask

# ---------------- Paths ---------------- #

train_images = "../Offroad_Segmentation_Training_Dataset/train/Color_Images"
train_masks  = "../Offroad_Segmentation_Training_Dataset/train/Segmentation"

val_images = "../Offroad_Segmentation_Training_Dataset/val/Color_Images"
val_masks  = "../Offroad_Segmentation_Training_Dataset/val/Segmentation"

train_dataset = OffroadDataset(train_images, train_masks)
val_dataset   = OffroadDataset(val_images, val_masks)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_loader   = DataLoader(val_dataset, batch_size=4)

# ---------------- Model ---------------- #

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=10,
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------- Training ---------------- #
# ---------------- IoU Function ---------------- #

def compute_iou(preds, masks, num_classes=10):
    preds = torch.argmax(preds, dim=1)

    ious = []

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        mask_cls = (masks == cls)

        intersection = (pred_cls & mask_cls).sum().item()
        union = (pred_cls | mask_cls).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0

    return sum(ious) / len(ious)


# ---------------- Training Loop ---------------- #
best_iou = 0.0

for epoch in range(5):

    # ---- Training ----
    

    model.train()
    train_loss = 0

    for batch_idx, (imgs, masks) in enumerate(train_loader):

        imgs = imgs.to(device)
        masks = masks.to(device)

        outputs = model(imgs)
        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss = 0
    total_iou = 0

    with torch.no_grad():
        for imgs, masks in val_loader:

            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, masks)

            val_loss += loss.item()
            total_iou += compute_iou(outputs.cpu(), masks.cpu())


    val_loss /= len(val_loader)
    mean_iou = total_iou / len(val_loader)

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val IoU: {mean_iou:.4f}")
    print("-"*30)

    # Save best model
    if mean_iou > best_iou:
        best_iou = mean_iou
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved!")

print("Training Complete!")

