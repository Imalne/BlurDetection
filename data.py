from torchvision import transforms
from PIL import Image
import random

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np

import glob

train_aug_paired = A.Compose([
    A.RandomResizedCrop((256,256), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=(-90, 90), p=0.5),
])

train_aug_input = A.Compose([
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=(3,7), sigma_limit=(0.1,2.0), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
])

val_aug = A.Compose([
    A.Resize(256,256),
])


class BlurDataset(Dataset):
    def __init__(self, img_dir, mask_dir, aug=None, aug_for_input=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.aug = aug
        self.aug_for_input = aug_for_input

        self.images = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.masks  = sorted(glob.glob(os.path.join(mask_dir, "*")))

        assert len(self.images) == len(self.masks), "Image and mask count mismatch!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        mask_path = self.masks[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"))  # grayscale mask
        if min(image.shape[0:2]) < 256:
            factor = 256 / min(image.shape[0:2])
            image = cv2.resize(image, (int(image.shape[1]*factor), int(image.shape[0]*factor)), interpolation=cv2.INTER_LINEAR)
            mask  = cv2.resize(mask,  (int(mask.shape[1]*factor),  int(mask.shape[0]*factor)),  interpolation=cv2.INTER_NEAREST)
        

        

        if self.aug:
            out = self.aug(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
            image = self.aug_for_input(image=image)["image"] if self.aug_for_input else image
        
        image = torch.from_numpy(image).permute(2,0,1).float()
        mask  = torch.from_numpy(mask).long()

        mask[mask <=128] = 0     # clear labeled as 0
        if "motion" in mask_path.lower():
            mask[mask> 128] = 1 # motion blur labeled as 2
        elif "out_of_focus" in mask_path.lower():
            mask[mask > 128] = 2 # defocus blur labeled as 1
        else:
            mask[mask > 128] = 2 # defocus blur labeled as 1
            # raise ValueError("Mask filename must contain 'motion' or 'defocus' to indicate blur type.")
        
        # print(image.shape)
        # print(np.max(mask.numpy()), np.min(mask.numpy()))
        return image, mask


