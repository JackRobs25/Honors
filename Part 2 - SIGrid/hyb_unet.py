import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
from torchvision.transforms import functional as TF  # Correct import for resizing
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy import ndimage
from skimage.segmentation import slic
from skimage.util import img_as_float
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import time
import datetime
import uuid
import json
from sklearn.metrics import jaccard_score
from skimage.measure import regionprops
import math 


class UNet(nn.Module):
    # compareSP=True means we are running a normal UNet with no superpixels but want to compare it to one with OH
    # False means normal run comparing to GS
    def __init__(self, input_channels, reduced_performance):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.reduced_performance = reduced_performance
        
        if self.reduced_performance:
            self.e11 = nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1)
            self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

            self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.d11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
            self.d12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

            self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.d21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
            self.d22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

            self.outconv = nn.Conv2d(64, 1, kernel_size=1)
        else:
            self.e11 = nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1)
            self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

            self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.d11 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            self.d12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

            self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.d21 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
            self.d22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

            self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.d31 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
            self.d32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

            self.outconv = nn.Conv2d(64, 1, kernel_size=1)

    def crop_to_match(self, larger_tensor, reference_tensor):
        """
        Center-crops the larger_tensor to match the spatial size of reference_tensor.
        """
        _, _, h, w = reference_tensor.size()
        _, _, H, W = larger_tensor.size()
        dh, dw = (H - h) // 2, (W - w) // 2
        return larger_tensor[:, :, dh:dh + h, dw:dw + w]
    
    def pad_to_multiple(self, x, multiple=8):
        _, _, h, w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        return F.pad(x, (0, pad_w, 0, pad_h)), h, w

    def crop_to_original(self, x, h, w):
        return x[:, :, :h, :w]

    def forward(self, x):
        x, orig_h, orig_w = self.pad_to_multiple(x, multiple=8)
        if self.reduced_performance:
            xe11 = F.relu(self.e11(x))
            xe12 = F.relu(self.e12(xe11))
            xp1 = self.pool1(xe12)

            xe21 = F.relu(self.e21(xp1))
            xe22 = F.relu(self.e22(xe21))
            xp2 = self.pool2(xe22)

            xe31 = F.relu(self.e31(xp2))
            xe32 = F.relu(self.e32(xe31))

            xu1 = self.upconv1(xe32)
            xe22 = self.crop_to_match(xe22, xu1)
            xu11 = torch.cat([xu1, xe22], dim=1)
            xd11 = F.relu(self.d11(xu11))
            xd12 = F.relu(self.d12(xd11))

            xu2 = self.upconv2(xd12)
            xe12 = self.crop_to_match(xe12, xu2)
            xu22 = torch.cat([xu2, xe12], dim=1)
            xd21 = F.relu(self.d21(xu22))
            xd22 = F.relu(self.d22(xd21))

            out = self.outconv(xd22)
        else:
            xe11 = F.relu(self.e11(x))
            xe12 = F.relu(self.e12(xe11))
            xp1 = self.pool1(xe12)

            xe21 = F.relu(self.e21(xp1))
            xe22 = F.relu(self.e22(xe21))
            xp2 = self.pool2(xe22)

            xe31 = F.relu(self.e31(xp2))
            xe32 = F.relu(self.e32(xe31))
            xp3 = self.pool3(xe32)

            xe41 = F.relu(self.e41(xp3))
            xe42 = F.relu(self.e42(xe41))

            xu1 = self.upconv1(xe42)
            xe32 = self.crop_to_match(xe32, xu1)
            xu11 = torch.cat([xu1, xe32], dim=1)
            xd11 = F.relu(self.d11(xu11))
            xd12 = F.relu(self.d12(xd11))

            xu2 = self.upconv2(xd12)
            xe22 = self.crop_to_match(xe22, xu2)
            xu22 = torch.cat([xu2, xe22], dim=1)
            xd21 = F.relu(self.d21(xu22))
            xd22 = F.relu(self.d22(xd21))

            xu3 = self.upconv3(xd22)
            xe12 = self.crop_to_match(xe12, xu3)
            xu33 = torch.cat([xu3, xe12], dim=1)
            xd31 = F.relu(self.d31(xu33))
            xd32 = F.relu(self.d32(xd31))

            out = self.outconv(xd32)
        
        out = self.crop_to_original(out, orig_h, orig_w)
        return out


class DatasetClass(Dataset):
    def __init__(self, dataset, n_segments, compactness, SIGrid_channels, og_image_dir, image_dir, mask_dir, seg_dir, img_transform=None, avg_color=True, area=False, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=False):
        self.dataset = dataset
        self.n_segments = n_segments
        self.compactness = compactness
        self.SIGrid_channels = SIGrid_channels
        self.og_image_dir = og_image_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.seg_dir = seg_dir 
        self.img_transform = img_transform
        self.avg_color = avg_color
        self.area = area
        self.width = width
        self.height = height 
        self.compac = compac
        self.solidity = solidity
        self.eccentricity = eccentricity
        self.hu = hu
        self.og_images = self._get_all_images(og_image_dir)
        self.images = self._get_all_images(image_dir)

    def _get_all_images(self, directory):
        image_paths = []
        
        if self.dataset == "CUB":
            # Walk through all subdirectories
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.npy')):
                        image_paths.append(os.path.join(root, file))
        else:
            # No subdirectories, images are in the initial folder
            for file in os.listdir(directory):
                if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.npy')):
                    image_paths.append(os.path.join(directory, file))
                    
        return image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        
        # Load .npy matrix file
        image = np.load(img_path)

        if image.ndim == 2:  # If grayscale, add a channel dimension
            image = np.expand_dims(image, axis=-1)
        elif image.ndim == 3 and image.shape[-1] not in [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:  # Ensure channel is last
            raise ValueError(f"Unexpected shape {image.shape} for .npy file: {img_path}")

        # select channels based off of selection for image
        image = self.extract_channels(image, self.avg_color, self.area, self.width, self.height, self.compac, self.solidity, self.eccentricity, self.hu)

        # Load mask and segmentation
        mask = self._load_mask(index)

        # Apply augmentations to image and mask
        augmentations = self.img_transform(image=image, mask=mask)
        transformed_sig = augmentations['image']
        transformed_sig_mask = augmentations['mask']

        return transformed_sig, transformed_sig_mask, index

    def extract_channels(self, image, avg_color, area, width, height,
                        compac, solidity, eccentricity, hu):
        """
        Extracts selected channels from the image based on provided booleans.
        Assumes channel ordering: [avg_color (3), area, width, height, compac, solidity, eccentricity, hu (7)]
        Total: 3 + 1*6 + 7 = 16 channels
        """
        selected = []
        idx = 0

        # avg_color (3 channels)
        if avg_color:
            selected.extend(range(idx, idx + 3))
        idx += 3

        # Individual scalar features (each 1 channel)
        for flag in [area, width, height, compac, solidity, eccentricity]:
            if flag:
                selected.append(idx)
            idx += 1

        # hu moments (7 channels)
        if hu:
            selected.extend(range(idx, idx + 7))

        # Select along the last axis (channel-last format)
        return image[:, :, selected]


    def _load_mask(self, index):
        img_path = self.images[index]

        train_path = f"{self.n_segments}_{self.compactness}_grid_train_16"
        test_path =  f"{self.n_segments}_{self.compactness}_grid_test_16"

    
        # Check if the path contains 'train_images/' or 'test_images/'
        if 'train_images/' in img_path:
            relative_image_path = img_path.split('train_images/')[1]
        elif 'test_images/' in img_path:
            relative_image_path = img_path.split('test_images/')[1]
        elif train_path in img_path:
            relative_image_path = img_path.split(train_path)[1]
        elif test_path in img_path:
            relative_image_path = img_path.split(test_path)[1]
        else:
            raise ValueError(f"Image path does not contain 'train_images/' or 'test_images/'. Found: {img_path}")
    
        # Replace the '.jpg' extension with '.png' for the mask
        if self.dataset == "Carvana":
            relative_mask_path = relative_image_path.replace('.jpg', '_mask.gif')
        else:
            relative_mask_path = relative_image_path.replace('.jpg', '.png')
    
        # Join the mask directory with the relative path
        mask_path = os.path.join(self.mask_dir, relative_mask_path.lstrip('/'))
    
        # Check if the mask file exists
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found for image: {mask_path}")
    
        mask = np.load(mask_path).astype(np.float32)
        return mask

def train_model(model, train_loader, optimizer, scaler, device):
    # for epoch in range(epochs):
    loop = tqdm(train_loader)
    for batch_idx, (data, targets, index) in enumerate(loop):
        data, targets = data.to(device), targets.float().unsqueeze(1).to(device)
        with torch.cuda.amp.autocast():
            predictions = model(data)
            M = (targets != -1) # where mask_processor is the grid_mask_train
            GT = targets
            loss = F.binary_cross_entropy_with_logits(predictions[M], GT[M])

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())
    return loss.detach().cpu().numpy()

def setup_transforms(channels):
    train_transform = A.Compose([
        # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0.0] * channels, std=[1.0] * channels, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    test_transform = A.Compose([
        # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0] * channels, std=[1.0] * channels, max_pixel_value=255.0),
        ToTensorV2(),
    ])

    return train_transform, test_transform
