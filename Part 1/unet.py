import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
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

# Hyperparameters

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_WORKERS = 1
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True


class UNet(nn.Module):
    # compareSP=True means we are running a normal UNet with no superpixels but want to compare it to one with OH
    # False means normal run comparing to GS
    def __init__(self, normal_UNET, input_channels, avg, fair, n, SP, compareOH, compareAvg, remember, fusionAdd, fusionCat, greyscale, k):
        super(UNet, self).__init__()

        self.normal_UNET = normal_UNET
        self.input_channels = input_channels
        self.avg = avg
        self.SP = SP
        self.compareOH = compareOH
        self.compareAvg = compareAvg
        self.greyscale = greyscale
        self.remember = remember
        self.fusionAdd = fusionAdd
        self.fusionCat = fusionCat
        self.fair = fair
        self.n = n
        self.k = k
        if self.remember == False:
            self.k = 0
        if self.fusionCat == True:
            t = 2
            cat = self.n
        else:
            t = 1
            cat = 0

        if self.SP:
            into = 6
        elif self.avg: 
            into = 3
        else:
            into = 1

        self.fusionconv1 = nn.Conv2d(into, self.n, kernel_size=3, padding=1)
        self.fusionconv2 = nn.Conv2d(self.n, self.n, kernel_size=3, padding=1)

        if self.normal_UNET:
            if self.remember:
                self.e11 = nn.Conv2d(9, 64, kernel_size=3, padding=1) # output: 570x570x64
                self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64
        
                # input: 284x284x64
                self.e21 = nn.Conv2d(70, 128, kernel_size=3, padding=1) # output: 282x282x128
                self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128
        
                # input: 140x140x128
                self.e31 = nn.Conv2d(134, 256, kernel_size=3, padding=1) # output: 138x138x256
                self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
                self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256
        
                # input: 68x68x256
                self.e41 = nn.Conv2d(262, 512, kernel_size=3, padding=1) # output: 66x66x512
                self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
                self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512
        
                # input: 32x32x512
                self.e51 = nn.Conv2d(518, 1024, kernel_size=3, padding=1) # output: 30x30x1024
                self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024
        
        
                # Decoder
                self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
                self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
                self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
                self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
                self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
                self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
                self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
                self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
                self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
                self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
                self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
                self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
                # Output layer
                self.outconv = nn.Conv2d(64, 1, kernel_size=1)
            else:
                # Encoder
                # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
                # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
                # -------
                # input: 572x572x3
                if self.SP:
                    j = 6
                else:
                    j=0
                self.e11 = nn.Conv2d(3+j, 64, kernel_size=3, padding=1) # output: 570x570x64
                self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64
        
                # input: 284x284x64
                self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
                self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128
        
                # input: 140x140x128
                self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
                self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
                self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256
        
                # input: 68x68x256
                self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
                self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
                self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512
        
                # input: 32x32x512
                self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
                self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024
        
        
                # Decoder
                self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
                self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
                self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
                self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
                self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
                self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
                self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
                self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
                self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
                self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
                self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
                self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
                # Output layer
                self.outconv = nn.Conv2d(64, 1, kernel_size=1)

        else:
            
            self.e11 = nn.Conv2d(self.input_channels, self.n, kernel_size=3, padding=1)
            self.e12 = nn.Conv2d(self.n, self.n, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
            self.e21 = nn.Conv2d((self.n+self.k)*t, (self.n+self.k)*2, kernel_size=3, padding=1)
            self.e22 = nn.Conv2d((self.n+self.k)*2, (self.n+self.k)*2, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    
            self.e31 = nn.Conv2d(((self.n+self.k)*2)+self.k, (((self.n+self.k)*2)+self.k)*2, kernel_size=3, padding=1)
            self.e32 = nn.Conv2d((((self.n+self.k)*2)+self.k)*2, (((self.n+self.k)*2)+self.k)*2, kernel_size=3, padding=1)
    
            self.upconv1 = nn.ConvTranspose2d((((self.n+self.k)*2)+self.k)*2, (((self.n+self.k)*2)+self.k), kernel_size=2, stride=2)
            self.d11 = nn.Conv2d((((self.n+self.k)*2)+self.k)*2 - self.k, (((self.n+self.k)*2)+self.k), kernel_size=3, padding=1)
            self.d12 = nn.Conv2d((((self.n+self.k)*2)+self.k), (((self.n+self.k)*2)+self.k), kernel_size=3, padding=1)
    
            self.upconv2 = nn.ConvTranspose2d((((self.n+self.k)*2)+self.k), self.n, kernel_size=2, stride=2)
            self.d21 = nn.Conv2d(2*self.n + cat, self.n, kernel_size=3, padding=1)
            self.d22 = nn.Conv2d(self.n, self.n, kernel_size=3, padding=1)
    
            self.outconv = nn.Conv2d(self.n, 1, kernel_size=1)

    def forward(self, x):
        if self.normal_UNET:
            if self.SP:
                sp_tensor = x[:, :6, :, :]
    
            # Encoder
            xe11 = F.relu(self.e11(x))
            xe12 = F.relu(self.e12(xe11))

            if self.fusionAdd:
                # pass sp_tensor through convolution
                f_sp = F.relu(self.fusionconv2(F.relu(self.fusionconv1(sp_tensor))))
                xe12 = xe12 + f_sp
            elif self.fusionCat:
                f_sp = F.relu(self.fusionconv2(F.relu(self.fusionconv1(sp_tensor))))
                xe12 = torch.cat((xe12, f_sp), dim=1) 

            xp1 = self.pool1(xe12)

            if self.remember:
                sp1 = self.pool1(sp_tensor)
                xp1_2 = torch.cat((xp1, sp1), dim=1)
                xe21 = F.relu(self.e21(xp1_2))
            else:
                xe21 = F.relu(self.e21(xp1))
                
            xe22 = F.relu(self.e22(xe21))
            xp2 = self.pool2(xe22)

            if self.remember:
                sp2 = self.pool2(sp1)
                xp2_2 = torch.cat((xp2, sp2), dim=1)
                xe31 = F.relu(self.e31(xp2_2))
            else:
                xe31 = F.relu(self.e31(xp2))
                
            xe32 = F.relu(self.e32(xe31))
            xp3 = self.pool3(xe32)

            if self.remember:
                sp3 = self.pool3(sp2)
                xp3_2 = torch.cat((xp3, sp3), dim=1)
                xe41 = F.relu(self.e41(xp3_2))
            else:
                xe41 = F.relu(self.e41(xp3))
                
            xe42 = F.relu(self.e42(xe41))
            xp4 = self.pool4(xe42)

            if self.remember:
                sp4 = self.pool4(sp3)
                xp4_2 = torch.cat((xp4, sp4), dim=1)
    
                xe51 = F.relu(self.e51(xp4_2))
            else:
                xe51 = F.relu(self.e51(xp4))
            xe52 = F.relu(self.e52(xe51))
            
            # Decoder
            xu1 = self.upconv1(xe52)
            xu11 = torch.cat([xu1, xe42], dim=1)
            xd11 = F.relu(self.d11(xu11))
            xd12 = F.relu(self.d12(xd11))
    
            xu2 = self.upconv2(xd12)
            xu22 = torch.cat([xu2, xe32], dim=1)
            xd21 = F.relu(self.d21(xu22))
            xd22 = F.relu(self.d22(xd21))
    
            xu3 = self.upconv3(xd22)
            xu33 = torch.cat([xu3, xe22], dim=1)
            xd31 = F.relu(self.d31(xu33))
            xd32 = F.relu(self.d32(xd31))
    
            xu4 = self.upconv4(xd32)
            xu44 = torch.cat([xu4, xe12], dim=1)
            xd41 = F.relu(self.d41(xu44))
            xd42 = F.relu(self.d42(xd41))
    
            # Output layer
            out = self.outconv(xd42)
    
            return out
        else:
            if self.fair or self.SP or self.avg:
                if self.fair:
                    if self.compareOH:
                        sp_tensor = x[:, :6, :, :]
                    elif self.compareAvg:
                        sp_tensor = x[:, :3, :, :]
                    else:
                        sp_tensor = x[:, :1, :, :]
                elif self.avg:
                    sp_tensor = x[:, :3, :, :]
                else:
                    sp_tensor = x[:, :1, :, :] if self.greyscale else x[:, :6, :, :]
    
                xe11 = F.relu(self.e11(x))
                xe12 = F.relu(self.e12(xe11))
    
                if self.fusionAdd:
                    # pass sp_tensor through convolution
                    f_sp = F.relu(self.fusionconv2(F.relu(self.fusionconv1(sp_tensor))))
                    xe12 = xe12 + f_sp
                elif self.fusionCat:
                    f_sp = F.relu(self.fusionconv2(F.relu(self.fusionconv1(sp_tensor))))
                    xe12 = torch.cat((xe12, f_sp), dim=1) 
                    
                    
                xp1 = self.pool1(xe12)
    
                if self.remember:
                    sp1 = self.pool1(sp_tensor)
                    xp1_2 = torch.cat((xp1, sp1), dim=1)
        
                    xe21 = F.relu(self.e21(xp1_2))
                else:
                    xe21 = F.relu(self.e21(xp1))
                    
                xe22 = F.relu(self.e22(xe21))
                xp2 = self.pool2(xe22)
    
                if self.remember:
                    sp2 = self.pool2(sp1)
                    xp2_2 = torch.cat((xp2, sp2), dim=1)
        
                    xe31 = F.relu(self.e31(xp2_2))
                else:
                    xe31 = F.relu(self.e31(xp2))
                xe32 = F.relu(self.e32(xe31))
    
            else:
                xe11 = F.relu(self.e11(x))
                xe12 = F.relu(self.e12(xe11))
                xp1 = self.pool1(xe12)
    
                xe21 = F.relu(self.e21(xp1))
                xe22 = F.relu(self.e22(xe21))
                xp2 = self.pool2(xe22)
    
                xe31 = F.relu(self.e31(xp2))
                xe32 = F.relu(self.e32(xe31))
    
            xu1 = self.upconv1(xe32)
            xu11 = torch.cat([xu1, xe22], dim=1)
            xd11 = F.relu(self.d11(xu11))
            xd12 = F.relu(self.d12(xd11))
    
            xu2 = self.upconv2(xd12)
            xu22 = torch.cat([xu2, xe12], dim=1)
            xd21 = F.relu(self.d21(xu22))
            xd22 = F.relu(self.d22(xd21))
    
            out = self.outconv(xd22)
            return out


class DatasetClass(Dataset):
    def __init__(self, dataset, image_dir, sp_dir, mask_dir, img_transform=None, sp_transform=None, SP=False, greyscale=False, fair=False, compareOH=False, compareAvg=False, avg=False):
        self.dataset = dataset
        self.SP = SP
        self.greyscale = greyscale
        self.fair = fair
        self.compareOH = compareOH
        self.compareAvg = compareAvg
        self.avg = avg
        self.image_dir = image_dir
        self.sp_dir = sp_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.sp_transform = sp_transform
        self.images = self._get_all_images(image_dir)
        if sp_dir:
            self.superpixelations = self._get_all_images(sp_dir)

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
        
        # Load regular image files (PNG, JPG, etc.)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.SP or self.avg:
            sp_tensor, combined_tensor = self._process_superpixels(index, image)
            image = combined_tensor.permute(1, 2, 0).numpy()  # Convert back to NumPy array
        else:
            # If SP is False and fair is True, concatenate the image with dummy tensor
            if self.fair:
                img_tensor = torch.tensor(image).permute(2, 0, 1)  # Convert image to tensor (C, H, W)
    
                # Create a dummy tensor with 6 channels if compareOH is True, or 1 channel if compareOH is False
                if self.compareOH:
                    dummy_channels = torch.zeros(6, img_tensor.shape[1], img_tensor.shape[2])  # (1, H, W)
                elif self.compareAvg:
                    dummy_channels = torch.zeros(3, img_tensor.shape[1], img_tensor.shape[2])  # (3, H, W)
                else:
                    dummy_channels = torch.zeros(1, img_tensor.shape[1], img_tensor.shape[2])  # (6, H, W)
    
                # Concatenate the dummy channels with the image tensor along the channel dimension (dim=0)
                img_tensor = torch.cat((dummy_channels, img_tensor), dim=0)
    
                # Convert the tensor back to a NumPy array for transformations
                image = img_tensor.permute(1, 2, 0).numpy()
                
                augmentations = self.sp_transform(image=image, mask=self._load_mask(index))
                
                return augmentations['image'], augmentations['mask']
            else:
                image = np.array(image, dtype='uint8')  # Ensure the image is a NumPy array
    
        mask = self._load_mask(index)

        augmentations = self.sp_transform(image=image, mask=mask) if (self.SP or self.avg) else self.img_transform(image=image, mask=mask)
        return augmentations['image'], augmentations['mask']

    def _process_superpixels(self, index, image):
        if not self.SP and not self.avg:
            return None, torch.tensor(image).permute(2, 0, 1)

        sp_path = os.path.join(self.sp_dir, self.superpixelations[index])
        if self.avg:
            sp = np.array(Image.open(sp_path).convert('RGB'))
        else:
            sp = np.array(Image.open(sp_path).convert('L'))
        sp_tensor = torch.tensor(sp, dtype=torch.long)
    
        if self.greyscale:
            img = torch.tensor(image).permute(2, 0, 1)
            sp_tensor = sp_tensor.unsqueeze(0).float()
            sp_tensor_resized = F.interpolate(sp_tensor.unsqueeze(0), size=(img.shape[1], img.shape[2]), mode='nearest')
            combined_tensor = torch.cat((sp_tensor_resized.squeeze(0), img), dim=0)
        elif self.avg:
            # Check if the image is grayscale (1 channel)
            if len(image.shape) == 2 or image.shape[2] == 1:
                # Convert grayscale image to RGB by repeating the channel 3 times
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

            img = torch.tensor(image).permute(2, 0, 1)
            sp_tensor = sp_tensor.permute(2,0,1).float()
            # Resize sp_tensor to match img's height and width
            sp_tensor = F.interpolate(sp_tensor.unsqueeze(0), size=(img.shape[1], img.shape[2]), mode='bilinear', align_corners=False).squeeze(0)
            combined_tensor = torch.cat((sp_tensor.squeeze(0), img), dim=0)
        elif self.SP:
            sp_tensor = F.one_hot(sp_tensor).permute(2, 0, 1).float()
            
            if sp_tensor.shape[0] == 5:
                # add a channel of 0's so all tensors have 9 channels at the end
                zs = np.zeros((1, sp_tensor.shape[1], sp_tensor.shape[2]))
                sp_tensor = torch.tensor(np.concatenate((sp_tensor, zs), axis=0))
            elif sp_tensor.shape[0] == 4:
                # Add 2 channels of 0's so all tensors have 9 channels at the end
                zs = np.zeros((2, sp_tensor.shape[1], sp_tensor.shape[2]))
                sp_tensor = torch.tensor(np.concatenate((sp_tensor, zs), axis=0))
            elif sp_tensor.shape[0] == 3:
                # Add 3 channels of 0's so all tensors have 9 channels at the end
                zs = np.zeros((3, sp_tensor.shape[1], sp_tensor.shape[2]))
                sp_tensor = torch.tensor(np.concatenate((sp_tensor, zs), axis=0))
            elif sp_tensor.shape[0] == 2:
                # Add 4 channels of 0's so all tensors have 9 channels at the end
                zs = np.zeros((4, sp_tensor.shape[1], sp_tensor.shape[2]))
                sp_tensor = torch.tensor(np.concatenate((sp_tensor, zs), axis=0))
            elif sp_tensor.shape[0] == 1:
                # Add 5 channels of 0's so all tensors have 9 channels at the end
                zs = np.zeros((5, sp_tensor.shape[1], sp_tensor.shape[2]))
                sp_tensor = torch.tensor(np.concatenate((sp_tensor, zs), axis=0))

            # Check if the image is grayscale (1 channel)
            if len(image.shape) == 2 or image.shape[2] == 1:
                # Convert grayscale image to RGB by repeating the channel 3 times
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

            img = torch.tensor(image).permute(2, 0, 1)
            
            # Resize the superpixel tensor to match the image dimensions
            sp_tensor_resized = F.interpolate(sp_tensor.unsqueeze(0), size=(img.shape[1], img.shape[2]), mode='nearest')
            
            # Concatenate along the channel dimension (dim=0)
            combined_tensor = torch.cat((sp_tensor_resized.squeeze(0), img), dim=0)
    
        return sp_tensor, combined_tensor

    def _load_mask(self, index):
        img_path = self.images[index]
    
        # Check if the path contains 'train_images/' or 'test_images/'
        if 'train_images/' in img_path:
            relative_image_path = img_path.split('train_images/')[1]
        elif 'test_images/' in img_path:
            relative_image_path = img_path.split('test_images/')[1]
        # elif 'grid_train/' in img_path:
        #     relative_image_path = img_path.split('grid_train/')[1]
        # elif 'grid_test/' in img_path:
        #     relative_image_path = img_path.split('grid_test/')[1]
        else:
            raise ValueError(f"Image path does not contain 'train_images/' or 'test_images/'. Found: {img_path}")
    
        # Replace the '.jpg' extension with '.png' for the mask
        if self.dataset == "Carvana":
            relative_mask_path = relative_image_path.replace('.jpg', '_mask.gif')
        else:
            relative_mask_path = relative_image_path.replace('.jpg', '.png')
    
        # Join the mask directory with the relative path
        mask_path = os.path.join(self.mask_dir, relative_mask_path)
    
        # Check if the mask file exists
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found for image: {mask_path}")
    
        
        # Load image mask and convert to grayscale float32
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask <= 127.0] = 0.0
        mask[mask > 127.0] = 1.0 # Normalize binary masks to 0 and 1
        return mask


def save_predictions_as_imgs(loader, model, folder, device):
    model.eval()
    os.makedirs(folder, exist_ok=True)
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = (torch.sigmoid(model(x)) > 0.5).float()
        save_image(preds, f'{folder}/pred_{idx}.png')
        save_image(y.unsqueeze(1), f'{folder}/truth_{idx}.png')


def train_model(model, train_loader, optimizer, scaler, device):
    loop = tqdm(train_loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.float().unsqueeze(1).to(device)
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = F.binary_cross_entropy_with_logits(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())


# def test_model(model, test_loader, criterion):
#     epoch_losses = []
#     model.eval()
#     loop = tqdm(test_loader)
#     for batch_idx, (data, targets) in enumerate(loop):
#           data = data.to(device=DEVICE)
#           targets = targets.float().unsqueeze(1).to(device=DEVICE)
#           preds = model(data)
#           loss = criterion(preds, targets)
#           epoch_losses.append(loss.detach().cpu().numpy())
    
#     print(f"average losses: {np.mean(epoch_losses)}")


def setup_transforms(SP, GS, avg, fair, compareOH, compareAvg):
    sp_train_transform = None
    sp_test_transform = None
    if fair:
        if compareAvg:
            sp_train_transform = A.Compose([
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(mean=[0.0] * 6, std=[1.0] * 6, max_pixel_value=255.0),
                ToTensorV2(),
            ])
            sp_test_transform = A.Compose([
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(mean=[0.0] * 6, std=[1.0] * 6, max_pixel_value=255.0),
                ToTensorV2(),
            ])
        elif compareOH:
            sp_train_transform = A.Compose([
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(mean=[0.0] * 9, std=[1.0] * 9, max_pixel_value=255.0),
                ToTensorV2(),
            ])
            sp_test_transform = A.Compose([
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(mean=[0.0] * 9, std=[1.0] * 9, max_pixel_value=255.0),
                ToTensorV2(),
            ])
        else: # grey scale compare
            sp_train_transform = A.Compose([
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(mean=[0.0] * 4, std=[1.0] * 4, max_pixel_value=255.0),
                ToTensorV2(),
            ])
            sp_test_transform = A.Compose([
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(mean=[0.0] * 4, std=[1.0] * 4, max_pixel_value=255.0),
                ToTensorV2(),
            ])
    elif GS:
        sp_train_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(mean=[0.0] * 4, std=[1.0] * 4, max_pixel_value=255.0),
            ToTensorV2(),
        ])
        sp_test_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0] * 4, std=[1.0] * 4, max_pixel_value=255.0),
            ToTensorV2(),
        ])
    elif SP:
        sp_train_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(mean=[0.0] * 9, std=[1.0] * 9, max_pixel_value=255.0),
            ToTensorV2(),
        ])
        sp_test_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0] * 9, std=[1.0] * 9, max_pixel_value=255.0),
            ToTensorV2(),
        ])
    elif avg:
        sp_train_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(mean=[0.0] * 6, std=[1.0] * 6, max_pixel_value=255.0),
            ToTensorV2(),
        ])
        sp_test_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0] * 6, std=[1.0] * 6, max_pixel_value=255.0),
            ToTensorV2(),
        ])
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0.0] * 3, std=[1.0] * 3, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    test_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0] * 3, std=[1.0] * 3, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    return sp_train_transform, sp_test_transform, train_transform, test_transform
