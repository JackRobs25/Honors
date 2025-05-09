import json
import torch
import os
from datetime import datetime
from unet import UNet, DatasetClass, setup_transforms, train_model, save_predictions_as_imgs
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import jaccard_score
import torch.nn as nn
import numpy as np
import uuid
import time
import torch.nn.functional as F


DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# DATASET_DIR = '/home/jroberts2/Carvana/carvana-image-masking-challenge/'
WORKING_DIR = '/Users/jroberts2/Jeova IS/Honors Project/Part 1/RESULTS'
#############################################
BASE_DIR = '/Users/jroberts2/Jeova IS/Honors Project/'
#############################################
CUB_TRAIN_IMG_DIR = BASE_DIR + 'CUB/train_images/'
CUB_TRAIN_MASK_DIR = BASE_DIR + 'CUB/train_masks/'

CUB_VAL_IMG_DIR = BASE_DIR + 'CUB/test_images/'
CUB_VAL_MASK_DIR = BASE_DIR + 'CUB/test_masks/'

CUB_SP_TRAIN_DIR_300_10 = BASE_DIR + 'CUB/sp_train_png'
CUB_SP_VAL_DIR_300_10 = BASE_DIR + 'CUB/sp_test_png'

CUB_AVG_TRAIN_DIR = BASE_DIR + 'CUB/avg_train_png'
CUB_AVG_TEST_DIR = BASE_DIR + 'CUB/avg_test_png'

#############################################
ECSSD_TRAIN_IMG_DIR = BASE_DIR + 'ECSSD/train_images/'
ECSSD_TRAIN_MASK_DIR = BASE_DIR + 'ECSSD/train_masks/'

ECSSD_VAL_IMG_DIR = BASE_DIR + 'ECSSD/test_images/'
ECSSD_VAL_MASK_DIR = BASE_DIR + 'ECSSD/test_masks/'

ECSSD_SP_TRAIN_DIR_300_10 = BASE_DIR + 'ECSSD/sp_train_png'
ECSSD_SP_VAL_DIR_300_10 = BASE_DIR + 'ECSSD/sp_test_png'

ECSSD_AVG_TRAIN_DIR = BASE_DIR + 'ECSSD/avg_train_png'
ECSSD_AVG_TEST_DIR = BASE_DIR + 'ECSSD/avg_test_png'

#############################################
CAR_TRAIN_IMG_DIR = BASE_DIR + 'Carvana/train_images/'
CAR_TRAIN_MASK_DIR = BASE_DIR + 'Carvana/train_masks/'

CAR_VAL_IMG_DIR = BASE_DIR + 'Carvana/test_images/'
CAR_VAL_MASK_DIR = BASE_DIR + 'Carvana/test_masks/'

CAR_SP_TRAIN_DIR_300_10 = BASE_DIR + 'Carvana/sp_train_png'
CAR_SP_VAL_DIR_300_10 = BASE_DIR + 'Carvana/sp_test_png'

CAR_AVG_TRAIN_DIR = BASE_DIR + 'Carvana/avg_train_png'
CAR_AVG_TEST_DIR = BASE_DIR + 'Carvana/avg_test_png'





class Experiment:
    def __init__(self, normal_UNET, dataset, avg, sp, greyscale, n, k, fair, compareOH, compareAvg, fusionAdd, fusionCat, remember, learning_rate, num_epochs, batch_size, device):
        self.normal_UNET = normal_UNET
        self.dataset = dataset
        self.model = None
        self.avg = avg
        self.sp = sp # true if adding OH or GS superpixels to input
        self.greyscale = greyscale # true if GS superpixels (can only be true if sp is true)
        self.n = n 
        self.k = k
        self.fair = fair # For non-sp test case, can only be true if sp is false.
        self.compareOH = compareOH # only true if fair is true, means we are comparing normal to onehot so add 6 dummy channels
        self.compareAvg = compareAvg
        self.fusionAdd = fusionAdd
        self.fusionCat = fusionCat
        self.remember = remember # true if passing sp into each encoder layer rather than just input
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.start_time = 0
        self.end_time = 0
        

        # Initialize results
        self.training_pixel_accuracies = []
        self.testing_pixel_accuracies = []
        self.training_iou = []
        self.testing_iou = []
        self.test_accuracy = -1
        self.test_losses = -1

    def run_experiment(self):
        # Setup the UNet model
        in_channels = 9 if self.sp else 3
        if self.fair:
            if self.compareOH:
                in_channels = 9
            elif self.compareAvg:
                in_channels = 6
            else:
                in_channels = 4
        if self.greyscale:
            in_channels = 4
        if self.avg:
            in_channels = 6

        if self.dataset == 'CUB':
            train_img_dir = CUB_TRAIN_IMG_DIR
            train_mask_dir = CUB_TRAIN_MASK_DIR
            val_img_dir = CUB_VAL_IMG_DIR
            val_mask_dir = CUB_VAL_MASK_DIR
            if self.sp or self.compareOH:
                sp_train_dir = CUB_SP_TRAIN_DIR_300_10
                sp_test_dir = CUB_SP_VAL_DIR_300_10
            elif self.avg or self.compareAvg:
                sp_train_dir = CUB_AVG_TRAIN_DIR
                sp_test_dir = CUB_AVG_TEST_DIR
            else:
                sp_train_dir = None
                sp_test_dir = None
        
        elif self.dataset == 'ECSSD':
            train_img_dir = ECSSD_TRAIN_IMG_DIR
            train_mask_dir = ECSSD_TRAIN_MASK_DIR
            val_img_dir = ECSSD_VAL_IMG_DIR
            val_mask_dir = ECSSD_VAL_MASK_DIR
            if self.sp or self.compareOH:
                sp_train_dir = ECSSD_SP_TRAIN_DIR_300_10
                sp_test_dir = ECSSD_SP_VAL_DIR_300_10
            elif self.avg or self.compareAvg:
                sp_train_dir = ECSSD_AVG_TRAIN_DIR
                sp_test_dir = ECSSD_AVG_TEST_DIR
            else:
                sp_train_dir = None
                sp_test_dir = None

        elif self.dataset == 'Carvana':
            train_img_dir = CAR_TRAIN_IMG_DIR
            train_mask_dir = CAR_TRAIN_MASK_DIR
            val_img_dir = CAR_VAL_IMG_DIR
            val_mask_dir = CAR_VAL_MASK_DIR
            if self.sp or self.compareOH:
                sp_train_dir = CAR_SP_TRAIN_DIR_300_10
                sp_test_dir = CAR_SP_VAL_DIR_300_10
            elif self.avg or self.compareAvg:
                sp_train_dir = CAR_AVG_TRAIN_DIR
                sp_test_dir = CAR_AVG_TEST_DIR 
            else:
                sp_train_dir = None
                sp_test_dir = None

        else:
            raise Exception("invalid dataset. Choose one of 'CUB', ECSSD', or 'Carvana'")
                    
        self.model = UNet(self.normal_UNET, in_channels, self.avg, self.fair, self.n, self.sp, self.compareOH, self.compareAvg, self.remember, self.fusionAdd, self.fusionCat, self.greyscale, self.k).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = F.binary_cross_entropy_with_logits #nn.BCEWithLogitsLoss()
        scaler = torch.cuda.amp.GradScaler()

        # Setup data transformations and loaders
        sp_train_transform, sp_test_transform, train_transform, test_transform = setup_transforms(self.sp, self.greyscale, self.avg, self.fair, self.compareOH, self.compareAvg)
        train_ds = DatasetClass(dataset=self.dataset, image_dir=train_img_dir, sp_dir=sp_train_dir, 
                                  mask_dir=train_mask_dir, img_transform=train_transform, 
                                  sp_transform=sp_train_transform, SP=self.sp, greyscale=self.greyscale, fair=self.fair, compareOH=self.compareOH,compareAvg=self.compareAvg, avg=self.avg)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, num_workers=1, pin_memory=True, shuffle=True)

        test_ds = DatasetClass(dataset=self.dataset, image_dir=val_img_dir, sp_dir=sp_test_dir, 
                                 mask_dir=val_mask_dir, img_transform=test_transform, 
                                 sp_transform=sp_test_transform, SP=self.sp, greyscale=self.greyscale, fair=self.fair, compareOH=self.compareOH, compareAvg=self.compareAvg, avg=self.avg)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, num_workers=1, pin_memory=True, shuffle=False)

        # Run training and store results
        self.start_time = time.time()
        for epoch in range(self.num_epochs):
            train_model(self.model, train_loader, optimizer, scaler, self.device)
            # Save training and testing accuracies after each epoch
            # train_iou, train_pixel_acc = self._check_accuracy(train_loader, self.model)
            # test_iou, test_pixel_acc = self._check_accuracy(test_loader, self.model)
            # self.training_pixel_accuracies.append(train_pixel_acc)
            # self.training_iou.append(train_iou)
            # self.testing_pixel_accuracies.append(test_pixel_acc)
            # self.testing_iou.append(test_iou)
            # print(f"Epoch {epoch + 1}/{self.num_epochs}, Training Pixel Accuracy: {train_pixel_acc:.2f}, Testing Pixel Accuracy: {test_pixel_acc:.2f}, Training IoU: {train_iou:.2f}, Testing IoU: {test_iou:.2f}")

        self.end_time = time.time()
        # self.test_losses = self._test_model(test_loader, model, criterion)
        test_loss, test_iou, test_pixel_accuracy = self._test_model(test_loader, self.model, criterion)
        print(f"Epoch {epoch + 1}/{self.num_epochs}, Avg loss: {test_loss:.2f}, Testing Pixel Accuracy: {test_pixel_accuracy:.2f}, Testing IoU: {test_iou:.2f}")
        self.test_losses = test_loss
        self.test_accuracy = test_pixel_accuracy
        self.test_iou = test_iou
        # Save final predictions as images
        save_predictions_as_imgs(test_loader, self.model, folder='/Users/jroberts2/Jeova IS/Honors Project/Part 1/' + self.dataset + '/saved_images', device=self.device)
        


    def save_data(self):
        # Prepare experiment data
        experiment_data = {
            'Dataset': self.dataset,
            'Normal UNet': self.normal_UNET,
            'SP': self.sp,
            'Greyscale': self.greyscale,
            'Average': self.avg,
            'Remember': self.remember,
            "Fair": self.fair,
            "FusionAdd": self.fusionAdd,
            "FusionCat": self.fusionCat,
            'Normal compared to OH': self.compareOH,
            'Normal compared to AVG': self.compareAvg,
            'Learning Rate': self.learning_rate,
            'Number of Epochs': self.num_epochs,
            'Batch Size': self.batch_size,
            'Training Pixel Accuracies': self.training_pixel_accuracies,
            'Testing Pixel Accuracies': self.testing_pixel_accuracies,
            'Training IoU': self.training_iou,
            'Testing IoU': self.testing_iou,
            'Test Iou': self.test_iou,
            'Testing Accuracy': self.test_accuracy,
            'Test Losses': float(self.test_losses) if isinstance(self.test_losses, (np.float32, np.float64)) else self.test_losses,  # Convert to float if necessary
            'Timestamp': str(datetime.now()),
            'Training Duration': self.end_time - self.start_time
        }

        # Generate a unique file name using UUID and timestamp
        unique_id = uuid.uuid4().hex  # Unique identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Timestamp
        file_name = f"experiment_results_{timestamp}_{unique_id}.json"
        model_file_name = f"model_{self.dataset}_{timestamp}_{unique_id}.pth"
        model_file_path = os.path.join(WORKING_DIR, model_file_name)
        torch.save(self.model.state_dict(), model_file_path)
        print(f"Model parameters saved to {model_file_path}")

        # Save to JSON
        file_path = os.path.join(WORKING_DIR, file_name)
        with open(file_path, 'w') as json_file:
            json.dump(experiment_data, json_file, indent=4)
        print(f"Experiment data saved to {file_path}")

    def _check_accuracy(self, loader, model):
        iou_scores = []
        num_correct = 0
        num_pixels = 0
        model.eval()
    
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device).unsqueeze(1)
            with torch.no_grad():  
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                # Flatten both predictions and ground truth
                preds_flat = preds.cpu().numpy().flatten()
                y_flat = y.cpu().numpy().flatten()
                num_correct += (preds == y).sum().item()
                num_pixels += torch.numel(preds)
            
            # Calculate IoU for the batch and add to list
            iou = jaccard_score(y_flat, preds_flat, average='macro')
            iou_scores.append(iou)
    
        # Calculate mean IoU across all batches
        mean_iou = np.mean(iou_scores)
        pixel_accuracy = num_correct / num_pixels * 100
        model.train()
        return mean_iou * 100, pixel_accuracy
    
    def _test_model(self, test_loader, model, criterion):
        batch_losses = []
        num_correct = 0
        num_pixels = 0
        iou_scores = []
        model.eval()
        
        with torch.no_grad():
            loop = tqdm(test_loader)
            for batch_idx, (data, targets) in enumerate(loop):
                data = data.to(self.device)
                targets = targets.float().unsqueeze(1).to(self.device)
    
                # Forward pass
                preds = model(data)
                loss = F.binary_cross_entropy_with_logits(preds, targets)
                
                # Store batch loss
                batch_losses.append(loss.detach().cpu().numpy())
                
                # Calculate IoU
                preds = torch.sigmoid(preds)
                preds = (preds > 0.5).float()
                num_correct += (preds == targets).sum().item()
                num_pixels += torch.numel(preds)
                preds_flat = preds.cpu().numpy().flatten()
                targets_flat = targets.cpu().numpy().flatten()
                
                # Calculate Jaccard index (IoU)
                iou = jaccard_score(targets_flat, preds_flat, average='macro')
                iou_scores.append(iou)
    
        # Compute average loss and mean IoU across all batches
        avg_loss = np.mean(batch_losses)
        mean_iou = np.mean(iou_scores)
        pixel_accuracy = num_correct / num_pixels * 100
        
        return avg_loss, mean_iou * 100, pixel_accuracy


if __name__ == "__main__":

    # self, dataset, avg, sp, greyscale, n, k, fair, compareOH, fusionAdd, fusionCat, remember, n_segments, compactness, learning_rate, num_epochs, batch_size, device
    UNET = Experiment(
        normal_UNET=False, dataset='CUB', avg=False, sp=True, greyscale=False, n=16, k=6, fair=False, compareOH=False, compareAvg=False, fusionAdd=False, fusionCat=False, remember=False, learning_rate=1e-4, num_epochs=1, batch_size=16, device='mps' if torch.backends.mps.is_available() else 'cpu'
    ) 

    UNET_OH_REMEMBER = Experiment(
        normal_UNET=True, dataset='CUB', avg=False, sp=True, greyscale=False, n=16, k=6, fair=False, compareOH=False, compareAvg=False, fusionAdd=False, fusionCat=True, remember=False, learning_rate=1e-4, num_epochs=1, batch_size=16, device='mps' if torch.backends.mps.is_available() else 'cpu'
    )

    UNET_OH_REMEMBER.run_experiment()
    # UNET_OH_REMEMBER.save_data()
    
    # UNET.run_experiment()
    # UNET.save_data()
    