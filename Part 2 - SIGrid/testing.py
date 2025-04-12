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
from tqdm import tqdm
import time
from datetime import datetime
import uuid
import json
from sklearn.metrics import jaccard_score
from hyb_unet import UNet, setup_transforms, DatasetClass, train_model
import torchvision.transforms.functional as TF


WORKING_DIR = '/Users/jroberts2/Jeova IS/hybrid/results'
#############################################
CUB_TRAIN_OG_IMG_DIR = '/Users/jroberts2/Jeova IS/Honors Project/CUB/train_images'
CUB_TRAIN_SEG_DIR = '/Users/jroberts2/Jeova IS/Honors Project/CUB/train_masks'
CUB_VAL_OG_IMG_DIR = '/Users/jroberts2/Jeova IS/Honors Project/CUB/test_images'
CUB_VAL_SEG_DIR = '/Users/jroberts2/Jeova IS/Honors Project/CUB/test_masks'


DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


class Experiment:
    def __init__(self, dataset, n_segments, compactness, SIGrid_channels, dim, reduced_performance, learning_rate, num_epochs, batch_size, device, avg_color, area, width, height, compac, solidity, eccentricity, hu):
        self.dataset = dataset
        self.n_segments = n_segments
        self.compactness = compactness
        self.SIGrid_channels = SIGrid_channels
        self.dim = dim 
        self.reduced_performance = reduced_performance
        self.model = None
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.avg_color = avg_color
        self.area = area
        self.width = width
        self.height = height 
        self.compac = compac
        self.solidity = solidity
        self.eccentricity = eccentricity
        self.hu = hu
        self.start_time = 0
        self.end_time = 0
        

        # Initialize results
        self.training_cell_accuracies = []
        self.training_cell_iou = []
        self.training_pixel_accuracies = []
        self.training_pixel_iou = []
        self.testing_cell_iou = []
        self.testing_cell_accuracies = []
        self.testing_pixel_iou = []
        self.testing_pixel_accuracies = []
        self.test_cell_iou = []
        self.test_cell_accuracy = []
        self.test_pixel_accuracy = []
        self.test_pixel_iou = []


    def run_experiment(self):
        # Setup the UNet model
        in_channels = self.SIGrid_channels

        if self.dataset == 'CUB':
            train_og_img_dir = CUB_TRAIN_OG_IMG_DIR
            train_seg_dir = CUB_TRAIN_SEG_DIR
            val_og_img_dir = CUB_VAL_OG_IMG_DIR
            val_seg_dir = CUB_VAL_SEG_DIR

            if self.n_segments in [300, 500] and self.compactness in [5, 10, 20, 100]:
                combo = f"{self.n_segments}_{self.compactness}_grid"
                split = "16"
                base_path = "/Users/jroberts2/Jeova IS/Honors Project/CUB"

                train_img_dir = f"{base_path}/{combo}_train_{split}"
                train_mask_dir = f"{base_path}/{combo}_mask_train_{split}"
                val_img_dir = f"{base_path}/{combo}_test_{split}"
                val_mask_dir = f"{base_path}/{combo}_mask_test_{split}"
            else:
                raise ValueError(f"Unsupported n_segments={self.n_segments}, compactness={self.compactness}")


                
        self.model = UNet(in_channels, self.reduced_performance).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scaler = torch.cuda.amp.GradScaler()

        # Setup data transformations and loaders
        train_transform, test_transform= setup_transforms(self.SIGrid_channels)
        train_ds = DatasetClass(dataset=self.dataset, n_segments=self.n_segments, compactness=self.compactness, SIGrid_channels = self.SIGrid_channels, og_image_dir = train_og_img_dir, image_dir=train_img_dir, mask_dir=train_mask_dir, seg_dir=train_seg_dir, img_transform=train_transform, avg_color=self.avg_color, area=self.area, width=self.width, height=self.height, compac=self.compac, solidity=self.solidity, eccentricity=self.eccentricity, hu=self.hu)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, num_workers=1, pin_memory=True, shuffle=True)

        test_ds = DatasetClass(dataset=self.dataset, n_segments=self.n_segments, compactness=self.compactness, SIGrid_channels = self.SIGrid_channels, og_image_dir=val_og_img_dir, image_dir=val_img_dir, mask_dir=val_mask_dir, seg_dir=val_seg_dir, img_transform=test_transform, avg_color=self.avg_color, area=self.area, width=self.width, height=self.height, compac=self.compac, solidity=self.solidity, eccentricity=self.eccentricity, hu=self.hu)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, num_workers=1, pin_memory=True, shuffle=False)

        # Run training and store results
        self.start_time = time.time()
        for epoch in range(self.num_epochs):
            loss = train_model(self.model, train_loader, optimizer, scaler, self.device)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss:.2f}")
            # Save training and testing accuracies after each epoch
            # train_cell_iou, train_cell_acc, train_pixel_iou, train_pixel_acc = self._check_accuracy(train_loader, self.model, testing=False, translate_cells=False)
            # testing_cell_iou, testing_cell_acc, testing_pixel_iou, testing_pixel_acc = self._check_accuracy(test_loader, self.model, testing=True, translate_cells=False)
            # self.training_cell_accuracies.append(train_cell_acc)
            # self.training_cell_iou.append(train_cell_iou)
            # self.training_pixel_accuracies.append(train_pixel_acc)
            # self.training_pixel_iou.append(train_pixel_iou)
            # self.testing_cell_iou.append(testing_cell_iou)
            # self.testing_cell_accuracies.append(testing_cell_acc)
            # self.testing_pixel_iou.append(testing_pixel_iou)
            # self.testing_pixel_accuracies.append(testing_pixel_acc)
            # print(f"Epoch {epoch + 1}/{self.num_epochs}, Training Pixel Accuracy: {train_pixel_acc:.2f}, Testing Pixel Accuracy: {testing_pixel_acc:.2f}, Training Pixel IoU: {train_pixel_iou:.2f}, Testing Pixel IoU: {testing_pixel_iou:.2f}, Training Cell Accuracy: {train_cell_acc:.2f}, Testing Cell Accuracy: {testing_cell_acc:.2f}, Training Cell IoU: {train_cell_iou:.2f}, Testing Cell IoU: {testing_cell_iou:.2f}")
            # print(f"Epoch {epoch + 1}/{self.num_epochs}, Training Pixel Accuracy: {train_pixel_acc:.2f}, Training Pixel IoU: {train_pixel_iou:.2f}, Training Cell Accuracy: {train_cell_acc:.2f}, Training Cell IoU: {train_cell_iou:.2f}")
        
        self.end_time = time.time()
        test_cell_iou, test_cell_accuracy, test_pixel_iou, test_pixel_acc = self._check_accuracy(test_loader, self.model, testing=True, translate_cells=True)
        self.test_cell_iou = test_cell_iou
        self.test_cell_accuracy = test_cell_accuracy
        self.test_pixel_accuracy = test_pixel_acc
        self.test_pixel_iou = test_pixel_iou


    def save_data(self):
        # Prepare experiment data
        experiment_data = {
            'Dataset': self.dataset,
            'N_segments': self.n_segments,
            'Compactness': self.compactness,
            'SIGrid Channels': self.SIGrid_channels,
            'SIGrid dimensions': self.dim,
            'Average Color': self.avg_color,
            'Area': self.area,
            'Width': self.width,
            'Height': self.height,
            'Compactness (shape descriptor)': self.compac,
            'Solidity': self.solidity,
            'Eccentricity': self.eccentricity,
            'Hu Moments: ': self.hu,
            'Reduced Performance': self.reduced_performance,
            'Learning Rate': self.learning_rate,
            'Number of Epochs': self.num_epochs,
            'Batch Size': self.batch_size,

            'Training Cell IoU': self.training_cell_iou,
            'Training Cell Accuracies': self.training_cell_accuracies,
            'Training Pixel IoU': self.training_pixel_accuracies,
            'Training Pixel Accuracies': self.training_pixel_iou,

            'Testing Cell IoU': self.testing_cell_iou,
            'Testing Pixel Accuracies': self.testing_pixel_accuracies,
            'Testing Cell IoU': self.testing_pixel_iou,
            'Testing Pixel Accuracies': self.testing_pixel_accuracies,

            'Test Cell Iou': self.test_cell_iou,
            'Test Cell Accuracy': self.test_cell_accuracy,
            'Test Pixel Iou': self.test_pixel_iou,
            'Test Pixel Accuracy': self.test_pixel_accuracy,
            'Timestamp': str(datetime.now()),
            'Training Duration': self.end_time - self.start_time
        }

        # Generate a unique file name using UUID and timestamp
        unique_id = uuid.uuid4().hex  # Unique identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Timestamp
        file_name = f"experiment_results_{timestamp}_{unique_id}_{self.n_segments}_{self.compactness}_{self.SIGrid_channels}.json"
        model_file_name = f"model_{self.dataset}_{timestamp}_{unique_id}.pth"
        model_file_path = os.path.join(WORKING_DIR, model_file_name)
        torch.save(self.model.state_dict(), model_file_path)
        print(f"Model parameters saved to {model_file_path}")

        # Save to JSON
        file_path = os.path.join(WORKING_DIR, file_name)
        with open(file_path, 'w') as json_file:
            json.dump(experiment_data, json_file, indent=4)
        print(f"Experiment data saved to {file_path}")

    def load_grid_and_slic(self, path, testing):
        """
        Extracts the relative image path, constructs the correct .npy file paths, 
        and loads the corresponding sp_label_grid and segments_slic matrices.

        Args:
            path (str): The full path of the image.
            testing (bool): Whether the model is in testing mode.

        Returns:
            tuple: sp_label_grid (numpy array), segments_slic (numpy array)
        
        Raises:
            FileNotFoundError: If any required .npy files are missing.
            ValueError: If the image path does not contain expected directory structures.
        """

        train_path = f"{self.n_segments}_{self.compactness}_grid_train_16"
        test_path =  f"{self.n_segments}_{self.compactness}_grid_test_16"

        # Extract the relative path based on known directory structure
        if train_path in path:
            relative_image_path = path.split(train_path)[1]
        elif test_path in path:
            relative_image_path = path.split(test_path)[1]
        else:
            raise ValueError(f"Image path does not contain expected directory structures. Found: {path}")

        # Remove the file extension to get the base name
        relative_base_path = os.path.splitext(relative_image_path)[0]

        # Define base directories for train and test grid/slic files
        valid_combinations = [(300, 10), (500, 10), (300, 5), (300, 20), (300, 100),
                      (500, 5), (500, 20), (500, 100)]

        if (self.n_segments, self.compactness) not in valid_combinations:
            raise ValueError(f"Unsupported combination: {self.n_segments=} {self.compactness=}")

        split = "test" if testing else "train"
        grid_dir = f"/Users/jroberts2/Jeova IS/Honors Project/CUB/sp_label_grid_{split}_{self.n_segments}_{self.compactness}"
        slic_dir = f"/Users/jroberts2/Jeova IS/Honors Project/CUB/segments_slic_{split}_{self.n_segments}_{self.compactness}"

        # Construct full paths for the .npy files
        grid_path = os.path.join(grid_dir, relative_base_path.lstrip('/') + ".npy")
        slic_path = os.path.join(slic_dir, relative_base_path.lstrip('/') + ".npy")

        # Load the .npy files
        if os.path.exists(grid_path) and os.path.exists(slic_path):
            sp_label_grid = torch.from_numpy(np.load(grid_path))  
            segments_slic = torch.from_numpy(np.load(slic_path))  
            return sp_label_grid, segments_slic
        else:
            missing_files = []
            if not os.path.exists(grid_path):
                missing_files.append(grid_path)
            if not os.path.exists(slic_path):
                missing_files.append(slic_path)
            raise FileNotFoundError(f"Missing the following files: {', '.join(missing_files)}")
        
    def load_gt(self, path, testing):
        """
        Extracts the relative image path, constructs the correct mask path, 
        and loads the corresponding ground truth segmentation map.

        Args:
            path (str): The full path of the image.
            testing (bool): Whether the model is in testing mode.

        Returns:
            numpy array: Ground truth segmentation map.
        
        Raises:
            FileNotFoundError: If the ground truth mask is missing.
            ValueError: If the image path does not contain expected directory structures.
        """
        train_path = f"{self.n_segments}_{self.compactness}_grid_train_16"
        test_path =  f"{self.n_segments}_{self.compactness}_grid_test_16"


        # Extract the relative path based on known directory structure
        if train_path in path:
            relative_image_path = path.split(train_path)[1]
        elif test_path in path:
            relative_image_path = path.split(test_path)[1]
        else:
            raise ValueError(f"Image path does not contain expected directory structures. Found: {path}")

        # Replace '.jpg' with '.png' for mask filename
        mask_filename = os.path.splitext(relative_image_path)[0] + ".png"

        # Define base directories for train and test masks
        mask_dir = "/Users/jroberts2/Jeova IS/Honors Project/CUB/train_masks" if not testing else "/Users/jroberts2/Jeova IS/CUB/test_masks"

        # Construct full path for the mask file
        mask_path = os.path.join(mask_dir, mask_filename.lstrip('/'))

        # Load the mask image
        if os.path.exists(mask_path):
            mask_image = Image.open(mask_path).convert("L")  # Convert to grayscale (single-channel)
            return np.array(mask_image)
        else:
            raise FileNotFoundError(f"Ground truth mask file not found: {mask_path}")


    def _check_accuracy(self, loader, model, testing, translate_cells):
        iou_cell_scores = []
        iou_pixel_scores = []
        num_correct_cells = 0
        num_correct_pixels = 0
        num_cells = 0
        num_pixels = 0
        model.eval()
        
        # Counter for saved images
        save_counter = 0
        save_path = '/Users/jroberts2/Jeova IS/Honors Project/Part 2 - SIGrid/preds'
        os.makedirs(save_path, exist_ok=True)

        for x, y, index in loader:
            x, y = x.to(self.device), y.to(self.device).unsqueeze(1)

            with torch.no_grad(): 
                preds = torch.sigmoid(model(x))
                preds = torch.where(preds < 0.5, torch.tensor(0.0, device=self.device), preds)
                preds = torch.where(preds >= 0.5, torch.tensor(1.0, device=self.device), preds)

                # --- Cell-Based Accuracy ---
                M = (y != -1)  # Mask to exclude -1 (ignored cells)
                preds_flat = preds[M].cpu().flatten()
                y_flat = y[M].cpu().flatten()
                num_correct_cells += (preds_flat == y_flat).sum().item()
                num_cells += torch.numel(preds_flat)

                # --- Resize Predictions & Ground Truth to `img_dim` and Compute Pixel Accuracy ---
                if translate_cells:
                    batch_size = preds.shape[0]

                    for i in range(batch_size):
                        pred = preds[i]
                        curr_index = index[i]
                        path = loader.dataset.images[curr_index]

                        sp_label_grid, segments_slic = self.load_grid_and_slic(path, testing)
                        ground_truth = self.load_gt(path, testing)
                        ground_truth = np.where(ground_truth > 175, 255, 0)
                        
                        if len(pred.shape) == 3:
                            pred = pred.squeeze(0)

                        pred_vec = pred[sp_label_grid != -1]

                        label_grid_vec = sp_label_grid[sp_label_grid != -1]

                        # Create a mapping tensor with default background (0)
                        mapping = torch.zeros(segments_slic.max() + 1, dtype=torch.uint8, device=pred_vec.device)

                        # Set foreground labels (255) based on predictions
                        mapping[label_grid_vec] = pred_vec.to(torch.uint8) * 255

                        # Apply the mapping to segments_slic
                        translated_pixels = mapping[segments_slic].cpu()

                        gt_tensor = torch.tensor(ground_truth) if not isinstance(ground_truth, torch.Tensor) else ground_truth
                        gt_tensor = gt_tensor.to(self.device)
                        tp_tensor = torch.tensor(translated_pixels) if not isinstance(translated_pixels, torch.Tensor) else translated_pixels
                        tp_tensor = tp_tensor.to(self.device)


                        num_correct_pixels += (gt_tensor == tp_tensor).sum().item()
                        num_pixels += torch.numel(torch.tensor(ground_truth))

                        # --- Compute IoU for This Sample ---
                        iou_pixel = jaccard_score(ground_truth.flatten(), translated_pixels.flatten(), average='macro')
                        iou_pixel_scores.append(iou_pixel)

                        # --- Save Predictions for Testing ---
                        if testing and save_counter < 5:
                            # Convert to torch tensor and move to CPU for saving
                            translated_pixels_tensor = translated_pixels_tensor = translated_pixels.clone().detach().float().cpu()
                            gt_tensor_cpu = gt_tensor.cpu().float()

                            # Save as images
                            pred_filename = os.path.join(save_path, f"pred_{save_counter}_{self.n_segments}_{self.compactness}_{self.SIGrid_channels}_{self.reduced_performance}.png")
                            gt_filename = os.path.join(save_path, f"gt_{save_counter}_{self.n_segments}_{self.compactness}_{self.SIGrid_channels}_{self.reduced_performance}.png")

                            final_pred = translated_pixels_tensor.to(torch.uint8)
                            TF.to_pil_image(final_pred).save(pred_filename)
                            gt_tensor_cpu = gt_tensor_cpu.to(torch.uint8)
                            TF.to_pil_image(gt_tensor_cpu).save(gt_filename)

                            save_counter += 1

                # --- Compute IoU Metrics for Cell-Based Accuracy ---
                iou_cell = jaccard_score(y_flat, preds_flat, average='macro')
                iou_cell_scores.append(iou_cell)

        # Compute final metrics
        mean_cell_iou = np.mean(iou_cell_scores)
        cell_accuracy = (num_correct_cells / num_cells) * 100
        if translate_cells:
            mean_pixel_iou = np.mean(iou_pixel_scores)
            pixel_accuracy = (num_correct_pixels / num_pixels) * 100
        else:
            mean_pixel_iou = -1
            pixel_accuracy = -1

        if not testing:
            model.train()

        return mean_cell_iou * 100, cell_accuracy, mean_pixel_iou * 100, pixel_accuracy


if __name__ == "__main__":

    # fixed_ns = [66, 57, 53, 33, 76, 69, 63, 42]

    a = Experiment(dataset='CUB', n_segments=300, compactness=5, SIGrid_channels=3, dim=66, reduced_performance=True, learning_rate=1e-3, num_epochs=1, batch_size=16, device=DEVICE, avg_color=True,  area=False, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=False)
    b = Experiment(dataset='CUB', n_segments=300, compactness=10, SIGrid_channels=3, dim=57, reduced_performance=True, learning_rate=1e-3, num_epochs=1, batch_size=16, device=DEVICE, avg_color=True,  area=False, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=False)
    c = Experiment(dataset='CUB', n_segments=300, compactness=20, SIGrid_channels=3, dim=53, reduced_performance=True, learning_rate=1e-3, num_epochs=1, batch_size=16, device=DEVICE, avg_color=True,  area=False, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=False)
    d = Experiment(dataset='CUB', n_segments=300, compactness=100, SIGrid_channels=3, dim=33, reduced_performance=True, learning_rate=1e-3, num_epochs=1, batch_size=16, device=DEVICE, avg_color=True,  area=False, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=False)
    e = Experiment(dataset='CUB', n_segments=500, compactness=5, SIGrid_channels=3, dim=76, reduced_performance=True, learning_rate=1e-3, num_epochs=1, batch_size=16, device=DEVICE, avg_color=True,  area=False, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=False)
    f = Experiment(dataset='CUB', n_segments=500, compactness=10, SIGrid_channels=3, dim=69, reduced_performance=True, learning_rate=1e-3, num_epochs=1, batch_size=16, device=DEVICE, avg_color=True,  area=False, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=False)
    g = Experiment(dataset='CUB', n_segments=500, compactness=20, SIGrid_channels=3, dim=63, reduced_performance=True, learning_rate=1e-3, num_epochs=1, batch_size=16, device=DEVICE, avg_color=True,  area=False, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=False)
    h = Experiment(dataset='CUB', n_segments=500, compactness=100, SIGrid_channels=3, dim=42, reduced_performance=True, learning_rate=1e-3, num_epochs=1, batch_size=16, device=DEVICE, avg_color=True,  area=False, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=False)

    a.run_experiment()
    a.save_data()
    b.run_experiment()
    b.save_data()
    c.run_experiment()
    c.save_data()
    d.run_experiment()
    d.save_data()
    e.run_experiment()
    e.save_data()
    f.run_experiment()
    f.save_data()
    g.run_experiment()
    g.save_data()
    h.run_experiment()
    h.save_data()

    ################################################################################################################ - after determining best hyperparams

    # e2 = Experiment(dataset='CUB', n_segments=, compactness=, SIGrid_channels=3, dim=, reduced_performance=False, learning_rate=1e-3, num_epochs=50, batch_size=16, device=DEVICE, avg_color=True,  area=False, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=False)
    
    # a1 = Experiment(dataset='CUB', n_segments=, compactness=, SIGrid_channels=4, dim=, reduced_performance=True, learning_rate=1e-3, num_epochs=1, batch_size=16, device=DEVICE, avg_color=True,  area=True, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=False)
    # a2 = Experiment(dataset='CUB', n_segments=, compactness=, SIGrid_channels=4, dim=, reduced_performance=False, learning_rate=1e-3, num_epochs=50, batch_size=16, device=DEVICE, avg_color=True,  area=True, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=False)

    # b1 = Experiment(dataset='CUB', n_segments=, compactness=, SIGrid_channels=5, dim=, reduced_performance=True, learning_rate=1e-3, num_epochs=50, batch_size=16, device=DEVICE, avg_color=True,  area=False, width=True, height=True, compac=False, solidity=False, eccentricity=False, hu=False)
    # b2 = Experiment(dataset='CUB', n_segments=, compactness=, SIGrid_channels=5, dim=, reduced_performance=False, learning_rate=1e-3, num_epochs=50, batch_size=16, device=DEVICE, avg_color=True,  area=False, width=True, height=True, compac=False, solidity=False, eccentricity=False, hu=False)

    # c1 = Experiment(dataset='CUB', n_segments=, compactness=, SIGrid_channels=6, dim=, reduced_performance=True, learning_rate=1e-3, num_epochs=50, batch_size=16, device=DEVICE, avg_color=True,  area=False, width=False, height=False, compac=True, solidity=True, eccentricity=True, hu=False)
    # c2 = Experiment(dataset='CUB', n_segments=, compactness=, SIGrid_channels=6, dim=, reduced_performance=False, learning_rate=1e-3, num_epochs=50, batch_size=16, device=DEVICE, avg_color=True,  area=False, width=False, height=False, compac=True, solidity=True, eccentricity=True, hu=False)

    # f1 = Experiment(dataset='CUB', n_segments=, compactness=, SIGrid_channels=6, dim=, reduced_performance=True, learning_rate=1e-3, num_epochs=50, batch_size=16, device=DEVICE, avg_color=True,  area=True, width=True, height=True, compac=False, solidity=False, eccentricity=False, hu=False)
    # f2 = Experiment(dataset='CUB', n_segments=, compactness=, SIGrid_channels=6, dim=, reduced_performance=False, learning_rate=1e-3, num_epochs=50, batch_size=16, device=DEVICE, avg_color=True,  area=True, width=True, height=True, compac=False, solidity=False, eccentricity=False, hu=False)

    # d1 = Experiment(dataset='CUB', n_segments=, compactness=, SIGrid_channels=7, dim=, reduced_performance=True, learning_rate=1e-3, num_epochs=50, batch_size=16, device=DEVICE, avg_color=False,  area=False, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=True)
    # d2 = Experiment(dataset='CUB', n_segments=, compactness=, SIGrid_channels=7, dim=, reduced_performance=False, learning_rate=1e-3, num_epochs=50, batch_size=16, device=DEVICE, avg_color=False,  area=False, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=True)

    # g1 = Experiment(dataset='CUB', n_segments=, compactness=, SIGrid_channels=10, dim=, reduced_performance=True, learning_rate=1e-3, num_epochs=50, batch_size=16, device=DEVICE, avg_color=True,  area=False, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=True)
    # g2 = Experiment(dataset='CUB', n_segments=, compactness=, SIGrid_channels=10, dim=, reduced_performance=False, learning_rate=1e-3, num_epochs=50, batch_size=16, device=DEVICE, avg_color=True,  area=False, width=False, height=False, compac=False, solidity=False, eccentricity=False, hu=True)

    # h1 = Experiment(dataset='CUB', n_segments=, compactness=, SIGrid_channels=16, dim=, reduced_performance=True, learning_rate=1e-3, num_epochs=50, batch_size=16, device=DEVICE, avg_color=True,  area=True, width=True, height=True, compac=True, solidity=True, eccentricity=True, hu=True)
    # h2 = Experiment(dataset='CUB', n_segments=, compactness=, SIGrid_channels=16, dim=, reduced_performance=False, learning_rate=1e-3, num_epochs=50, batch_size=16, device=DEVICE, avg_color=True,  area=True, width=True, height=True, compac=True, solidity=True, eccentricity=True, hu=True)

    # e2.run_experiment()
    # e2.save_data()

    # a1.run_experiment()
    # a1.save_data()

    # a2.run_experiment()
    # a2.save_data()

    # b1.run_experiment()
    # b1.save_data()

    # b2.run_experiment()
    # b2.save_data()

    # c1.run_experiment()
    # c1.save_data()

    # c2.run_experiment()
    # c2.save_data()

    # f1.run_experiment()
    # f1.save_data()

    # f2.run_experiment()
    # f2.save_data()

    # d1.run_experiment()
    # d1.save_data()

    # d2.run_experiment()
    # d2.save_data()

    # g1.run_experiment()
    # g1.save_data()

    # g2.run_experiment()
    # g2.save_data()

    # h1.run_experiment()
    # h1.save_data()

    # h2.run_experiment()
    # h2.save_data()




    
    