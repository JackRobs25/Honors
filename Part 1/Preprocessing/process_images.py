import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import torch
from skimage.segmentation import slic
from PIL import Image
import os

def create_adj_matrix(segments_slic, kernel_size, n_sp, width, length):
    G = np.zeros((n_sp, n_sp))  # Represents neighboring relationship between superpixels
    for seg in np.unique(segments_slic):
        mask = segments_slic == seg
        xy = np.where(mask)
        max_x, min_x = np.max(xy[0]), np.min(xy[0])
        max_y, min_y = np.max(xy[1]), np.min(xy[1])
        min_x = max(0, min_x - kernel_size)
        min_y = max(0, min_y - kernel_size)
        max_x = min(width, max_x + kernel_size)
        max_y = min(length, max_y + kernel_size)
        region_of_interest = mask[min_x:max_x, min_y:max_y]
        dilated = ndimage.binary_dilation(region_of_interest)
        diff = dilated - region_of_interest.astype(int)
        neig = np.unique(segments_slic[min_x:max_x, min_y:max_y][diff != 0])
        G[seg, neig] = 1
    return G

def generate_colored_G(G):
    degree = [sum(G[i]) for i in range(len(G))]
    colorDict = {i: [0, 1, 2, 3, 4, 5] for i in range(len(G))}
    sortedNode = sorted(range(len(degree)), key=lambda k: degree[k], reverse=True)
    solution = {}
    for n in sortedNode:
        setTheColor = colorDict[n]
        solution[n] = setTheColor[0]
        adjacentNode = G[n]
        for j in range(len(adjacentNode)):
            if adjacentNode[j] == 1 and (setTheColor[0] in colorDict[j]):
                colorDict[j].remove(setTheColor[0])
    return solution

def sp_encode(img, n_segments=300, compactness=10):
    if len(img.shape) == 3:  # RGB image
        width, length, _ = img.shape
        channel_axis = -1
    elif len(img.shape) == 2:  # Grayscale image
        width, length = img.shape
        channel_axis = None
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    segments_slic = slic(img, n_segments=n_segments, compactness=compactness, sigma=1, start_label=1, channel_axis=channel_axis) - 1
    
    n_sp = len(np.unique(segments_slic))
    kernel_size = 3
    G = create_adj_matrix(segments_slic, kernel_size, n_sp, width, length)

    solution = generate_colored_G(G)

    segments_slic_1d = segments_slic.flatten()
    color_pix = np.array([solution[segment] for segment in segments_slic_1d])
    color_pix = color_pix.reshape(segments_slic.shape)

    return color_pix

def process_images(source_dir, destination_dir, has_subfolders=False):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    if has_subfolders:
        # If the dataset contains subfolders, traverse them
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Process only image files
                    image_path = os.path.join(root, file)
                    try:
                        # Open the image and convert to NumPy array
                        image = np.array(Image.open(image_path))
                    except (IOError, OSError):
                        print(f"Skipping non-image file: {file}")
                        continue

                    # Compute the superpixelated image
                    sp_tensor = sp_encode(image)

                    # Convert to NumPy array if necessary
                    if isinstance(sp_tensor, torch.Tensor):
                        sp_tensor = sp_tensor.numpy()

                    # Convert to a PIL image and ensure it's in 'L' mode (grayscale)
                    sp_image = Image.fromarray(sp_tensor.astype('uint8'), mode='L')

                    # Define the output path with the folder structure preserved
                    relative_folder_path = os.path.relpath(root, source_dir)  # Get folder path relative to source_dir
                    output_folder = os.path.join(destination_dir, relative_folder_path)

                    # Create the corresponding folder in the destination directory if it doesn't exist
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    # Save the image in the destination folder
                    output_path = os.path.join(output_folder, f"sp_img_{file[:-4]}.png")

                    sp_image.save(output_path, format='PNG')
                    print(f"Grayscale image saved at: {output_path}")
    else:
        # Process all images directly in the source directory
        for file in os.listdir(source_dir):
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Process only image files
                image_path = os.path.join(source_dir, file)
                try:
                    # Open the image and convert to NumPy array
                    image = np.array(Image.open(image_path))
                except (IOError, OSError):
                    print(f"Skipping non-image file: {file}")
                    continue

                # Compute the superpixelated image
                sp_tensor = sp_encode(image)

                # Convert to NumPy array if necessary
                if isinstance(sp_tensor, torch.Tensor):
                    sp_tensor = sp_tensor.numpy()

                # Convert to a PIL image and ensure it's in 'L' mode (grayscale)
                sp_image = Image.fromarray(sp_tensor.astype('uint8'), mode='L')

                # Save the image in the destination directory
                output_path = os.path.join(destination_dir, f"sp_img_{file[:-4]}.png")

                sp_image.save(output_path, format='PNG')
                print(f"Grayscale image saved at: {output_path}")

def main():
    # Define source directories for train and test images
    train_source_dir = '/Users/jroberts2/Jeova IS/ECSSD/train_images'
    test_source_dir = '/Users/jroberts2/Jeova IS/ECSSD/test_images'
    
    train_destination_dir = "/Users/jroberts2/Jeova IS/ECSSD/sp_train_png/"
    test_destination_dir = "/Users/jroberts2/Jeova IS/ECSSD/sp_test_png/"

    # Set whether folders have subfolders
    has_subfolders = False

    # Process train and test images
    process_images(train_source_dir, train_destination_dir, has_subfolders)
    process_images(test_source_dir, test_destination_dir, has_subfolders)

if __name__ == "__main__":
    main()