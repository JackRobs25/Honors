import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from PIL import Image
import os

import numpy as np
from skimage.segmentation import slic

def sp_average_color(img, n_segments=300, compactness=10):
    # Check if the image is grayscale or RGB
    if len(img.shape) == 3:  # RGB image
        is_rgb = True
        img_flattened = img.reshape(-1, 3)  # Flatten the image to a (N, 3) array, where N is the number of pixels
        channel_axis = -1  # Use channel_axis=-1 for RGB images
    elif len(img.shape) == 2:  # Grayscale image
        is_rgb = False
        img_flattened = img.flatten()  # Flatten the image to a (N,) array
        channel_axis = None  # Use channel_axis=None for grayscale images
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    # Perform superpixelation with SLIC
    segments_slic = slic(img, n_segments=n_segments, compactness=compactness, sigma=1, start_label=1, channel_axis=channel_axis)

    # Flatten the segments_slic array to match img_flattened
    segments_slic_flattened = segments_slic.flatten()

    # Get the number of segments
    num_segments = np.max(segments_slic_flattened) + 1

    # Calculate the mean color (or intensity) for each segment
    if is_rgb:
        avg_colors = np.zeros_like(img_flattened)
        for channel in range(3):
            # Sum of pixel values per segment
            sums = np.bincount(segments_slic_flattened, weights=img_flattened[:, channel], minlength=num_segments)
            # Number of pixels per segment
            counts = np.bincount(segments_slic_flattened, minlength=num_segments)
            # Avoid division by zero
            avg_colors[:, channel] = sums[segments_slic_flattened] / np.where(counts[segments_slic_flattened] == 0, 1, counts[segments_slic_flattened])
    else:
        # Grayscale version
        sums = np.bincount(segments_slic_flattened, weights=img_flattened, minlength=num_segments)
        counts = np.bincount(segments_slic_flattened, minlength=num_segments)
        avg_colors = sums[segments_slic_flattened] / np.where(counts[segments_slic_flattened] == 0, 1, counts[segments_slic_flattened])

    # Reshape avg_colors back to the original image shape
    if is_rgb:
        avg_colors_reshaped = avg_colors.reshape(img.shape)
    else:
        avg_colors_reshaped = avg_colors.reshape(img.shape)

    return avg_colors_reshaped

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

                    # Compute the superpixelated image with average colors
                    avg_color_img = sp_average_color(image)

                    # Convert the result to a PIL image
                    sp_image_rgb = Image.fromarray((avg_color_img).astype('uint8'))

                    # Define the output path with the folder structure preserved
                    relative_folder_path = os.path.relpath(root, source_dir)  # Get folder path relative to source_dir
                    output_folder = os.path.join(destination_dir, relative_folder_path)

                    # Create the corresponding folder in the destination directory if it doesn't exist
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    # Save the image in the destination folder
                    output_path = os.path.join(output_folder, f"sp_rgb_img_{file[:-4]}.png")

                    sp_image_rgb.save(output_path, format='PNG')
                    print(f"RGB image saved at: {output_path}")
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

                # Compute the superpixelated image with average colors
                avg_color_img = sp_average_color(image)

                # Convert the result to a PIL image
                sp_image_rgb = Image.fromarray((avg_color_img).astype('uint8'))

                # Save the image in the destination directory
                output_path = os.path.join(destination_dir, f"sp_rgb_img_{file[:-4]}.png")

                sp_image_rgb.save(output_path, format='PNG')
                print(f"RGB image saved at: {output_path}")

def main():
    # Define source and destination pairs for both train and test images
    paths = [     
        # # ECSSD Dataset
        # {
        #     "train_source": "/Users/jroberts2/Jeova IS/ECSSD/train_images",
        #     "train_destination": "/Users/jroberts2/Jeova IS/ECSSD/avg_train_png",
        #     "test_source": "/Users/jroberts2/Jeova IS/ECSSD/test_images",
        #     "test_destination": "/Users/jroberts2/Jeova IS/ECSSD/avg_test_png",
        #     "has_subfolders": False
        # },   
        # Carvana Dataset
        {
            "train_source": "/Users/jroberts2/Jeova IS/Carvana/train_images",
            "train_destination": "/Users/jroberts2/Jeova IS/Carvana/avg_train_png",
            "test_source": "/Users/jroberts2/Jeova IS/Carvana/test_images",
            "test_destination": "/Users/jroberts2/Jeova IS/Carvana/avg_test_png",
            "has_subfolders": False
        },
        # CUB Dataset
        {
            "train_source": "/Users/jroberts2/Jeova IS/CUB/train_images",
            "train_destination": "/Users/jroberts2/Jeova IS/CUB/avg_train_png",
            "test_source": "/Users/jroberts2/Jeova IS/CUB/test_images",
            "test_destination": "/Users/jroberts2/Jeova IS/CUB/avg_test_png",
            "has_subfolders": True  # CUB dataset has subfolders
        }
    ]

    # Process each dataset
    for path in paths:
        # Process train images
        process_images(path["train_source"], path["train_destination"], path["has_subfolders"])

        # Process test images
        process_images(path["test_source"], path["test_destination"], path["has_subfolders"])

if __name__ == "__main__":
    main()