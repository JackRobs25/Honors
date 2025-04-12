import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops, regionprops_table
from skimage.segmentation import relabel_sequential
import math
from scipy.spatial.distance import pdist, squareform

class MergingImageProcessor:
    def __init__(self, image_path, n_segments, compactness, sigma=1, merge_threshold=5):
        self.image_path = image_path
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.merge_threshold = merge_threshold
        self.img = None
        self.segments_slic = None
        self.superpixel_centers = []
        self.grid = None

    def load_image(self):
        self.img = io.imread(self.image_path)

    def compute_superpixels_and_merge(self):
        if len(self.img.shape) == 3:
            channel_axis = -1
        elif len(self.img.shape) == 2:
            channel_axis = None
        else:
            raise ValueError(f"Unexpected image shape: {self.img.shape}")

        segments = slic(self.img, n_segments=self.n_segments, compactness=self.compactness,
                        sigma=self.sigma, start_label=1, channel_axis=channel_axis)

        regions = regionprops(segments)
        centers = np.array([region.centroid for region in regions])
        labels = np.array([region.label for region in regions])

        dists = squareform(pdist(centers))
        merged = {label: label for label in labels}

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                if dists[i, j] < self.merge_threshold:
                    li, lj = merged[labels[i]], merged[labels[j]]
                    if li != lj:
                        lower, higher = sorted((li, lj))
                        for k in merged:
                            if merged[k] == higher:
                                merged[k] = lower

        label_map = {old: new for old, new in merged.items()}
        new_segments = np.copy(segments)
        for old, new in label_map.items():
            new_segments[segments == old] = new

        new_segments, _, _ = relabel_sequential(new_segments)
        self.segments_slic = new_segments
        self.superpixel_centers = [region.centroid for region in regionprops(new_segments)]

    def show_sp_centres(self):
        if self.segments_slic is None:
            raise ValueError("You must compute superpixels first.")
        image_with_boundaries = mark_boundaries(self.img, self.segments_slic, color=(1, 1, 0), mode='thick')
        plt.figure(figsize=(10, 10))
        plt.imshow(image_with_boundaries)
        for center in self.superpixel_centers:
            y, x = center
            plt.plot(x, y, 'ro', markersize=5)
        plt.title('Superpixelation with Superpixel Centers')
        plt.axis('off')
        plt.show()

    def create_grid(self, fixed_n):
        if self.segments_slic is None:
            raise ValueError("You must compute superpixels first.")

        img_height, img_width = self.img.shape[:2]
        grid_height = np.ceil(img_height / fixed_n)
        grid_width = np.ceil(img_width / fixed_n)

        # Create an empty n x n grid (to store the average color values)
        self.grid = np.full((fixed_n, fixed_n, 16), -1, dtype=np.float32)  # 3 channels 
        # self.grid = np.full((n, n, 9), -1, dtype=np.float32)  # 6 channels


        average_colors = np.zeros((np.max(self.segments_slic) + 1, 3), dtype=np.float32)

        ### 6 channels 
        widths = np.zeros((np.max(self.segments_slic) + 1), dtype=np.float32) ###
        heights = np.zeros((np.max(self.segments_slic) + 1), dtype=np.float32) ###
        ###

        for label in np.unique(self.segments_slic):
            mask = self.segments_slic == label
            average_colors[label] = self.img[mask].mean(axis=0)
            coords = np.column_stack(np.where(mask)) ###
            widths[label] = (coords[:, 1].max() - coords[:, 1].min()) / img_width ###
            heights[label] = (coords[:, 0].max() - coords[:, 0].min()) / img_height ###

        regions = regionprops(self.segments_slic)

        for region in regions:
            y, x = region.centroid
            label = region.label
            grid_row = min(int(y // grid_height), self.grid.shape[0] - 1)
            grid_col = min(int(x // grid_width), self.grid.shape[1] - 1)

            if np.all(self.grid[grid_row, grid_col] == -1):
                width = widths[label]
                height = heights[label]
                normalised_area = region.area / (img_height*img_width)
                normalised_compactness = ((region.perimeter ** 2) / (4 * np.pi * region.area)) * (region.area / (img_height*img_width)) if region.area > 0 else 0 # how circular is the superpixel 
                solidity = region.solidity
                eccentricity = region.eccentricity 
                hu_moments = region.moments_hu

                self.grid[grid_row, grid_col, :3] = average_colors[label]
                ### 
                self.grid[grid_row, grid_col, 3] = normalised_area
                self.grid[grid_row, grid_col, 4] = width
                self.grid[grid_row, grid_col, 5] = height
                self.grid[grid_row, grid_col, 6] = normalised_compactness
                self.grid[grid_row, grid_col, 7] = solidity 
                self.grid[grid_row, grid_col, 8] = eccentricity 
                ###
                self.grid[grid_row, grid_col, 9:16] = hu_moments
                
    def plot_grid(self):
        if self.grid is None:
            raise ValueError("You must create the grid first.")

        plot_grid = self.grid[:, :, :3].copy()
        plot_grid[plot_grid == -1] = 0
        plt.figure(figsize=(10, 10))
        plt.imshow(plot_grid / 255.0)
        plt.title(f'Grid Representation ({self.grid.shape[0]}x{self.grid.shape[1]})')
        plt.axis('off')
        plt.show()

# --- Precompute fixed n ---
def compute_fixed_n_for_dataset(image_dirs, n_segments, compactness, merge_threshold=5, sigma=1):
    print("\n📏 Computing fixed grid size (n) from all images...")
    max_grid_size = 0
    for split, image_dir in image_dirs.items():
        for root, _, files in os.walk(image_dir):
            for fname in files:
                if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                image_path = os.path.join(root, fname)
                try:
                    processor = MergingImageProcessor(image_path, n_segments, compactness, sigma, merge_threshold)
                    processor.load_image()
                    processor.compute_superpixels_and_merge()
                    centers = processor.superpixel_centers
                    height, width = processor.img.shape[:2]

                    def is_valid(n):
                        cell_h = height / n
                        cell_w = width / n
                        seen = set()
                        for y, x in centers:
                            r = int(y // cell_h)
                            c = int(x // cell_w)
                            if (r, c) in seen:
                                return False
                            seen.add((r, c))
                        return True

                    n = math.ceil(math.sqrt(len(centers)))
                    while not is_valid(n):
                        n += 1

                    max_grid_size = max(max_grid_size, n)
                    print(f"✅ {fname}: requires {n}x{n}")

                except Exception as e:
                    print(f"❌ Failed to process {fname}: {e}")

    print(f"\n📐 Final fixed grid size: {max_grid_size}x{max_grid_size}")
    return max_grid_size


class MaskProcessor:
    def __init__(self, mask_path, n):
        self.mask_path = mask_path
        self.mask = None
        self.segments_slic = None  # This will come from the MergingImageProcessor
        self.superpixel_centers = []
        self.grid = None
        self.n = n

    def load_mask(self):
        """Load the mask from the provided path."""
        self.mask = io.imread(self.mask_path)

    def set_superpixels(self, segments_slic):
        """Set the precomputed superpixel segmentation from the ImageProcessor."""
        self.segments_slic = segments_slic
        regions = regionprops(self.segments_slic)
        self.superpixel_centers = [region.centroid for region in regions]  # List of (y, x) coordinates

    def show_sp_centres(self):
        """Display the mask image with transparent superpixel boundaries, superpixel centers, and highlighted grid cells containing 1's."""
        if self.segments_slic is None:
            raise ValueError("You must set the superpixels first using set_superpixels().")

        # Mark the boundaries of superpixels with a transparency (alpha)
        image_with_boundaries = mark_boundaries(self.mask, self.segments_slic, color=(1, 0.1, 0.1), mode='thin')  # Red boundaries

        # Get mask dimensions
        mask_height, mask_width = self.mask.shape[:2]
        n = self.n

        # Grid cell size
        grid_height = np.ceil(mask_height / n)
        grid_width = np.ceil(mask_width / n)

        # Plot the mask image with transparent superpixel boundaries
        plt.figure(figsize=(10, 10))
        plt.imshow(self.mask, cmap='gray')  # Show the mask image
        plt.imshow(image_with_boundaries, alpha=0.6)  # Overlay the superpixel boundaries with 60% opacity (transparency)

        # Overlay the superpixel centers
        for center in self.superpixel_centers:
            y, x = center  # y, x coordinates of the superpixel center
            plt.plot(x, y, color='yellow', marker='o', markersize=4, markeredgecolor='black', markeredgewidth=0.5)  # Brighter yellow with black outline

        # Add rectangles around grid cells based on their values
        for i in range(n):
            for j in range(n):
                # Get the grid cell's value (1, 0, -1)
                value = self.grid[i, j]

                # Compute the rectangle coordinates (upper-left corner of each grid cell)
                top_left_y = i * grid_height
                top_left_x = j * grid_width

                if value == 1:  # Highlight grid cells with 1 in green (boldest)
                    edge_color = 'green'
                    line_width = 3  # Bold outline for green boxes
                elif value == -1:  # Highlight grid cells with -1 in light grey
                    edge_color = 'lightgrey'
                    line_width = 1  # Thin outline for grey boxes
                else:  # Highlight grid cells with 0 in blue (narrower)
                    edge_color = 'blue'
                    line_width = 2  # Thin outline for blue boxes

                # Draw a rectangle around the grid cell
                plt.gca().add_patch(
                    plt.Rectangle((top_left_x, top_left_y), grid_width, grid_height, 
                                fill=False, edgecolor=edge_color, linewidth=line_width)
                )

        plt.title('Mask with Transparent Superpixel Boundaries, Centers, and Highlighted Grid Cells')
        plt.axis('off')
        plt.show()

    def create_grid(self):
        """Create a grid with 1 for foreground, 0 for background, and -1 for empty cells.
        A superpixel is considered foreground only if at least 80% of its pixels belong to the object.
        """
        if self.segments_slic is None:
            raise ValueError("You must set the superpixels first using set_superpixels().")

        # Get mask dimensions
        mask_height, mask_width = self.mask.shape[:2]
        n = self.n

        # Grid cell size
        grid_height = np.ceil(mask_height / n)
        grid_width = np.ceil(mask_width / n)

        # Create an empty n x n grid (to store values)
        self.grid = np.full((n, n), -1, dtype=np.int32)

        # Threshold for foreground classification (80% of the superpixel must be foreground)
        foreground_threshold = 0.8

        for region in regionprops(self.segments_slic):
            y, x = region.centroid  # Superpixel center
            label = region.label

            # Get all pixels belonging to this superpixel
            mask_pixels = self.mask[self.segments_slic == label]
            
            # Compute the proportion of foreground pixels
            total_pixels = mask_pixels.size
            foreground_pixels = np.sum(mask_pixels>127)
            foreground_ratio = foreground_pixels / total_pixels

            # Determine if the superpixel is part of the foreground (1) or background (0)
            is_foreground = foreground_ratio >= foreground_threshold  # Must be at least 80%

            # Get the corresponding grid cell
            grid_row = min(int(y // grid_height), n - 1)
            grid_col = min(int(x // grid_width), n - 1)

            if np.all(self.grid[grid_row, grid_col] == -1):  # Only update if cell is empty
                self.grid[grid_row, grid_col] = 1 if is_foreground else 0  # Foreground (1), Background (0)

    
    def visualize_superpixels_with_grid(self):
        """
        Combined visualization:
        - Left subplot: Mask with superpixel boundaries, centers, and highlighted grid cells.
        - Right subplot: Foreground (white) vs. Background (black) superpixels.
        """
        if self.segments_slic is None or self.grid is None:
            raise ValueError("Superpixels must be computed and grid must be created first.")

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # ---- First Subplot: Mask with Superpixel Boundaries, Centers, and Grid ----
        image_with_boundaries = mark_boundaries(self.mask, self.segments_slic, color=(1, 1, 0), mode='thick')  # Yellow boundaries
        axes[0].imshow(self.mask, cmap="gray")  # Show mask image
        axes[0].imshow(image_with_boundaries, alpha=0.6)  # Overlay boundaries

        # Overlay superpixel centers
        for center in self.superpixel_centers:
            y, x = center
            axes[0].plot(x, y, 'ro', markersize=5)  # Red dots at superpixel centers

        # Grid dimensions
        mask_height, mask_width = self.mask.shape[:2]
        grid_height = np.ceil(mask_height / self.n)
        grid_width = np.ceil(mask_width / self.n)

        # Draw rectangles around grid cells based on values
        for i in range(self.n):
            for j in range(self.n):
                value = self.grid[i, j]
                top_left_y = i * grid_height
                top_left_x = j * grid_width

                if value == 1:
                    edge_color, line_width = 'green', 3  # Foreground (bold green)
                elif value == -1:
                    edge_color, line_width = 'red', 1  # Background (thin red)
                else:
                    edge_color, line_width = 'blue', 1  # Empty (thin blue)

                axes[0].add_patch(
                    plt.Rectangle((top_left_x, top_left_y), grid_width, grid_height,
                                  fill=False, edgecolor=edge_color, linewidth=line_width)
                )

        axes[0].set_title("Superpixel Boundaries, Centers, and Grid")
        axes[0].axis("off")

        # ---- Second Subplot: Foreground (White) vs. Background (Black) Superpixels ----
        output_image = np.zeros_like(self.mask, dtype=np.uint8)

        for region in regionprops(self.segments_slic):
            label = region.label
            y, x = map(int, region.centroid)

            grid_row = min(int(y // grid_height), self.n - 1)
            grid_col = min(int(x // grid_width), self.n - 1)

            if self.grid[grid_row, grid_col] == 1:
                output_image[self.segments_slic == label] = 255  # White for foreground

        axes[1].imshow(output_image, cmap="gray")
        axes[1].set_title("Foreground (White) vs. Background (Black) Superpixels")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
    
    def plot_grid(self):
        """Plot the grid with black (0), white (1), and gray (-1) squares."""
        if self.grid is None:
            raise ValueError("You must create the grid first.")

        # Create a copy of the grid for visualization
        plot_grid = np.zeros(self.grid.shape + (3,), dtype=np.float32)

        # Color mapping
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                value = self.grid[i, j]
                if value == 1:      # Foreground (white)
                    plot_grid[i, j] = [1, 1, 1]    # White
                elif value == 0:    # Background (black)
                    plot_grid[i, j] = [0, 0, 0]    # Black
                else:               # Empty (-1)
                    plot_grid[i, j] = [0]  # Gray

        # Plot the grid
        plt.figure(figsize=(10, 10))
        plt.imshow(plot_grid)
        plt.title('Mask Grid Representation (Foreground, Background, Empty)')
        plt.axis('off')
        plt.show()


# # Define the paths for ECSSD, CUB, and Carvana datasets
# image_dirs = {
#     'CUB_train': '/Users/jroberts2/Jeova IS/CUB/train_images',
#     'CUB_test': '/Users/jroberts2/Jeova IS/CUB/test_images',
# }

# grid_save_dirs = {
#     'CUB_train': '/Users/jroberts2/Jeova IS/CUB/500_10_grid_train_9',  
#     'CUB_test': '/Users/jroberts2/Jeova IS/CUB/500_10_grid_test_9',
# }
# grid_mask_save_dirs = {
#     'CUB_train': '/Users/jroberts2/Jeova IS/CUB/500_10_grid_mask_train_9',
#     'CUB_test': '/Users/jroberts2/Jeova IS/CUB/500_10_grid_mask_test_9',
# }

# mask_dirs = {
#     'CUB_train': '/Users/jroberts2/Jeova IS/CUB/train_masks',
#     'CUB_test': '/Users/jroberts2/Jeova IS/CUB/test_masks',
# }

def process_images(image_dir, mask_dir, grid_dir, grid_mask_dir, dataset_name, fixed_n, n_segments, compactness):

    # max_n = 0  # Track the largest n
    # unique_n = set()
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(grid_mask_dir, exist_ok=True)
    
    for root, subdirs, files in os.walk(image_dir):
        # print("new folder: ", root)
        for image_name in files:
            if image_name.endswith('.jpg') or image_name.endswith('.png'):
                image_path = os.path.join(root, image_name)

                # Handle CUB dataset's nested mask structure
                if 'CUB' in dataset_name:
                    # Preserve subfolder structure
                    relative_path = os.path.relpath(root, image_dir)
                    grid_subdir = os.path.join(grid_dir, relative_path)
                    grid_mask_subdir = os.path.join(grid_mask_dir, relative_path)
                    
                    os.makedirs(grid_subdir, exist_ok=True)
                    os.makedirs(grid_mask_subdir, exist_ok=True)

                    mask_path = find_mask_in_subfolders(mask_dir, image_name)
                    grid_save_path = os.path.join(grid_subdir, os.path.splitext(image_name)[0] + '.npy')
                    grid_mask_save_path = os.path.join(grid_mask_subdir, os.path.splitext(image_name)[0] + '.npy')
                else:
                    # Save in a flat directory structure
                    if 'Carvana' in dataset_name:
                        mask_path = os.path.join(mask_dir, os.path.splitext(image_name)[0] + '_mask.gif')
                    else:
                        mask_path = os.path.join(mask_dir, os.path.splitext(image_name)[0] + '.png')
                    grid_save_path = os.path.join(grid_dir, os.path.splitext(image_name)[0] + '.npy')
                    grid_mask_save_path = os.path.join(grid_mask_dir, os.path.splitext(image_name)[0] + '.npy')

                if not os.path.exists(mask_path):
                    print(f"Mask not found for {image_name}, skipping...")
                    continue

                # print("Processing:", image_path)
                processor = MergingImageProcessor(image_path, n_segments, compactness)
                processor.load_image()
                processor.compute_superpixels_and_merge()
                processor.create_grid(fixed_n)
                # processor.show_sp_centres()
                # processor.plot_grid()
                np.save(grid_save_path, processor.grid)
                print(f"Saved image grid to: {grid_save_path}")

                mask_processor = MaskProcessor(mask_path, fixed_n)
                mask_processor.load_mask()
                mask_processor.set_superpixels(processor.segments_slic)
                mask_processor.create_grid()
                # mask_processor.plot_grid()
                # mask_processor.show_sp_centres()
                np.save(grid_mask_save_path, mask_processor.grid)
                print(f"Saved mask grid to: {grid_mask_save_path}")
    print("fixed_n used: ", fixed_n)
    # print(f"Unique values of n encountered: {sorted(unique_n)}")  # Print all unique values of n
    # return max_n

def find_mask_in_subfolders(mask_dir, image_name):
    """Search for the corresponding mask file in nested directories for CUB dataset."""
    base_name = os.path.splitext(image_name)[0]  # Remove file extension
    for root, _, files in os.walk(mask_dir):
        for file in files:
            if file.startswith(base_name):  # Match filename prefix
                return os.path.join(root, file)
    return None  # Return None if not found

# Define hyperparameter sweep
# n_segments_list = [300, 500, 700]
# compactness_list = [5, 10, 20, 100]
n_segments_list = [500]
compactness_list = [20]

# Base directories
base_image_dirs = {
    'CUB_train': '/Users/jroberts2/Jeova IS/CUB/train_images',
    'CUB_test': '/Users/jroberts2/Jeova IS/CUB/test_images',
}
base_mask_dirs = {
    'CUB_train': '/Users/jroberts2/Jeova IS/CUB/train_masks',
    'CUB_test': '/Users/jroberts2/Jeova IS/CUB/test_masks',
}

def main():
    overall_max_n = 0
    # fixed_ns = [66, 57, 53, 33, 76, 69, 63, 42, 85, 77, 70, 50]
    fixed_ns = [63] # just for 500_20

    idx = 0

    for n_segments in n_segments_list:
        for compactness in compactness_list:
            param_key = f"{n_segments}_{compactness}"

            # Build directories for this combination
            image_dirs = {
                f'CUB_train_{param_key}': base_image_dirs['CUB_train'],
                f'CUB_test_{param_key}': base_image_dirs['CUB_test'],
            }
            mask_dirs = {
                f'CUB_train_{param_key}': base_mask_dirs['CUB_train'],
                f'CUB_test_{param_key}': base_mask_dirs['CUB_test'],
            }
            grid_save_dirs = {
                f'CUB_train_{param_key}': f'/Users/jroberts2/Jeova IS/CUB/{param_key}_grid_train_16',
                f'CUB_test_{param_key}': f'/Users/jroberts2/Jeova IS/CUB/{param_key}_grid_test_16',
            }
            grid_mask_save_dirs = {
                f'CUB_train_{param_key}': f'/Users/jroberts2/Jeova IS/CUB/{param_key}_grid_mask_train_16',
                f'CUB_test_{param_key}': f'/Users/jroberts2/Jeova IS/CUB/{param_key}_grid_mask_test_16',
            }

            # print(f"\n🔍 Starting processing for SLIC params: n_segments={n_segments}, compactness={compactness}")
            # fixed_n = compute_fixed_n_for_dataset(image_dirs, merge_threshold=5,
            #                                       n_segments=n_segments, compactness=compactness)
            # print(f"🔁 Using fixed grid size: {fixed_n}x{fixed_n} for all images")
            fixed_n = fixed_ns[idx]
            for split in image_dirs:
                process_images(
                    image_dirs[split],
                    mask_dirs[split],
                    grid_save_dirs[split],
                    grid_mask_save_dirs[split],
                    split,
                    fixed_n,
                    n_segments,
                    compactness
                )
            idx += 1
                # overall_max_n = max(overall_max_n, max_n)

    # print(f"\n✅ All processing complete. Overall max grid size encountered: {overall_max_n}x{overall_max_n}")

if __name__ == "__main__":
    main()



# def main():
#     fixed_n = compute_fixed_n_for_dataset(image_dirs, merge_threshold=5, n_segments=500, compactness=10)
#     print(f"\n🔁 Using fixed grid size: {fixed_n}x{fixed_n} for all images")
#     overall_max_n = 0  # Track the largest n across all datasets
#     for split in image_dirs.keys():
#         max_n = process_images(image_dirs[split], mask_dirs[split], grid_save_dirs[split], grid_mask_save_dirs[split], split, fixed_n)
#         overall_max_n = max(overall_max_n, max_n)
#     print(f"Overall max grid dimensions: {overall_max_n}x{overall_max_n}")



# if __name__ == "__main__":
#     main()

   