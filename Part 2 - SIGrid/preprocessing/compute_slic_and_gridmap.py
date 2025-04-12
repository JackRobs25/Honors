# # compute segments_slic and sp_label_grid folders
# # segments_slic --> maps pixels to superpixel labels
# # sp_label_grid --> maps grid cells to superpixel labels 

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
import os
from skimage.segmentation import relabel_sequential
from scipy.spatial.distance import pdist, squareform

class ImageProcessor:
    def __init__(self, image_path, grid_size, n_segments, compactness, sigma=1, merge_threshold=5):
        self.image_path = image_path
        self.grid_size = grid_size
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.merge_threshold = merge_threshold
        self.img = None
        self.segments_slic = None
        self.superpixel_centers = []
        self.grid = None
        self.sp_label_grid = None

    def load_image(self):
        self.img = io.imread(self.image_path)

    def compute_superpixels(self):
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

    def create_grid(self):
        if self.segments_slic is None:
            raise ValueError("You must compute superpixels first.")

        img_height, img_width = self.img.shape[:2]
        n = self.grid_size

        grid_height = np.ceil(img_height / n)
        grid_width = np.ceil(img_width / n)

        self.grid = np.full((n, n, 3), -1, dtype=np.float32)
        self.sp_label_grid = np.full((n, n), -1, dtype=np.int32)

        average_colors = np.zeros((np.max(self.segments_slic) + 1, 3), dtype=np.float32)
        for label in np.unique(self.segments_slic):
            mask = self.segments_slic == label
            average_colors[label] = self.img[mask].mean(axis=0)

        for region in regionprops(self.segments_slic):
            y, x = region.centroid
            label = region.label
            grid_row = min(int(y // grid_height), self.grid.shape[0] - 1)
            grid_col = min(int(x // grid_width), self.grid.shape[1] - 1)
            if np.all(self.grid[grid_row, grid_col] == -1):
                self.sp_label_grid[grid_row, grid_col] = label

image_dirs = {
    'CUB_train': '/Users/jroberts2/Jeova IS/CUB/train_images',
    'CUB_test': '/Users/jroberts2/Jeova IS/CUB/test_images'
}

fixed_ns = [66, 57, 53, 33, 76, 69, 63, 42]
slic_params = [(300, 5), (300, 10), (300, 20), (300, 100), (500, 5), (500, 10), (500, 20), (500, 100)]

def process_images(image_dir, sp_label_grid_dir, segments_slic_dir, n_segments, compactness, fixed_n):
    for root, _, files in os.walk(image_dir):
        for image_name in files:
            if image_name.endswith('.jpg') or image_name.endswith('.png'):
                image_path = os.path.join(root, image_name)
                relative_path = os.path.relpath(root, image_dir)
                sp_label_grid_subdir = os.path.join(sp_label_grid_dir, relative_path)
                segments_slic_subdir = os.path.join(segments_slic_dir, relative_path)
                os.makedirs(sp_label_grid_subdir, exist_ok=True)
                os.makedirs(segments_slic_subdir, exist_ok=True)

                sp_label_grid_save_path = os.path.join(sp_label_grid_subdir, os.path.splitext(image_name)[0] + '.npy')
                segments_slic_save_path = os.path.join(segments_slic_subdir, os.path.splitext(image_name)[0] + '.npy')

                processor = ImageProcessor(image_path, fixed_n, n_segments, compactness)
                processor.load_image()
                processor.compute_superpixels()
                processor.create_grid()

                np.save(sp_label_grid_save_path, processor.sp_label_grid)
                np.save(segments_slic_save_path, processor.segments_slic)
                print(f"Saved grids for {image_name} in {relative_path} with n_segments={n_segments}, compactness={compactness}")

def main():
    for idx, (n_segments, compactness) in enumerate(slic_params):
        fixed_n = fixed_ns[idx]
        suffix = f"{n_segments}_{compactness}"

        sp_label_grid_dirs = {
            'CUB_train': f'/Users/jroberts2/Jeova IS/CUB/sp_label_grid_train_{suffix}',
            'CUB_test': f'/Users/jroberts2/Jeova IS/CUB/sp_label_grid_test_{suffix}'
        }

        segments_slic_dirs = {
            'CUB_train': f'/Users/jroberts2/Jeova IS/CUB/segments_slic_train_{suffix}',
            'CUB_test': f'/Users/jroberts2/Jeova IS/CUB/segments_slic_test_{suffix}'
        }

        for split in image_dirs:
            process_images(
                image_dirs[split],
                sp_label_grid_dirs[split],
                segments_slic_dirs[split],
                n_segments,
                compactness,
                fixed_n
            )

if __name__ == "__main__":
    main()



# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io
# from skimage.segmentation import slic, mark_boundaries
# from skimage.measure import regionprops
# import os
# from skimage.segmentation import relabel_sequential
# from scipy.spatial.distance import pdist, squareform

# class ImageProcessor:
#     def __init__(self, image_path, grid_size, n_segments, compactness, sigma=1):
#         self.image_path = image_path
#         self.grid_size = grid_size
#         self.n_segments = n_segments
#         self.compactness = compactness
#         self.sigma = sigma
#         self.img = None
#         self.segments_slic = None
#         self.superpixel_centers = []
#         self.grid = None
#         self.sp_label_grid = None

#     def load_image(self):
#         """Load the image from the provided path."""
#         self.img = io.imread(self.image_path)

#     def compute_superpixels(self):
#         """Perform SLIC segmentation to compute superpixels."""
#         if len(self.img.shape) == 3:  # RGB image
#             channel_axis = -1
#         elif len(self.img.shape) == 2:  # Grayscale image
#             channel_axis = None
#         else:
#             raise ValueError(f"Unexpected image shape: {self.img.shape}")
#         # self.segments_slic = slic(self.img, n_segments=self.n_segments, compactness=self.compactness, sigma=self.sigma, start_label=1, channel_axis=channel_axis)
#         # regions = regionprops(self.segments_slic)
#         # self.superpixel_centers = [region.centroid for region in regions]  # List of (y, x) coordinates
#         segments = slic(self.img, n_segments=self.n_segments, compactness=self.compactness,
#                         sigma=self.sigma, start_label=1, channel_axis=channel_axis)

#         regions = regionprops(segments)
#         centers = np.array([region.centroid for region in regions])
#         labels = np.array([region.label for region in regions])

#         dists = squareform(pdist(centers))
#         merged = {label: label for label in labels}

#         for i in range(len(centers)):
#             for j in range(i + 1, len(centers)):
#                 if dists[i, j] < self.merge_threshold:
#                     li, lj = merged[labels[i]], merged[labels[j]]
#                     if li != lj:
#                         lower, higher = sorted((li, lj))
#                         for k in merged:
#                             if merged[k] == higher:
#                                 merged[k] = lower

#         label_map = {old: new for old, new in merged.items()}
#         new_segments = np.copy(segments)
#         for old, new in label_map.items():
#             new_segments[segments == old] = new

#         new_segments, _, _ = relabel_sequential(new_segments)
#         self.segments_slic = new_segments
#         self.superpixel_centers = [region.centroid for region in regionprops(new_segments)]

#     def create_grid(self):
#         if self.segments_slic is None:
#             raise ValueError("You must compute superpixels first.")

#         # Get image dimensions
#         img_height, img_width = self.img.shape[:2]

#         # k = len(self.superpixel_centers)

#         # # Binary search for optimal grid size
#         # lower_bound = int(np.ceil(np.sqrt(k)))
#         # upper_bound = 16 * lower_bound

#         # def is_valid_grid_size(n):
#         #     grid_height = img_height / n
#         #     grid_width = img_width / n
#         #     occupied_cells = set()

#         #     for region in regionprops(self.segments_slic):
#         #         y, x = region.centroid
#         #         grid_row = int(y // grid_height)
#         #         grid_col = int(x // grid_width)
#         #         if (grid_row, grid_col) in occupied_cells:
#         #             return False  # More than one superpixel center in a cell
#         #         occupied_cells.add((grid_row, grid_col))
#         #     return True

#         # for _ in range(5):  # Perform 5 binary search iterations
#         #     mid = (lower_bound + upper_bound) / 2
#         #     if is_valid_grid_size(mid):
#         #         upper_bound = mid
#         #     else:
#         #         lower_bound = mid + 1

#         # n = upper_bound

#         # # Ensure no cell has more than one superpixel center
#         # while not is_valid_grid_size(n):
#         #     n += 1

#         # n = math.ceil(n)

#         n = self.grid_size

#         # Grid cell size
#         grid_height = np.ceil(img_height / n)
#         grid_width = np.ceil(img_width / n)

#         # Initialize grids
#         self.grid = np.full((n, n, 3), -1, dtype=np.float32)  # Stores average color
#         self.sp_label_grid = np.full((n, n), -1, dtype=np.int32)  # Stores superpixel label

#         # Calculate the average color of each superpixel
#         average_colors = np.zeros((np.max(self.segments_slic) + 1, 3), dtype=np.float32)
#         for label in np.unique(self.segments_slic):
#             mask = self.segments_slic == label
#             average_colors[label] = self.img[mask].mean(axis=0)

#         # Populate the grids
#         for region in regionprops(self.segments_slic):
#             y, x = region.centroid
#             label = region.label

#             # Ensure indices stay within valid grid bounds
#             grid_row = min(int(y // grid_height), self.grid.shape[0] - 1)
#             grid_col = min(int(x // grid_width), self.grid.shape[1] - 1)

#             if np.all(self.grid[grid_row, grid_col] == -1):
#                 # self.grid[grid_row, grid_col] = average_colors[label]
#                 self.sp_label_grid[grid_row, grid_col] = label  # Store superpixel label


# # ---- DIRECTORY SETUP ----
# image_dirs = {
#     'CUB_train': '/Users/jroberts2/Jeova IS/CUB/train_images',
#     'CUB_test': '/Users/jroberts2/Jeova IS/CUB/test_images'
# }

# sp_label_grid_dirs = {
#     'CUB_train': '/Users/jroberts2/Jeova IS/CUB/sp_label_grid_train_500_10',
#     'CUB_test': '/Users/jroberts2/Jeova IS/CUB/sp_label_grid_test_500_10'
# }

# segments_slic_dirs = {
#     'CUB_train': '/Users/jroberts2/Jeova IS/CUB/segments_slic_train_500_10',
#     'CUB_test': '/Users/jroberts2/Jeova IS/CUB/segments_slic_test_500_10'
# }

# fixed_ns = [66, 57, 53, 33, 76, 69, 63, 42] #, 85, 77, 70, 50]

# # ---- PROCESSING FUNCTION ----
# def process_images(image_dir, sp_label_grid_dir, segments_slic_dir):
#     for root, _, files in os.walk(image_dir):
#         for image_name in files:
#             if image_name.endswith('.jpg') or image_name.endswith('.png'):
#                 image_path = os.path.join(root, image_name)

#                 # Preserve subfolder structure
#                 relative_path = os.path.relpath(root, image_dir)
#                 sp_label_grid_subdir = os.path.join(sp_label_grid_dir, relative_path)
#                 segments_slic_subdir = os.path.join(segments_slic_dir, relative_path)

#                 # Ensure directories exist
#                 os.makedirs(sp_label_grid_subdir, exist_ok=True)
#                 os.makedirs(segments_slic_subdir, exist_ok=True)

#                 # Define full save paths
#                 sp_label_grid_save_path = os.path.join(sp_label_grid_subdir, os.path.splitext(image_name)[0] + '.npy')
#                 segments_slic_save_path = os.path.join(segments_slic_subdir, os.path.splitext(image_name)[0] + '.npy')

#                 # Process the image
#                 processor = ImageProcessor(image_path, fixed_n)
#                 processor.load_image()
#                 processor.compute_superpixels()
#                 processor.create_grid()

#                 # Save grids and segmentation map
#                 np.save(sp_label_grid_save_path, processor.sp_label_grid) # maps cells to corresponding sp label 
#                 np.save(segments_slic_save_path, processor.segments_slic) # maps pixels to corresponding sp label 
#                 print(f"Saved grids and segmentation map for {image_name} in {relative_path}")


# # ---- MAIN FUNCTION ----
# def main():
#     for split in image_dirs.keys():
#         process_images(
#             image_dirs[split],
#             sp_label_grid_dirs[split],
#             segments_slic_dirs[split]
#         )

# if __name__ == "__main__":
#     main()
