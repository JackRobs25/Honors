import os
import numpy as np

# Base path
base_path = '/Users/jroberts2/Jeova IS/CUB'

# Desired combinations of grid sizes and thresholds
grid_sizes = [300, 500, 700]
thresholds = [5, 10, 20, 100]

# Store results
results = {}

for g in grid_sizes:
    for t in thresholds:
        folder_prefix = f"{g}_{t}_grid_mask_test_3"
        folder_path = os.path.join(base_path, folder_prefix)
        
        if not os.path.isdir(folder_path):
            print(f"Skipping missing directory: {folder_path}")
            continue

        # Walk through species subfolders
        for species_folder in os.listdir(folder_path):
            species_path = os.path.join(folder_path, species_folder)
            if not os.path.isdir(species_path):
                continue

            # Look for the first .npy file
            for file_name in os.listdir(species_path):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(species_path, file_name)
                    data = np.load(file_path)
                    if data.ndim == 2 and data.shape[0] == data.shape[1]:
                        results[f"{g}_{t}"] = data.shape[0]
                        print(f"{g}_{t}: grid size = {data.shape[0]}x{data.shape[1]}")
                    else:
                        print(f"{g}_{t}: invalid shape {data.shape} in {file_path}")
                    break  # Only take the first .npy file
            break  # Only check one species folder

# Optional: print all results
print("\nSummary:")
for key, size in results.items():
    print(f"{key}: {size}x{size}")
