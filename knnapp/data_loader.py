import nibabel as nib
import numpy as np
import os

def load_nii(file_path):
    """Loads a NIfTI file and returns the 3D volume as a numpy array."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def load_dataset(folder_path, label):
    """Load all .nii.gz files from a folder and assign labels."""
    dataset = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".nii.gz"):
                full_path = os.path.join(root, file)
                data = load_nii(full_path)
                # Select a central slice to reduce dimensionality
                slice_index = data.shape[2] // 2
                slice_data = data[:, :, slice_index]
                dataset.append((slice_data, label))
    return dataset