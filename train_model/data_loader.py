import os
import numpy as np
import nibabel as nib
from sklearn.utils import shuffle
import random
import scipy.ndimage

def load_data(healthy_dir, tumor_dir, max_per_class = 50, target_shape = (64,64,64)):
    X = []
    y = []

    healthy_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(healthy_dir)
        for file in files if file.endswith(".nii.gz")
    ]

    tumor_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(tumor_dir)
        for file in files if file.endswith(".nii.gz")
    ]

#Step 2: Sample the desired number per class
    random.seed(42)
    healthy_sample = random.sample(healthy_files, min(max_per_class, len(healthy_files)))
    tumor_sample = random.sample(tumor_files, min(max_per_class, len(tumor_files)))
#Step 3: Load and flatten each sampled image
    def load_and_flatten(path):
        img =nib.load(path).get_fdata()
        
        if img.ndim == 4:
            img = img[..., 0]
        
        img = np.squeeze(img)

        if img.ndim != 3:
            raise ValueError(f"[ERROR] Expected 3D image but got shape {img.shape} from {path}")
        
        zoom_factors = [t / s for t,s in zip(target_shape, img.shape)]
        img_resized = scipy.ndimage.zoom(img, zoom = zoom_factors)
        
        if img_resized.shape != target_shape:
            raise ValueError(f"[ERROR] After resizing, shape is {img_resized.shape} instead of {target_shape} for file {path}")
        
        flat = img_resized.flatten()
        return flat

    for path in healthy_sample:
        X.append(load_and_flatten(path))
        y.append(0)
    for path in tumor_sample:
        X.append(load_and_flatten(path))
        y.append(1)

#Step 4: Final shape check and shuffle
    print(f"✅ All MRI volumes flattened to vectors of length: {len(X[0])}")
    print()
#Step 5: Shuffle and return as NumPy arrays
    X, y = shuffle(X, y, random_state=42)
    return np.array(X), np.array(y)