import os
import numpy as np
import nibabel as nib
from sklearn.utils import shuffle
import random
import scipy.ndimage as ndi
import re

#helper functions
def robust_zscore(x, eps = 1e-6):
    """robust per volume normalization: (x - median) / (1.4826*MAD)"""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med)/(1.4826*mad)

def brain_mask_heuristic(vol, pct=20, min_size_vox=1000):
    """fast, per volume mask:
        1) robust normalize
        2)threshold by percentile (acts like 0tsu without dataset globals)
        3) keep largest connected component
        4) light marphology to fill small holes
    returns bolean mask.
    """
    z = robust_zscore(vol)
    thr = np.percentile(z, pct)
    rough = z > thr

    lbl, n = ndi.label(rough)
    if n == 0:
        return np.ones_like(vol, dytpe=bool)
    #largest component
    sizes = ndi.sum(np.ones_like(vol), lbl, index=np.arange(1,n+1))
    mask = (lbl ==(1+np.argmax(sizes)))

    #clean up
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=2)

    if mask.sum() < min_size_vox:
        return np.ones_like(vol, dtype=bool)
    
    return mask

def resize_3d(vol, target_shape=(170,170,170)):
    """3D resize with linear interpolation"""
    zoom_factors = [t/s for t, s in zip(target_shape, vol.shape)]
    return ndi.zoom(vol, zoom_factors, order=3)
    
def load_single_nifti(path, target_shape=(170,170,170), time_strategy="mean4d"):
    """
    load .nii.gz -> 3d brian volume (masked, normalized, resized).
        -time_strategy: "first4d" or "mean4d" for 4D inputs.
    """
    vol = nib.load(path).get_fdata()
    #handle 4D volumes
    if vol.ndim == 4:
        if time_strategy == "first4d":
            vol = vol[..., 0]
        else:  # "mean4d"
            vol = vol.mean(axis=3)
    elif vol.ndim == 3:
        pass  # already 3D
    elif vol.ndim == 2:
        # Rare edge case: promote to 3D with a singleton z-dimension
        vol = vol[..., np.newaxis]
    else:
        raise ValueError(f"[ERROR] Unexpected ndim={vol.ndim} for {path}")
    #Per-volume mask applied to all scans (healthy & tumor)
    vol = np.squeeze(vol)

    if vol.ndim != 3:
        raise ValueError(f"[ERROR] Expected 3D after reduction, got {vol.shape} from {path}")

    mask = brain_mask_heuristic(vol)
    #nomralize inside mask only (prevents global leakage)
    if mask.any():
        m = vol[mask].mean()
        s = vol[mask].std() + 1e-6
        vol = (vol - m) / s

    # zero out non-brain voxels uniformly
    vol = vol * mask
    #Resize to common grid
    vol = resize_3d(vol, target_shape)

    if vol.shape != target_shape:
        raise ValueError(f"[Error] After resizing{path}, got {vol.shape}, expected {target_shape}")
    return vol

def extract_subject_id(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]      # removes only the last extension (e.g., .gz), OK
    stem = re.sub(r'\.nii$', '', stem)    # remove leftover .nii if present

    # BIDS-style: sub-XXXX
    m = re.search(r'(sub-[A-Za-z0-9]+)', stem)
    if m:
        return m.group(1)

    # YG_<alnum+>_[rest...]  -> capture 'YG_<alnum+>'
    m = re.match(r'(YG_[A-Za-z0-9]+)', stem)
    if m:
        return m.group(1)

    # Fallback: first two tokens if they look like PREFIX_ID
    parts = stem.split('_')
    if len(parts) >= 2 and re.fullmatch(r'[A-Za-z]+', parts[0]) and re.fullmatch(r'[A-Za-z0-9]+', parts[1]):
        return f"{parts[0]}_{parts[1]}"

    # Last resort: first token
    return parts[0] if parts else stem

def load_data(healthy_dir, tumor_dir, max_per_class=120, target_shape=(170, 170, 170), time_strategy="first4d"):
    """
    Loads NIfTI files from healthy_dir (label=0) and tumor_dir (label=1),
    resizes to target_shape, flattens, and returns X, y, paths.
    The returned arrays/lists are aligned and shuffled together.
    """
    import numpy as np
    import nibabel as nib
    import random
    from sklearn.utils import shuffle
    import scipy.ndimage as ndi
    import os

    def resize_3d(vol, out_shape):
        zoom_factors = [t / s for t, s in zip(out_shape, vol.shape)]
        return ndi.zoom(vol, zoom=zoom_factors, order=1)

    def brain_mask_heuristic(vol):
        # simple per-volume intensity threshold; tweak as needed
        thr = np.percentile(vol, 40)
        return vol > thr

    def load_single_nifti(path, target_shape=(170,170,170), time_strategy="first4d"):
        vol = nib.load(path).get_fdata()

        # Handle 4D volumes
        if vol.ndim == 4:
            if time_strategy == "first4d":
                vol = vol[..., 0]
            elif time_strategy == "mean4d":
                vol = vol.mean(axis=3)
            else:
                # default: first volume
                vol = vol[..., 0]

        vol = np.squeeze(vol)
        if vol.ndim != 3:
            raise ValueError(f"[Error] Expected 3D after squeeze, got {vol.shape} from {path}")

        # Per-volume mask (same rule for all classes to avoid leakage)
        mask = brain_mask_heuristic(vol)

        # Normalize inside mask only (no global stats)
        if mask.any():
            m = vol[mask].mean()
            s = vol[mask].std() + 1e-6
            vol = (vol - m) / s

        # zero-out background uniformly
        vol = vol * mask

        # Resize
        vol = resize_3d(vol, target_shape)

        if vol.shape != target_shape:
            raise ValueError(f"[Error] After resizing {path}, got {vol.shape}, expected {target_shape}")
        return vol

    # 1) Enumerate files
    healthy_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(healthy_dir)
        for f in files if f.endswith(".nii.gz")
    ]
    tumor_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(tumor_dir)
        for f in files if f.endswith(".nii.gz")
    ]

    print(f"🧠 Found {len(healthy_files)} healthy brain images.")
    print(f"🧠 Found {len(tumor_files)} tumor brain images.")
    print("Healthy sample files (up to 3):", healthy_files[:3])
    print("Tumor sample files (up to 3):", tumor_files[:3])

    # 2) Sample per class
    random.seed(42)
    healthy_sample = random.sample(healthy_files, min(max_per_class, len(healthy_files)))
    tumor_sample   = random.sample(tumor_files,   min(max_per_class, len(tumor_files)))

    # 3) Load, flatten, collect paths in the SAME order
    X, y, paths = [], [], []

    def append_one(p, label):
        vol = load_single_nifti(p, target_shape=target_shape, time_strategy=time_strategy)
        X.append(vol.flatten())
        y.append(label)
        paths.append(p)

    for p in healthy_sample:
        append_one(p, 0)
    for p in tumor_sample:
        append_one(p, 1)

    if not X:
        raise ValueError("❌ No MRI data was loaded. Check directories and .nii.gz files.")

    print(f"✅ All MRI volumes flattened to vectors of length: {len(X[0])}")

    # 4) Shuffle X, y, paths together to preserve alignment
    X, y, paths = shuffle(X, y, paths, random_state=42)
    return np.array(X), np.array(y), np.array(paths)