import os
import numpy as np
import nibabel as nib
import h5py
from tqdm import tqdm
from glob import glob

# Updated output size based on your knee dimensions [384, 384, 160]
output_size = [384, 384, 160]  # Keep original size or adjust as needed

image_dir = "/home/asmak/Documents/Datasets/Shapemedknee/MRIs/50_samples_train"
label_dir = "/home/asmak/Documents/Datasets/Shapemedknee/segs_original_split/50_segmentations_train"
output_dir = "/home/asmak/Documents/Methods/MT/data/knee/train"

os.makedirs(output_dir, exist_ok=True)

def center_crop_or_pad(image, label, target_size):
    """Center crop or pad to target size while preserving aspect ratio"""
    current_shape = image.shape
    
    processed_image = image.copy()
    processed_label = label.copy()
    
    for dim in range(3):
        current = current_shape[dim]
        target = target_size[dim]
        
        if current > target:
            # Need to crop
            start = (current - target) // 2
            end = start + target
            if dim == 0:
                processed_image = processed_image[start:end, :, :]
                processed_label = processed_label[start:end, :, :]
            elif dim == 1:
                processed_image = processed_image[:, start:end, :]
                processed_label = processed_label[:, start:end, :]
            else:
                processed_image = processed_image[:, :, start:end]
                processed_label = processed_label[:, :, start:end]
        elif current < target:
            # Need to pad
            pad_before = (target - current) // 2
            pad_after = target - current - pad_before
            pad_width = [(0, 0), (0, 0), (0, 0)]
            pad_width[dim] = (pad_before, pad_after)
            processed_image = np.pad(processed_image, pad_width, mode='constant', constant_values=0)
            processed_label = np.pad(processed_label, pad_width, mode='constant', constant_values=0)
    
    return processed_image, processed_label

def convert_nii_to_h5():
    # Debug: Check if directories exist
    print(f"Checking image directory: {image_dir}")
    print(f"Directory exists: {os.path.exists(image_dir)}")
    
    print(f"\nChecking label directory: {label_dir}")
    print(f"Directory exists: {os.path.exists(label_dir)}")
    
    # Try different glob patterns to find files
    print(f"\n--- Searching for image files ---")
    image_paths = sorted(glob(os.path.join(image_dir, "*.nii.gz")))
    print(f"Found {len(image_paths)} .nii.gz files")
    
    # If no .nii.gz files, try .nii
    if len(image_paths) == 0:
        image_paths = sorted(glob(os.path.join(image_dir, "*.nii")))
        print(f"Found {len(image_paths)} .nii files")
    
    # Show first few files found
    if len(image_paths) > 0:
        print(f"\nFirst 3 image files found:")
        for i, path in enumerate(image_paths[:3]):
            print(f"  {i+1}. {os.path.basename(path)}")
    else:
        print("\nNo image files found! Listing all files in directory:")
        all_files = os.listdir(image_dir) if os.path.exists(image_dir) else []
        for f in all_files[:10]:
            print(f"  - {f}")
        return
    
    # Check label directory
    print(f"\n--- Checking label files ---")
    label_files = sorted(glob(os.path.join(label_dir, "*.nii.gz")))
    if len(label_files) == 0:
        label_files = sorted(glob(os.path.join(label_dir, "*.nii")))
    
    print(f"Found {len(label_files)} label files")
    if len(label_files) > 0:
        print(f"First 3 label files:")
        for i, path in enumerate(label_files[:3]):
            print(f"  {i+1}. {os.path.basename(path)}")
    
    # Process files
    print(f"\n--- Starting conversion ---")
    print("Using Label 1 for BONE segmentation")
    
    successful = 0
    failed = 0
    
    for image_path in tqdm(image_paths):
        basename = os.path.basename(image_path)
        # Remove extension (.nii.gz or .nii)
        if basename.endswith('.nii.gz'):
            basename_no_ext = basename.replace(".nii.gz", "")
        else:
            basename_no_ext = basename.replace(".nii", "")
        
        # Try multiple label naming patterns
        label_patterns = [
            f"{basename_no_ext}-label.nii.gz",
            f"{basename_no_ext}-label.nii",
            f"{basename_no_ext}_label.nii.gz",
            f"{basename_no_ext}_label.nii",
            f"{basename_no_ext}_seg.nii.gz",
            f"{basename_no_ext}_seg.nii",
        ]
        
        label_path = None
        for pattern in label_patterns:
            potential_path = os.path.join(label_dir, pattern)
            if os.path.exists(potential_path):
                label_path = potential_path
                break
        
        if label_path is None:
            print(f"\nSegmentation not found for {basename_no_ext}")
            print(f"  Tried patterns: {label_patterns}")
            failed += 1
            continue
        
        try:
            # Load image and label
            image = nib.load(image_path).get_fdata()
            label = nib.load(label_path).get_fdata()
            
            # Keep only label 1 (bone)
            label = (label == 1).astype(np.uint8)
            
            # Normalize image
            image = (image - np.mean(image)) / np.std(image)
            image = image.astype(np.float32)
            
            # Center crop or pad to target size
            image_processed, label_processed = center_crop_or_pad(image, label, output_size)
            
            # Save to .h5
            save_path = os.path.join(output_dir, basename_no_ext + ".h5")
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('image', data=image_processed, compression='gzip')
                f.create_dataset('label', data=label_processed, compression='gzip')
            
            successful += 1
            
        except Exception as e:
            print(f"\nError processing {basename_no_ext}: {str(e)}")
            failed += 1
    
    print(f"\n--- Conversion Complete ---")
    print(f"Successfully converted: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    convert_nii_to_h5()

# import numpy as np
# from glob import glob
# from tqdm import tqdm
# import h5py
# import nrrd

# output_size =[112, 112, 80]

# def covert_h5():
#     listt = glob('/home/asmak/Documents/Methods/MT/data/knee/train/*.nii.gz')
#     for item in tqdm(listt):
#         image, img_header = nrrd.read(item)
#         label, gt_header = nrrd.read(item.replace('lgemri.nrrd', 'laendo.nrrd'))
#         label = (label == 255).astype(np.uint8)
#         w, h, d = label.shape

#         tempL = np.nonzero(label)
#         minx, maxx = np.min(tempL[0]), np.max(tempL[0])
#         miny, maxy = np.min(tempL[1]), np.max(tempL[1])
#         minz, maxz = np.min(tempL[2]), np.max(tempL[2])

#         px = max(output_size[0] - (maxx - minx), 0) // 2
#         py = max(output_size[1] - (maxy - miny), 0) // 2
#         pz = max(output_size[2] - (maxz - minz), 0) // 2
#         minx = max(minx - np.random.randint(10, 20) - px, 0)
#         maxx = min(maxx + np.random.randint(10, 20) + px, w)
#         miny = max(miny - np.random.randint(10, 20) - py, 0)
#         maxy = min(maxy + np.random.randint(10, 20) + py, h)
#         minz = max(minz - np.random.randint(5, 10) - pz, 0)
#         maxz = min(maxz + np.random.randint(5, 10) + pz, d)

#         image = (image - np.mean(image)) / np.std(image)
#         image = image.astype(np.float32)
#         image = image[minx:maxx, miny:maxy]
#         label = label[minx:maxx, miny:maxy]
#         print(label.shape)
#         f = h5py.File(item.replace('lgemri.nrrd', 'mri_norm2.h5'), 'w')
#         f.create_dataset('image', data=image, compression="gzip")
#         f.create_dataset('label', data=label, compression="gzip")
#         f.close()

# if __name__ == '__main__':
#     covert_h5()