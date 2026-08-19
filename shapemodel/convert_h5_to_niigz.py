import h5py
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm

def convert_h5_to_nifti(image_list, save_path, dataset='la'):
    """
    Convert .h5 files containing 'image' and 'label' datasets into .nii.gz format.
    
    Parameters:
        image_list (list): List of paths to .h5 files.
        save_path (str): Directory where the converted files will be saved.
        dataset (str): Dataset type 
    """
    os.makedirs(save_path, exist_ok=True)

    for image_path in tqdm(image_list):
        print(f"Processing: {image_path}")
        id = os.path.basename(image_path).split('.')[0]

        # Load image and label from the .h5 file
        with h5py.File(image_path, 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:]

        # Save the image and label as .nii.gz files
        nib.save(nib.Nifti1Image(image.astype(np.float32), np.eye(4)), os.path.join(save_path, f"{id}_img.nii.gz"))
        nib.save(nib.Nifti1Image(label.astype(np.float32), np.eye(4)), os.path.join(save_path, f"{id}_gt.nii.gz"))

        print(f"Saved: {id}_img.nii.gz and {id}_gt.nii.gz")

# original:
# input_directory = '/home/sci/asmak/Documents/Methods/ssm/Ground_Truth/20_percent_gt/org_20_percent_training_h5files'
# output_directory = '/home/sci/asmak/Documents/Methods/ssm/Ground_Truth/20_percent_gt'

input_directory = '/home/sci/asmak/Documents/Methods/ssm/Ground_Truth/femur/40_percent_gt/src data/org_40_percent_training_h5files'
output_directory = '/home/sci/asmak/Documents/Methods/ssm/Ground_Truth/femur/40_percent_gt'
# Get the list of .h5 files in the input directory
image_list = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.h5')]

convert_h5_to_nifti(image_list, output_directory)
