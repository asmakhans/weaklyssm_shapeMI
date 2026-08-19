import os
import glob
import shapeworks as sw
import numpy as np

import h5py
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("complete_shapes.csv")
df["name"] = df["image"].apply(lambda x: x.split('/')[-2])

with open("vertebrae_L1/train.out", "r") as file:
    file_names = [line.strip().replace(".particles", "") for line in file if line.strip().endswith(".particles")]

selected_files = df[df["name"].isin(file_names)]

for index in tqdm(range(len(selected_files))):
    row = selected_files.iloc[index]
    if pd.isna(row["vertebrae_L1"]):
        continue
    image = sw.Image("../../../DeepSSM/Janmesh_Data/pre_images/"+row["name"]+".nii.gz")
    label = sw.Image(row["vertebrae_L1"])

    iso_spacing = [1, 1, 1]
    label.resample(iso_spacing, sw.InterpolationType.Linear)
    image.resample(iso_spacing, sw.InterpolationType.Linear)
    label.binarize()
    
    bounding_box = sw.ImageUtils.boundingBox([label], 0.5).pad(5)
    label.crop(bounding_box)
    image.crop(bounding_box)

    image = image.toArray().transpose()
    label = label.toArray().transpose()


    image = (image - np.mean(image)) / np.std(image)
    image = image.astype(np.float32)

    f = h5py.File(os.path.join("../../benchmark_self-supervise/datasets/vertebrae_L1/h5/train", row["name"]+".h5"), 'w')
    f.create_dataset('image', data=image, compression="gzip")
    f.create_dataset('label', data=label, compression="gzip")
    f.close()