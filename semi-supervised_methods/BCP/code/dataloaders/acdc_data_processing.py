#!/usr/bin/env python3
"""Convert ACDC NIfTI volumes into per-slice HDF5 files.

The original upstream utility embedded a developer workstation path. This
public-release version requires explicit input/output directories.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir.expanduser().resolve()
    label_dir = args.label_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.nii.gz"))
    if not image_paths:
        raise FileNotFoundError(f"No .nii.gz files found in: {image_dir}")

    slice_num = 0
    for image_path in image_paths:
        img_itk = sitk.ReadImage(str(image_path))
        image = sitk.GetArrayFromImage(img_itk)
        item = image_path.name.removesuffix(".nii.gz")
        label_path = label_dir / f"{item}_gt.nii.gz"
        if not label_path.exists():
            print(f"Skipping {item}: label not found at {label_path}")
            continue

        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
        if image.shape != mask.shape:
            raise ValueError(f"Shape mismatch for {item}: {image.shape} vs {mask.shape}")

        denom = float(image.max() - image.min())
        image = ((image - image.min()) / denom if denom > 0 else image * 0).astype(np.float32)
        for slice_ind in range(image.shape[0]):
            output_path = output_dir / f"{item}_slice_{slice_ind}.h5"
            with h5py.File(output_path, "w") as handle:
                handle.create_dataset("image", data=image[slice_ind], compression="gzip")
                handle.create_dataset("label", data=mask[slice_ind], compression="gzip")
            slice_num += 1

    print("Converted all ACDC volumes to 2D slices")
    print(f"Total {slice_num} slices")


if __name__ == "__main__":
    main()
