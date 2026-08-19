#!/usr/bin/env python3
"""Convert HDF5 image/label pairs to NIfTI without machine-specific paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing .h5 files with `image` and `label` datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated *_img.nii.gz and *_gt.nii.gz files.",
    )
    return parser.parse_args()


def convert_h5_to_nifti(image_list: list[Path], save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    for image_path in tqdm(image_list):
        print(f"Processing: {image_path}")
        sample_id = image_path.stem
        with h5py.File(image_path, "r") as h5f:
            image = h5f["image"][:]
            label = h5f["label"][:]

        nib.save(
            nib.Nifti1Image(image.astype(np.float32), np.eye(4)),
            save_path / f"{sample_id}_img.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(label.astype(np.float32), np.eye(4)),
            save_path / f"{sample_id}_gt.nii.gz",
        )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    image_list = sorted(input_dir.glob("*.h5"))
    if not image_list:
        raise FileNotFoundError(f"No .h5 files found in: {input_dir}")
    convert_h5_to_nifti(image_list, output_dir)


if __name__ == "__main__":
    main()
