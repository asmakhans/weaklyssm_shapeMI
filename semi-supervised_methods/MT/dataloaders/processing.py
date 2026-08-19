#!/usr/bin/env python3
"""Convert paired NIfTI images/labels to HDF5 for MT experiments.

All input/output directories are command-line arguments; no local workstation
paths are embedded in the public version.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=3,
        default=(384, 384, 160),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--label-value",
        type=int,
        default=1,
        help="Label value treated as foreground (default: 1).",
    )
    return parser.parse_args()


def center_crop_or_pad(image: np.ndarray, label: np.ndarray, target_size) -> tuple[np.ndarray, np.ndarray]:
    """Center crop or pad each dimension to target_size."""
    processed_image = image.copy()
    processed_label = label.copy()
    for dim, target in enumerate(target_size):
        current = processed_image.shape[dim]
        if current > target:
            start = (current - target) // 2
            end = start + target
            slices = [slice(None)] * 3
            slices[dim] = slice(start, end)
            processed_image = processed_image[tuple(slices)]
            processed_label = processed_label[tuple(slices)]
        elif current < target:
            before = (target - current) // 2
            after = target - current - before
            pad_width = [(0, 0)] * 3
            pad_width[dim] = (before, after)
            processed_image = np.pad(processed_image, pad_width, mode="constant", constant_values=0)
            processed_label = np.pad(processed_label, pad_width, mode="constant", constant_values=0)
    return processed_image, processed_label


def find_label(label_dir: Path, basename: str) -> Path | None:
    candidates = [
        f"{basename}-label.nii.gz",
        f"{basename}-label.nii",
        f"{basename}_label.nii.gz",
        f"{basename}_label.nii",
        f"{basename}_seg.nii.gz",
        f"{basename}_seg.nii",
    ]
    return next((label_dir / name for name in candidates if (label_dir / name).exists()), None)


def convert_nii_to_h5(image_dir: Path, label_dir: Path, output_dir: Path, target_size, label_value: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(image_dir.glob("*.nii.gz")) or sorted(image_dir.glob("*.nii"))
    if not image_paths:
        raise FileNotFoundError(f"No NIfTI images found in: {image_dir}")

    successful = 0
    failed = 0
    for image_path in tqdm(image_paths):
        basename = image_path.name.removesuffix(".nii.gz").removesuffix(".nii")
        label_path = find_label(label_dir, basename)
        if label_path is None:
            print(f"Segmentation not found for {basename}")
            failed += 1
            continue
        try:
            image = nib.load(str(image_path)).get_fdata()
            label = nib.load(str(label_path)).get_fdata()
            label = (label == label_value).astype(np.uint8)
            std = float(np.std(image))
            image = ((image - np.mean(image)) / std if std > 0 else image - np.mean(image)).astype(np.float32)
            image, label = center_crop_or_pad(image, label, target_size)
            with h5py.File(output_dir / f"{basename}.h5", "w") as handle:
                handle.create_dataset("image", data=image, compression="gzip")
                handle.create_dataset("label", data=label, compression="gzip")
            successful += 1
        except Exception as exc:
            print(f"Error processing {basename}: {exc}")
            failed += 1

    print(f"Successfully converted: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")


def main() -> None:
    args = parse_args()
    convert_nii_to_h5(
        args.image_dir.expanduser().resolve(),
        args.label_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        tuple(args.target_size),
        args.label_value,
    )


if __name__ == "__main__":
    main()
