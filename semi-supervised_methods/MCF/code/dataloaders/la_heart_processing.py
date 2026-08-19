"""Preprocess LA Heart NRRD cases into the HDF5 layout used by MCF.

The original preprocessing script contained a developer-local Windows drive
path. This version is path-agnostic and writes processed cases under a supplied
output directory.
"""

import argparse
from pathlib import Path

import h5py
import nrrd
import numpy as np
from tqdm import tqdm

OUTPUT_SIZE = [112, 112, 80]


def convert_h5(data_root: Path, output_root: Path, output_name: str) -> None:
    image_paths = sorted(data_root.glob("*/lgemri.nrrd"))
    if not image_paths:
        raise FileNotFoundError(f"No */lgemri.nrrd files found under: {data_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths):
        label_path = image_path.with_name("laendo.nrrd")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")

        image, _ = nrrd.read(str(image_path))
        label, _ = nrrd.read(str(label_path))
        label = (label == 255).astype(np.uint8)
        w, h, d = label.shape

        nonzero = np.nonzero(label)
        if not nonzero[0].size:
            raise ValueError(f"Label contains no foreground voxels: {label_path}")
        minx, maxx = np.min(nonzero[0]), np.max(nonzero[0])
        miny, maxy = np.min(nonzero[1]), np.max(nonzero[1])
        minz, maxz = np.min(nonzero[2]), np.max(nonzero[2])

        px = max(OUTPUT_SIZE[0] - (maxx - minx), 0) // 2
        py = max(OUTPUT_SIZE[1] - (maxy - miny), 0) // 2
        pz = max(OUTPUT_SIZE[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        image = image.astype(np.uint8)
        image = image[minx:maxx, miny:maxy, minz:maxz]
        label = label[minx:maxx, miny:maxy, minz:maxz]

        case_output = output_root / image_path.parent.name
        case_output.mkdir(parents=True, exist_ok=True)
        with h5py.File(case_output / output_name, "w") as handle:
            handle.create_dataset("image", data=image, compression="gzip")
            handle.create_dataset("label", data=label, compression="gzip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-name", default="wyc_mri_norm0_255.h5")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_h5(
        args.data_root.expanduser().resolve(),
        args.output_root.expanduser().resolve(),
        args.output_name,
    )
