"""Preprocess LA Heart NRRD cases into the HDF5 layout used by BCP.

The script is intentionally path-agnostic for public release. Point ``--data-root``
at a directory containing one subdirectory per case. Each case is expected to
contain ``lgemri.nrrd`` and ``laendo.nrrd``. The processed HDF5 file is written
inside each case directory by default.
"""

import argparse
from pathlib import Path

import h5py
import nrrd
import numpy as np
from tqdm import tqdm

OUTPUT_SIZE = [112, 112, 80]


def convert_h5(data_root: Path, output_name: str = "mri_norm2.h5") -> None:
    image_paths = sorted(data_root.glob("*/lgemri.nrrd"))
    if not image_paths:
        raise FileNotFoundError(
            f"No */lgemri.nrrd files found under data root: {data_root}"
        )

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

        std = float(np.std(image))
        if std == 0:
            raise ValueError(f"Image has zero standard deviation: {image_path}")
        image = ((image - np.mean(image)) / std).astype(np.float32)
        image = image[minx:maxx, miny:maxy, minz:maxz]
        label = label[minx:maxx, miny:maxy, minz:maxz]

        output_path = image_path.parent / output_name
        with h5py.File(output_path, "w") as handle:
            handle.create_dataset("image", data=image, compression="gzip")
            handle.create_dataset("label", data=label, compression="gzip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing one LA case directory per subject.",
    )
    parser.add_argument(
        "--output-name",
        default="mri_norm2.h5",
        help="Output HDF5 filename written inside each case directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_h5(args.data_root.expanduser().resolve(), args.output_name)
