"""Convert image/segmentation pairs listed in a CSV to normalized HDF5 cases.

This utility was originally tied to a developer-local vertebra dataset. The
public-release version takes every filesystem location from the command line.
It is not required for the paper's FEMUR/NAMIC ShapeWorks workflow.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import shapeworks as sw
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="CSV containing image/label metadata.")
    parser.add_argument(
        "--particle-list",
        type=Path,
        required=True,
        help="Text file containing selected case names or .particles filenames.",
    )
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing input NIfTI images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for output .h5 files.")
    parser.add_argument(
        "--label-column",
        default="vertebrae_L1",
        help="CSV column containing segmentation paths (default: vertebrae_L1).",
    )
    parser.add_argument(
        "--image-column",
        default="image",
        help="CSV column used to derive case names (default: image).",
    )
    parser.add_argument(
        "--isotropic-spacing",
        type=float,
        default=1.0,
        help="Isotropic resampling spacing (default: 1.0).",
    )
    parser.add_argument("--crop-padding", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.expanduser().resolve()
    particle_list = args.particle_list.expanduser().resolve()
    image_dir = args.image_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if args.image_column not in df or args.label_column not in df:
        raise KeyError(
            f"CSV must contain columns {args.image_column!r} and {args.label_column!r}."
        )
    df["name"] = df[args.image_column].apply(lambda value: Path(str(value)).parent.name)

    selected_names = {
        line.strip().removesuffix(".particles")
        for line in particle_list.read_text().splitlines()
        if line.strip()
    }
    selected_files = df[df["name"].isin(selected_names)]

    for _, row in tqdm(selected_files.iterrows(), total=len(selected_files)):
        if pd.isna(row[args.label_column]):
            continue

        image = sw.Image(str(image_dir / f"{row['name']}.nii.gz"))
        label = sw.Image(str(Path(str(row[args.label_column])).expanduser()))

        spacing = [args.isotropic_spacing] * 3
        label.resample(spacing, sw.InterpolationType.Linear)
        image.resample(spacing, sw.InterpolationType.Linear)
        label.binarize()

        bounding_box = sw.ImageUtils.boundingBox([label], 0.5).pad(args.crop_padding)
        label.crop(bounding_box)
        image.crop(bounding_box)

        image_array = image.toArray().transpose()
        label_array = label.toArray().transpose()
        std = float(np.std(image_array))
        if std == 0:
            raise ValueError(f"Image has zero standard deviation: {row['name']}")
        image_array = ((image_array - np.mean(image_array)) / std).astype(np.float32)

        with h5py.File(output_dir / f"{row['name']}.h5", "w") as handle:
            handle.create_dataset("image", data=image_array, compression="gzip")
            handle.create_dataset("label", data=label_array, compression="gzip")


if __name__ == "__main__":
    main()
