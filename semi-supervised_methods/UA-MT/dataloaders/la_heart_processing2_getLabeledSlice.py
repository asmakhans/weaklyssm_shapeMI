"""Create sparsely labelled LA Heart HDF5 files for UA-MT.

All input/output locations are supplied on the command line so the script can be
used outside the original developer environment.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

np.random.seed(309)


def get_range_image_depth(image):
    first_flag = True
    start_position = 0
    end_position = 0
    for z in range(image.shape[2]):
        nonzero_flag = np.max(image[..., z])
        if nonzero_flag and first_flag:
            start_position = z
            first_flag = False
        if nonzero_flag:
            end_position = z
    return start_position, end_position


def process(input_root: Path, output_root: Path, list_file: Path, num_labeled: int) -> None:
    image_list = [line.strip() for line in list_file.read_text().splitlines() if line.strip()]
    output_root.mkdir(parents=True, exist_ok=True)

    for index, image_name in enumerate(image_list):
        input_path = input_root / image_name / "mri_norm2.h5"
        if not input_path.exists():
            raise FileNotFoundError(f"Input case not found: {input_path}")

        with h5py.File(input_path, "r") as h5f:
            image = h5f["image"][:]
            label = h5f["label"][:]

        start_position, end_position = get_range_image_depth(label)
        random_start = int(start_position + (end_position - start_position) / 5)
        random_end = int(end_position - (end_position - start_position) / 5)
        if random_end <= random_start:
            raise ValueError(f"Foreground depth range is too small for case: {image_name}")
        label_index = np.random.randint(random_start, random_end, size=1)

        new_label = np.zeros(label.shape, dtype=label.dtype)
        if index < num_labeled:
            new_label[..., label_index] = label[..., label_index]

        case_output = output_root / image_name
        case_output.mkdir(parents=True, exist_ok=True)
        with h5py.File(case_output / "mri_norm2.h5", "w") as save_file:
            save_file.create_dataset("image", data=image)
            save_file.create_dataset("label_full", data=label)
            save_file.create_dataset("label", data=new_label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--list-file",
        type=Path,
        required=True,
        help="Text file containing case IDs, one per line.",
    )
    parser.add_argument(
        "--num-labeled",
        type=int,
        default=32,
        help="Number of cases that retain a sampled labelled slice (default: 32).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(
        args.input_root.expanduser().resolve(),
        args.output_root.expanduser().resolve(),
        args.list_file.expanduser().resolve(),
        args.num_labeled,
    )
