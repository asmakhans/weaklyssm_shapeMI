"""Create sparsely labelled femur HDF5 files for Mean Teacher.

The original script referenced a developer-local benchmark directory. This
public-release version takes all filesystem locations as command-line arguments.
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


def process(input_dir: Path, output_dir: Path, num_labeled: int) -> None:
    image_list = sorted(input_dir.glob("*.h5"))
    if not image_list:
        raise FileNotFoundError(f"No .h5 files found in: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, input_path in enumerate(image_list):
        with h5py.File(input_path, "r") as h5f:
            image = h5f["image"][:]
            label = h5f["label"][:]

        start_position, end_position = get_range_image_depth(label)
        random_start = int(start_position + (end_position - start_position) / 5)
        random_end = int(end_position - (end_position - start_position) / 5)
        if random_end <= random_start:
            raise ValueError(f"Foreground depth range is too small for: {input_path}")
        label_index = np.random.randint(random_start, random_end, size=1)

        new_label = np.zeros(label.shape, dtype=label.dtype)
        if index < num_labeled:
            new_label[..., label_index] = label[..., label_index]

        with h5py.File(output_dir / input_path.name, "w") as save_file:
            save_file.create_dataset("image", data=image)
            save_file.create_dataset("label_full", data=label)
            save_file.create_dataset("label", data=new_label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-labeled",
        type=int,
        default=16,
        help="Number of cases that retain a sampled labelled slice (default: 16).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(
        args.input_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.num_labeled,
    )
