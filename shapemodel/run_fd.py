#!/usr/bin/env python3
"""Build a ShapeWorks training statistical shape model from segmentations.

This script is the first ShapeWorks stage used by both paper strategies. It is
path-agnostic: all input/output locations are derived from ``--train-dir``.

Expected input
--------------
``--train-dir`` should contain the training segmentations as ``*.nii.gz``.
The script writes grooming outputs, the selected reference shape, and the
ShapeWorks project under the same directory.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import shapeworks as sw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Groom training segmentations and build a ShapeWorks training SSM."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="Directory containing training segmentations (*.nii.gz).",
    )
    parser.add_argument(
        "--project-name",
        default="la",
        help="ShapeWorks project/spreadsheet base name (default: la).",
    )
    parser.add_argument("--particles", type=int, default=1024)

    # Grooming parameters reported for the paper experiments.
    parser.add_argument("--iso-value", type=float, default=0.5)
    parser.add_argument("--antialias-iterations", type=int, default=30)
    parser.add_argument("--isotropic-spacing", type=float, default=1.0)
    parser.add_argument("--initial-pad", type=int, default=30)
    parser.add_argument("--post-registration-pad", type=int, default=10)
    parser.add_argument("--icp-iterations", type=int, default=200)
    parser.add_argument("--gaussian-sigma", type=float, default=1.5)

    # Optimization parameters. Defaults match the implementation details
    # reported in the paper; they can be overridden from the command line.
    parser.add_argument("--iterations-per-split", type=int, default=1000)
    parser.add_argument("--optimization-iterations", type=int, default=500)
    parser.add_argument("--starting-regularization", type=float, default=1000)
    parser.add_argument("--ending-regularization", type=float, default=10)
    parser.add_argument("--recompute-regularization-interval", type=int, default=2)
    parser.add_argument(
        "--procrustes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable Procrustes alignment during optimization.",
    )
    parser.add_argument(
        "--run-optimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run `shapeworks optimize` after writing the project.",
    )
    parser.add_argument(
        "--launch-studio",
        action="store_true",
        help="Open the generated project in ShapeWorksStudio after optimization.",
    )
    return parser.parse_args()


def require_segmentations(train_dir: Path) -> list[Path]:
    files = sorted(train_dir.glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(
            f"No .nii.gz training segmentations were found in: {train_dir}"
        )
    return files


def main() -> None:
    args = parse_args()
    train_dir = args.train_dir.expanduser().resolve()
    train_dir.mkdir(parents=True, exist_ok=True)
    train_files = require_segmentations(train_dir)

    groom_dir = train_dir / "groomed"
    groom_dir.mkdir(parents=True, exist_ok=True)

    shape_segmentations: list[sw.Image] = []
    shape_names: list[str] = []
    spacing = [args.isotropic_spacing] * 3

    for shape_path in train_files:
        shape_name = shape_path.name.removesuffix(".nii.gz")
        print(f"Loading: {shape_path}")
        shape = sw.Image(str(shape_path))
        shape_names.append(shape_name)
        shape_segmentations.append(shape)

        print(f"Grooming: {shape_name}")
        bbox = sw.ImageUtils.boundingBox([shape], args.iso_value).pad(2)
        shape.crop(bbox)
        shape.antialias(args.antialias_iterations).resample(
            spacing, sw.InterpolationType.Linear
        ).binarize()
        shape.pad(args.initial_pad, 0)

    ref_index = sw.find_reference_image_index(shape_segmentations)
    ref_name = shape_names[ref_index]
    ref_seg = shape_segmentations[ref_index].write(str(groom_dir / f"{ref_name}.nrrd"))
    ref_seg.write(str(groom_dir / "reference.nrrd"))
    (groom_dir / "reference_shape.txt").write_text(ref_name + "\n")
    print(f"Reference found: {ref_name}")

    # Images are explicitly transformed into the reference image geometry, so
    # the project stores identity grooming transforms below.
    for shape, shape_name in zip(shape_segmentations, shape_names):
        print(f"Finding alignment transform from {shape_name} to {ref_name}")
        shape.antialias(args.antialias_iterations)
        rigid_transform = shape.createRigidRegistrationTransform(
            ref_seg, args.iso_value, args.icp_iterations
        )
        shape.applyTransform(
            rigid_transform,
            ref_seg.origin(),
            ref_seg.dims(),
            ref_seg.spacing(),
            ref_seg.coordsys(),
            sw.InterpolationType.Linear,
        )
        shape.binarize()
        bbox = sw.ImageUtils.boundingBox([shape], args.iso_value).pad(2)
        shape.isolate()
        shape.crop(bbox).pad(args.post_registration_pad, 0)

        print(f"Converting {shape_name} to distance transform")
        shape.antialias(args.antialias_iterations).computeDT(0).gaussianBlur(
            args.gaussian_sigma
        )

    groomed_files = sw.utils.save_images(
        str(groom_dir / "distance_transforms"),
        shape_segmentations,
        shape_names,
        extension="nrrd",
        compressed=True,
        verbose=True,
    )
    domain_type, groomed_files = sw.data.get_optimize_input(groomed_files, False)

    project_dir = train_dir / f"shape_models_{args.particles}"
    project_dir.mkdir(parents=True, exist_ok=True)

    subjects = []
    for original_path, groomed_path in zip(train_files, groomed_files):
        subject = sw.Subject()
        subject.set_number_of_domains(1)
        subject.set_original_filenames([str(original_path)])
        subject.set_groomed_filenames([str(groomed_path)])
        subject.set_groomed_transforms([np.eye(4).flatten()])
        subjects.append(subject)

    project = sw.Project()
    project.set_subjects(subjects)
    parameters = sw.Parameters()
    parameter_dictionary = {
        "number_of_particles": args.particles,
        "use_normals": 0,
        "checkpointing_interval": 200,
        "keep_checkpoints": 0,
        "iterations_per_split": args.iterations_per_split,
        "optimization_iterations": args.optimization_iterations,
        "starting_regularization": args.starting_regularization,
        "ending_regularization": args.ending_regularization,
        "recompute_regularization_interval": args.recompute_regularization_interval,
        "relative_weighting": 1.0,
        "initial_relative_weighting": 0.05,
        "save_init_splits": 0,
        "verbosity": 1,
        "procrustes": int(args.procrustes),
    }
    for key, value in parameter_dictionary.items():
        parameters.set(key, sw.Variant([value]))
    parameters.set("domain_type", sw.Variant(domain_type[0]))
    project.set_parameters("optimize", parameters)

    spreadsheet = project_dir / f"{args.project_name}.xlsx"
    project.save(str(spreadsheet))
    print(f"Saved ShapeWorks project: {spreadsheet}")

    if args.run_optimize:
        subprocess.run(
            ["shapeworks", "optimize", "--name", str(spreadsheet)], check=True
        )

    if args.launch_studio:
        subprocess.run(["ShapeWorksStudio", str(spreadsheet)], check=True)


if __name__ == "__main__":
    main()
