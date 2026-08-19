#!/usr/bin/env python3
"""Run fixed-domain ShapeWorks optimization on test segmentations.

This is the second ShapeWorks stage used in both experimental strategies:

* Strategy 1: ``--train-dir`` is the manual/ground-truth training SSM and
  ``--test-dir`` contains semi-supervised test predictions.
* Strategy 2: ``--train-dir`` is the SSM built from semi-supervised training
  predictions and ``--test-dir`` contains predictions from the same method on
  the held-out test set.

The fixed training particles are used to initialize/anchor optimization of the
new test subjects. No machine-specific paths are embedded in this script.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import shapeworks as sw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add test segmentations to an existing training SSM using fixed domains."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="Directory containing the previously built training SSM.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Directory containing test segmentations (*.nii.gz).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for fixed-domain grooming/project outputs. Defaults to "
            "--train-dir. Set this for Strategy 1 to reuse one manual training SSM "
            "across multiple semi-supervised test predictions without overwriting results."
        ),
    )
    parser.add_argument("--project-name", default="la")
    parser.add_argument("--particles", type=int, default=1024)

    parser.add_argument("--iso-value", type=float, default=0.5)
    parser.add_argument("--antialias-iterations", type=int, default=30)
    parser.add_argument("--isotropic-spacing", type=float, default=1.0)
    parser.add_argument("--initial-pad", type=int, default=30)
    parser.add_argument("--post-registration-pad", type=int, default=30)
    parser.add_argument("--icp-iterations", type=int, default=200)
    parser.add_argument("--gaussian-sigma", type=float, default=1.5)

    parser.add_argument("--iterations-per-split", type=int, default=1000)
    parser.add_argument("--optimization-iterations", type=int, default=500)
    parser.add_argument("--starting-regularization", type=float, default=1000)
    parser.add_argument("--ending-regularization", type=float, default=10)
    parser.add_argument("--recompute-regularization-interval", type=int, default=2)
    parser.add_argument(
        "--procrustes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable Procrustes alignment during fixed-domain optimization.",
    )
    parser.add_argument(
        "--run-optimize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--launch-studio",
        action="store_true",
        help="Open the generated fixed-domain project in ShapeWorksStudio.",
    )
    parser.add_argument(
        "--keep-fixed-particles",
        action="store_true",
        help="Keep copies of fixed training particles in the fixed-domain output folder.",
    )
    return parser.parse_args()


def particle_files(model_dir: Path) -> tuple[list[Path], list[Path]]:
    local = sorted(
        p for p in model_dir.glob("*local*.particles") if "meanshape" not in p.name
    )
    world = sorted(
        p for p in model_dir.glob("*world*.particles") if "meanshape" not in p.name
    )
    if not local:
        raise FileNotFoundError(f"No local particle files found in {model_dir}")
    return local, world


def write_mean_shape(files: list[Path], destination: Path) -> Path:
    arrays = [np.loadtxt(path) for path in files]
    mean_shape = np.mean(np.stack(arrays, axis=0), axis=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, mean_shape)
    return destination


def main() -> None:
    args = parse_args()
    train_dir = args.train_dir.expanduser().resolve()
    test_dir = args.test_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else train_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    test_files = sorted(test_dir.glob("*.nii.gz"))
    if not test_files:
        raise FileNotFoundError(f"No .nii.gz test segmentations found in: {test_dir}")

    train_groom_dir = train_dir / "groomed"
    reference_path = train_groom_dir / "reference.nrrd"
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Missing training reference image: {reference_path}. Run run_fd.py first."
        )
    ref_seg = sw.Image(str(reference_path))

    train_groomed = sorted((train_groom_dir / "distance_transforms").glob("*.nrrd"))
    model_dir = train_dir / f"shape_models_{args.particles}" / f"{args.project_name}_particles"
    local_particles, world_particles = particle_files(model_dir)

    if len(train_groomed) != len(local_particles):
        raise RuntimeError(
            "Training groomed-file count does not match local-particle count: "
            f"{len(train_groomed)} vs {len(local_particles)}"
        )

    mean_shape_path = write_mean_shape(local_particles, model_dir / "meanshape.particles")

    test_groom_dir = output_dir / "groomed_test"
    test_groom_dir.mkdir(parents=True, exist_ok=True)
    spacing = [args.isotropic_spacing] * 3

    test_segmentations: list[sw.Image] = []
    test_names: list[str] = []
    for shape_path in test_files:
        shape_name = shape_path.name.removesuffix(".nii.gz")
        print(f"Loading: {shape_path}")
        shape = sw.Image(str(shape_path))
        test_segmentations.append(shape)
        test_names.append(shape_name)

        print(f"Grooming: {shape_name}")
        shape.isolate()
        bbox = sw.ImageUtils.boundingBox([shape], args.iso_value).pad(2)
        shape.crop(bbox)
        shape.antialias(args.antialias_iterations).resample(
            spacing, sw.InterpolationType.Linear
        ).binarize()
        shape.pad(args.initial_pad, 0)

    for shape, shape_name in zip(test_segmentations, test_names):
        shape.isolate()
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

    test_groomed = sw.utils.save_images(
        str(test_groom_dir / "distance_transforms"),
        test_segmentations,
        test_names,
        extension="nrrd",
        compressed=True,
        verbose=True,
    )

    subjects = []
    for groomed_path, local_path in zip(train_groomed, local_particles):
        subject = sw.Subject()
        subject.set_number_of_domains(1)
        subject.set_groomed_filenames([str(groomed_path)])
        subject.set_local_particle_filenames([str(local_path)])
        # The original implementation intentionally used local particles for
        # both local and world fixed particles after pre-alignment.
        subject.set_world_particle_filenames([str(local_path)])
        subject.set_extra_values({"fixed": "yes"})
        subjects.append(subject)

    for original_path, groomed_path in zip(test_files, test_groomed):
        subject = sw.Subject()
        subject.set_number_of_domains(1)
        subject.set_original_filenames([str(original_path)])
        subject.set_groomed_filenames([str(groomed_path)])
        subject.set_local_particle_filenames([str(mean_shape_path)])
        subject.set_world_particle_filenames([str(mean_shape_path)])
        subject.set_extra_values({"fixed": "no"})
        subjects.append(subject)

    project_dir = output_dir / f"shape_models_{args.particles}_fd"
    project_dir.mkdir(parents=True, exist_ok=True)
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
        "domains_per_shape": 1,
        "relative_weighting": 1,
        "initial_relative_weighting": 0.05,
        "save_init_splits": 0,
        "verbosity": 1,
        "procrustes": int(args.procrustes),
        "use_fixed_subjects": 1,
        "narrow_band": 1e50,
    }
    for key, value in parameter_dictionary.items():
        parameters.set(key, sw.Variant([value]))
    parameters.set("domain_type", sw.Variant(1))
    project.set_parameters("optimize", parameters)

    spreadsheet = project_dir / f"{args.project_name}.xlsx"
    project.save(str(spreadsheet))
    print(f"Saved fixed-domain ShapeWorks project: {spreadsheet}")

    if args.run_optimize:
        subprocess.run(
            ["shapeworks", "optimize", "--name", str(spreadsheet)], check=True
        )

        if not args.keep_fixed_particles:
            fd_particle_dir = project_dir / f"{args.project_name}_particles"
            for src in local_particles + world_particles:
                candidate = fd_particle_dir / src.name
                if candidate.exists():
                    candidate.unlink()

    if args.launch_studio:
        subprocess.run(["ShapeWorksStudio", str(spreadsheet)], check=True)


if __name__ == "__main__":
    main()
