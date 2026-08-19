#!/usr/bin/env python3
"""Calculate ShapeWorks SSM metrics for a completed train/fixed-domain pair.

The script expects the directory layout created by ``run_fd.py`` and
``run_fd2.py`` and computes compactness, specificity, test generalization, and
training reconstruction error. The resulting ``stats_new.npz`` is compatible
with the original analysis notebooks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapeworks as sw
from joblib import Parallel, delayed
from tqdm import tqdm





def create_data_matrix(particle_files: list[Path] | list[str]) -> np.ndarray:
    """Load ShapeWorks particle files as [subjects, particles, xyz]."""
    return np.stack([np.loadtxt(str(path)) for path in particle_files], axis=0)


class PCAEmbedder:
    """Minimal PCA helper used for reconstruction/generalization metrics."""

    def __init__(self, point_matrix: np.ndarray, num_dim: int):
        if point_matrix.ndim < 2:
            raise ValueError("Expected a subject-by-feature particle matrix.")
        self.point_shape = point_matrix.shape[1:]
        flat = point_matrix.reshape(point_matrix.shape[0], -1)
        self.mean = flat.mean(axis=0)
        centered = flat - self.mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        rank = min(num_dim, max(0, point_matrix.shape[0] - 1), vt.shape[0])
        self.eigen_vectors = vt[:rank].T
        self.PCA_scores = centered @ self.eigen_vectors

    def project(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float).reshape(-1)
        basis = self.eigen_vectors[:, : scores.size]
        reconstructed = self.mean + basis @ scores
        return reconstructed.reshape(self.point_shape)


@dataclass(frozen=True)
class MetricContext:
    root: Path
    train_mesh_dir: Path
    test_mesh_dir: Path
    train_particle_dir: Path
    test_particle_dir: Path
    template_mesh: Path
    template_particles: Path
    work_dir: Path
    n_jobs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute compactness, specificity and generalization metrics."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Experiment/output root containing groomed_test/ and "
            "shape_models_<N>_fd/. When --train-root is omitted, this same root "
            "must also contain groomed/ and shape_models_<N>/."
        ),
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=None,
        help=(
            "Root containing the training grooming and training particle model. "
            "Defaults to --root. Use a separate path when run_fd2.py wrote "
            "fixed-domain test outputs to --output-dir."
        ),
    )
    parser.add_argument(
        "--reference-shape",
        default=None,
        help=(
            "Reference subject basename, without extension. If omitted, the value "
            "written by run_fd.py to groomed/reference_shape.txt is used."
        ),
    )
    parser.add_argument("--project-name", default="la")
    parser.add_argument("--particles", type=int, default=1024)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--max-modes",
        type=int,
        default=None,
        help="Optional cap on PCA modes. Default is number of training shapes - 1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Default: <root>/groomed_test/stats_new.npz",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Temporary/generated metric workspace. Default: <root>/metrics_work",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep intermediate specificity samples and reconstruction particles.",
    )
    return parser.parse_args()


def get_particle_files(model_dir: Path) -> list[Path]:
    files = sorted(
        p for p in model_dir.glob("*local*.particles") if "meanshape" not in p.name
    )
    if not files:
        raise FileNotFoundError(f"No local particle files found in: {model_dir}")
    return files


def closest_point_distance(mesh: sw.Mesh, points: np.ndarray) -> float:
    distances = np.zeros(points.shape[0], dtype=float)
    for i, point in enumerate(points):
        closest = mesh.closestPoint(point)
        distances[i] = np.linalg.norm(point - closest[0])
    return float(np.mean(distances))


def convert_to_mesh(dt_dir: Path, mesh_dir: Path, n_jobs: int) -> int:
    mesh_dir.mkdir(parents=True, exist_ok=True)
    distance_transforms = sorted(dt_dir.glob("*.nrrd"))
    if not distance_transforms:
        raise FileNotFoundError(f"No distance transforms found in: {dt_dir}")

    def worker(dt_path: Path) -> None:
        output_path = mesh_dir / f"{dt_path.stem}.vtk"
        dt = sw.Image(str(dt_path))
        dt.binarize()
        dt.isolate()
        dt.antialias(30).computeDT(0).gaussianBlur(1.0)
        dt.toMesh(0).remesh(10000, 1.0).write(str(output_path))

    Parallel(n_jobs=n_jobs)(delayed(worker)(path) for path in distance_transforms)
    return len(distance_transforms)


def relative_for_log(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def generalization(
    num_modes: int,
    point_embedder,
    particle_dir: Path,
    mesh_dir: Path,
    save_dir: Path,
    ctx: MetricContext,
) -> float:
    particle_files = get_particle_files(particle_dir)
    meshes = sorted(mesh_dir.glob("*.vtk"))
    if len(particle_files) != len(meshes):
        raise RuntimeError(
            f"Particle/mesh count mismatch for {particle_dir}: "
            f"{len(particle_files)} particles vs {len(meshes)} meshes"
        )

    point_matrix = create_data_matrix(particle_files)
    centered = point_matrix.reshape(point_matrix.shape[0], -1).T - point_embedder.mean.reshape(
        point_embedder.mean.shape[0], 1
    )
    pca_scores = point_embedder.eigen_vectors.T @ centered
    pca_scores = pca_scores[:num_modes, :].T
    save_dir.mkdir(parents=True, exist_ok=True)

    def worker(i: int) -> dict:
        current_particles = point_embedder.project(pca_scores[i])
        pred_path = save_dir / f"p{i}.particles"
        np.savetxt(pred_path, current_particles)
        gt_mesh = sw.Mesh(str(meshes[i]))
        distance = closest_point_distance(gt_mesh, current_particles)
        return {
            "pred_particles": relative_for_log(pred_path, ctx.root),
            "gt_mesh": relative_for_log(meshes[i], ctx.root),
            "distance": distance,
            "template_mesh": relative_for_log(ctx.template_mesh, ctx.root),
            "template_particles": relative_for_log(ctx.template_particles, ctx.root),
        }

    infos = Parallel(n_jobs=ctx.n_jobs)(
        delayed(worker)(i)
        for i in tqdm(range(len(particle_files)), desc=f"Generalization k={num_modes}")
    )
    (save_dir / "info.json").write_text(json.dumps({"all_infos": infos}, indent=2))
    return float(np.mean([item["distance"] for item in infos]))


def calculate_mode(num_modes: int, ctx: MetricContext) -> list[float]:
    train_particles = get_particle_files(ctx.train_particle_dir)
    particle_system = sw.ParticleSystem([str(p) for p in train_particles])

    compactness = sw.ShapeEvaluation.ComputeCompactness(
        particleSystem=particle_system, nModes=num_modes
    )

    point_matrix = create_data_matrix(train_particles)
    point_embedder = PCAEmbedder(point_matrix, num_dim=num_modes)

    test_generalization = generalization(
        num_modes,
        point_embedder,
        ctx.test_particle_dir,
        ctx.test_mesh_dir,
        ctx.work_dir / "generalization_test" / str(num_modes),
        ctx,
    )
    train_generalization = generalization(
        num_modes,
        point_embedder,
        ctx.train_particle_dir,
        ctx.train_mesh_dir,
        ctx.work_dir / "generalization_train" / str(num_modes),
        ctx,
    )

    specificity_dir = ctx.work_dir / "specificity" / str(num_modes)
    if specificity_dir.exists():
        shutil.rmtree(specificity_dir)
    specificity_dir.mkdir(parents=True, exist_ok=True)

    sw.ShapeEvaluation.ComputeSpecificity(
        particleSystem=particle_system,
        nModes=num_modes,
        saveTo=str(specificity_dir),
    )
    xml_files = sorted(specificity_dir.glob("*.xml"))
    generated_particles = sorted(specificity_dir.glob("*.particles"))
    if len(xml_files) != len(generated_particles):
        raise RuntimeError(
            f"Specificity output mismatch in {specificity_dir}: "
            f"{len(xml_files)} XML files vs {len(generated_particles)} particles"
        )

    def specificity_worker(i: int) -> dict:
        first_line = xml_files[i].read_text().splitlines()[0].strip()
        closest_file = Path(first_line).name
        sampled_particles = np.loadtxt(generated_particles[i])
        gt_mesh_path = ctx.train_mesh_dir / closest_file.replace(
            "_local.particles", ".vtk"
        )
        gt_mesh = sw.Mesh(str(gt_mesh_path))
        distance = closest_point_distance(gt_mesh, sampled_particles)
        return {
            "pred_particles": relative_for_log(generated_particles[i], ctx.root),
            "gt_mesh": relative_for_log(gt_mesh_path, ctx.root),
            "distance": distance,
            "template_mesh": relative_for_log(ctx.template_mesh, ctx.root),
            "template_particles": relative_for_log(ctx.template_particles, ctx.root),
        }

    specificity_infos = Parallel(n_jobs=ctx.n_jobs)(
        delayed(specificity_worker)(i)
        for i in tqdm(range(len(xml_files)), desc=f"Specificity k={num_modes}")
    )
    (specificity_dir / "info.json").write_text(
        json.dumps({"all_infos": specificity_infos}, indent=2)
    )
    specificity = float(
        np.mean([item["distance"] for item in specificity_infos])
    )

    return [float(compactness), specificity, test_generalization, train_generalization]


def resolve_reference_shape(train_root: Path, supplied: str | None) -> str:
    if supplied:
        return supplied
    reference_file = train_root / "groomed" / "reference_shape.txt"
    if not reference_file.exists():
        raise FileNotFoundError(
            "No --reference-shape was provided and groomed/reference_shape.txt "
            "does not exist. Run run_fd.py or pass --reference-shape explicitly."
        )
    value = reference_file.read_text().strip()
    if not value:
        raise ValueError(f"Reference-shape file is empty: {reference_file}")
    return value


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    train_root = (
        args.train_root.expanduser().resolve() if args.train_root else root
    )
    reference_shape = resolve_reference_shape(train_root, args.reference_shape)

    train_dt_dir = train_root / "groomed" / "distance_transforms"
    test_dt_dir = root / "groomed_test" / "distance_transforms"
    train_mesh_dir = train_root / "groomed" / "mesh"
    test_mesh_dir = root / "groomed_test" / "mesh"
    train_particle_dir = (
        train_root / f"shape_models_{args.particles}" / f"{args.project_name}_particles"
    )
    test_particle_dir = (
        root
        / f"shape_models_{args.particles}_fd"
        / f"{args.project_name}_particles"
    )

    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir
        else root / "metrics_work"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    convert_to_mesh(train_dt_dir, train_mesh_dir, args.n_jobs)
    convert_to_mesh(test_dt_dir, test_mesh_dir, args.n_jobs)

    template_mesh = train_mesh_dir / f"{reference_shape}.vtk"
    template_particles = train_particle_dir / f"{reference_shape}_local.particles"
    if not template_mesh.exists():
        raise FileNotFoundError(f"Reference mesh not found: {template_mesh}")
    if not template_particles.exists():
        raise FileNotFoundError(f"Reference particles not found: {template_particles}")

    train_particles = get_particle_files(train_particle_dir)
    test_particles = get_particle_files(test_particle_dir)
    if len(train_particles) < 2:
        raise ValueError("At least two training shapes are required for PCA metrics.")
    if not test_particles:
        raise ValueError("No test particle files are available for generalization.")

    max_modes = len(train_particles) - 1
    if args.max_modes is not None:
        max_modes = min(max_modes, args.max_modes)
    if max_modes < 1:
        raise ValueError("No valid PCA modes to evaluate.")

    ctx = MetricContext(
        root=root,
        train_mesh_dir=train_mesh_dir,
        test_mesh_dir=test_mesh_dir,
        train_particle_dir=train_particle_dir,
        test_particle_dir=test_particle_dir,
        template_mesh=template_mesh,
        template_particles=template_particles,
        work_dir=work_dir,
        n_jobs=max(1, args.n_jobs),
    )

    print(f"Calculating metrics for {max_modes} PCA modes")
    output_rows = [
        calculate_mode(mode, ctx)
        for mode in tqdm(range(1, max_modes + 1), desc="PCA modes")
    ]
    output = np.asarray(output_rows, dtype=float)

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else root / "groomed_test" / "stats_new.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        comp=output[:, 0],
        spec=output[:, 1],
        gen=output[:, 2],
        gen_train=output[:, 3],
    )
    print(f"Saved metrics: {output_path}")

    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
