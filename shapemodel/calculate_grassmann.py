#!/usr/bin/env python3
"""Compute Grassmannian distance between two ShapeWorks PCA subspaces.
Compares local particle correspondences from two SSMs and saves both the raw
Grassmannian distance and the log-normalized value used by the analysis
notebooks/paper figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-particles",
        type=Path,
        required=True,
        help="Directory containing reference/GT local *.particles files.",
    )
    parser.add_argument(
        "--comparison-particles",
        type=Path,
        required=True,
        help="Directory containing comparison local *.particles files.",
    )
    parser.add_argument(
        "--max-modes",
        type=int,
        default=None,
        help="Optional maximum PCA subspace dimension.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npz file.",
    )
    return parser.parse_args()


def particle_files(directory: Path) -> list[Path]:
    files = sorted(
        path
        for path in directory.expanduser().resolve().glob("*local*.particles")
        if "meanshape" not in path.name
    )
    if not files:
        raise FileNotFoundError(f"No local particle files found in: {directory}")
    return files


def pca_basis(files: list[Path]) -> np.ndarray:
    matrix = np.stack([np.loadtxt(path).reshape(-1) for path in files], axis=0)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    # Rows are subjects; right singular vectors are PCA directions in feature space.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(len(files) - 1, vt.shape[0])
    return vt[:rank].T


def grassmann_curve(reference_basis: np.ndarray, comparison_basis: np.ndarray, max_modes: int) -> tuple[np.ndarray, np.ndarray]:
    raw = []
    log_normalized = []
    for k in range(1, max_modes + 1):
        a = reference_basis[:, :k]
        b = comparison_basis[:, :k]
        singular_values = np.linalg.svd(a.T @ b, compute_uv=False)
        singular_values = np.clip(singular_values, -1.0, 1.0)
        principal_angles = np.arccos(singular_values)
        distance = float(np.sqrt(np.sum(principal_angles ** 2)))
        raw.append(distance)
        # The original notebooks plotted log(distance / k).
        log_normalized.append(float(np.log(max(distance / k, np.finfo(float).tiny))))
    return np.asarray(raw), np.asarray(log_normalized)


def main() -> None:
    args = parse_args()
    ref_files = particle_files(args.reference_particles)
    cmp_files = particle_files(args.comparison_particles)
    if len(ref_files) != len(cmp_files):
        print(
            "Warning: reference and comparison SSMs contain different numbers of "
            f"subjects ({len(ref_files)} vs {len(cmp_files)}). PCA ranks will be "
            "limited by the smaller available rank."
        )

    ref_basis = pca_basis(ref_files)
    cmp_basis = pca_basis(cmp_files)
    max_modes = min(ref_basis.shape[1], cmp_basis.shape[1])
    if args.max_modes is not None:
        max_modes = min(max_modes, args.max_modes)
    if max_modes < 1:
        raise ValueError("No valid PCA subspace dimensions are available.")

    raw, log_normalized = grassmann_curve(ref_basis, cmp_basis, max_modes)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        modes=np.arange(1, max_modes + 1),
        grassmann=raw,
        log_grassmann=log_normalized,
    )
    print(f"Saved Grassmannian distances: {output}")


if __name__ == "__main__":
    main()
