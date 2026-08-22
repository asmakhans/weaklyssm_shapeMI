# Semi-Supervised Segmentation for Statistical Shape Modeling Benchmark

This repository is for **“On the Viability of Semi-Supervised Segmentation Methods for Statistical Shape Modeling”** and contains the code used to study whether semi-supervised anatomical segmentations can replace manual segmentations for downstream statistical shape model (SSM) construction.

The repository has two parts:

1. **Semi-supervised segmentation methods** — code for the eight methods evaluated in the paper, plus benchmark data-split files and links to pretrained checkpoints.
2. **Shape-model evaluation** — a path-agnostic ShapeWorks pipeline that builds training/test SSMs, runs fixed-domain optimization, computes SSM metrics, and supports Grassmannian subspace comparison.

> **Data are not distributed in this repository.** The public NAMIC/CARMA left-atrium dataset must be obtained from its original source. The FEMUR dataset used in the paper is in-house and is not publicly released. Only split files and code are included here.

## Repository layout

```text
.
├── LICENSE
├── README.md
├── semi-supervised_methods/
│   ├── BCP/
│   ├── CAML/
│   ├── DeSCO/
│   ├── DTC/
│   ├── MCF/
│   ├── MT/
│   ├── SASSnet/
│   ├── UA-MT/
│   └── datasets/
│       └── namic/                 # benchmark split files only; no images/labels
└── shapemodel/
    ├── run_fd.py                  # build a training SSM
    ├── run_fd2.py                 # fixed-domain test SSM
    ├── calculate_metric4.py       # compactness/specificity/generalization
    ├── calculate_grassmann.py     # PCA-subspace / Grassmannian distance
    ├── settingI.ipynb             # optional Strategy 1 analysis notebook
    ├── settingII.ipynb            # optional Strategy 2 analysis notebook
    ├── convert_h5_to_niigz.py     # optional HDF5 -> NIfTI conversion utility
    └── ...                        # legacy compatibility entry points/helpers
```

## Benchmark methods

The paper evaluates:

- Mean Teacher (**MT**)
- Uncertainty-Aware Mean Teacher (**UA-MT**)
- Mutual Correction Framework (**MCF**)
- Orthogonal Annotation (**DeSCO**)
- Bidirectional Copy-Paste (**BCP**)
- Correlation-Aware Mutual Learning (**CAML**)
- Dual-task Consistency (**DTC**)
- Shape-Aware Semi-supervised segmentation (**SASSnet**)

The method folders are based on the corresponding authors' implementations. **Use the README/instructions inside each method folder for its environment, preprocessing, training, and inference commands.** This benchmark repository does not attempt to merge the eight training pipelines into one environment.

The benchmark-specific change is that dataset locations are now passed through command-line arguments rather than developer-machine paths. The supplied NAMIC split files are under each method's `data/` or `data_split/` folder and are also collected under `semi-supervised_methods/datasets/namic/`.

Pretrained model locations are provided as Google Drive links in each method's `model/` or `models/` directory. The repository intentionally does not commit `.pth` checkpoint files.

## Paper data split

The paper uses fixed train/test splits and two annotation budgets:

| Dataset | Train | Test | 20% labelled training | 40% labelled training |
|---|---:|---:|---:|---:|
| NAMIC / left atrium | 50 | 9 | 10 labelled + 40 unlabelled | 20 labelled + 30 unlabelled |
| FEMUR | 40 | 9 | 8 labelled + 32 unlabelled | 16 labelled + 24 unlabelled |

The NAMIC split files in this repository reproduce the benchmark split organization. The FEMUR data and labels are not distributed.

## End-to-end workflow

The benchmark is intentionally separated into a **segmentation stage** and a **shape-model stage**.

```text
images + limited labels
        |
        v
semi-supervised method
        |
        +--> predicted TRAIN segmentations  (needed for Strategy 2)
        |
        +--> predicted TEST segmentations   (needed for Strategies 1 and 2)
                     |
                     v
          orientation normalization
                     |
                     v
             ShapeWorks pipeline
         run_fd.py -> run_fd2.py
                     |
                     v
     calculate_metric4.py + calculate_grassmann.py
```

### 1. Prepare a method-specific dataset

For the method you want to evaluate:

1. Obtain your images and segmentations from the appropriate source.
2. Arrange them in the format expected by that method's original implementation.
3. Use the benchmark split files where applicable, or create equivalent train/test and labelled/unlabelled lists for your own dataset.
4. Pass the dataset root through the method's `--root_path`, `--data-root`, or equivalent command-line option.
5. Train using the desired annotation budget.
6. Run inference to generate segmentation masks for the required cohort(s).

For **Strategy 1**, only semi-supervised **test predictions** are required for the shape-model comparison because the training SSM is built from manual segmentations.

For **Strategy 2**, generate semi-supervised predictions for **both the training and test subjects** because both SSM stages are based on predicted segmentations.

### 2. Normalize anatomical orientation before shape modeling

Before passing segmentations to ShapeWorks, make sure all subjects represent the anatomy in a consistent orientation.

This is particularly important for bilateral anatomies. For example, the FEMUR experiments were normalized so the femurs had a consistent right-facing orientation before SSM construction. The correct flip axis depends on the anatomy, image affine/coordinate convention, and preprocessing pipeline, so this repository does **not** apply a universal automatic flip.

Verify orientation visually before proceeding.

### 3. Convert predictions to NIfTI if necessary

The public ShapeWorks scripts expect segmentation masks as `*.nii.gz` files. If a method produces HDF5 files, `shapemodel/convert_h5_to_niigz.py` can be used as a path-agnostic conversion utility; check the expected HDF5 dataset key before use.

## ShapeWorks setup

The paper used **ShapeWorks v6.5.1**. Install ShapeWorks according to the official ShapeWorks instructions and run these scripts from an environment where both the Python `shapeworks` package and the `shapeworks` command-line executable are available.

After activating the ShapeWorks environment, install the additional packages used by the canonical utilities if they are not already available:

```bash
pip install -r shapemodel/requirements.txt
```

`run_fd.py` and `run_fd2.py` can optionally open **ShapeWorksStudio** with `--launch-studio`. Studio is useful for interactive model inspection and qualitative mode-of-variation analysis, but it is not required to create the project files or calculate the numerical metrics.

The defaults in the public scripts match the paper's reported primary ShapeWorks settings:

| Parameter | Default |
|---|---:|
| ShapeWorks version used in paper | 6.5.1 |
| Correspondence particles | 1024 |
| Grooming isovalue | 0.5 |
| Antialias iterations | 30 |
| Iterations per split | 1000 |
| Optimization iterations | 500 |
| Starting regularization | 1000 |
| Ending regularization | 10 |
| Recompute regularization interval | 2 |
| Procrustes alignment | enabled |

The scripts expose these values as command-line arguments so other anatomies can be evaluated without editing source code.

## First build the manual reference SSMs

The paper defines the manual training SSM, `Phi_train`, from all manual training segmentations. The manual test reference, `Phi_test`, is **not** an independent unconstrained test model: it is optimized on manual test segmentations using the manual training SSM as the fixed-domain initialization.

### A. Manual training SSM

Place the manual training segmentations in one directory:

```text
/path/to/manual_train/
├── subject_001.nii.gz
├── subject_002.nii.gz
└── ...
```

Run:

```bash
python shapemodel/run_fd.py \
  --train-dir /path/to/manual_train
```

This creates, among other outputs:

```text
/path/to/manual_train/
├── groomed/
│   ├── reference.nrrd
│   ├── reference_shape.txt
│   └── distance_transforms/
└── shape_models_1024/
    ├── la.xlsx
    └── la_particles/
```

`groomed/reference_shape.txt` stores the selected reference subject name. `calculate_metric4.py` reads it automatically, so the reference shape no longer needs to be hardcoded.

### B. Manual test reference SSM

Use the manual test segmentations with the manual training SSM as the fixed domain:

```bash
python shapemodel/run_fd2.py \
  --train-dir /path/to/manual_train \
  --test-dir /path/to/manual_test \
  --output-dir /path/to/results/manual_test_reference
```

This produces the reference fixed-domain test correspondences used for comparison with semi-supervised methods.

## Strategy 1: manual training SSM is available

In **Strategy 1**, the manual training SSM is retained. For each semi-supervised method, only the manual test segmentations are replaced by that method's predicted test segmentations.

Conceptually:

```text
manual training segmentations
        |
        v
     run_fd.py
        |
        v
  manual Phi_train  -------------------------+
                                              |
semi-supervised predicted TEST segmentations  |
                   |                          |
                   +--------> run_fd2.py <----+
                                  |
                                  v
                      predicted-test SSM
```

Example for a method named `BCP`:

```bash
python shapemodel/run_fd2.py \
  --train-dir /path/to/manual_train \
  --test-dir /path/to/BCP/test_predictions \
  --output-dir /path/to/results/strategy1/BCP_20
```

Using a separate `--output-dir` lets one manual training SSM be reused for every method and annotation budget without overwriting previous fixed-domain outputs.

Calculate the reconstruction metrics for this pairing with:

```bash
python shapemodel/calculate_metric4.py \
  --root /path/to/results/strategy1/BCP_20 \
  --train-root /path/to/manual_train
```

For the paper's **Strategy 1** comparison, the relevant downstream quantities are **generalization** and **Grassmannian distance**, because the same manual training SSM is used by the reference and semi-supervised test models.

To compare the predicted-test PCA subspace with the manual-test reference:

```bash
python shapemodel/calculate_grassmann.py \
  --reference-particles /path/to/results/manual_test_reference/shape_models_1024_fd/la_particles \
  --comparison-particles /path/to/results/strategy1/BCP_20/shape_models_1024_fd/la_particles \
  --output /path/to/results/strategy1/BCP_20/grassmann.npz
```

Repeat this fixed-domain test step for each method and for the 20%/40% annotation settings.

## Strategy 2: manual training SSM is not available

In **Strategy 2**, the semi-supervised method supplies predicted segmentations for both training and test subjects. A new training SSM is therefore constructed from that method's predicted training segmentations, and its predicted test segmentations are then optimized using that predicted training SSM as the fixed domain.

Conceptually:

```text
semi-supervised predicted TRAIN segmentations
                   |
                   v
                run_fd.py
                   |
                   v
          method-specific Phi_train
                   |
                   +-------------------------+
                                             |
semi-supervised predicted TEST segmentations |
                   |                         |
                   +-------> run_fd2.py <----+
                                |
                                v
                    method-specific test SSM
```

Example:

```bash
python shapemodel/run_fd.py \
  --train-dir /path/to/BCP/train_predictions

python shapemodel/run_fd2.py \
  --train-dir /path/to/BCP/train_predictions \
  --test-dir /path/to/BCP/test_predictions
```

Then calculate compactness, specificity, and generalization:

```bash
python shapemodel/calculate_metric4.py \
  --root /path/to/BCP/train_predictions
```

The default output is:

```text
/path/to/BCP/train_predictions/groomed_test/stats_new.npz
```

with arrays:

- `comp` — compactness across PCA modes
- `spec` — specificity across PCA modes
- `gen` — test generalization across PCA modes
- `gen_train` — training reconstruction/generalization diagnostic

For Grassmannian analysis, compare the Strategy 2 fixed-domain test particles against the **manual-test reference** created above:

```bash
python shapemodel/calculate_grassmann.py \
  --reference-particles /path/to/results/manual_test_reference/shape_models_1024_fd/la_particles \
  --comparison-particles /path/to/BCP/train_predictions/shape_models_1024_fd/la_particles \
  --output /path/to/BCP/train_predictions/grassmann.npz
```

`grassmann.npz` contains:

- `modes`
- `grassmann` — raw Grassmannian distance
- `log_grassmann` — `log(distance / k)`, where `k` is the PCA-subspace dimension, matching the legacy analysis convention

For Strategy 2, the paper evaluates **compactness, specificity, generalization, and Grassmannian distance**.

## ShapeWorksStudio and qualitative analysis

To inspect a generated model interactively:

```bash
python shapemodel/run_fd.py \
  --train-dir /path/to/segmentations \
  --launch-studio
```

or:

```bash
python shapemodel/run_fd2.py \
  --train-dir /path/to/training_ssm \
  --test-dir /path/to/test_segmentations \
  --launch-studio
```

You can also open the generated `.xlsx` project directly in ShapeWorksStudio. The paper's qualitative analysis compares the first two PCA modes at `-2 sigma`, `-1 sigma`, the mean, `+1 sigma`, and `+2 sigma`.

## Optional analysis notebooks

`shapemodel/settingI.ipynb` and `shapemodel/settingII.ipynb` are retained as legacy plotting/analysis notebooks corresponding to the two experimental strategies. For public release:

- developer-machine paths were removed;
- cell outputs and execution counts were cleared;
- paths are resolved relative to the `WEAKLYSSM_RESULTS_ROOT` environment variable.

Example:

```bash
export WEAKLYSSM_RESULTS_ROOT=/path/to/all/experiment/results
jupyter lab shapemodel/settingI.ipynb
```

The notebooks contain exploratory/legacy analysis code and may require additional packages such as PyTorch3D, scikit-learn, matplotlib, or seaborn. For new runs, the command-line scripts (`calculate_metric4.py` and `calculate_grassmann.py`) are the recommended reproducible path.

## Legacy script names

Several historical scripts are retained so older commands do not immediately break:

- `run_shapeworks.py` and `run_shapework2.py` delegate to `run_fd.py`.
- `run_fd3.py` delegates to `run_fd2.py`.
- `calculate_metric1.py`, `calculate_metric2.py`, `calculate_metric3.py`, `calculate_metric5.py`, and `calc_new.py` delegate to `calculate_metric4.py`.

New experiments should use the canonical script names documented above.

## Outputs and release hygiene

The repository `.gitignore` excludes common medical-image data, checkpoints, predictions, ShapeWorks outputs, notebook checkpoints, caches, and temporary result files. This is intentional: public Git history should contain **source code, split metadata, and small configuration files**, not patient data or trained model binaries unless they have been explicitly approved for redistribution.

Before publishing a fork or release:

1. Verify that no private/in-house data are staged.
2. Verify the contents and sharing permissions of the Google Drive checkpoint folder.
3. Keep the FEMUR dataset out of the public repository.
4. Review each imported method's original license and citation requirements.
5. If a sensitive path/file was ever committed previously, remember that deleting it from the current working tree does **not** remove it from Git history; rewrite/purge the history before making the repository public.

See `PUBLIC_RELEASE_CHECKLIST.md` for a short final review checklist.

## Reproducibility notes

- The primary benchmark used one predefined train/test split and fixed random seed `1337` for the reported segmentation experiments.
- Shape-model results depend on consistent segmentation orientation and preprocessing.
- Keep training and test subject membership fixed when comparing methods.
- Keep the ShapeWorks particle count and optimization parameters fixed when reproducing paper comparisons.
- For Strategy 2, use the **same semi-supervised method and annotation setting** for both predicted training and predicted test segmentations.
- For Grassmannian comparison, preserve subject ordering/membership between the reference and comparison test cohorts.

## Citation

If you use this benchmark, please cite the accompanying paper:

> @article{khan2024viability,
  title={On the Viability of Semi-Supervised Segmentation Methods for Statistical Shape Modeling},
  author={Khan, Asma and Kataria, Tushar and Ukey, Janmesh and Elhabian, Shireen Y},
  journal={arXiv preprint arXiv:2407.15260},
  year={2024}
}

Please also cite the original publication/codebase for each semi-supervised method you use. The method-specific READMEs contain the relevant upstream information where available.

Disclaimer: README for this repository was formatted with the help of OpenAI's ChatGPT.

## License

The top-level repository license is provided in `LICENSE` (GNU GPL v3 in this release). Some imported method folders also contain their own upstream license files. Users and redistributors are responsible for complying with the applicable upstream licenses and attribution requirements for third-party code.
