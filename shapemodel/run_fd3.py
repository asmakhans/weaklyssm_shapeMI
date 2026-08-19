#!/usr/bin/env python3
"""Backward-compatible entry point for fixed-domain SSM construction.

The previous version depended on a locally generated spreadsheet containing
absolute paths. The public-release version delegates to ``run_fd2.py``, which
reconstructs the required project from ``--train-dir`` and ``--test-dir``.
"""

from run_fd2 import main


if __name__ == "__main__":
    main()
