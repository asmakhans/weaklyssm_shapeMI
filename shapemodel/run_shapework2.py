#!/usr/bin/env python3
"""Backward-compatible entry point for training SSM construction.

Use ``--particles 256`` or ``--particles 1024`` with the same arguments as
``run_fd.py``.
"""

from run_fd import main


if __name__ == "__main__":
    main()
