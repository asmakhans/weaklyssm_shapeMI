#!/usr/bin/env python3
"""Backward-compatible entry point for training SSM construction.

This legacy filename now delegates to ``run_fd.py`` so path handling and
ShapeWorks parameters are maintained in one place.
"""

from run_fd import main


if __name__ == "__main__":
    main()
