#!/usr/bin/env python3
"""Legacy metric entry point.

The public release consolidates metric calculation in ``calculate_metric4.py``
to avoid duplicated machine-specific paths and inconsistent experimental
entry points. This filename is retained for compatibility.
"""

from calculate_metric4 import main


if __name__ == "__main__":
    main()
