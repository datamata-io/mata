"""Test configuration file."""

import os
import sys
from pathlib import Path

# Force non-interactive matplotlib backend for all tests
# (prevents TkAgg / tkinter failures on headless/CI environments)
os.environ.setdefault("MPLBACKEND", "Agg")

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
