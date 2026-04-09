"""Sphinx configuration for Jano."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "Jano"
author = "Marcos Manuel Muraro"
copyright = "2026, Marcos Manuel Muraro"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_title = "Jano documentation"
html_static_path = ["_static"]

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
