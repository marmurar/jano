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
release = "0.3.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_title = "Jano"
html_short_title = "Jano"
html_static_path = ["_static"]
html_logo = "_static/jano_logo.png"
html_css_files = ["jano-docs.css"]
html_theme_options = {
    "logo_name": False,
    "description": "Temporal partitions, backtesting and simulation reporting for time-correlated data.",
    "fixed_sidebar": True,
    "page_width": "1180px",
    "sidebar_width": "290px",
}
html_sidebars = {
    "**": [
        "sidebar_about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
    ]
}

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_typehints = "description"
