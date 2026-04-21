# Configuration file for the Sphinx documentation builder.
#
# This configuration keeps dependencies light and works with the project
# structure used by AgencityLab.

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = "AgencityLab"
author = "AgencityLab Contributors"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

language = "en"

html_theme = "alabaster"
html_static_path = ["_static"]

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "deflist",
    "tasklist",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable", {}),
}

html_title = "AgencityLab Documentation"
html_short_title = "AgencityLab"
