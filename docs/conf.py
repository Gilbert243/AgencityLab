# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agencitylab.version import __version__

project = "AgencityLab"
author = "AgencityLab Contributors"
copyright = f"{datetime.now().year}, {author}"
release = __version__
version = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

templates_path = []
exclude_patterns = ["_build", "README.md", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"

html_theme = "alabaster"
html_static_path = []

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
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_title = "AgencityLab Documentation"
html_short_title = "AgencityLab"

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")
