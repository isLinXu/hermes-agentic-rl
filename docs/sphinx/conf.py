from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = "hermes-agentic-rl"
author = "hermes-agentic-rl contributors"
copyright = "2026, hermes-agentic-rl contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# MyST configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

# Use furo theme if available, fall back to alabaster
try:
    import furo  # noqa: F401

    html_theme = "furo"
except ImportError:
    html_theme = "alabaster"

html_static_path = ["_static"]
html_title = "hermes-agentic-rl Documentation"
html_short_title = "hermes-agentic-rl"

# Intersphinx links
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "transformers": ("https://huggingface.co/docs/transformers/en", None),
}

autodoc_mock_imports = [
    "datasets",
    "pyarrow",
    "tensorboard",
    "torch",
    "transformers",
    "vllm",
    "wandb",
    "optuna",
    "pytest",
    "pytest_benchmark",
    "pydantic",
    "megatron",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False

nitpicky = False
master_doc = "index"
