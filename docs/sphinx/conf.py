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
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

html_theme = "alabaster"
html_static_path = ["_static"]

autodoc_mock_imports = [
    "datasets",
    "pyarrow",
    "tensorboard",
    "torch",
    "transformers",
    "vllm",
    "wandb",
]

nitpicky = False
master_doc = "index"
