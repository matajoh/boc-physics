"""Sphinx configuration for the bocphysics API reference."""

from importlib import metadata

project = "bocphysics"
author = "Matthew Johnson"
copyright = "2026, Matthew Johnson"  # noqa: A001

try:
    release = metadata.version("bocphysics")
except metadata.PackageNotFoundError:
    release = "0.4"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

autodoc_mock_imports = ["pyglet"]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "tutorial"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"bocphysics {release}"
