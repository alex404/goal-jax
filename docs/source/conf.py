# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Goal"
copyright = "2025, Sacha Sokoloski"
author = "Sacha Sokoloski"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_math_dollar",
    "myst_parser",  # For Markdown support
]

extensions.extend(
    [
        "sphinx.ext.inheritance_diagram",
        "sphinx.ext.graphviz",
    ]
)

# Set better resolution for inheritance diagrams
graphviz_output_format = "svg"  # PNG has better browser compatibility

inheritance_graph_attrs = {
    "rankdir": "TB",  # Top to bottom layout
    "size": '"24.0, 32.0"',  # Larger size
    "bgcolor": "transparent",
    "fontsize": 18,
}

# Extension configurations
add_module_names = False
autodoc_member_order = "bysource"
myst_enable_extensions = ["dollarmath", "amsmath"]

# Intersphinx mapping for cross-referencing to other docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
}

# The theme to use for HTML and HTML Help pages
html_theme = "furo"

# Additional templates
templates_path = ["_templates"]

# Files to exclude
exclude_patterns = []

# Source file parsers
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}

# Settings for math
mathjax3_config = {"tex": {"macros": {}, "tags": "ams"}}
