# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import canopy_factory

project = 'canopy_factory'
copyright = '2025, Meagan Lang'
author = 'Meagan Lang'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
release = canopy_factory.__version__
# The short X.Y version.
version = release.split('+')[0]

# Substitutions
# .. _Docs: http://yggdrasil.readthedocs.io/en/latest/
rst_epilog = """
.. _Docs: https://langmm.github.io/canopy_factory/
.. _Meagan Lang: langmm.astro@gmail.com
"""


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    # 'sphinxarg.ext',
    "myst_parser",
    "numpydoc",
    "sphinx_copybutton",
]

templates_path = ['_templates']
exclude_patterns = []
pygments_style = 'sphinx'
todo_include_todos = True
# napoleon_include_private_with_doc = True
napoleon_custom_sections = [
    ('Class Attributes', "params_style"),
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ['_static']
html_title = f"{project} {release}"
html_theme_options = {
    "repository_url": "https://github.com/langmm/canopy_factory",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
}
html_context = {
    "github_user": "langmm",
    "github_repo": project,
    "github_version": "main",
    "doc_path": "docs",
}
