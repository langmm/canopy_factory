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
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    # 'sphinxarg.ext',
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

html_theme = 'alabaster'
html_static_path = ['_static']
