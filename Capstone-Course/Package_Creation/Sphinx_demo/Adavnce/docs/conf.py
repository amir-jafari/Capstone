# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath("../src"))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Sphinx Demo'
copyright = '2024, EP'
author = 'EP'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        "sphinx.ext.autodoc",
        "sphinx.ext.napoleon",
        "sphinx.ext.linkcode",
        "sphinx_gallery.gen_gallery"
    ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


napoleon_google_docstring = True 

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "elpham6",  # Username of repo's owner
    "github_repo": "sphinx_demo",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/src/",  # Path in the checkout to the docs root
}
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    # return "https://somesite/sourcerepo/%s.py" % filename
    return f"https://github.com/{html_context['github_user']}/{html_context['github_repo']}/blob/{html_context['github_version']}/{html_context['conf_py_path']}/{filename}.py"

sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': ['../examples'],
    # path to where to save gallery generated output
    'gallery_dirs': ['auto_examples'],
    'filename_pattern': '.py',
    'plot_gallery': 'False',
    }

