"""Build the documentation without importing the scientific runtime."""

import os
from pathlib import Path
import shutil
import sys
import subprocess
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

project = "SEAREG"
author = "SEAREG contributors"
copyright = "2026, SEAREG contributors"
with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as metadata:
    release = tomllib.load(metadata)["project"]["version"]
version = ".".join(release.split(".")[:2])

# Prefer the build revision; archives without Git use the documented main fallback.
source_revision = os.environ.get("READTHEDOCS_GIT_COMMIT_HASH")
if not source_revision:
    try:
        source_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision = "main"
extlinks = {
    "source": (f"https://github.com/xmyjy/SEAREG/blob/{source_revision}/%s", "%s"),
}

extensions = [
    "nbsphinx", "nbsphinx_link", "seareg_notebooks",
    "sphinx_copybutton", "sphinx_design", "sphinx.ext.extlinks",
]
master_doc = "index"
source_suffix = {".rst": "restructuredtext"}
exclude_patterns = ["_build", "**/.ipynb_checkpoints"]
language = "en"

# nbsphinx calls the Pandoc executable. The docs-only dependency supplies it
# when Pandoc is not already installed on the documentation builder.
if shutil.which("pandoc") is None:
    import pypandoc

    pandoc_directory = str(Path(pypandoc.get_pandoc_path()).parent)
    os.environ["PATH"] = pandoc_directory + os.pathsep + os.environ.get("PATH", "")

nbsphinx_execute = "never"
nbsphinx_allow_errors = False
nbsphinx_prolog = r"""
{% set notebook = env.docname.split('/')[-1] %}

.. only:: html

   .. admonition:: Download and run
      :class: note

      :download:`Download this notebook <../../tutorials/{{ notebook }}.ipynb>`.
      Set up the :doc:`tutorial environment <../installation>` before running it.

      {% if notebook[:2] == "T7" %}
      Configure the saved T3/T4 analysis paths and the external input files listed below.
      {% else %}
      Configure section 1 and choose ``train`` (default) or ``provided``.
      See :doc:`../quickstart` for the difference between these modes and reopening saved analysis.
      {% endif %}

"""

html_theme = "furo"
html_title = "SEAREG"
html_static_path = ["_static"]
html_css_files = ["seareg.css"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#006d77",
        "color-brand-content": "#006d77",
        "color-sidebar-background": "#f5f7f8",
        "content-width": "52rem",
    },
    "dark_css_variables": {
        "color-brand-primary": "#63c7ca",
        "color-brand-content": "#63c7ca",
        "color-sidebar-background": "#181d20",
    },
    "source_repository": "https://github.com/xmyjy/SEAREG/",
    "source_branch": source_revision,
    "top_of_page_buttons": ["view"],
    "source_directory": "docs/",
}
html_show_sourcelink = False

# Copy inputs from notebook and terminal examples without their prompts.
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
copybutton_selector = "div.highlight pre"


def setup(app):
    # Executed Python kernels advertise "ipython3" in notebook metadata.
    # These tutorials use plain Python, so the bundled lexer is sufficient
    # without adding IPython or the scientific runtime to the docs build.
    from pygments.lexers import PythonLexer

    app.add_lexer("ipython3", PythonLexer)
