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
{% set tutorials = {
    "T1": {
        "purpose": "Identify spatial chromatin-accessibility domains in an E13 mouse embryo and compare selected peak and motif spatial scores with observed peak accessibility.",
        "inputs": "ATAC counts, spatial coordinates and peak–motif associations. Co-profiled RNA is not used in this analysis.",
        "workflow": "Prepare the graph, train or load embeddings, then identify domains and fit spot and joint UMAPs. Inspect selected feature maps.",
        "outputs": "Spatial-domain labels, embeddings, UMAPs, feature-score and peak-accessibility tables, and editable figures."
    },
    "T2": {
        "purpose": "Identify spatial chromatin-accessibility domains in an E13 mouse embryo section and visualize their organization.",
        "inputs": "Spatial ATAC counts, native spatial coordinates and peak–motif associations.",
        "workflow": "Prepare the graph, train or load embeddings, then identify domains and fit spot and joint UMAPs.",
        "outputs": "Spatial-domain labels, spot/feature embeddings, UMAPs and editable figures."
    },
    "T3": {
        "purpose": "Identify spatial chromatin-accessibility domains in a P22 mouse brain section and examine their spatial organization.",
        "inputs": "ATAC counts, spatial coordinates and peak–motif associations. Co-profiled RNA is not used in this analysis.",
        "workflow": "Prepare the graph, train or load embeddings, then identify domains and fit spot and joint UMAPs.",
        "outputs": "Spatial-domain labels, spot/feature embeddings, UMAPs and editable figures."
    },
    "T4": {
        "purpose": "Identify spatial domains from H3K27ac profiles in a P22 mouse brain section and visualize their organization.",
        "inputs": "H3K27ac CUT&Tag counts, spatial coordinates and peak–motif associations. Co-profiled RNA is not used in this analysis.",
        "workflow": "Prepare the graph, train or load embeddings, then identify domains and fit spot and joint UMAPs.",
        "outputs": "Spatial-domain labels, spot/feature embeddings, UMAPs and editable figures."
    },
    "T5": {
        "purpose": "Identify broad spatial chromatin-accessibility domains in a human tonsil section and compare their spatial and embedding patterns.",
        "inputs": "Spatial ATAC counts, native spatial coordinates and peak–motif associations.",
        "workflow": "Prepare the graph, train or load embeddings, then identify domains and fit spot and joint UMAPs.",
        "outputs": "Spatial-domain labels, spot/feature embeddings, UMAPs and editable figures."
    },
    "T6": {
        "purpose": "Integrate chromatin accessibility profiles from four developing mouse brain sections to identify spatial domains and compare their spatial organization across developmental stages.",
        "inputs": "Combined ATAC counts with shared peak IDs, section IDs and native coordinates; tissue annotations support reference plots. No RNA or motif matrix is used.",
        "workflow": "Prepare section graphs, train or load embeddings, identify domains and fit UMAPs, then compare section-specific maps.",
        "outputs": "Reference and domain maps, spot/joint UMAPs, embeddings and reloadable analysis results."
    }
} %}
{% set tutorial = tutorials[notebook[:2]] %}

.. only:: html

   .. admonition:: {{ notebook[:2] }} — Tutorial at a glance
      :class: note

      **Purpose:** {{ tutorial.purpose }}

      **Inputs:** {{ tutorial.inputs }} See :doc:`../input_data` and :doc:`../data_access`.

      **Workflow:** {{ tutorial.workflow }}

      **Outputs:** {{ tutorial.outputs }}

      **Run info:** This page displays recorded outputs. Download the notebook, configure section 1 and run cells in order: ``train`` (default) fits a model; ``provided`` reanalyzes existing embeddings. See :doc:`../quickstart`.

      :download:`Download this notebook <../../tutorials/{{ notebook }}.ipynb>`

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
