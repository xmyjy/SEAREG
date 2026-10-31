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
    'T1': {
        'purpose': 'Resolve domains in an E13 embryo and inspect peak/motif spatial associations.',
        'inputs': 'Single-section ATAC counts, native XY and peak–motif associations in ``data/ME13_joint_2187/``. RNA is not a model input.',
        'profile': '``ME13_joint_2187.json``; 300 epochs; target K13.',
        'figures': 'Four saved figures: spatial domains, spot UMAP, joint UMAP and feature spatial scores/accessibility.',
        'provenance': 'Saved training outputs identify ``github_fresh_20261019T182236Z``. Optional provided exports are under ``result/ME13_joint_2187/training/embeddings/300_epoch/``.'
    },
    'T2': {
        'purpose': 'Resolve spatial domains from an E13 embryo accessibility section.',
        'inputs': 'Single-section ATAC counts, native XY and peak–motif associations in ``data/ME13_ATAC_2275/``.',
        'profile': '``ME13_ATAC_2275.json``; 300 epochs; target K17.',
        'figures': 'Three saved figures: spatial domains, spot UMAP and joint UMAP.',
        'provenance': 'Saved training outputs identify ``github_fresh_20261019T182113Z``. Optional provided exports are under ``result/ME13_ATAC_2275/training/embeddings/300_epoch/``.'
    },
    'T3': {
        'purpose': 'Resolve accessibility domains in a P22 mouse brain section.',
        'inputs': 'Single-section ATAC counts, native XY and peak–motif associations in ``data/P22_ATAC/``. RNA is not a model input.',
        'profile': '``P22_ATAC.json``; 500 epochs; target K13.',
        'figures': 'Three saved figures: spatial domains, spot UMAP and joint UMAP.',
        'provenance': 'Saved training outputs identify ``github_fresh_20261019T181535Z``. Optional provided exports are under ``result/P22_ATAC/training/embeddings/500_epoch/``.'
    },
    'T4': {
        'purpose': 'Resolve domains from spatial H3K27ac enrichment in a P22 mouse brain section.',
        'inputs': 'Single-section CUT&Tag H3K27ac counts, native XY and peak–motif associations in ``data/P22_H3K27ac/``. RNA is not a model input.',
        'profile': '``P22_H3K27ac.json``; 500 epochs; target K11.',
        'figures': 'Three saved figures: spatial domains, spot UMAP and joint UMAP.',
        'provenance': 'Saved training outputs identify ``github_fresh_20261019T181535Z``. Optional provided exports are under ``result/P22_H3K27ac/training/embeddings/500_epoch/``.'
    },
    'T5': {
        'purpose': 'Inspect broad spatial organization in a human tonsil accessibility section.',
        'inputs': 'Single-section ATAC counts, native XY and peak–motif associations in ``data/Tonsil/``.',
        'profile': '``Tonsil.json``; 500 epochs; target K3.',
        'figures': 'Three saved figures: spatial domains, spot UMAP and joint UMAP.',
        'provenance': 'Saved training outputs identify ``github_fresh_20261019T182113Z``. Optional provided exports are under ``result/Tonsil/training/embeddings/500_epoch/``.'
    },
    'T6': {
        'purpose': 'Integrate four developmental brain sections using an adaptive gating mechanism to reduce the influence of spots with locally anomalous sequencing depth.',
        'inputs': 'ATAC counts, shared peaks, section IDs and native XY in ``data/Embryo_multislice/``; no motifs or RNA. Reference labels are display/evaluation inputs.',
        'profile': '``Embryo_multislice_adaptive_depth_gate.json``; 500 epochs; train/provided pipelines use exact-K target16 with no fixed resolution.',
        'figures': 'Reference maps, spatial domains and three-view spot/joint UMAPs retain all 7,118 spots, including 81 Unknowns, and gray joint shared peaks.',
        'provenance': 'Sections 1–16 follow counts, training, analysis, figures and saved-result reopening. The current example outputs are under ``result/Embryo_multislice/adaptive_depth_gate/``; ``tutorial_execution.json`` identifies their inputs, training and analysis. Optional provided mode reads the final ``500_epoch/`` exports and fits a new analysis. See :doc:`../multislice`.'
    }
} %}
{% set tutorial = tutorials[notebook[:2]] %}

.. only:: html

   .. admonition:: {{ notebook[:2] }} — Tutorial at a glance
      :class: note

      **Purpose:** {{ tutorial.purpose }}

      **Inputs:** {{ tutorial.inputs }} See :doc:`../input_data` for the schema and :doc:`../data_access` for bundled sources.

      **Profile:** {{ tutorial.profile }} Profiles are in ``tutorials/configs/``.

      **Saved outputs:** {{ tutorial.figures }} {{ tutorial.provenance }}

      **Execution:** This page displays saved outputs without running cells. The original notebook defaults to ``RUN_MODE="train"``. Optional ``provided`` mode analyzes selected exports and fits new analysis; reopening saved analysis uses its existing labels and coordinates. See :doc:`../quickstart` for these choices and :doc:`../tutorials/index` for prerequisites.

      :download:`Download the original notebook <../../tutorials/{{ notebook }}.ipynb>`

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
