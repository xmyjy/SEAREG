"""Resolve repository Notebook links when rendered beneath docs/tutorials."""

import re

import nbformat


def read_linked_notebook(text):
    """Read a Notebook and rebase its Markdown links for the Sphinx page.

    Source notebooks live in tutorials/ and link to ../docs/. Linked Sphinx
    pages live one directory below docs/, where those links resolve to ../.
    Only the in-memory Markdown representation changes; source notebooks and
    their downloadable copies retain the paths used by Jupyter and GitHub.
    """
    notebook = nbformat.reads(text, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            cell.source = re.sub(
                r"(!?\[[^\]]*\]\()\.\./docs/",
                r"\g<1>../",
                cell.source,
            )
    return notebook


def configure_linked_converter(app):
    # nbsphinx-link owns the .nblink parser. Set its notebook converter after
    # source suffix registration so nbsphinx does not register that suffix twice.
    app.config.nbsphinx_custom_formats[".nblink"] = (
        "seareg_notebooks.read_linked_notebook"
    )


def setup(app):
    app.connect("builder-inited", configure_linked_converter)
    return {"version": "1", "parallel_read_safe": True}
