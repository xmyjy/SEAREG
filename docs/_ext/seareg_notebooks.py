"""Resolve repository Notebook links when rendered beneath docs/tutorials."""

from hashlib import sha256
from html import escape, unescape
from pathlib import Path
import re
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

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


def version_image_urls(app, exception):
    """Bind local rendered image links to their bytes without changing images."""
    if exception is not None or app.builder.format != "html":
        return
    output = Path(app.builder.outdir).resolve()
    digests = {}
    for page in output.rglob("*.html"):
        def replace(match):
            uri = urlsplit(unescape(match.group("url")))
            if uri.scheme or uri.netloc or "_images/" not in uri.path:
                return match.group(0)
            image = (page.parent / unquote(uri.path)).resolve()
            if not image.is_relative_to(output) or not image.is_file():
                return match.group(0)
            if image not in digests:
                digests[image] = sha256(image.read_bytes()).hexdigest()
            query = [(key, value) for key, value in parse_qsl(uri.query, keep_blank_values=True)
                     if key != "v"]
            query.append(("v", digests[image]))
            url = urlunsplit(("", "", uri.path, urlencode(query), uri.fragment))
            return f'{match.group("attribute")}="{escape(url, quote=True)}"'

        source = page.read_text(encoding="utf-8")
        updated = re.sub(r'(?P<attribute>\b(?:src|href))="(?P<url>[^"\n]+)"', replace, source)
        if updated != source:
            page.write_text(updated, encoding="utf-8")


def setup(app):
    app.connect("builder-inited", configure_linked_converter)
    app.connect("build-finished", version_image_urls)
    return {"version": "2", "parallel_read_safe": True, "parallel_write_safe": True}
