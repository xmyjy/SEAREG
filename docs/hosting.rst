Building and hosting
=====================

The documentation renders the tutorial instructions, code and saved outputs without
executing the notebooks.
It can be built without installing the scientific runtime or preparing datasets.

Build locally
--------------

Use a complete checkout or GitHub Download ZIP. Documentation dependencies are separate from the
scientific environment. From the repository root, create a docs-only
environment (Python 3.11 matches the Read the Docs configuration):

.. code-block:: bash

   python3.11 -m venv .venv/docs
   source .venv/docs/bin/activate
   python -m pip install -r docs/requirements.txt
   python tools/check_docs.py
   python -m sphinx -n -W --keep-going -b html docs docs/_build/html

On Windows, use ``py -3.11 -m venv .venv/docs`` and run the remaining commands
with ``.venv/docs/Scripts/python.exe`` instead of activating the Linux path.
The documentation build itself does not need WSL2, GPU, R or scientific
packages.

Open ``docs/_build/html/index.html``. Notebook sources in ``tutorials/`` are
linked into Sphinx using the ``.nblink`` files in ``docs/tutorials/``.
``nbsphinx_execute='never'`` prevents notebook execution during the build.
The static checker also rejects notebook/cell metadata that would override
that policy. The GitHub documentation workflow builds and uploads HTML; it
does not itself deploy a public website.

Source links use ``READTHEDOCS_GIT_COMMIT_HASH`` when supplied, otherwise the
checkout's Git commit. For a Download ZIP or a builder without Git, set that
environment variable to the archive's known commit SHA before building to
retain revision-specific links. If the revision is unknown, the build still
succeeds, but source links point to ``main`` and may differ from the archive;
they do not identify the archive's revision.

Hosting requirements
----------------------

Read the Docs Community supports public Git repositories and public
documentation. Read the Docs Business can build private repositories and
serve public or private documentation. See the official
`platform comparison <https://about.readthedocs.com/choosing-a-platform/>`_
and `private repository guide <https://docs.readthedocs.com/platform/stable/guides/creating-project-private-repository.html>`_.
Private repositories require Business or another suitable host;
manual import does not remove Community's public-repository requirement. Keeping the source private
requires a suitable Business account or another authorized host for the
built static HTML. Making the repository public is a separate owner decision.

Import into Read the Docs
--------------------------

After the hosting account and repository access are settled:

1. Check the account's existing projects and reuse a matching project if one
   exists. For a new project, connect the GitHub account/app with access to
   the intended repository, then choose **Add project** and search for SEAREG.
2. Confirm the repository URL, project owner and default branch. Use the
   committed ``.readthedocs.yaml``: Ubuntu 24.04, Python 3.11,
   ``docs/requirements.txt``, ``docs/conf.py`` and failure on Sphinx warnings.
   No installation of SEAREG or its scientific dependencies is needed.
3. Confirm the Git integration/webhook and trigger a build of the intended
   commit. With manual import, configure the webhook explicitly. Inspect the
   build log for the commit SHA, environment, output directory and errors.

The official `project import instructions <https://docs.readthedocs.com/platform/stable/intro/add-project.html>`_
describe both automatic and manual setup. Do not substitute a guessed project
slug or URL for a successful build.

Verify a deployment
---------------------

Before adding a hosted URL to the repository's project links, record the
project owner, actual URL, version, build ID and commit SHA. Check that the
deployed SHA matches the reviewed source, then visit the homepage, all six
tutorials, their figures, internal links, search and notebook downloads.
Test desktop and narrow layouts and both themes. Downloaded notebooks must
match the source files; viewing/building pages must not execute their cells.
For a public site, check those routes without signing in. For a private
Business site, check the intended viewer permissions.

Versions and rollback
-----------------------

Use ``latest`` for the development branch. Activate a release tag and use
``stable`` only when there is an appropriate actual release; the current dev
version alone is not one. Confirm the default version and links in the
project settings. See Read the Docs'
`version guide <https://docs.readthedocs.com/platform/stable/versions.html>`_.

Keep the last verified build's SHA and version in the deployment handoff.
If a new deployment fails, inspect its logs and keep the previous verified
version as the default where available. To restore earlier content, build
the recorded known-good ref as a version and verify it before switching the
default, or revert the faulty documentation commit through the repository's
normal review process and rebuild. Do not rewrite scientific results or
release tags to repair a website. Record the restored SHA, build ID, URL and
checks so another maintainer can continue.
