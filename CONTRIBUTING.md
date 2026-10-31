# Building and contributing

Use a complete GitHub checkout for notebooks and documentation. Wheel and source distributions contain the installable package; the source archive also includes environment specifications.

Create the [Linux Python environment](docs/installation.rst), then build the package:

```bash
python -m pip install -e . --no-deps
python -m pip install build
python -m pip check
python -m build
```

GitHub Actions builds the wheel and source distribution, checks clean installation on the supported Python versions, and builds the documentation.

Build the documentation with its dependencies:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The documentation build renders saved notebook outputs without running notebooks. Keep unrelated cells and saved outputs intact when editing a notebook. See [package architecture](docs/architecture.rst) for module responsibilities.
