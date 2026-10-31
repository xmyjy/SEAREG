Installation
============

Runtime
-------

Use Linux x86_64 and Python 3.10.22 for the pinned tutorial environment below.
The default GPU build is PyTorch 2.2.0 with CUDA 12.1. The supplied Conda file
selects the exact Python version; ``requirements.txt`` pins the scientific and
notebook packages. The package accepts Python 3.10–3.12 for general installs.
Windows users can run the workflow in a Linux environment under WSL2.
SEAREG uses Linux ``renameat2`` for publishing completed
output directories and POSIX file locks. Native Windows and macOS execution are
not supported by this release.

The tutorial dependencies include PyTorch, NumPy, SciPy, pandas, AnnData,
scikit-learn, h5py, Scanpy, UMAP, PyNNDescent, Numba/llvmlite, igraph, Louvain,
Matplotlib and JupyterLab. A compatible NVIDIA GPU is useful for the full
datasets; CPU training is available. Package versions and fixed seeds do not
guarantee identical floating-point values across hardware and numerical runtimes.

Install from the repository
---------------------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/xmyjy/SEAREG.git
   cd SEAREG

Create and activate the default Python 3.10.22 environment with Conda:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate seareg

If Python 3.10.22 is already installed, a virtual environment is an alternative:

.. code-block:: bash

   python3.10 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

Both paths use the same ``requirements.txt`` in the following steps.

The complete GitHub checkout or Download ZIP includes the six notebooks,
compressed input matrices and provided embeddings. Wheel and source
distributions contain the installable ``seareg`` package and scanner resources.
The source archive also includes environment specifications. Both omit
notebooks, documentation, data and results. Use the complete repository for
the tutorial commands below.

Select one PyTorch build
------------------------

By default, install the CUDA 12.1 wheel and Triton in the activated environment:

.. code-block:: bash

   python -m pip install "torch==2.2.0+cu121" --index-url https://download.pytorch.org/whl/cu121
   python -m pip install "triton==2.2.0"

The CUDA path requires an NVIDIA driver compatible with the selected wheel.
The PyTorch wheel supplies its CUDA runtime libraries; a separately installed
system CUDA toolkit does not choose the wheel's runtime. See the official
`PyTorch 2.2.0 installation commands <https://pytorch.org/get-started/previous-versions/#v220>`_
for other available builds. Changing the PyTorch version or CUDA build creates
a different environment from this pinned setup.

For CPU execution, select this wheel instead of the CUDA commands above:

.. code-block:: bash

   python -m pip install "torch==2.2.0" --index-url https://download.pytorch.org/whl/cpu

Continue with the same shared requirements below.

Install the tutorial stack and package
--------------------------------------

After selecting PyTorch, install the shared pins, then install SEAREG from this
checkout without resolving or replacing those dependencies:

.. code-block:: bash

   python -m pip install -r requirements.txt
   python -m pip install --no-build-isolation --no-deps -e .
   python -m pip check

``torch==2.2.0`` in the requirements accepts the selected CPU or CUDA 2.2.0
wheel. The CUDA path additionally uses Triton 2.2.0; the CPU path does not
require it. The shared pins include the neighbor-graph dependencies
(``pynndescent==0.5.13``, ``numba==0.61.2``, ``llvmlite==0.44.0``) and the
notebook renderer ``matplotlib-inline==0.1.7``. These are selected dependency
pins, rather than a complete lock of every transitive package.

Use ``--no-deps`` only after installing the requirements. An unrestricted
``pip install -e ".[analysis,cuda,notebooks]"`` can resolve different versions.
For a general package installation without the tutorial pins, the ``analysis``,
``cuda`` and ``notebooks`` extras remain available.

Package-only installation
-------------------------

For Python/CLI use on your own data, use a separate Linux environment with
Python 3.10–3.12. From the repository root, this CPU recipe installs SEAREG
and its analysis dependencies:

.. code-block:: bash

   python3 -m venv .venv-package
   source .venv-package/bin/activate
   python -m pip install --upgrade pip
   python -m pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
   python -m pip install ".[analysis]"
   python -m pip check
   seareg --help

This resolves versions within the package's dependency ranges; it does not
reproduce the pinned tutorial environment above. Jupyter and the example
datasets are unnecessary for the Python/CLI workflow. See
:doc:`continuous_pipeline` for independent count inputs, optional motifs and
detached training jobs. Select a compatible CUDA PyTorch build and the
``cuda`` extra when configuring GPU training.

When no compatible wheel is available, ``louvain`` may compile from source.
This requires a C/C++ compiler, CMake 3.18 or newer and Python development
headers matching the chosen interpreter. On Debian/Ubuntu with the distribution's
Python, these are commonly supplied by ``build-essential``, ``cmake`` and
``python3-dev``. A separately
installed Python needs its own matching headers, including ``Python.h``.

Optional motif-scanning dependencies
------------------------------------

Generating a motif matrix from genomic peaks additionally requires R,
bedtools and the R/Bioconductor packages listed in :doc:`motif_annotation`.
Create the supplied R environment separately while keeping your SEAREG
Python environment active:

.. code-block:: bash

   conda env create -f environment-motif.yml
   SEAREG_MOTIF_PREFIX="$(conda run -n seareg-motif python -c 'import sys; print(sys.prefix)')"
   export SEAREG_RSCRIPT="$SEAREG_MOTIF_PREFIX/bin/Rscript"
   export PATH="$PATH:$SEAREG_MOTIF_PREFIX/bin"
   "$SEAREG_RSCRIPT" --version
   "$SEAREG_RSCRIPT" -e 'library(motifmatchr); library(JASPAR2024)'
   bedtools --version

These commands work with either the Conda or virtual-environment Python
setup above. Appending the R environment's executable directory keeps the
active Python environment first on ``PATH`` while making bedtools available.
Launch Jupyter from this shell; the notebooks read
``SEAREG_RSCRIPT``. For the CLI, pass ``--rscript "$SEAREG_RSCRIPT"``; for
Python or JSON configuration, set ``rscript`` to that executable's absolute
path.

The R scanner is included in the wheel and source distribution. Initialize
the JASPAR2024 database cache, then select a matching reference, assembly and species as described in
:doc:`motif_annotation`. The five single-section tutorials already include motif
matrices; R resources are needed when generating an annotation.

Check the installation
----------------------

.. code-block:: bash

   python -c "import seareg; print(seareg.__version__)"
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   seareg --help

Prepare the :doc:`included inputs <data_access>` and run an input check from
the repository root:

.. code-block:: bash

   python tools/prepare_data.py --dataset ME13_joint_2187
   seareg --config tutorials/configs/ME13_joint_2187.json check

This checks files and dependencies. GPU availability is checked by the explicit
training worker when training starts. Choose the execution device in the
configuration as described in :doc:`workflow`.

Multislice coordinate preflight
-------------------------------

The motif-free adaptive_depth_gate model uses the pinned Python dependencies;
R, JASPAR, motif matrices and bedtools are unnecessary for T6.

For real data inspect
``seareg.multislice.geometry.spatial_coverage_report(coordinates, spatial,
slice_codes)`` on the native within-section spatial graph. It reports separate
eight-sector and unit four-side coverage by section, with tolerance 1e-6.
No four-side coverage means zero additional correction on those rows; the coherence correction
remains. Do not rotate/scale coordinates silently or use plotting transforms
as training inputs. Alpha and actual matched budgets are logged only after
full-message refresh. See :doc:`input_data` and :doc:`multislice`.

The T6 profile selects physical GPU1; choose an available card with
``SEAREG_PHYSICAL_GPU``. CPU and CUDA training use tmux. Reading saved
reference labels and coordinates needs no GPU, tmux or model checkpoint.
Read the per-run handoff before resuming.

Detached training jobs
-----------------------

The notebooks default to ``RUN_MODE="train"``, which uses independent tmux jobs.
Install tmux before opening a tutorial for training. In a Conda environment:

.. code-block:: bash

   conda install -c conda-forge tmux

For the virtual-environment path, install tmux with your Linux distribution's
package manager and ensure it is available on ``PATH``. Both CPU and CUDA
training use tmux. See :doc:`workflow` for reconnecting and exact-state recovery.

Open a tutorial
---------------

.. code-block:: bash

   jupyter lab tutorials/T1_Mouse_Embryo_E13_Spatial_ATAC_RNA.ipynb

Keep ``RUN_MODE="train"`` in the first cell. The tutorial prepares the selected
counts, motif associations and network, starts an independent training job,
then reads that job's final embeddings for analysis and plotting. Configure
the device and inputs before running the cells in order.

Use the Jupyter kernel belonging to the environment where SEAREG was installed.
Register it explicitly when Jupyter also has kernels from other environments:

.. code-block:: bash

   python -m ipykernel install --user --name seareg --display-name "Python (SEAREG)"

Select ``Python (SEAREG)`` in the notebook. The first cell accepts either the
example paths or your own inputs and output base. For CPU training, set
``TRAINING_OPTIONS["device"] = "cpu"`` and
``EXECUTION_OPTIONS["physical_gpu"] = None`` there; the profiles select CUDA
by default. The saved outputs
remain available when opening a notebook without running its cells.
The :doc:`tutorial index <tutorials/index>` lists the other five notebooks.

Use existing embeddings
-----------------------

For an existing result, set ``RUN_MODE="provided"``, choose its
``EMBEDDING_DIR`` and select the matching ``ATAC_PATH`` in the first cell.
This alternative performs downstream analysis without training, tmux, a GPU
or a complete model checkpoint. Its execution record identifies the selected
export. T6 records ``training_executed=false`` and ``training_run=null`` while
retaining its configured training and execution options. See :doc:`workflow`.
