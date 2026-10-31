Citation and license
====================

Software citation
------------------

This documentation describes SEAREG |release|, the first public release.
Use the repository's :download:`CITATION.cff <../CITATION.cff>` as the software
citation metadata and record the version and Git commit used for your analysis.
The current metadata identifies SEAREG contributors as the software author
and the `SEAREG repository <https://github.com/xmyjy/SEAREG>`_ as its source.
It does not specify a software DOI or a separate formal-paper citation.

Dataset and method references
------------------------------

Citing the software does not replace citing the sequencing studies that
produced the data. :doc:`data_access` lists the original Zhang et al. (2023),
Deng et al. (2022) and Jiang et al. (2023) papers, their study accessions and
the modality used in each tutorial. Use those study references when using
their datasets. The :doc:`method guide <unified_framework>` explains the
representation and interpretation of model-derived scores.

Software and third-party licenses
----------------------------------

SEAREG's code and documentation use the
:download:`MIT License <../LICENSE>` (copyright 2026 SEAREG contributors).
Keep the copyright and permission notice in copies or substantial portions
of the software, as required by that license.

Bundled components retain their own terms. In particular, the unmodified
NVIDIA ``ptxas`` executable for the optional CUDA 11 scoring path is covered
by the NVIDIA CUDA Toolkit EULA, not by SEAREG's MIT License. See the
:download:`third-party notices <../seareg/THIRD_PARTY_NOTICES.md>` and the
:download:`bundled NVIDIA license <../seareg/_compiler/cuda11/LICENSE>` for
the component's terms and provenance. The software license does not replace
the original data providers' usage terms.
