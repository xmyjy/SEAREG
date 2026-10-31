SEAREG
======

.. rst-class:: seareg-lead

   Shared spot–peak representations for spatial chromatin regulation.

SEAREG learns from chromatin counts, tissue coordinates and optional
transcription factor motif associations using a multi-relational heterogeneous
graph. Its representations connect spatial tissue organization with genomic
regulatory features in single-section and multislice analyses.

.. grid:: 1 1 3 3
   :gutter: 2
   :class-container: seareg-entry-grid

   .. grid-item-card:: Install SEAREG
      :link: installation
      :link-type: doc

      Set up the tutorial environment or install the Python package.

   .. grid-item-card:: Explore tutorials
      :link: tutorials/index
      :link-type: doc

      Choose a dataset and inspect saved analyses and figures.

   .. grid-item-card:: Python & CLI
      :link: continuous_pipeline
      :link-type: doc

      Build a workflow for your own inputs and manage training jobs.

`Source code <https://github.com/xmyjy/SEAREG>`_ ·
`Software citation <https://github.com/xmyjy/SEAREG/blob/main/CITATION.cff>`_

From tissue to regulatory features
----------------------------------

.. figure:: _static/figure1_homepage.svg
   :alt: Chromatin counts, spatial neighbors and optional motif associations form a heterogeneous graph; shared spot, peak and motif representations support spatial and regulatory analyses.
   :align: center
   :width: 100%
   :figclass: seareg-method-figure

   SEAREG connects spatial neighborhoods and chromatin features through a
   shared representation. :download:`Download the original SVG <_static/figure1_homepage.svg>`.

Spots and peaks are linked by chromatin counts; spatial edges connect tissue
neighbors. Sequence-informed analyses also link peaks to motif nodes.
The :doc:`method guide <unified_framework>` explains the graph, learned
representations and score interpretation.

What can you study?
-------------------

**Spatial domains.** Cluster spot representations and compare the resulting
domains with tissue organization. The :doc:`single-section tutorials <tutorials/index>`
show domain maps; :doc:`T6 <tutorials/T6_Mouse_Brain_Development_MISAR_seq>`
jointly models four developing mouse brain sections using shared peaks and
within-section spatial graphs.

**Peak profiles.** Use spot–peak representation scores to inspect specified
chromatin features across tissue. :doc:`T1 <tutorials/T1_Mouse_Embryo_E13_Spatial_ATAC_RNA>`
shows feature-score maps alongside domains and spot/joint UMAPs. These are
model-derived profiles; see the method guide for their interpretation.

**Motif-informed profiles.** When motif associations are included, spot–motif
scores summarize motif-associated patterns. T1 demonstrates these maps.
Motif scores alone do not establish transcription factor binding or causality.
See :doc:`motif_annotation` for annotation inputs and scanning options.

**Candidate regulatory relationships.** Shared peak and motif representations
can support downstream prioritization. Candidate TF–element–target
relationships additionally require evidence such as external peak–gene links;
the package's domain, embedding and spatial-score outputs are described in
the :doc:`API reference <api>`.

Choose a tutorial
-----------------

T1–T5 model one section each with optional motif information. T6 uses a
dedicated motif-free adaptive_depth_gate multislice workflow. The ATAC–RNA and CUT&Tag–RNA study
names identify data sources: these tutorials use chromatin counts and spatial
coordinates as model inputs, rather than automatically adding RNA to training.

.. grid:: 1 1 2 2
   :gutter: 2
   :class-container: seareg-tutorial-grid

   .. grid-item-card:: T1 · Mouse embryo, E13
      :link: tutorials/T1_Mouse_Embryo_E13_Spatial_ATAC_RNA
      :link-type: doc

      Spatial ATAC–RNA study · single section.
      Start here for domains, UMAPs and peak/motif score maps.

   .. grid-item-card:: T2 · Mouse embryo, E13
      :link: tutorials/T2_Mouse_Embryo_E13_Spatial_ATAC
      :link-type: doc

      Spatial ATAC · single section.
      Follow the workflow on an independent ATAC dataset.

   .. grid-item-card:: T3 · Mouse brain, P22
      :link: tutorials/T3_Mouse_Brain_P22_Spatial_ATAC_RNA
      :link-type: doc

      Spatial ATAC–RNA study · single section.
      Explore a larger chromatin dataset and its spatial domains.

   .. grid-item-card:: T4 · Mouse brain, P22
      :link: tutorials/T4_Mouse_Brain_P22_Spatial_CUTTag_RNA_H3K27ac
      :link-type: doc

      Spatial CUT&Tag–RNA study · single section.
      Model H3K27ac counts independently of the ATAC example.

   .. grid-item-card:: T5 · Human tonsil
      :link: tutorials/T5_Human_Tonsil_Spatial_ATAC
      :link-type: doc

      Spatial ATAC · single section.
      Apply the same workflow to human tissue.

   .. grid-item-card:: T6 · Developing mouse brain
      :link: tutorials/T6_Mouse_Brain_Development_MISAR_seq
      :link-type: doc

      MISAR-seq · four sections, E11.0–E18.5.
      Compare reference annotations, domains and joint representations.

See :doc:`tutorials/index` for dataset sizes and :doc:`multislice` for the
multislice protocol and saved-result provenance.

Train a model or analyze existing embeddings
--------------------------------------------

**Train on your data.** Install the runtime, prepare :doc:`input_data` and
follow a matching tutorial or the :doc:`Python workflow <continuous_pipeline>`.
All six notebooks default to ``RUN_MODE="train"`` and use independent training
jobs. Each execution writes its models, exports, analysis tables and figures
to a unique run directory under the selected output base.

**Analyze existing embeddings.** Set ``RUN_MODE="provided"`` and select an
``EMBEDDING_DIR`` with matching ``ATAC_PATH`` counts. The repository includes
named exports for the examples. This path performs downstream analysis
without training. Provided embeddings are analysis inputs; exact training
recovery requires a complete checkpoint. See :doc:`workflow` for result assets
and recovery.

These web pages display saved notebook outputs. Browsing or building the
documentation does not execute notebook cells.

Data and project
----------------

The full repository includes compressed tutorial inputs and named embeddings;
package distributions contain the software and scanner resources.
:doc:`data_access` lists sample identities, original studies and accessions.
Input and output paths can also point outside the repository.

This documentation describes SEAREG |release|, the first public release.
Use the `software citation metadata <https://github.com/xmyjy/SEAREG/blob/main/CITATION.cff>`_
when citing the software and the original studies when using their data.
Source code is available under the
`MIT License <https://github.com/xmyjy/SEAREG/blob/main/LICENSE>`_; bundled
components retain the licenses listed in the
`third-party notices <https://github.com/xmyjy/SEAREG/blob/main/seareg/THIRD_PARTY_NOTICES.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   data_access

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: User guide

   input_data
   motif_annotation
   workflow
   multislice
   continuous_pipeline
   cli

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Method & reference

   unified_framework
   api
   architecture

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Project

   troubleshooting
   citation
   changelog
