.. repliclust documentation master file, created by
   sphinx-quickstart on Tue Dec 13 21:09:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Natural Language-Based Synthetic Data Generation for Cluster Analysis
=================================================================

**repliclust** is a Python package for generating synthetic datasets with clusters based on **high-level descriptions**. Instead of manually setting low-level parameters like cluster centroids or covariance matrices, you can simply describe the desired characteristics of your data, and **repliclust** will automatically generate datasets that match those specifications.

.. code-block:: python

    import repliclust as rpl

    X, y, _ = rpl.generate("three highly separated oblong clusters in 10D")

Try our demo `here <https://demo.repliclust.org>`_!


Key Features
------------

- **Generate Data from High-Level Descriptions**: Create datasets by specifying scenarios such as "clusters with very different shapes and sizes" or "highly overlapping oblong clusters."

- **Data Set Archetypes**: Use archetypes to define the overall geometry of your datasets with intuitive parameters that summarize cluster overlaps, shapes, sizes, and distributions.

- **Integration with Large Language Models (LLMs)**: Leverage LLMs to map verbal descriptions onto data set archetypes, enabling automated dataset generation from natural language inputs.

- **Flexible Cluster Shapes**: Go beyond convex, blob-like clusters by applying nonlinear transformations, such as random neural networks for distortion or stereographic projections to create directional data.

- **Reproducible and Informative Benchmarks**: Independently manipulate different aspects of the data to create benchmarks that effectively evaluate and compare clustering algorithms under various conditions.


Getting Started
---------------

- **Installation**: Follow the instructions in the `Installation Guide <install>`_ to set up **repliclust**.

- **User Guide**: Learn how to generate datasets from high-level descriptions in the `User Guide <user_guide>`_.

- **API Reference**: Explore the detailed API documentation in the `Reference <reference>`_ section.

- **Source Code**: Visit our `GitHub repository <https://github.com/mzelling/repliclust>`_ to view the source code and contribute.


Why Use **repliclust**?
-------------------

- **Simplify Synthetic Data Generation**: Eliminate the need to fine-tune low-level simulation parameters. Describe your desired scenario, and let **repliclust** handle the rest.

- **Enhance Benchmark Quality**: By controlling high-level aspects of the data, you can create more informative benchmarks that reveal the strengths and weaknesses of clustering algorithms under various conditions.

- **Accelerate Research**: Quickly generate diverse datasets to test hypotheses, validate models, and perform robustness checks.


Reference
---------

For more details, check out our paper: `Natural Language-Based Synthetic Data Generation for Cluster Analysis <https://arxiv.org/abs/2303.14301>`_.


.. toctree::
    :hidden:

    install
    user_guide
    reference
    source_code
