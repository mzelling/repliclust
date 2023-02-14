User Guide
==========

Generating synthetic data with **repliclust** is different from the 
experience you may have had with other cluster generators. Our software
is based on *data set archetypes*, high-level geometric descriptions
of whole classes of data sets. 

More specifically, an archetype represents a probability distribution
over mixture models with similar geometry (same number of clusters,
overlaps between clusters, cluster probability distributions, ...). To
generate individual synthetic data sets, **repliclust** first generates
a probabilistic mixture model based on the data set archetype. This
mixture model, in turn, samples the actual data set. The figure below
sketches our workflow.

.. image:: workflow.svg

The following brief tutorials demonstrate how to use **repliclust**. The
section :doc:`basic_usage` will get you started generating your own
synthetic data sets within minutes.

.. toctree::

    basic_usage
    generating_multiple_datasets
    specifying_an_archetype
