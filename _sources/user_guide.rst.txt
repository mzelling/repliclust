User Guide
==========

Generating synthetic data with **repliclust** is different from the 
experience you may have had with other cluster generators. Our software
is based on *data set archetypes*, high-level geometric descriptions
of whole classes of data sets. 

To understand what we mean, try the demo here: `here <https://demo.repliclust.org>`_!

But since you're still here, let me explain. An archetype represents a
probability distribution over mixture models with similar overall geometry.
For example, data sets that looks like "seven "
To generate individual synthetic data sets, **repliclust** first generates
a probabilistic mixture model that matches the desired archetype. This
mixture model, in turn, samples the actual data set(s). The figure below
sketches our workflow.

.. image:: workflow.svg

The following brief tutorials demonstrate how to use **repliclust**. The
section :doc:`getting_started` will get you started generating your own
data sets within minutes.

.. toctree::

    getting_started
    generating_multiple_datasets
    specifying_an_archetype
