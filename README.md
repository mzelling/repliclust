[![Tests](https://github.com/mzelling/repliclust/actions/workflows/tests.yml/badge.svg)](https://github.com/mzelling/repliclust/actions/workflows/tests.yml)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/mzelling/7c9e8b5cc4b5124352fb58b0c753f79f/raw/bfe0b3e29f0f92f8acad677c47e864148fecefec/repliclust__heads_dev.json)

                                                                  
    ██████  ███████ ██████  ██      ██  ██████ ██      ██    ██ ███████ ████████ 
    ██   ██ ██      ██   ██ ██      ██ ██      ██      ██    ██ ██         ██    
    ██████  █████   ██████  ██      ██ ██      ██      ██    ██ ███████    ██    
    ██   ██ ██      ██      ██      ██ ██      ██      ██    ██      ██    ██    
    ██   ██ ███████ ██      ███████ ██  ██████ ███████  ██████  ███████    ██    
                                                                                 
                                                                              
              
## Description

**repliclust** is a Python package for generating synthetic datasets with clusters. It allows you to generate many different datasets that are geometrically similar, but *without* ever touching low-level parameters like cluster centroids or covariance matrices.

## Features

- Reproducibly generate clusters with defined geometric characteristics
- Manage cluster overlaps, shapes, and probability distributions through intuitive, high-level controls
- Define custom dataset archetypes to power reproducible and informative benchmarks

## Installation

```bash
pip install repliclust
```

## Quickstart

```python
from repliclust import Archetype, DataGenerator

# Create archetype for 5 oblong clusters with typical "aspect ratio" of 3
oblong_clusters = Archetype(n_clusters=5, dim=2, n_samples=500,
                            aspect_ref=3, aspect_maxmin=1.5,
                            name="oblong")
# Define the data generator
data_generator = DataGenerator(archetype=oblong_clusters)

# Sample data points X and class labels y
X, y, _ = data_generator.synthesize()
```

## User Guide / Documentation

For a full user guide and documentation, visit the project website: [https://repliclust.org](https://repliclust.org).

## Citation

To reference **repliclust** in your work, please cite:
```
@article{Zellinger:2023,
  title   = {repliclust: Synthetic Data for Cluster Analysis},
  author  = {Zellinger, Michael J and B{\"u}hlmann, Peter},
  journal = {arXiv preprint arXiv:2303.14301},
  doi     = {10.48550/arXiv.2303.14301},
  year    = {2023}
}
```
