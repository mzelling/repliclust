[![Tests](https://github.com/mzelling/repliclust/actions/workflows/tests.yml/badge.svg)](https://github.com/mzelling/repliclust/actions/workflows/tests.yml)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/mzelling/7c9e8b5cc4b5124352fb58b0c753f79f/raw/bfe0b3e29f0f92f8acad677c47e864148fecefec/repliclust__heads_dev.json)

                                                                  
    ██████  ███████ ██████  ██      ██  ██████ ██      ██    ██ ███████ ████████ 
    ██   ██ ██      ██   ██ ██      ██ ██      ██      ██    ██ ██         ██    
    ██████  █████   ██████  ██      ██ ██      ██      ██    ██ ███████    ██    
    ██   ██ ██      ██      ██      ██ ██      ██      ██    ██      ██    ██    
    ██   ██ ███████ ██      ███████ ██  ██████ ███████  ██████  ███████    ██    
                                                                                 
                                                                              
## Natural Language-Based Synthetic Data Generation for Cluster Analysis

**repliclust** is a Python package for generating synthetic datasets with clusters based on high-level descriptions. Instead of manually setting low-level parameters like cluster centroids or covariance matrices, you can simply describe the desired characteristics of your data, and repliclust will automatically generate datasets that match those specifications.

## What can this software do for you?
+ **Simplify Synthetic Data Generation**: Eliminate the need to fine-tune low-level simulation parameters. Describe your desired scenario, and let **repliclust** handle the rest.

+ **Enhance Benchmark Quality**: By controlling high-level aspects of the data, you can create more informative benchmarks that reveal the strengths and weaknesses of clustering algorithms under various conditions.

+ **Accelerate Research**: Quickly generate diverse datasets to test hypotheses, validate models, and perform robustness checks.


## Key Features
+ **Generate Data from High-Level Descriptions**: Create datasets by specifying scenarios such as "clusters with very different shapes and sizes" or "highly overlapping oblong clusters."

+ **Data Set Archetypes**: Use archetypes to define the overall geometry of your datasets with intuitive parameters that summarize cluster overlaps, shapes, sizes, and distributions.

+ **Flexible Cluster Shapes**: Go beyond convex, blob-like clusters by applying nonlinear transformations, such as random neural networks for distortion or stereographic projections to create directional data.

+ **Reproducible and Informative Benchmarks**: Independently manipulate different aspects of the data to create benchmarks that effectively evaluate and compare clustering algorithms under various conditions.


## Demo

Try our demo [here](https://demo.repliclust.org)!


## Installation

```bash
pip install repliclust
```

Optional features are split into extras to keep the core lightweight:

- NLP-powered archetypes (OpenAI):

```bash
pip install "repliclust[nlp]"
```

- Neural-network distortion (PyTorch):

```bash
pip install "repliclust[distort]"
```

- Everything optional:

```bash
pip install "repliclust[all]"
```

## Quickstart

The easiest way to get started using repliclust is to create synthetic datasets from high-level descriptions in English. These NLP features require the `nlp` extra and an OpenAI API key. Install with `pip install "repliclust[nlp]"`. You can set the key as ``OPENAI_API_KEY=<your-api-key>`` in an .env file, or pass it to individual functions as a keyword argument ``openai_api_key="<your-api-key>"``.

+ **Generating data directly**:

```python
import repliclust as rpl

X, y, _ = rpl.generate("three highly separated oblong clusters in 10D", openai_api_key="<your-api-key>")
rpl.plot(X,y)
```

+ **Creating an archetype**:

```python
archetype = rpl.Archetype.from_verbal_description(
    "seven gamma-distributed clusters in 2D of very different shapes",
    openai_api_key="<your-api-key>"
)
```

+ **Generating data from the archetype**:

```python
X, y = archetype.synthesize()
```

+ **Making cluster shapes more irregular**:

```python
# distort() requires the 'distort' extra (PyTorch)
X_irregular = rpl.distort(X)
X_directional = rpl.wrap_around_sphere(X)  # available in core
```

## Documentation

+ **User Guide**: Learn how to generate datasets from high-level descriptions in the [User Guide](https://repliclust.org/getting_started.html).
+ **Reference**: Explore the package [Reference](https://repliclust.org/reference.html).


## Citation
To reference **repliclust** in your work, please cite:
```
@article{Zellinger:2023,
  title   = {Natural Language-Based Synthetic Data Generation for Cluster Analysis},
  author  = {Zellinger, Michael J and B{\"u}hlmann, Peter},
  journal = {arXiv preprint arXiv:2303.14301},
  doi     = {10.48550/arXiv.2303.14301},
  year    = {2023}
}
```
