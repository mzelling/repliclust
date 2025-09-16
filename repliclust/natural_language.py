"""Utilities for constructing Archetypes from natural language (optional)."""

# Optional dependencies: openai and python-dotenv
try:
    from openai import OpenAI  # type: ignore
except Exception:  # ImportError when extra not installed
    OpenAI = None  # type: ignore

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv():  # type: ignore
        return False

OPENAI_CLIENT = None

def load_openai_client(api_key=None):
    """Load the OpenAI client if the optional dependency is available."""
    global OPENAI_CLIENT

    load_dotenv()
    try:
        OPENAI_CLIENT = OpenAI(api_key=api_key)
    except Exception:
        OPENAI_CLIENT = None
        print(
            "Failed to initialize OpenAI client."
            + " Either put OPENAI_API_KEY=<...> into the .env file"
            + " or pass openai_api_key=<...> as an argument in a function call."
        )

# Initialize client on import (no-op if OpenAI not installed)
load_openai_client()

FEW_SHOT_EXAMPLES = [
    {"input":"five oblong clusters in two dimensions","output":"{\n  \"n_clusters\": 5,\n  \"dim\": 2,\n  \"n_samples\": 500,\n  \"aspect_ref\": 3,\n  \"aspect_maxmin\": 1.5,\n}"},
    {"input":"three spherical clusters with significant overlap in two dimensions","output":"{\n  \"n_clusters\": 3,\n  \"dim\": 2,\n  \"n_samples\": 300,\n  \"max_overlap\": 0.2,\n  \"min_overlap\": 0.1,\n  \"aspect_ref\": 1.0,\n  \"aspect_maxmin\": 1.0\n}"},
    {"input":"eight spherical clusters of different sizes with significant overlap in two dimensions","output":"{\n  \"n_clusters\": 8,\n  \"dim\": 2,\n  \"n_samples\": 800,\n  \"max_overlap\": 0.25,\n  \"min_overlap\": 0.1,\n  \"aspect_ref\": 1.0,\n  \"aspect_maxmin\": 1.0,\n  \"radius_maxmin\": 2.0\n}"},
    {"input":"ten clusters which are all highly oblong (and equally so) but of very different sizes, with moderate overlap","output":"{\n  \"n_clusters\": 10,\n  \"n_samples\": 1000,\n  \"aspect_ref\": 5,\n  \"aspect_maxmin\": 1.0,\n  \"max_overlap\": 0.10,\n  \"min_overlap\": 0.05,\n  \"radius_maxmin\": 4.0\n}"},
    {"input":"five clusters with significant class imbalance","output":"{\n  \"n_clusters\": 5,\n  \"n_samples\": 500,\n  \"imbalance_ratio\": 5,\n  \"aspect_ref\": 1.5,\n  \"aspect_maxmin\": 1.5\n}"},
    {"input":"five clusters with perfect class balance","output":"{\n  \"n_clusters\": 5,\n  \"n_samples\": 500,\n  \"imbalance_ratio\": 1.0,\n  \"aspect_ref\": 1.4,\n  \"aspect_maxmin\": 1.6\n}"},
    {"input":"eight clusters of which 70% are exponentially distributed and 30% are normally distributed","output":"{\n  \"n_clusters\": 8,\n  \"n_samples\": 800,\n  \"aspect_ref\": 1.7,\n  \"aspect_maxmin\": 1.5,\n  \"distributions\": [\"exponential\", \"normal\"],\n  \"distribution_proportions\": [0.7, 0.3],\n}"},
    {"input":"eight clusters with 1000 total samples of which half are exponentially distributed and half are normally distributed","output":"{\n  \"n_clusters\": 8,\n  \"n_samples\": 1000,\n  \"aspect_ref\": 1.7,\n  \"aspect_maxmin\": 1.5,\n  \"distributions\": [\"exponential\", \"normal\"],\n  \"distribution_proportions\": [0.5, 0.5]\n}"},
    {"input":"two clusters of different sizes in 10 dimensions that are well-separated","output":"{\n  \"n_clusters\": 2,\n  \"dim\": 10,\n  \"n_samples\": 200,\n  \"aspect_ref\": 2\n  \"aspect_maxmin\": 2,\n  \"radius_maxmin\": 4.0,\n  \"max_overlap\": 0.001,\n  \"min_overlap\": 0.0001\n}"},
    {"input":"very oblong clusters that overlap heavily","output":"{\n  \"n_clusters\": 6,\n  \"n_samples\": 600,\n  \"aspect_ref\": 7,\n  \"aspect_maxmin\": 1.4,\n  \"max_overlap\": 0.4,\n  \"min_overlap\": 0.3\n}"},
    {"input":"highly separated and very oblong clusters","output":"{\n  \"n_clusters\": 4,\n  \"n_samples\": 400,\n  \"aspect_ref\": 6,\n  \"aspect_maxmin\": 1.6,\n  \"max_overlap\": 1e-4,\n  \"min_overlap\": 1e-5\n}"},
    {"input":"ten clusters with very different shapes","output":"{\n  \"n_clusters\": 10,\n  \"n_samples\": 1000,\n  \"aspect_ref\": 1.5,\n  \"aspect_maxmin\": 3.0,\n  \"radius_maxmin\": 3.0\n}"},
    {"input":"twelve well-separated clusters with very different shapes", "output":"{\n  \"n_clusters\": 12,\n  \"n_samples\": 1200,\n  \"aspect_ref\": 1.5,\n  \"aspect_maxmin\": 5.0,\n \"radius_maxmin\": 5.0, \n \"max_overlap\": 1e-4,\n  \"min_overlap\": 1e-5\n}}"},
    {"input": "twelve highly separated Gaussian clusters with very different shapes", "output": "{\n  \"n_clusters\": 12,\n  \"n_samples\": 1200,\n  \"aspect_ref\": 1.5,\n  \"aspect_maxmin\": 5.0,\n \"radius_maxmin\": 5.0, \n \"max_overlap\": 1e-4,\n  \"min_overlap\": 1e-5\n \"distributions\": [\"normal\"]}}"},
    {"input": "five heavy-tailed clusters", "output": "{\n  \"n_clusters\": 5,\n  \"n_samples\": 500,\n  \"aspect_ref\": 1.5,\n \"distributions\": [\"standard_t\", \"lognormal\", \"pareto\"]}}"},
    {"input": "clusters with holes", "output": "{\"distributions\": [\"f\"]}"},
    {"input": "clusters from a variety of distributions", "output": "{\"distributions\": [\"normal\", \"exponential\", \"gamma\", \"weibull\", \"lognormal\"]}"},
    {"input": "clusters from all different distributions", "output": "{\"distributions\": ['normal', 'standard_t', 'exponential', 'beta', 'uniform', 'chisquare', 'gumbel', 'weibull', 'gamma', 'f', and 'lognormal']}"},
    {"input": "clusters from different distributions", "output": "{\"distributions\": ['normal', 'exponential', 'beta', 'uniform', 'chisquare', 'gumbel', 'weibull', 'gamma', 'f', and 'lognormal']}"},
    {"input": "highly separated clusters from all different distributions but no heavy tails", "output": "{\"max_overlap\": 1e-4,\n  \"min_overlap\": 1e-5,\n \"distributions\": ['normal', 'exponential', 'beta', 'uniform', 'chisquare', 'gumbel', 'weibull', 'gamma', 'f', and 'lognormal']}"},
    {"input": "seven clusters with uniform distribution with light overlap", "output": "{ \"max_overlap\": 0.025, \n\"min_overlap\": 0.0025,\n \"distributions\": [\"uniform\"]}"},
    {"input": "clusters with bounded support", "output": "{\"distributions\": [\"beta\", \"uniform\"]}"},
]

FEW_SHOT_NAME_EXAMPLES = [
    {"input":"five oblong clusters in two dimensions","output":"five_oblong_2d"},
    {"input":"three spherical clusters with significant overlap in two dimensions","output":"three_spherical_significant_overlap_2d"},
    {"input":"eight spherical clusters of different sizes with significant overlap in two dimensions","output":"eight_spherical_different_sizes_significant_overlap_2d"},
    {"input":"ten clusters which are all highly oblong (and equally so) but of very different sizes, with moderate overlap","output":"ten_highly_oblong_very_different_shapes_moderate_overlap"},
    {"input":"five clusters with significant class imbalance","output":"five_significant_class_imbalance"},
    {"input":"five clusters with perfect class balance","output":"five_perfect_class_balance"},
    {"input":"eight clusters of which 70% are exponentially distributed and 30% are normally distributed","output":"eight_exp_and_norm"},
    {"input":"eight clusters with 1000 total samples of which half are exponentially distributed and half are normally distributed","output":"eight_exp_and_norm_1000_samples"},
    {"input":"two clusters of different sizes in 10 dimensions that are well-separated","output":"two_different_sizes_well_separated_10d"},
    {"input":"very oblong clusters that overlap heavily","output":"very_oblong_heavy_overlap"},
    {"input":"ten clusters with very different shapes","output":"ten_very_different_shapes"},
]

def format_few_shot_examples(examples):
    return "\n\n".join(
        [ "Description: {}\nArchetype JSON: {}".format(example['input'], example['output'])
            for example in examples ]
        )

MAKE_ARCHETYPE_SYSTEM_PROMPT = (
   """Your task is to turn a verbal description of a data set archetype from Repliclust into a precise JSON that specifies which parameter settings to use to create the desired data set archetype in Repliclust. These are the allowed parameters: 
   
n_clusters: int >= 1, the number of clusters to generate
dim: int >= 2, the dimensionality of the data
n_samples: int >= 1, the number of data samples to generate
aspect_ref: float >= 1, the eccentricity of a typical cluster (how oblong vs spherical it is)
aspect_maxmin: float >= 1, how much the eccentricity varies across clusters in a data set
radius_maxmin: float >= 1, how much cluster radius (and thereby cluster volume) varies across the clusters
max_overlap: float > 0, the maximum allowed overlap between any pair of clusters (0.1-0.2 is significant overlap, 0.01-0.05 is little overlap, 0.001 is very little overlap, and 0.0001 and lower is well-separated)
min_overlap: float > 0, the minimum amount of overlap each cluster should have with some other cluster, preventing a cluster from being too far away from all other clusters
imbalance_ratio: float >= 1, specifies how imbalanced the number of data points per cluster is
distributions: list[str], determines the probability distributions to use for the clusters; the available distributions are 'normal', 'standard_t', 'exponential', 'beta', 'uniform', 'chisquare', 'gumbel', 'weibull', 'gamma', 'pareto', 'f', and 'lognormal'

IMPORTANT NOTES:
Any words like "separated", "far away", "close together", or "overlapping" refer to the overlap between clusters. Far apart means that max_overlap is 1e-4 or less
Always make min_overlap smaller than max_overlap, usually ten times smaller!
ONLY include the Pareto ('pareto') distribution if the user specifically asks for heavy tails!
   
EXAMPLES:

{examples}""".format(examples=format_few_shot_examples(FEW_SHOT_EXAMPLES))
)

MAKE_ARCHETYPE_PROMPT_TEMPLATE = """Description: {description}\nArchetype JSON: """

MAKE_ARCHETYPE_NAME_SYSTEM_PROMPT = (
    """Your task is to turn a description of a data set archetype into an identifier for
the archetype. The identifier should be short yet descriptive, and not contain any whitespace
(but underscores are OK). IMPORTANT: the identifier should be a valid Python variable name.
Specifically, it may NOT start with a number, nor contain any special character except for
underscores.

EXAMPLES:

{examples}""".format(examples=format_few_shot_examples(FEW_SHOT_NAME_EXAMPLES))
)

MAKE_ARCHETYPE_NAME_PROMPT_TEMPLATE = """Description: {description}\nArchetype identifier: """