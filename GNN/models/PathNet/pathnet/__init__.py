"""PathNet node-classification adaptation for the Robust-Representations baselines.

Faithful port of the node-classification model from zjunet/PathNet
("Beyond Homophily: Structure-aware Path Aggregation Graph Neural Network",
IJCAI 2022), wired to the same MAGNN-preprocessed IMDB / Freebase graphs used by
``PathNNs_expressive`` so the two path-based baselines are directly comparable.
"""

from .model import PathNetNC
from .walks import WalkData, build_walk_data

__all__ = ["PathNetNC", "WalkData", "build_walk_data"]
