# Information-based Invariance for Graph Neural Networks and Graph Foundation Models

Code for *Information-based Invariance, Equivariance, and Expressivity for Graph Neural Networks*, and for
evaluating graph foundation models on the same question.

The same information can be stored in graphs that are not isomorphic. A movie's director can be attached to the
movie node or to a copy of it, a fact can be an edge or a node, and a two-hop path can also appear as a single
shortcut edge. Models that read graph structure can give different answers on such information-equivalent
representations.

| folder | contents |
| --- | --- |
| [`GNN/`](GNN/README.md) | seven heterogeneous GNNs (RGCN, MAGNN, SlotGAT, SeHGNN, CMPNN, PathNet, PAIN) made invariant by running on a compiled semantic program; IMDb, DBLP, WN18RR and Freebase variants; four experimental arms |
| [`GFM/`](GFM/README.md) | pretrained graph foundation models evaluated zero-shot across representations, and with every representation compiled back to one canonical graph; one subfolder per model (`GFM/ULTRA/`, ...) |

Each folder is self-contained, with its own README, environment files and scripts. `GFM/` reads the WN18RR variants
from `GNN/data/` so both use identical graphs; everything else in `GFM/` is downloaded or built by
`GFM/scripts/`.
