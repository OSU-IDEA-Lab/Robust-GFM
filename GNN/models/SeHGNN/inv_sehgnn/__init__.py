"""Invariant SeHGNN (node classification): channel compiler, model, bundle I/O.

See docs/INVARIANT_MECHANISMS.md for the design. The compiler turns a graph (a physical variant, the
union graph, or the compiled semantic program of a variant) into SeHGNN's precomputed channel
tensors under the full-K channel set (`full`; relation-aware `full_rel` on Freebase); the
runner trains the unchanged SeHGNN on them.
"""
