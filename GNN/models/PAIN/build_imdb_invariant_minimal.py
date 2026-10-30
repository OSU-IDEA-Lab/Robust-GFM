#!/usr/bin/env python3
"""PAIN IMDb invariant artifacts: each variant compiled into the semantic program.

Each variant's graph is compiled with the PAIN compiler (``preprocessing.imdb_node_classification.compile_invariant_edges``),
keeping the conditional director / actor edges, and PAIN's invariant preprocessing (per-variant compilation, hash
check, exhaustive rooted simple paths with L = 3) writes the result to ``$PAIN_PREPROCESSED_ROOT/IMDB_invariant_minimal``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pain_nc import paths  # noqa: E402
import preprocessing.imdb_invariant as inv  # noqa: E402
from preprocessing.imdb_node_classification import compile_invariant_edges as pain_compiler  # noqa: E402

LINK, YEAR = 3, 4


def conditional_compiler(physical_edges, variant, node_types):
    semantic, audit = pain_compiler(physical_edges, variant, node_types)
    nt = np.asarray(node_types)
    dropped = {e for e in semantic if {int(nt[e[0]]), int(nt[e[1]])} == {LINK, YEAR}}
    semantic = semantic - dropped
    audit = dict(audit, compiler="imdb_movie_link_conditional_semantic_MINIMAL (closure minus projected link-year)",
                 dropped_link_year_edges=len(dropped), semantic_edges=len(semantic))
    return semantic, audit


def main():
    out = paths.preprocessed("IMDB_invariant_minimal")
    if (out / "metadata.json").exists():
        sys.exit(f"{out} already exists - refusing to overwrite")
    inv.compile_invariant_edges = conditional_compiler          # module-level name used inside inv.preprocess
    inv.preprocess(paths.imdb_csv(), out, validate=True, schema="year")
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["status"] == "PASS" and meta["semantic_equals_universal"], "invariant artifacts failed the check"
    print("invariant program check:", meta["status"],
          "| semantic hash", set(meta["semantic_graph_sha256"].values()), "| path program", set(meta["selected_path_program_sha256"].values()))


if __name__ == "__main__":
    main()
