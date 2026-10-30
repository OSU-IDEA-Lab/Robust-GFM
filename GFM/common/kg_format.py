"""Shared on-disk format for evaluating graph foundation models across representations.

Layout of one dataset directory (``GFM/data/<DATASET>/``)::

    base/
        entities.txt      one token per line; line i = entity id i (the ORIGINAL entities only)
        relations.txt     one token per line; line i = relation id i (the ORIGINAL relations only)
        train.tsv         h r t as integer ids (official training triples)
        valid.tsv         h r t (official validation queries)
        test.tsv          h r t (official test queries)
        filter_extra.tsv  (optional) known-true triples used ONLY for filtering, e.g. held-out queries
                          left out of a sampled query set
        negatives_<split>.npy   int32 [num_queries, 2, K]: K fixed filtered negatives per query and
                                direction (0 = predict tail, 1 = predict head), identical for every variant
        manifest.json
    variants/<VARIANT>/
        entities.txt      base entities first (same ids), then any nodes the transformation adds
        relations.txt     base relations first (same ids), then any relations the transformation adds
        graph.tsv         h r t: the message-passing graph the model sees for this variant
        manifest.json     what the transformation did, counts, hashes, leakage statistics

Every variant keeps the base entity and relation ids, so queries, filters and fixed negatives are
shared and per-query scores are directly comparable across variants. Queries never change; only the
graph the model passes messages over does.
"""

import hashlib
import json
import os

import numpy as np


def write_lines(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(f"{x}\n")


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_triples(path, triples):
    arr = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
    np.savetxt(path, arr, fmt="%d", delimiter="\t")
    return arr


def read_triples(path):
    arr = np.loadtxt(path, dtype=np.int64, delimiter="\t", ndmin=2)
    return arr.reshape(-1, 3)


def file_sha256(path, length=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def key_seed(*parts):
    """Deterministic 64-bit seed from arbitrary parts (keyed, not positional, randomness)."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little")


def make_fixed_negatives(queries, true_tails, true_heads, num_entities, k, seed, split):
    """K filtered negatives per (query, direction), keyed on the query itself.

    queries: int array [Q, 3] (h, r, t); true_tails[(h, r)] / true_heads[(t, r)]: sets of known answers
    over train + valid + test. A negative is never a known answer. Keyed seeding makes the draw for a
    query independent of its position in the file and of every other query.
    """
    out = np.empty((len(queries), 2, k), dtype=np.int32)
    for i, (h, r, t) in enumerate(queries.tolist()):
        for d, (anchor, banned) in enumerate(((h, true_tails[(h, r)]), (t, true_heads[(t, r)]))):
            rng = np.random.default_rng(key_seed(seed, split, d, h, r, t))
            picked = []
            seen = set()
            while len(picked) < k:
                cand = rng.integers(0, num_entities, size=2 * k).tolist()
                for c in cand:
                    if c in banned or c in seen:
                        continue
                    seen.add(c)
                    picked.append(c)
                    if len(picked) == k:
                        break
            out[i, d] = picked
    return out


def answer_sets(*triple_arrays):
    tails, heads = {}, {}
    for arr in triple_arrays:
        for h, r, t in arr.tolist():
            tails.setdefault((h, r), set()).add(t)
            heads.setdefault((t, r), set()).add(h)
    return tails, heads


def write_base(out_dir, entities, relations, train, valid, test, extra_manifest, num_negatives=50,
               neg_seed=1566911444, filter_extra=None):
    base = os.path.join(out_dir, "base")
    os.makedirs(base, exist_ok=True)
    write_lines(os.path.join(base, "entities.txt"), entities)
    write_lines(os.path.join(base, "relations.txt"), relations)
    tr = write_triples(os.path.join(base, "train.tsv"), train)
    va = write_triples(os.path.join(base, "valid.tsv"), valid)
    te = write_triples(os.path.join(base, "test.tsv"), test)
    fx = np.zeros((0, 3), dtype=np.int64)
    if filter_extra is not None and len(filter_extra):
        fx = write_triples(os.path.join(base, "filter_extra.tsv"), filter_extra)
    tails, heads = answer_sets(tr, va, te, fx)
    for split, q in (("valid", va), ("test", te)):
        neg = make_fixed_negatives(q, tails, heads, len(entities), num_negatives, neg_seed, split)
        np.save(os.path.join(base, f"negatives_{split}.npy"), neg)
    manifest = {
        "num_entities": len(entities),
        "num_relations": len(relations),
        "num_train": int(len(tr)),
        "num_valid": int(len(va)),
        "num_test": int(len(te)),
        "num_fixed_negatives": num_negatives,
        "negative_seed": neg_seed,
        "num_filter_extra": int(len(fx)),
        "negatives_are_filtered_against": "train + valid + test (+ filter_extra)",
        "hashes": {name: file_sha256(os.path.join(base, name))
                   for name in ("entities.txt", "relations.txt", "train.tsv", "valid.tsv", "test.tsv",
                                "negatives_valid.npy", "negatives_test.npy")},
    }
    manifest.update(extra_manifest)
    write_json(os.path.join(base, "manifest.json"), manifest)
    return manifest


def write_variant(out_dir, name, entities, relations, graph, manifest):
    vdir = os.path.join(out_dir, "variants", name)
    os.makedirs(vdir, exist_ok=True)
    write_lines(os.path.join(vdir, "entities.txt"), entities)
    write_lines(os.path.join(vdir, "relations.txt"), relations)
    g = write_triples(os.path.join(vdir, "graph.tsv"), graph)
    manifest = dict(manifest)
    manifest.update({
        "variant": name,
        "num_entities": len(entities),
        "num_relations": len(relations),
        "num_graph_edges": int(len(g)),
        "hashes": {n: file_sha256(os.path.join(vdir, n)) for n in ("entities.txt", "relations.txt", "graph.tsv")},
    })
    write_json(os.path.join(vdir, "manifest.json"), manifest)
    return manifest


def load_base(dataset_dir):
    base = os.path.join(dataset_dir, "base")
    fx_path = os.path.join(base, "filter_extra.tsv")
    return {
        "entities": read_lines(os.path.join(base, "entities.txt")),
        "relations": read_lines(os.path.join(base, "relations.txt")),
        "filter_extra": read_triples(fx_path) if os.path.exists(fx_path) else np.zeros((0, 3), dtype=np.int64),
        "train": read_triples(os.path.join(base, "train.tsv")),
        "valid": read_triples(os.path.join(base, "valid.tsv")),
        "test": read_triples(os.path.join(base, "test.tsv")),
        "manifest": read_json(os.path.join(base, "manifest.json")),
        "dir": base,
    }


def load_variant(dataset_dir, name):
    vdir = os.path.join(dataset_dir, "variants", name)
    return {
        "entities": read_lines(os.path.join(vdir, "entities.txt")),
        "relations": read_lines(os.path.join(vdir, "relations.txt")),
        "graph": read_triples(os.path.join(vdir, "graph.tsv")),
        "manifest": read_json(os.path.join(vdir, "manifest.json")),
        "dir": vdir,
    }


def check_prefix(variant, base):
    """Variant ids must extend the base ids unchanged (shared queries depend on it)."""
    nb, rb = len(base["entities"]), len(base["relations"])
    assert variant["entities"][:nb] == base["entities"], "variant entity ids do not extend the base ids"
    assert variant["relations"][:rb] == base["relations"], "variant relation ids do not extend the base ids"
