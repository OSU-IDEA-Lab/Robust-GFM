#!/usr/bin/env python3
"""Rebuild Freebase `exact_3` from the clean `range_2_3`, and build the true union.

`exact_3/link.dat` is unusable: a single contiguous 3.78 GB NUL write hole, and it
holds the UNFILTERED 143-new-relation enumeration while its own `info.dat`, its
`new_transitive_relations.*` and `RGCN/registries/exact_3.jsonl` all describe the
FILTERED 22-new-relation variant (`exact_2` is consistently filtered at 6).

`range_2_3/link.dat` is clean and is a strict superset, so `exact_3` is exactly
recoverable by keeping the 36 base relations plus the 22 named relations and
renumbering them to the ids `info.dat` already declares (36..57).

`range_2_3` itself is a superset of the variant union (182 new relations vs 6+22),
so it is NOT the union graph the `{model}*` column claims. This script also emits
`union_exact_2_3` = base ∪ exact_2's 6 ∪ exact_3's 22 = 64 relations, which is the
honest universal-graph comparison point. `range_2_3` is left untouched and keeps
serving as the semantic catalog.

Subcommands:
  plan          validate the sources and emit the relation-id map files
  verify-counts compare a built link.dat's per-relation row counts against the
                registries (this is the acceptance test; a mismatch aborts)
  union-info    write info.dat / registry for the union directory
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_link_section(info_dat: Path) -> dict[int, tuple[int, int, str]]:
    """id -> (start_type, end_type, name) from an info.dat link.dat section."""
    text = info_dat.read_text()
    if "link.dat" not in text:
        raise ValueError(f"{info_dat}: no link.dat section")
    section = text.split("link.dat", 1)[1]
    out: dict[int, tuple[int, int, str]] = {}
    for line in section.splitlines():
        parts = [p for p in line.split("\t") if p != ""]
        if len(parts) == 4 and all(parts[i].isdigit() for i in range(3)):
            out[int(parts[0])] = (int(parts[1]), int(parts[2]), parts[3])
    if not out:
        raise ValueError(f"{info_dat}: parsed no relations")
    return out


def load_registry(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cmd_plan(args) -> None:
    root = Path(args.variants_root)
    reg_root = Path(args.registry_root)

    rels = {v: parse_link_section(root / v / "info.dat")
            for v in ("unchanged", "exact_2", "exact_3", "range_2_3")}
    base = rels["unchanged"]
    if len(base) != 36:
        raise SystemExit(f"expected 36 base relations, got {len(base)}")

    # Base relation ids must be identical everywhere, or renumbering is unsafe.
    for v, m in rels.items():
        for rid, spec in base.items():
            if m.get(rid) != spec:
                raise SystemExit(f"base relation {rid} differs in {v}: {m.get(rid)} != {spec}")
    print(f"[ok] 36 base relations identical across all four variants")

    r23_by_name = {spec[2]: rid for rid, spec in rels["range_2_3"].items()}
    if len(r23_by_name) != len(rels["range_2_3"]):
        raise SystemExit("range_2_3 has duplicate relation names")

    e3_reg = load_registry(reg_root / "exact_3.jsonl")
    e2_reg = load_registry(reg_root / "exact_2.jsonl")
    print(f"[ok] registries: exact_2={len(e2_reg)} new, exact_3={len(e3_reg)} new")

    # exact_3's declared ids must agree between info.dat and the registry.
    e3_info_new = {rid: spec for rid, spec in rels["exact_3"].items() if rid not in base}
    if len(e3_info_new) != len(e3_reg):
        raise SystemExit(f"exact_3 info.dat has {len(e3_info_new)} new relations, "
                         f"registry has {len(e3_reg)}")
    for entry in e3_reg:
        rid, name = int(entry["new_relation_id"]), entry["new_relation_name"]
        if e3_info_new.get(rid, (None, None, None))[2] != name:
            raise SystemExit(f"exact_3 id {rid}: info.dat/registry name mismatch")
        if name not in r23_by_name:
            raise SystemExit(f"exact_3 relation {name!r} absent from range_2_3 — not recoverable")
    print(f"[ok] all {len(e3_reg)} exact_3 relations present in range_2_3")

    for entry in e2_reg:
        if entry["new_relation_name"] not in r23_by_name:
            raise SystemExit(f"exact_2 relation {entry['new_relation_name']!r} absent from range_2_3")
    print(f"[ok] all {len(e2_reg)} exact_2 relations present in range_2_3")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Map 1: range_2_3 relation id -> exact_3 relation id (base is identity).
    with open(out / "map_r23_to_exact3.tsv", "w") as fh:
        for rid in sorted(base):
            fh.write(f"{rid}\t{rid}\n")
        for entry in sorted(e3_reg, key=lambda e: int(e["new_relation_id"])):
            fh.write(f"{r23_by_name[entry['new_relation_name']]}\t{entry['new_relation_id']}\n")

    # Map 2: exact_3 id -> union id  (base identity; exact_3's 36..57 -> 42..63)
    # Map 3: exact_2 id -> union id  (base identity; exact_2's 36..41 -> 36..41)
    e2_ids = sorted(int(e["new_relation_id"]) for e in e2_reg)
    e3_ids = sorted(int(e["new_relation_id"]) for e in e3_reg)
    union_of_e2 = {rid: 36 + i for i, rid in enumerate(e2_ids)}
    union_of_e3 = {rid: 36 + len(e2_ids) + i for i, rid in enumerate(e3_ids)}
    with open(out / "map_exact3_to_union.tsv", "w") as fh:
        for rid in sorted(base):
            fh.write(f"{rid}\t{rid}\n")
        for rid, u in sorted(union_of_e3.items()):
            fh.write(f"{rid}\t{u}\n")
    with open(out / "map_exact2_to_union.tsv", "w") as fh:
        # base rows are already supplied by the exact_3 side; take only the new ones
        for rid, u in sorted(union_of_e2.items()):
            fh.write(f"{rid}\t{u}\n")

    # Expected per-relation row counts, used by verify-counts.
    base_counts = {}
    with open(root / "unchanged" / "link.dat") as fh:
        for line in fh:
            p = line.split("\t")
            if len(p) >= 3:
                base_counts[int(p[2])] = base_counts.get(int(p[2]), 0) + 1
    expect_e3 = {str(rid): n for rid, n in base_counts.items()}
    for e in e3_reg:
        expect_e3[str(e["new_relation_id"])] = int(e["num_new_edges"])
    expect_un = {str(rid): n for rid, n in base_counts.items()}
    for e in e2_reg:
        expect_un[str(union_of_e2[int(e["new_relation_id"])])] = int(e["num_new_edges"])
    for e in e3_reg:
        expect_un[str(union_of_e3[int(e["new_relation_id"])])] = int(e["num_new_edges"])
    (out / "expected_counts_exact_3.json").write_text(json.dumps(expect_e3, indent=1, sort_keys=True))
    (out / "expected_counts_union.json").write_text(json.dumps(expect_un, indent=1, sort_keys=True))

    meta = {
        "union_variant_name": args.union_name,
        "union_num_relations": 36 + len(e2_reg) + len(e3_reg),
        "union_id_assignment": {
            "base": "0..35 (identity)",
            "exact_2": {str(k): v for k, v in union_of_e2.items()},
            "exact_3": {str(k): v for k, v in union_of_e3.items()},
        },
        "expected_rows_exact_3": sum(expect_e3.values()),
        "expected_rows_union": sum(expect_un.values()),
    }
    (out / "rebuild_plan.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[ok] exact_3 target: {sum(expect_e3.values()):,} rows over {len(expect_e3)} relations")
    print(f"[ok] union  target: {sum(expect_un.values()):,} rows over {len(expect_un)} relations")
    print(f"[ok] wrote maps + expectations to {out}")


def cmd_verify(args) -> None:
    expected = {int(k): v for k, v in json.loads(Path(args.expected).read_text()).items()}
    actual = {int(k): v for k, v in json.loads(Path(args.actual).read_text()).items()}
    bad = []
    for rid in sorted(set(expected) | set(actual)):
        e, a = expected.get(rid, 0), actual.get(rid, 0)
        if e != a:
            bad.append((rid, e, a))
    if bad:
        print(f"FAIL: {len(bad)} relation(s) with wrong row counts")
        for rid, e, a in bad[:20]:
            print(f"  rel {rid}: expected {e:,} got {a:,}")
        raise SystemExit(2)
    print(f"[ok] all {len(expected)} relations match expected row counts "
          f"({sum(expected.values()):,} rows total)")


def cmd_union_info(args) -> None:
    root = Path(args.variants_root)
    plan = json.loads((Path(args.out_dir) / "rebuild_plan.json").read_text())
    rels_e2 = parse_link_section(root / "exact_2" / "info.dat")
    rels_e3 = parse_link_section(root / "exact_3" / "info.dat")
    base = parse_link_section(root / "unchanged" / "info.dat")

    lines = []
    for rid in sorted(base):
        s, e, n = base[rid]
        lines.append(f"{rid}\t\t{s}\t\t{e}\t{n}")
    reg = []
    for src_rels, mapping, hop, src in ((rels_e2, plan["union_id_assignment"]["exact_2"], 2, "exact_2"),
                                        (rels_e3, plan["union_id_assignment"]["exact_3"], 3, "exact_3")):
        for old, new in sorted(mapping.items(), key=lambda kv: kv[1]):
            s, e, n = src_rels[int(old)]
            lines.append(f"{new}\t\t{s}\t\t{e}\t{n}")
            reg.append({"new_relation_id": int(new), "new_relation_name": n,
                        "start_type_id": s, "end_type_id": e, "hop_count": hop,
                        "source_variant": src, "source_relation_id": int(old)})

    header = (root / "unchanged" / "info.dat").read_text().split("link.dat", 1)[0]
    tail = "\n\nTargeting: Label Type 0\n"
    dest = Path(args.union_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "info.dat").write_text(
        header + "link.dat\n\nLINK\tSTART\tEND\tMEANING\n" + "\n".join(lines) + tail)
    with open(dest / "new_transitive_relations.jsonl", "w") as fh:
        for r in reg:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    (dest / "variant_overview.json").write_text(json.dumps({
        "variant": plan["union_variant_name"],
        "description": "True union of the filtered variants: base ∪ exact_2 ∪ exact_3. "
                       "This is the universal-graph ({model}*) comparison point; "
                       "range_2_3 is a strict superset and is NOT the union.",
        "num_total_relations": plan["union_num_relations"],
        "num_new_relations": plan["union_num_relations"] - 36,
        "num_total_edges": plan["expected_rows_union"],
    }, indent=2, sort_keys=True) + "\n")
    print(f"[ok] wrote {dest}/info.dat ({len(lines)} relations), registry, overview")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants-root",
                    default=str(Path(os.environ.get("INV_GNN_DATA", str(Path(__file__).resolve().parents[2] / "data")))
                                / "dataset_variant_3hops_filtered"))
    ap.add_argument("--registry-root", default="registries")
    ap.add_argument("--out-dir", default="freebase_rebuild")
    ap.add_argument("--union-name", default="union_exact_2_3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    v = sub.add_parser("verify-counts")
    v.add_argument("--expected", required=True)
    v.add_argument("--actual", required=True)
    u = sub.add_parser("union-info")
    u.add_argument("--union-dir", required=True)
    args = ap.parse_args()
    {"plan": cmd_plan, "verify-counts": cmd_verify, "union-info": cmd_union_info}[args.cmd](args)


if __name__ == "__main__":
    main()
