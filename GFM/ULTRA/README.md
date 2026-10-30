# ULTRA

Zero-shot evaluation of [ULTRA](https://github.com/DeepGraphLearning/ULTRA) (Galkin et al., ICLR 2024), a
pretrained knowledge-graph foundation model, on information-equivalent representations of the same knowledge
graph. No training is involved: the released checkpoints are evaluated as they are.

Two experiments:

1. **Sensitivity.** Each checkpoint ranks the same test queries on several representations of the same facts:
   named inverse edges, facts reified through a node, materialized 2- and 3-hop shortcuts. The outputs are
   compared query by query (Kendall τ, ΔMRR).
2. **Compiled view.** Every representation is compiled back to one canonical graph through its declared
   transformations (`../common/compile_back.py`), and the unchanged checkpoint is run on that graph.

## Model

| checkpoint | pretraining graphs | held out here |
| --- | --- | --- |
| `ultra_3g` | FB15k-237, WN18RR, CoDEx-M | Hetionet (documented by the upstream pretraining configs) |
| `ultra_4g` | the 3g graphs + NELL-995 | Hetionet (documented) |
| `ultra_50g` | 50 graphs; the list is not published (the upstream code base supports 57 KGs, including Hetionet) | none claimed |

- Upstream code: commit `427966ad` (2025-02-02), fetched untouched by `setup_upstream.sh`. Checkpoints:
  `upstream/ckpts/`. Every `metrics.json` records the checkpoint's SHA-256.
- Architecture: about 168K parameters, the same for every checkpoint. Relation and entity models are 6-layer
  NBFNets of width 64 (DistMult messages, sum aggregation, shortcut connections, layer norm). The config is read
  from `upstream/config/transductive/inference.yaml` (model block only).
- **What ULTRA reads:** graph structure only. It has no entity or relation embeddings and never reads names;
  relation representations come from a graph of how relations share heads and tails. A transformation is therefore
  anything that changes the edge list; renaming a relation changes nothing.
- The released checkpoints are PyG re-implementations: the released 3g gives WN18RR MRR 0.369 against 0.480 in the
  ULTRA paper. `scripts/run_ref_check.sh` confirms this is the checkpoint's own number.

On WN18RR and FB15k-237 every checkpoint has seen the graph during pretraining, so a change there is "seen graph,
new representation". Hetionet is the zero-shot test for 3g and 4g.

## Datasets and representations

Built by `../scripts/build_data.sh` (see `../README.md`). Every representation is built from the official training
triples only and keeps the base entity and relation ids.

| dataset | variant | what changes | message-graph edges |
| --- | --- | --- | --- |
| WN18RR | `no_changes` | reference: the official training graph | 86,835 |
| WN18RR | `all_inverse_edges` | + a named inverse relation for each of the 11 relations | 173,670 |
| WN18RR | `transitive_edges` | + 526 shortcut relations (2- and 3-hop compositions) | 706,128 |
| WN18RR | `universal_edges` | union graph of all representations | 893,629 |
| FB15k-237 | `original` | reference | 272,115 |
| FB15k-237 | `inverse` | + `(t, r__inverse, h)` for every edge | 544,230 |
| FB15k-237 | `reified_cvt_light` | 10 Freebase mediator relations through a fact node, chosen to remove few queries | 285,718 (+13,603 nodes) |
| FB15k-237 | `reified_cvt_strong` | the 10 most frequent mediator relations through a fact node | 361,586 (+89,471 nodes) |
| FB15k-237 | `shortcut2_c5k` / `c50k` / `c200k` | + 2-hop shortcuts for the top 20 compositions, at most 5K / 50K / 200K per composition | 372,115 / 1,232,897 / 3,474,893 |
| Hetionet | `original` | reference (5,000 keyed-sampled test queries; the other held-out triples filter only) | 2,025,177 |
| Hetionet | `inverse` | + named inverses | 4,050,354 |
| Hetionet | `reified_regulation` | compound / disease up- and down-regulation (CuG, CdG, DuG, DdG) as event nodes | 2,074,842 (+49,665 nodes) |
| Hetionet | `shortcut2_c50k` | + 2-hop shortcuts, top 20 compositions with ≤ 1M paths each | 3,007,160 |

144 of FB15k-237's 237 relations (67.9% of training edges) are Freebase mediator paths collapsed into one relation
name. Reifying them partly restores Freebase's own encoding; one fact node per edge approximates Freebase's
multi-role mediator nodes. A reified relation no longer exists as an edge type, so its queries cannot be posed to
ULTRA on that variant: 26.7% of test queries for `reified_cvt_strong`, 0.3% for `reified_cvt_light`, 2.6% for
`reified_regulation`. Each pair of variants is compared on the queries answerable on both.

## Setup

```bash
pip install -r requirements.txt        # or reuse the GNN environment and add: easydict ninja jinja2
bash setup_upstream.sh                 # ULTRA at commit 427966ad, checkpoints included
```

ULTRA's `rspmm` CUDA kernel is JIT-compiled on first use into `torch_extensions/`. It needs a C++17 compiler
(`export CC=gcc CXX=g++` with gcc ≥ 9) and `nvcc` matching your PyTorch CUDA version. To build once for several
GPU types, set e.g. `TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;9.0"` before the first run.

## Run

```bash
# 0. data (from GFM/): bash scripts/download_raw.sh && bash scripts/build_data.sh
# 1. smoke test: 64 queries
LIMIT=64 bash scripts/run_zero_shot.sh WN18RR ultra_4g
# 2. upstream reference check (prints the test MRR of ULTRA's own script)
bash scripts/run_ref_check.sh ultra_4g
# 3. every (dataset, checkpoint) pair
for ds in WN18RR FB15k237 Hetionet; do for ck in ultra_3g ultra_4g ultra_50g; do
  bash scripts/run_zero_shot.sh $ds $ck
done; done
# 4. compiled views
for ds in WN18RR FB15k237 Hetionet; do bash scripts/run_compiled.sh $ds ultra_4g; done
# 5. table + figure
bash scripts/summarize.sh
```

Each script is plain bash; set `ENV_ACTIVATE=/path/to/env/bin/activate` to activate an environment and wrap it in
`sbatch` / `srun` on a cluster. Finished variants are skipped, so re-running a script resumes it. One evaluation by
hand:

```bash
python ultra_eval.py --dataset-dir ../data/FB15k237 --variant inverse --ckpt upstream/ckpts/ultra_4g.pth \
    --out /tmp/x --limit 64
```

**Resources.** One GPU with ≥ 4 GB (peak 2.4 GB on the largest variant), 4 CPUs, 12 GB RAM. On an RTX 2080 Ti:
WN18RR ~10 min, Hetionet ~30 min and FB15k-237 ~1.5 h per checkpoint; evaluation time grows about linearly with
message-graph edges. Results take a few MB per (dataset, checkpoint): only ranks and the 51 candidate scores per
query are saved, not full score matrices.

## How a variant is evaluated (`ultra_eval.py`)

1. **Graph.** Load the variant's `graph.tsv`, check that its ids extend the base ids, add ULTRA's unnamed inverse
   edges (relation `r + R`), and build ULTRA's relation graph with upstream `build_relation_graph`. On the
   `inverse` variant this gives four relation types per base relation (r, r__inverse and the unnamed inverses of
   both), two of them redundant.
2. **Queries.** Every test query `(h, r, t)` is scored in both directions, `(h, r, ?)` and `(t, r⁻¹, ?)`, in batches
   of 8. The model scores every node of the variant graph; only base entities are kept, so nodes a transformation
   adds are never candidates.
3. **Filtering.** Other known answers are removed with the base filter set (train + valid + test, plus the
   filter-only held-out triples on Hetionet), identical for every variant; derived edges are never answers.
4. **Ranks.** `rank_pess` is ULTRA's own `compute_ranking` (ties count against the answer; comparable with
   published ULTRA numbers). `rank_tie` = 1 + #better + #tied / 2; every MRR and ΔMRR below uses it.
5. **Candidate scores.** The scores of the true answer and the 50 fixed filtered negatives
   (`base/negatives_test.npy`, keyed on the query, the same for every variant) are saved as `cand_scores`;
   Kendall τ is computed on them.
6. **Determinism.** `torch.use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
   `torch.manual_seed(0)`, `PYTHONHASHSEED=0`. The `rspmm` kernel uses atomics only in the backward pass, so
   inference is reproducible.

`../common/compare_variants.py` then reports, for each variant against the reference, on the common queries: MRR and
Hits@k, mean Kendall τ-b over the 51-candidate vectors (queries × 2 directions; identical vectors count as 1), ΔMRR,
the share of Hit@1 / Hit@10 outcomes that flip, and whether the scores are bitwise identical.

## Expected results

`scripts/summarize.sh` writes `results/summary.{csv,md}` and `results/figure.{pdf,png}`. Kendall τ and ΔMRR against
the reference (tie-aware MRR; **held out** = graph not in the checkpoint's documented pretraining set). The repeat
run of the reference gives τ = 1.000 with bitwise-identical scores for every (dataset, checkpoint).

| dataset | variant | τ 3g | τ 4g | τ 50g | ΔMRR 3g | ΔMRR 4g | ΔMRR 50g |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WN18RR | `all_inverse_edges` | 0.506 | 0.581 | 0.388 | −0.064 | −0.050 | −0.092 |
| WN18RR | `transitive_edges` | 0.132 | 0.327 | 0.235 | −0.066 | −0.141 | −0.080 |
| WN18RR | `universal_edges` | 0.060 | 0.334 | 0.225 | +0.002 | −0.096 | −0.051 |
| FB15k-237 | `inverse` | 0.819 | 0.808 | 0.798 | −0.004 | −0.009 | −0.015 |
| FB15k-237 | `reified_cvt_light` | 0.866 | 0.886 | 0.814 | −0.002 | −0.003 | −0.002 |
| FB15k-237 | `reified_cvt_strong` | 0.537 | 0.712 | 0.586 | −0.039 | −0.021 | −0.038 |
| FB15k-237 | `shortcut2_c5k` | 0.800 | 0.785 | 0.810 | −0.013 | −0.014 | −0.010 |
| FB15k-237 | `shortcut2_c50k` | 0.590 | 0.573 | 0.657 | −0.047 | −0.053 | −0.041 |
| FB15k-237 | `shortcut2_c200k` | 0.538 | 0.494 | 0.542 | −0.073 | −0.083 | −0.068 |
| Hetionet (held out for 3g, 4g) | `inverse` | 0.848 | 0.820 | 0.803 | −0.012 | −0.019 | −0.068 |
| Hetionet (held out for 3g, 4g) | `reified_regulation` | 0.916 | 0.887 | 0.886 | −0.003 | +0.001 | −0.007 |
| Hetionet (held out for 3g, 4g) | `shortcut2_c50k` | 0.736 | 0.723 | 0.747 | −0.013 | −0.042 | −0.120 |

Reference MRR (tie-aware, all queries): WN18RR 0.370 / 0.485 / 0.429, FB15k-237 0.369 / 0.368 / 0.367, Hetionet
0.254 / 0.274 / 0.394 for 3g / 4g / 50g. `expected_results.md` has the full per-row table, including query counts
and Hit@1 flip rates.

**Compiled view** (`ultra_4g`). All 15 transformed variants compile to graphs byte-identical to the compiled
reference, and every pair gives τ = 1.000 with bitwise-identical scores. The reified FB15k-237 queries become
answerable again (20,466 instead of 14,995). MRR: WN18RR 0.4854, FB15k-237 0.3684, Hetionet 0.2741.

**Checks.**

| check | expected |
| --- | --- |
| `run_ref_check.sh` (upstream `script/run.py`, WN18RR) vs pessimistic MRR of `run_zero_shot.sh` | 4g: 0.483783 vs 0.4838; 3g: 0.369067 vs 0.3691 |
| FB15k-237, 3g, reference MRR | 0.369 (ULTRA paper: ~0.37) |
| Hetionet, 3g, reference MRR / Hits@10 on the 5,000-query sample | 0.254 / 0.377 (ULTRA paper, Table 10: 0.257 / 0.379) |
| edge order: `original` vs its sorted copy `original__compiled` | scores differ by ≤ 5e-6, τ = 1.000, same MRR |

## Scope and caveats

- Zero-shot only; `../docs/EVALUATION_GUIDE.md` §C describes fine-tuning.
- WN18RR and FB15k-237 were seen in pretraining. Hetionet is held out for 3g and 4g only.
- One test split, one negative set, no confidence intervals. τ is over 51 candidates; MRR uses the full filtered
  ranking over all base entities.
- FB15k-237 reification uses one fact node per edge; Hetionet uses a 5,000-query sample.
- ULTRA is relation-agnostic, so name-only changes are invisible to it. A GFM that reads text needs renaming
  transformations as well.
- The compiled view here is canonicalization at inference by declared transformations. It works with the
  unmodified backbone because ULTRA is relation-agnostic.

## Files

| file | what |
| --- | --- |
| `setup_upstream.sh` | fetches ULTRA at the pinned commit into `upstream/` |
| `ultra_eval.py` | one checkpoint × one variant: filtered ranks and candidate scores (output contract in `../README.md`) |
| `ref_check.py`, `ref_check_inference.yaml` | runs upstream `script/run.py` zero-shot through the same shims |
| `aggregate.py`, `make_figure.py` | `results/summary.{csv,md}`, `results/figure.{pdf,png}` |
| `scripts/` | `run_zero_shot.sh`, `run_compiled.sh`, `run_ref_check.sh`, `summarize.sh` |
| `expected_results.md` | the full summary table of the sensitivity experiment |
