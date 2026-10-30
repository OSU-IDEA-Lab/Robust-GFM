# Evaluating a graph foundation model for representation sensitivity

This guide describes how to test whether a graph foundation model (GFM) gives the same answers when the same facts
are written as different graphs. `ULTRA/README.md` is a complete worked example. The guide has three parts:

- **A.** The general process, step by step, with the reason for each step.
- **B.** Zero-shot evaluation on the transformed graphs of the GNN experiments (IMDb, DBLP, Freebase, WN18RR), which
  are mostly not pretraining graphs.
- **C.** Evaluation with fine-tuning.

Vocabulary:

- A **representation** is one way of writing the facts as a graph (a *variant* in `GNN/`).
- A **transformation** is an information-preserving map between representations (a *switching* in `GNN/docs/THEORY.md`).
- The **reference** is the representation everything else is compared with.
- The **compiled view** is the canonical graph obtained by undoing declared transformations (the *compiled semantic
  program* in `GNN/`).

---

## A. The general process

### A0. Checklist

1. Pin the model: commit, checkpoint hash, config, and **which graphs it was pretrained on** (A1).
2. Reproduce one published number with the model's *own* code before writing anything (A2).
3. Work out what the model actually reads, and therefore which transformations are visible to it (A3).
4. Build representations in the shared-id format, from training data only, with the declaration recorded (A4).
5. Fix the comparison sets: the same queries or labelled nodes, the same filtered candidates, the same negatives,
   keyed (A5).
6. Run every representation, plus a repeat of the reference, and save per-query scores, not just metrics (A6).
7. Compare: accuracy deltas, Kendall τ, flip rates, paired bootstrap CIs (A7).
8. Run the controls: noise floor, upstream match, edge order, compile-back (A8).
9. Report with held-out status and caveats (A9).

### A1. Pin the model and its pretraining set

- **Record the identity.** Note the git commit, checkpoint file and SHA-256, and the exact config used. Write the
  checkpoint hash into every `metrics.json` (`ULTRA/ultra_eval.py` does).
- **Find the pretraining graphs from primary sources.** Check, in this order, the training configs in the model's
  repository, the appendix tables of its paper, then the model card.
  - Write down which evaluation graphs are **seen** (pretrained on), **held out (documented)**, or **unknown**.
  - For ULTRA, the 3g and 4g pretraining sets are documented by configs, while the 50g list is not published. That
    is why Hetionet counts as held out for 3g and 4g but not for 50g.
  - Do not infer "held out" from a model's name.
- **Released weights may differ from the published model.** ULTRA's released 3g checkpoint is a re-implementation:
  WN18RR gives 0.369 where the ULTRA paper reports 0.480. Always evaluate the checkpoint you will cite.

### A2. Reproduce upstream first

**Before writing your own harness,** run the model's own evaluation script zero-shot on its own copy of one
dataset and record the number (ULTRA: `ULTRA/scripts/run_ref_check.sh`, which calls upstream
`script/run.py --epochs 0`).

**Then make your harness reproduce that number** on the reference representation, to 3–4 decimals, using the
model's rank convention. If the two disagree, find out why before going further: the usual causes are filtering,
the inverse-edge convention, tie handling, or the split files. A sensitivity result from a harness that does not
reproduce upstream is not meaningful.

Practical tips:

- Keep the upstream checkout untouched. Apply compatibility shims from outside it (`ULTRA/ultra_eval.py` patches
  the PyG version string at import time).
- Install packages the shared environment lacks with `pip install --target <GFM>/pydeps` rather than modifying the
  environment.
- Custom CUDA kernels (ULTRA's `rspmm`) need a matching compiler. Build them once for every GPU architecture you
  may run on (`TORCH_CUDA_ARCH_LIST`).

### A3. Work out what the model reads

A transformation only matters if it changes something the model consumes. Classify the GFM before choosing
transformations:

| what the GFM reads | examples (verify current versions) | transformations it can see | transformations it cannot see |
| --- | --- | --- | --- |
| structure only, relation-agnostic (relation representations computed from how relations co-occur) | ULTRA and other "double equivariant" KG GFMs | adding or removing edges, inverses, reification, shortcuts, splitting or merging relations | renaming relations or entities, permuting ids |
| structure + node features | node-classification GFMs that work from features and labels in context (e.g. GraphAny) | relocating attributes between node types, adding link nodes, changing which nodes carry features | renaming |
| text (node and relation descriptions rendered as text) | text-attributed or LLM-based GFMs (e.g. OFA-style, LLM graph prompting) | all of the above **plus** renaming, rewording relation descriptions, changing schema vocabulary | none, in principle |
| typed schema / metapaths | heterogeneous GNNs used as pretrained backbones | anything that changes node or edge types or metapath sets | ids |

Two consequences:

- **Name-only changes need their own tests.** For a text-reading GFM, add renaming and paraphrasing
  transformations. For a relation-agnostic GFM, a renaming test is a free sanity check: τ should be 1.000.
- **Know how the model handles inverses.** ULTRA adds unnamed inverse edges internally, so a "named inverse"
  representation is a pure restatement for it. If a model does *not* add inverses, adding them changes what is
  reachable in k hops. That is still a fair transformation, but say so.

### A4. Build representations in the shared-id format

Use `common/kg_format.py`. Its contract is what makes per-query comparison valid:

- **One `base/`** holds the vocabularies, splits, filter set and fixed negatives. **One `variants/<V>/`** per
  representation holds that representation's graph.
- **Shared ids.** Every variant keeps the base entity and relation ids as a prefix and appends anything new (fact
  nodes, link nodes, new relations). `check_prefix` enforces this.
- **Training edges only.** Build every variant from training edges only. Validation and test edges must never enter
  a message graph, directly or through a derived edge (a shortcut or inverse of a test edge).
- **Keyed randomness.** Any random choice (sampled queries, capped shortcuts, negatives) must be keyed on the item's
  identity (`key_seed`), not on its position in a file. Then adding a variant or reordering a file changes nothing
  else.
- **Record the declaration** in a form a program can read back. The builders encode it in relation names
  (`r__inverse`, `r__subject` / `r__object`, `a>>b`, `__transitive__a|||b`), which is what makes compile-back
  possible. A JSON list of transformations in the manifest would work equally well.
- **Grade the strength of a transformation.** Include at least one mild and one strong version of the same
  transformation (`reified_cvt_light` vs `reified_cvt_strong`; shortcut caps 5K / 50K / 200K). A dose-response is
  much more convincing than a single point.
- **Write a manifest per variant** with counts, hashes, the number of queries made unanswerable, and leakage
  statistics (`write_variant` does this).

**Choosing transformations.** Prefer transformations that real data actually varies by:

- Freebase mediator (CVT) nodes versus collapsed relations.
- Link nodes versus direct edges (IMDb).
- Event reification in biomedical KGs.
- Materialized inferred edges, as reasoners and ETL jobs produce.

Purely synthetic perturbations are a weaker test.

### A5. Fix the comparison sets

- **Queries.** The same test queries for every representation. For node classification, the same labelled nodes
  and the same class set.
- **Candidates.** Rank only **base entities**. Nodes added by a transformation (fact nodes, link nodes) are never
  candidates.
- **Filter set.** Filter against the **base** truth set (train + valid + test, plus any held-out triples left out of
  a query sample). Never filter against the variant graph: derived edges are not answers.
- **Fixed negatives.** Store K filtered negatives per query and direction (`base/negatives_<split>.npy`, K = 50).
  Kendall τ is computed on these candidate vectors; MRR uses the full filtered ranking.
- **Unanswerable queries.** If a transformation removes the query relation (reification), the query cannot be posed
  natively. List the relation in the manifest (`unanswerable_query_relations`), skip its queries, and compare each
  pair on the queries answerable on **both** representations. Report the share lost.

### A6. Run

- **Determinism.** Use `torch.use_deterministic_algorithms(True, warn_only=True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed seeds and `PYTHONHASHSEED=0`.
- **Noise-floor repeat.** Evaluate the reference twice, in separate processes, in the same job
  (`<ref>__repeat`). Inference should give τ = 1.000. If it does not, the model has nondeterministic inference
  kernels or the GPU is faulty, and every comparison must be judged against the measured floor rather than 1.
- **Save enough to recompute everything:** per-query ranks under both tie conventions and the K + 1 candidate
  scores (the output contract in `GFM/README.md`). Full score matrices are gigabytes and unnecessary.
- **Smoke test first.** Run `LIMIT=64` queries before the full run, and record peak GPU memory and seconds per
  query to size the full run.

### A7. Compare

`common/compare_variants.py` computes all of these except the bootstrap and the per-relation breakdown.

| metric | what it answers | notes |
| --- | --- | --- |
| ΔMRR, ΔHits@k (tie-aware) | did accuracy change? | variant − reference, on the common subset |
| Kendall τ-b over the K + 1 candidate scores, mean over queries × directions | did the ranking change? | identical vectors count as 1; NaN as 0 |
| share of Hit@1 / Hit@10 outcomes that flip | how many answers changed? | easy to read for non-specialists |
| bitwise identical | is it literally the same output? | expected only for the repeat and the compiled view |
| paired bootstrap CI on ΔMRR (resample queries) | is the change real? | cheap to add |
| per-relation breakdown | where does it break? | e.g. Hetionet Compound–treats–Disease |

For node classification use instead:

- Δaccuracy and Δmacro-F1.
- The share of nodes whose predicted class changes.
- Kendall τ (or total-variation distance) between the class-probability vectors of each node.

### A8. Controls

1. **Noise floor:** the repeat run (A6).
2. **Upstream match:** A2.
3. **Edge order:** shuffle or sort the reference edge list. The effect should be at float-rounding level (ULTRA:
   ≤ 5e-6). Otherwise "representation effects" could be ordering effects.
4. **Id permutation or renaming:** relabel entity ids (or relation names, for a relation-agnostic model). Scores
   should be the same up to rounding.
5. **Compile-back:** undo every declared transformation (`common/compile_back.py`, which verifies each removed
   edge). Every variant should then give byte-identical graphs and τ = 1.000. This shows that the sensitivity is
   removable without retraining, and it checks that the variants really were information-preserving. A variant that
   cannot be compiled back was either not information-preserving or is incompletely declared.

### A9. Report

For every number, state:

- the checkpoint;
- held-out status (seen / documented held out / unknown);
- the queries used and the share lost to unanswerability;
- the rank convention;
- the noise floor.

Quote τ together with ΔMRR. τ alone can look alarming while accuracy barely moves (FB15k-237 `inverse`: τ 0.81,
ΔMRR −0.004).

---

## B. Zero-shot on the GNN datasets

The GNN folder has transformed graphs with full provenance (`GNN/docs/DATASET_STATISTICS.md`,
`GNN/docs/DATASET_CONSTRUCTION_FREEBASE_WORDNET.md`). Most are not GFM pretraining graphs, so they are natural
zero-shot tests.

| dataset | GNN task(s) | representations | size | GFM fit |
| --- | --- | --- | --- | --- |
| WN18RR | LP | `no_changes`, `all_inverse_edges`, `transitive_edges`, universal | 87K–894K edges | in `ULTRA/`; **seen** by ULTRA |
| IMDb (year graph) | NC (3 genres); LP Movie–Year; LP Movie–Director | v1–v4 (director/actor on the movie vs on its Link node), universal | 15,789 nodes, ~50K directed edges per variant | LP: KG GFMs. NC: feature / in-context GFMs |
| DBLP | LP Paper–Conference | v1–v3, universal | 26,076 nodes, 268K–287K edges | KG GFMs (20-way ranking) |
| Freebase (books) | NC (8 genres) | `unchanged`, `exact_2`, `exact_3`, union | 180K nodes; 1.06M / 47.4M / 447M edges | NC GFMs; `exact_3` needs a large GPU allocation |

### B1. Converting a heterogeneous graph to the shared-id format

- **Node vocabulary.** Prefix node ids with their type (`movie:123`, `director:45`) so the vocabulary is unique.
  Keep the **same base entity list in every variant**: movie, director, actor and year nodes. Link nodes are base
  entities on IMDb, because v1 already has them; check the variant tables before deciding what counts as "added".
- **Relations.** Use the GNN relation names. A relation that moves (movie–director vs link–director) is a
  *different relation* in the variant; that is exactly the transformation being tested.
- **Edge direction.** Write each undirected pair once, as a forward triple, and let the GFM add inverses if it does
  (ULTRA does). The GNN bundles store both directions; feeding both directions *and* letting ULTRA add inverses
  silently creates a "named inverse" variant.
- **Write a builder like `common/build_wn18rr_variants.py`.** Export from the RGCN bundles under
  `GNN/data/preprocessed/` read-only, and assert that each variant's facts compile to the semantic-program hash of
  the GNN experiments (IMDb NC `942317043f15`, DBLP `308cec36b62b`). This proves the variants are the same ones the
  GNNs were evaluated on.

### B2. Posing the GNN tasks to a link-prediction GFM

- **LP tasks map directly.**
  - Movie–Year becomes queries `(movie, has_year, ?)`.
  - Paper–Conference becomes `(paper, published_in, ?)`.
  - Movie–Director becomes `(movie, directed_by, ?)`.
  - The held-out target edges must be absent from every variant graph. The GNN bundles already exclude them; keep
    it that way, including through link nodes (no `link → year` edge for a test movie).
- **Reuse the GNN candidate sets.** IMDb LP uses 19 fixed negatives; DBLP ranks all 20 conferences. Store them as
  `negatives_test.npy`. Then:
  - zero-shot numbers are on the same footing as the trained GNNs (original / universal / augmentation /
    invariant);
  - candidates are **type-correct** (only years for Movie–Year). A KG GFM ranking all entities is partly measuring
    whether it can tell a year from an actor, which is not the question.
  - Report the GNN-candidate metric first and full-entity ranking as a secondary number.
- **A query relation that moves is unanswerable natively.** On IMDb v2/v4 the director hangs off the Link node, so
  `(movie, directed_by, ?)` is not an edge type there, as with reification. Either pose it on the compiled view
  (B4) or count it as unanswerable. Movie–Year does not have this problem: the year stays on the movie in every
  variant.

### B3. Node classification

- **KG GFMs like ULTRA do link prediction.** Node classification can be posed as link prediction ("label as an
  entity"): add class nodes and `(movie, has_genre, genre_k)` edges for the training movies, then rank genres for
  the test movies.
  - This is a reformulation; say so explicitly. It is not the same task as the GNN node classification.
  - It drops IMDb's bag-of-words features, which carry most of the signal. Expect low absolute accuracy and focus
    on **variant-to-variant differences**.
- **Feature-based NC GFMs** consume the features plus a few labelled nodes in context. For them:
  - use the canonical feature table (`GNN/common/imdb_canonical_features.py`, identical in every variant), so only
    structure changes between variants;
  - use the GNN splits (2,926 / 418 / 836 on IMDb; 4,772 / 1,591 / 1,591 on Freebase) as the labelled-context /
    validation / test sets.

### B4. Zero-shot analogues of the GNN arms

| GNN arm | zero-shot analogue | what it tells you |
| --- | --- | --- |
| original | the GFM on each physical variant | sensitivity: τ and ΔMRR across variants |
| universal | the GFM on the union graph | does "give it everything" help or hurt? On WN18RR the union graph gave the lowest τ (0.06–0.33) |
| invariant | the GFM on the compiled view | identical input for every variant, so τ = 1 by construction; the interesting number is accuracy compared with the typical variant |
| augmentation | — | needs training; see C3 |

The compiled views of these datasets are the RGCN bundles' semantic programs: export them directly. For Freebase,
check the program's edge count before exporting.

### B5. Size and cost

- **Cost scales with edges.** Full-graph message passing as in ULTRA costs roughly edges × layers × width per
  query batch; time grew about linearly with edges:
  - Hetionet, 2M edges: about 0.03 s per query-direction on an RTX 2080 Ti;
  - FB15k-237 `shortcut2_c200k`, 3.5M edges: about 0.03 s.
- **IMDb, DBLP, WN18RR:** minutes on any GPU.
- **Freebase `unchanged` (1M edges) and `exact_2` (47M):** feasible on a 32 GB GPU. Time 16 queries first.
- **Freebase `exact_3` (447M edges) and the union (493M):**
  - roughly 200× Hetionet's time per query;
  - several GB for the edge indices alone, before activations;
  - sample queries (keyed, as for Hetionet) or use a large-memory GPU allocation;
  - the compiled view is far smaller than the 447M-edge physical graph, so this is also where compiling saves
    the most compute.

### B6. Contamination and "held out"

- **ULTRA has no entity embeddings,** so "seen" means its weights were tuned on that graph's structure, not that it
  memorized facts. Contamination is weaker than for text models, but report it.
- **Freebase overlap.** The Freebase-books graph shares an origin with FB15k-237, but the relations and the task
  differ. Call it "related to a pretraining source", not held out, unless relation overlap has been checked.
- **IMDb and DBLP** are not in ULTRA's documented 3g / 4g pretraining sets. For 50g and other GFMs, check the
  configs (A1).
- **Text GFMs.** Contamination is real: IMDb and DBLP entity names are on the web. Use renamed or anonymized
  entities as a control.

### B7. Other tips

- **Noisy and partial declarations.** Corrupt a fraction ε of a declaration and ask how much compile-back still
  helps. Its verification step (refusing unverifiable edges) then becomes a measurement: the share of edges that
  verify at each ε.
- **Hub filters for shortcuts.** On new graphs skip compositions through huge hubs (`--shortcut-max-paths`).
  Otherwise one composition can add tens of millions of edges and dominate the variant.
- **Per-relation and per-degree breakdowns.** Sensitivity usually concentrates on a few relations and on
  low-degree entities.

---

## C. Evaluation with fine-tuning

Zero-shot asks whether the pretrained model is invariant. Fine-tuning asks a different question: **does adapting to
a representation remove the sensitivity, or bake it in?** This is also where the GNN arms map most directly. ULTRA
ships fine-tuning settings in `upstream/script/run_many.py` (`default_finetuning_config`):

- FB15k-237 and WN18RR: 1 epoch over the full loader.
- Hetionet: 1 epoch × 4,000 batches.

Run them through `script/run.py --epochs E --bpe B --ckpt <ckpt>`, or add a training loop to the evaluation harness.

### C1. Regimes

| regime | train on | test on | answers | GNN arm |
| --- | --- | --- | --- | --- |
| C-a per-variant fine-tune | variant V | variant V | best accuracy per representation; do variants still disagree after adaptation? | original |
| C-b cross-representation transfer | variant V | variant W ≠ V | does fine-tuning on one encoding hurt another? (train on one ETL output, serve another) | — |
| C-c augmentation | a mix of all variants | each variant | does seeing all encodings buy invariance? | augmentation |
| C-d universal | union graph | union graph | cost reference | universal |
| C-e compiled | compiled view of V | compiled view of any variant | invariance by construction | invariant |

**C-e is the strongest check.** Every variant compiles to the same bytes, so with the same seed and data order the
fine-tuned models should be identical (identical structural input and identical parameters give identical models,
`GNN/docs/THEORY.md`). In practice see C4.

### C2. Keep the objective fixed; vary only the message graph

This is the most important design rule, and the easiest to get wrong.

- **Training targets** are the **base training triples**, the same set for every variant. Inverse, shortcut and
  fact-node edges are message-passing context, never targets. Otherwise a variant changes both the graph and the
  loss, and differences cannot be attributed to representation.
- **Target-edge removal must cover derived edges.** ULTRA's training loop removes the target edge `(h, r, t)` and
  its internal inverse from the graph for each training query (`remove_easy_edges` in `ultra/models.py`). It does
  **not** know about:
  - the named inverse `(t, r__inverse, h)`;
  - shortcut edges whose path goes through `(h, r, t)`;
  - the fact node `h → n → t` that reifies it.

  These edges leak the answer during training: variants with derived edges then train on an easier task, or learn
  to rely on leaks that are absent at test time. **Fix:** extend the removal to every edge derived from the target,
  using the declaration to precompute a map `base edge → derived edge ids`, or train with `remove_one_hop` (drop all
  edges between h and t), which covers inverses and direct shortcuts but not fact nodes.
- **Equal budgets in steps, not epochs.** Variants have different edge counts, so an epoch costs different time. Fix
  the number of optimizer steps, the batch size and the data order (keyed), and report wall-clock separately.
- **Model selection on base validation queries** answerable on every variant, with the same checkpoint-selection
  rule for every arm.
- **Unanswerable relations under reification.** Keep the base targets, and exclude the reified relation's queries
  from both training and test on that variant (as in zero-shot).

### C3. Augmentation (C-c)

- Sample batches across variants by a keyed rule, with equal steps per variant, and remove each base triple's
  derived edges *in the variant it is drawn from* (C2).
- Test on each variant separately and report both the mean and the spread across variants. Augmentation reduces the
  spread but does not guarantee invariance; the question is how much spread remains.

### C4. Noise floor under training

- **Training is not bitwise reproducible.** ULTRA's `rspmm` backward pass uses atomics, so two fine-tunes with the
  same seed differ slightly. The zero-shot floor of τ = 1.000 does not carry over.
- **Measure the training floor:** fine-tune the reference twice with the same seed, and with ≥ 3 different seeds,
  and report variant-to-variant τ and ΔMRR against this seed-to-seed spread.
- **Report a ratio:** representation effect / seed effect. A ratio well above 1 means representation matters more
  than randomness.
- **C-e (compiled) fine-tunes** should differ only at the seed-floor level. If they differ more, something in the
  pipeline depends on the representation (data order, edge order, or a variant-specific vocabulary size leaking into
  initialization).
- **Hardware.** Run all arms on one GPU type, and repeat a run on a second node before trusting a bitwise claim:
  some GPUs give run-to-run differences even with deterministic settings.

### C5. Cost

- **ULTRA is small** (about 168K parameters), so full fine-tuning is cheap; the per-step cost is dominated by
  message passing over the graph.
- **Rough sizing.** A training step costs about 3× a zero-shot query batch (forward + backward), about 0.07 s per
  batch of 8 on FB15k-237 `original` (RTX 2080 Ti). ULTRA's default FB15k-237 fine-tune is one epoch, about 34K
  batches of 8 or roughly 2 h; expect about 3× that on the 3.5M-edge shortcut variant. Arms × variants × 3 seeds on
  FB15k-237 therefore come to a few hundred GPU-hours. Time one fine-tune before committing.
- **Freebase `exact_3`** at full graph size needs a large multi-GPU allocation. The compiled-view arm (C-e) is the
  cheap one.
- **Larger, text-reading GFMs.** Parameter-efficient fine-tuning (heads only, or adapters) is reasonable; use the
  same method for every arm.

### C6. What to report

- Accuracy per arm and per variant, as mean ± sd over seeds.
- Cross-variant τ and ΔMRR per arm, next to the seed-to-seed floor.
- The transfer matrix (train V → test W) for C-b.
- Steps, wall-clock and peak memory per arm.
- The leak check: the number of derived edges removed per training batch, as evidence that C2 was applied.
