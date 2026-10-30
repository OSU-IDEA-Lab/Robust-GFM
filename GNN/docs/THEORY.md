# Theory of Information-Based Invariance, and How It Maps to Each Architecture

Companion to `docs/INVARIANT_MECHANISMS.md` (per-model implementation details). This page
summarises the theory and explains how architecture-level invariance is implemented for
RGCN, MAGNN, SlotGAT, SeHGNN, CMPNN, PathNet and PAIN (code in `models/`).

## 1. The core theory

### 1.1 Information equivalence (beyond isomorphism)
Two labeled directed graphs are **τ-equivalent** if a bijective mapping τ (not necessarily
a permutation) maps one to the other. τ is represented by a set of **switchings**
τ = {(Pᵢ, Pᵢ′)}: pattern-graph pairs; G maps to H by replacing every occurrence
(homomorphism image) of Pᵢ with Pᵢ′. Example: (year(u), paper(v)) and (paper(v), venue(w))
maps to (year(u), venue(w)) and (paper(v), venue(w)) — same information, non-isomorphic
topology. Edge-direction reversal is the simplest switching. Switchings are *minimal*
(no two Pᵢ/Pⱼ isomorphic). τ-equivalence is an equivalence relation; permutations Πⁿ plus
compositions of switchings form a group (Prop. 2), so it suffices to reason about single
switchings.

### 1.2 τ-invariance of a GNN
f is **τ-invariant** if f(τ(A)) = f(A) for all inputs; τ-equivariant if output permutes
with input. Standard MPNNs are permutation-invariant but NOT τ-invariant (Props. 3, 7, 8):
τ changes neighborhoods, so AGGREGATE sees different multisets on G vs τ(G).

### 1.3 Structural bias abstraction (Eq. 2)
A GNN's **structural bias** S^f = set of rooted pattern graphs Sᵘ. Each layer:
`h_v^(l+1) = UPDATE(h_v^(l), TAGG({{PAGG(hom(Sᵘ, Gᵛ), Sᵘ)}}))` — PAGG aggregates over all
occurrences of pattern Sᵘ rooted at v, TAGG aggregates across patterns. This unifies:
1-hop MPNN (S = single edge), K-hop, path-based, metapath-based (MAGNN/SeHGNN),
homomorphism-based GNNs.

**Invariance criterion (Lemma 1):** f is τ-invariant iff there is an isomorphism-preserving
mapping b_τ : S^f → S^f such that PAGG gets equal inputs on G and τ(G):
`PAGG(hom(Sᵘ,Gᵛ), Sᵘ) = PAGG(hom(b_τ(Sᵘ), τ(Gᵛ)), b_τ(Sᵘ))`. For ANY fixed structural
bias there exists a τ breaking this (Prop. 4) — hence the fix must *extend* the bias.

### 1.4 Conditional graphs (CGs): the fix
Generalize each pattern Sᵘ into a **conditional graph**: edges partitioned into
**context edges** (E_C) and **propagation edges** (E_P). Semantics: for every occurrence
of the *context subgraph* in Gᵛ, the model propagates information along the *propagation
subgraph* (whose nodes ⊆ context nodes). Occurrences of context generate propagation
homomorphisms: `pom(Sᵘ, Gᵛ)` replaces `hom` as PAGG's input. Context = "check this pattern
exists"; propagation = "actually pass messages between these endpoints".

### 1.5 The algorithm (Sec. 4.3 of the paper) — minimally extend S^f given τ
For each switching (Pᵢ, Pᵢ′) and each Sᵘ ∈ S^f:
1. **Map bias through switching**: if Pᵢ iso-maps into Sᵘ, apply switching to Sᵘ to get
   Rᵢᵘ; add both (all edges context+propagation).
2. **Switchings as context**: else use Pᵢ (per node) as context subgraph with Sᵘ's
   propagation marked inside it → set **R**ᵢ^f.
3. **Map propagation patterns**: for each Rᵘ with context Pᵢᵘ, add CG with context Pᵢ′ᵘ
   and propagation edges (y,z) iff (pᵢ(y),pᵢ(z)) was a propagation edge in Rᵘ (and the
   symmetric direction via pᵢ⁻¹).
Result S_τ^f: no isomorphic CGs → τ-invariant (Thm 1), *minimal* (Prop. 5), *complete*
(learns rep for every node, Prop. 9), and at least as expressive as f (Thm 2) — sometimes
strictly more (the δ edge-reversal example). Cost: quadratic in |S^f|·|switchings|; the
patterns are tiny (paths), so cheap.

### 1.6 What this means operationally
- The input graph is never altered. Preprocessing may *compile occurrences* of context
  subgraphs in each physical variant (this is per-variant and cheap-ish), producing a
  **shared semantic message program**: the multiset of (src, dst, semantic-relation)
  propagation messages must be *identical across variants* (verifiable by hashing).
- If the semantic program + supervision + initialization + reduction order are identical,
  training trajectories are bit-identical across variants → Kendall-τ = 1 exactly, not
  approximately. This is the validation suite the RGCN code implements (identical
  semantic hashes, identical logits/loss/gradients/one Adam step, exact resume).
- **Skip nodes**: when a metapath/pattern in variant A passes through an intermediate node
  that in variant B doesn't exist (e.g. Link node), that node is *context-only* — its
  existence licenses the message, but its features are excluded from the message content.
  Notation for metapaths: parenthesized nodes are skip nodes, e.g. `ML(M)D(M)LM`.
- **Loss propagation**: gradients flow through whatever the propagation edges touch; if a
  piece of information is represented by several virtual/physical realizations, all of
  them must receive the gradient of the shared semantic message (in practice: project
  per-variant occurrences onto shared semantic edges *inside* the model so autograd
  handles it; don't precompute messages outside the graph of computation).

### 1.7 Degree-of-invariance metric
Pairwise **Kendall-τ** over output-score rankings across variants (per test instance,
averaged over variant pairs, instances, runs); Kendall-τ@K for top-K in LP. Target
for real invariance: 1.000 ± 0.000 *with per-variant independently trained models*. (The
`universal` arm, {model}*, gets τ=1 trivially by training once on the union graph; the
`invariant` arm reaches it with a model trained separately on each variant.)

## 2. Architecture-by-architecture mapping

### 2.1 RGCN — local node-type-pair message passing
- **Structural bias**: one rooted edge per relation type (relation = node-type pair or
  named KG relation). PAGG = per-relation mean of W_r h_u; TAGG = sum over relations.
- **Why not invariant**: a switching rewires 1-hop neighborhoods (e.g. venue no longer
  1 hop from paper; inverse edges double neighborhoods; Freebase exact-k hops change
  which pairs are adjacent).
- **Invariant instantiation** (`models/RGCN`): a **conditional
  compiler**. A catalog of dependencies q = r₁;r₂;r₃ (relation compositions, mined from
  the MAGNN universal selections) defines *semantic relations*. For each physical variant
  independently, preprocessing reconstructs endpoint sets B_q(h,G) = {t : h→x→y→t} — using
  virtual transposes in `no_changes`, physical inverses in `all_inverse_edges`, verified
  shortcuts in `universal_edges` — collapsing distinct interior witnesses (set semantics)
  and applying a shared deterministic cap. Every variant thus compiles to the SAME
  (h, semantic_rel, t) message list, consumed by a standard basis-decomposed RGCN with a
  fixed deterministic reduction order (segment sums, not atomics). IMDb version: context
  occurrences projected to shared semantic propagation edges
  (`preprocess_IMDB_rgcn_structural.py`, `run_rgcn_structural.py`); skip nodes are
  context-only. Freebase: block-conditional variant (`freebase_conditional_rgcn.py`).
- **Invariance argument**: identical semantic tensors + identical supervision + identical
  init/ordering ⇒ identical training ⇒ τ=1 exactly. Matches Lemma 1: PAGG input equalized
  by compiling pom(Sᵘ,·) per variant.

### 2.2 MAGNN — metapath-instance message passing (the reference implementation)
- **Structural bias**: set of rooted symmetric metapaths (label sequences). Instance
  encoder (RotatE0) ingests ALL nodes on each instance; intra-metapath attention over
  instances; inter-metapath attention.
- **Why not invariant**: variants change which label sequences realize a semantic
  relation (M–D direct in IMDb1 vs M–L–D through Link in IMDb2), so metapath instance
  sets differ.
- **Invariant instantiation** (`models/MAGNN`, `INVARIANT_MECHANISMS.md` §3): per-variant metapath sets
  with **skip nodes** `(X)`: the metapath is matched in the physical
  variant *including* the intermediate nodes, but skip nodes are excluded from the
  instance encoder's feature sequence. So variant A's `MDM` and variant B's `M(L)D(L)M`
  yield the same encoded instance (endpoints + non-skip intermediates identical) — same
  PAGG input across variants. Union vs Intersect = high-recall vs minimal invariant
  metapath sets (both τ=1; Intersect cheaper, Union sometimes more accurate).
- This is the template: *pattern matched on physical graph = context; features used =
  propagation.* Skip nodes are exactly context-minus-propagation nodes from §1.4.
- Scope: no WordNet. On a single-node-type KG the type-sequence metapaths degenerate to relation paths
  and the two-type LP decoder to a self-similarity ranking (Hits@1 ≈ 0).

### 2.3 SlotGAT — node-type-aware (slot) attention
- **Structural bias**: 1-hop edges, but each node keeps per-type feature *slots*;
  neighbor features land in the slot of their type; attention combines slots.
- **Why not invariant**: 1-hop neighborhoods differ per variant; extra node types (Link,
  Area) add/remove slots and re-route features.
- **Invariant instantiation** (`models/SlotGAT`, `INVARIANT_MECHANISMS.md` §5): conditional 2/3-hop context patterns compiled
  per variant into shared semantic edges typed by (source-type slot, semantic relation);
  skip-node types contribute no features (their slot is context-only). The attention must
  see the same (slot, message) multiset in every variant.
- Scope: IMDb NC + Freebase NC; link prediction (IMDb MY/MD, DBLP, WN18RR) with the deterministic
  parity SlotGAT on the RGCN bundles, τ = 1 everywhere (`INVARIANT_MECHANISMS.md` §5a).

### 2.4 SeHGNN — metapath feature propagation (precomputed) + transformer fusion
- **Structural bias**: for each metapath P, a *feature channel* = normalized adjacency
  product A_{P₁}·…·A_{P_k} applied to raw features (one-shot propagation before training);
  transformer fuses channels.
- **Why not invariant**: the set of channels and each channel's operator depend on the
  physical layout (hop counts change; Link/Area nodes insert extra hops with their own
  features).
- **Invariant instantiation** (`models/SeHGNN`, `INVARIANT_MECHANISMS.md` §6): compute each *semantic* channel per variant by
  composing the physical adjacencies along the variant's realization of the semantic
  metapath, **skipping feature injection at skip nodes** (use their adjacency for
  connectivity/normalization but not their features). If per-variant semantic channel
  matrices are numerically identical, the downstream training is identical → τ=1. Stock
  full-K SeHGNN on the union graph (all channels up to length K) is the universal proxy; the
  invariant version replaces "run on union graph" with "compile per variant, prove equality".
- Scope: IMDb NC + Freebase NC.

### 2.5 CMPNN — source/relation-conditioned path propagation (Bellman-Ford style, torchdrug)
- **Structural bias**: for query (h, r, ?), rooted relational paths from h; per-layer
  relation-conditioned messages (like NBFNet/C-MPNN).
- **Why not invariant**: path sets from h differ per variant (inverse edges create new
  paths; shortcut/transitive edges shorten paths; hop counts change path lengths).
- **Invariant instantiation** (`models/CMPNN`, `INVARIANT_MECHANISMS.md` §4): semantic relation program again — per variant,
  compile the composed-relation reachability so each layer's conditioned message multiset
  matches across variants; inverse traversal virtualized in `no_changes`, verified
  physical in `all_inverse_edges`, shortcut-audited in `universal_edges` (same recipe as
  RGCN WordNet). Because CMPNN is query-conditioned, the compilation must hold for the
  *query-rooted* subgraphs, not just globally.
- Scope: LP only (DBLP, IMDb, WordNet). Separate conda env (torchdrug).
- WN18RR: the invariant program is the 11 base relations only — the `path ↔ shortcut` and
  `r ↔ r⁻¹` switchings are inside a path model's `S^f` once inverse traversal is virtual, so shortcuts and
  inverses are context (audited equal to the chain / transpose), never propagation edges. Shortcuts mined
  from the training graph would otherwise leak every training triple through its derived copies
  (train/held-out signature 96.9 % vs 35.3 %). The base-only program is byte-identical to the `no_changes`
  graph and reproduces that original bitwise (.4238, τ = 1).

### 2.6 PathNet — MERW walk aggregation
- **Structural bias**: per target, maximal-entropy random walks with distance-bucketed nodes;
  LSTM per walk, gated mean across walks (`PathNet_homo`), concatenation with the ego representation.
- **Why not invariant**: a switching changes the adjacency the walks are sampled from.
- **Invariant instantiation** (`models/PathNet`, `INVARIANT_MECHANISMS.md` §6a): the unchanged sampler
  runs on each variant's compiled semantic program (the RGCN program), which is hash-identical
  across variants; with the RNG keyed on (seed, epoch) the per-epoch walks and training are identical → τ=1.
- Scope: IMDb NC + Freebase NC.

### 2.7 PAIN — rooted simple-path aggregation
- **Structural bias**: every rooted simple path with up to L = 3 edges (the path-based row of Eq. 2);
  each path is read by an LSTM and the path encodings are summed at the root.
- **Why not invariant**: a switching changes which simple paths exist (a director on the Link copy is
  one edge further from the movie), so the multiset of rooted paths changes.
- **Invariant instantiation** (`models/PAIN`, `INVARIANT_MECHANISMS.md` §6b): the same path enumeration
  (exhaustive on IMDb, the semantically keyed sampler on Freebase) runs on each variant's compiled
  program (Freebase: the family program). The path programs are byte-identical across variants → τ=1.
- Scope: IMDb NC + Freebase NC.

## 3. Dataset variants as switchings

- **IMDb1–4**: switchings move the Link node's attachments (M–L vs D–L vs A–L etc.);
  Link is informationally redundant re-encoding of movie relations.
- **DBLP1–3**: Area attached to Paper (1), Venue (2), or Author (3) — switching
  (paper,area) ↔ (venue,area) etc.; recovering P–Ar in DBLP2 needs P–V–Ar (2-hop),
  in DBLP3 P–A–Ar.
- **WordNet/WN18RR**: `no_changes` = official train triples; `all_inverse_edges` = +named
  physical inverse relations (switching: r ↔ r⁻¹ materialization); `transitive_edges`
  = + transitive/shortcut composition edges (switching: r₁;r₂[;r₃] path ↔ single
  named shortcut edge); `universal_edges` = union of all (the universal graph). Invariant
  model must reconstruct e.g. inverse traversal virtually where physical inverses are
  absent, and verify/ignore-as-context shortcut edges where present.
- **Freebase**: `unchanged`; `exact_2` / `exact_3` = entity pairs connected by type-filtered
  paths of exactly length 2 / 3 re-encoded as direct shortcut relations
  (`DATASET_CONSTRUCTION_FREEBASE_WORDNET.md`); universal = `union_exact_2_3`. Invariant model:
  semantic relations = composed 2-hop and 3-hop programs compiled from whichever physical
  encoding the variant provides.

## 4. Validation checklist (applies to every architecture)

1. Raw physical graph hashes DIFFER across variants (the variants really are different inputs).
2. Semantic message program hashes IDENTICAL across variants.
3. Supervision hashes identical (splits, pos/neg samples, labels, batch order).
4. With fixed init: logits, loss, gradients, one optimizer step — exactly equal.
5. Full training: identical checkpoints per epoch; exact interrupt/resume equality.
6. Aggregate: per-variant metrics equal (spread 0), pairwise Kendall-τ = 1.000,
   candidate-logit exact equality.
7. Performance: ≥ universal-graph run; ≈ or ≥ mean of original per-variant runs.

## 5. Pitfalls

- Nondeterministic CUDA scatter-adds break exact equality → use sorted segment reductions
  with fixed order (the RGCN code does this deliberately).
- Interior-witness multiplicity: two different physical paths realizing the same semantic
  message must collapse (set semantics) or be capped identically across variants —
  otherwise message *counts* differ and attention/normalization diverges.
- Caps/sampling in preprocessing must be deterministic functions of the *semantic*
  identity (e.g. per-source rotation), never of the physical layout or dict order.
- Data augmentation baseline: naive per-variant sampling gives inconsistent
  positives/negatives, which makes Kendall-τ estimates meaningless. Fix the sample sets and
  batch schedule globally.
- The universal graph inflates some tasks (WN18RR: inverse edges make MRR jump 31→84):
  compare invariant runs against the right baseline set, and expect the variant mix to
  change task difficulty.
- **Pattern redundancy** (`INVARIANT_MECHANISMS.md` §0): `S_τ^f` contains
  τ-equivalent (non-isomorphic) pattern pairs — a pattern and its switched image — that compute
  the same function. Edge-level compilers dedupe them automatically (set semantics on messages);
  pattern-parameterised models (SeHGNN channels, MAGNN metapaths) must take the quotient
  explicitly (skip representatives), or they pay in capacity while remaining invariant
  (SeHGNN IMDb: one channel per τ-class = 25 channels).
- **Noisy / partial / lossy / unknown switchings**:
  the theory assumes total, bijective, exact, known switchings. Partial switchings are
  invariant for free given the quotient (N1); lossy ones force a choice between an invariance tax
  and a certificate — DBLP2 already uses a certificate (N2); corrupted ones admit no invariance
  (N3) but the compiled bias is O(ε)-stable where the physical bias is Θ(1)-different (N4) and
  derived encodings allow exact rejection (N5); an incomplete catalog gives block-structured τ
  (N6); randomness in the bias must be semantically keyed (N7).

## 6. When the invariant arm and the union graph (`{model}*`) must agree — and when not

`{model}*` is the stock model on the physical union graph; the invariant arm is the model on the
compiled semantic program. They are the **same computation** only when (a) the program the compiler
emits for every variant is, tensor for tensor, what the stock model extracts from the union graph —
same relations / metapaths / channels, same instance tables, same ids and order — and (b) init, seed
and batch schedule coincide. Under (a)+(b) they must agree **bitwise** on deterministic hardware
(the "recoverability" reading of §2). When (a) fails they are legitimately
different models and are *expected* to differ: the invariant bias adds or removes structure relative
to the union graph. Only a difference in the (a)+(b) case is diagnostic of a bug or of hardware
nondeterminism. Where each cell stands:

**Holds — bitwise, checked by program / checkpoint hash:**

| cell | evidence |
|---|---|
| RGCN DBLP (.4024 = .4024); RGCN IMDb Movie–Year (.2003 = .2003) | semantic program == union graph edge for edge (`sem == union`) |
| CMPNN DBLP; CMPNN IMDb Movie–Year | checkpoint hashes equal per seed |
| MAGNN Freebase `invariant` (intersection) vs `universal`; MAGNN IMDb Movie–Year `invariant_union` vs `universal` | identical instance tables (`9fa90f3c62f7`, `d571cfbafb`) |
| SlotGAT LP: IMDb Movie–Year (.2171 = .2171), DBLP (.5565 = .5565) | final checkpoint hashes equal per seed |
| CMPNN WN18RR (context-only program): invariant ≡ the **`no_changes` original** (.4238 = .4238, checkpoints `2c7315ad` / `2c1dbe47` / `38bfdc9c`) | the base-only program is byte-identical to the `no_changes` physical graph; the union graph is *not* the reference on this cell (leaky under the stock mask, .3524) |

**Does not hold, by cause (all by design except the last):**

1. *Semantic program ⊋ union graph.* On the IMDb year graph the compiler projects across the bridge
   (`link_to_year`, `link_to_director`) — edges no variant and no union materialises
   (NC 91,924 semantic vs 83,570 union edges; MD 86,908 vs 72,702). Hence
   RGCN IMDb NC (.6531 vs .6535) and MD (.2001 vs .1766), CMPNN MD (.4228 vs .4171). Movie–Year is
   the exception only because the target-projection rule suppresses the extra edges.
2. *Skip nodes prune what the union graph carries redundantly.* SeHGNN IMDb `invariant_full`: 25
   channels over the canonical schema with Link skipped vs 101 channels (80 duplicates) on the union
   graph — .6535 vs .6431. SeHGNN Freebase `full_rel`: the invariant arm is the union of
   the variants' own biases (511 channels, Sec. 4.3 step 1) while the universal arm enumerates every type path over
   the union graph (597). MAGNN IMDb NC `invariant_union` (.6308) vs universal (.6320); MAGNN Freebase
   `invariant_union` (.6107) vs universal (.6021): the union *metapath set* is not the union
   graph's stock set.
3. *Same message multiset, different representation → same function class, different computation.*
   SlotGAT Freebase: invariant = `exact_union` chunked schedule, universal = `physical` schedule of
   `union_exact_2_3`. RGCN WordNet: the semantic program has exactly the
   universal graph's 893,629 edges yet the one-runner gives .3802 vs .3761 — relation labelling /
   decoder set (`task_only`) differ, so parameters are assigned differently (mechanism not traced). SlotGAT LP
   study on WN18RR (.3662 vs .3635) and SlotGAT IMDb Movie–Director (.2514 vs .2610, program ⊋ union — category 1)
   likewise differ by representation / program, not by bug.
   RGCN Freebase: the program uses the family catalog `exact_3` (the union graph's 64 relations) and is
   category 1, marginally: 12,716 extra walk-composed pairs through the self-looped `BOOK-and-BOOK`
   (0.0013 % of edges) that the materialised shortcuts lack — .6895 (τ = 1, exact logits) vs universal .6864.
   **SlotGAT IMDb NC:** the universal row is the parity model on the union graph with its
   *physical* message schedule (.6519) — the honest `{model}*` proxy — while the invariant arm runs the conditional
   schedule on each variant (.6611, τ = 1). Same architecture and message multiset, different schedule, so the two
   are not the same computation.
4. *Hardware nondeterminism* — the only problematic category: an identical computation giving
   different numbers. Not observed in the V100 runs: every "holds" cell agrees bitwise for every seed.
   Bitwise claims are made per GPU architecture.

**Summary.** In the "holds" cells the invariant arm is *exactly* `{model}*`, and its
contribution is that each variant alone recovers it (recoverability + exact invariance under
independent training). In the other cells the invariant arm is a different — richer (1) or pruned
(2) — structural bias, to be compared against the originals and augmentation, with the union graph as
a reference rather than an identity check. A cross-variant check should use the invariant arm's own
variants; a `universal`-vs-`invariant` identity check is meaningful only in the "holds" cells, and
there any difference at all means a bug or nondeterminism.
