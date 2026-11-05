# How we run node classification with MOTIF

MOTIF is a **link-prediction** model: a knowledge-graph foundation model that answers
questions of the form *"(head, relation, ?) — which node completes this edge?"*. It has no
classification head. This document explains how we made it classify nodes (IMDb movie
genre, Freebase BOOK category), why the setup is built the way it is, and what the
results can and cannot be compared with.

Each section has a **plain-language** part (🧒) and a **technical** part (🔧).

---

## 1. The core trick: turn classes into nodes

🧒 **Plain language.** MOTIF only knows how to answer "who is connected to whom?". So we
don't ask "what genre is this movie?". We add three new dots to the graph, labelled
*Action*, *Comedy* and *Drama*, and draw a line from every **training** movie to its genre's
dot. Then, for a test movie, we ask MOTIF: *"which genre dot should this movie be connected
to?"*. Whichever dot MOTIF thinks is the most likely link is our prediction.

🔧 **Technical.**
* For a task with C classes, we append C new nodes to the graph, with ids
  `num_nodes … num_nodes + C − 1` (IMDb: 15789–15791; Freebase: 180098–180104).
* We add one new relation type, `node_to_class`, and one edge `(v, node_to_class, class_node[y_v])`
  for every **training** node v. IMDb has 2,926 such edges, Freebase 4,772.
* A test query is `(v, node_to_class, ?)`, scored against **all C class nodes**: a [C]
  score vector per node. The prediction is the argmax (ties go to the smallest class id).
* MOTIF adds inverse edges internally, as for every relation, so a class node also points
  back to its training members. That reverse direction is what lets information flow from
  "movies already known to be Dramas" to a test movie.

This is the same setup as DBLP paper→conference link prediction, just with 3 (IMDb) or
7 (Freebase) candidates instead of 20 conferences.

---

## 2. Only training nodes get a label edge (no leakage)

🧒 **Plain language.** The label lines are an answer key. If a test movie already had a
line to its genre dot, MOTIF could just read the answer off the graph. So only training
movies get lines; validation and test movies stay unconnected to any genre dot, and MOTIF
has to work their genre out from their neighbours.

🔧 **Technical.** `nc_motif.py export` builds `train_nodes × class_nodes` edges only, then
checks every variant and **fails the export** if any validation or test node has a
`node_to_class` edge, or if the source graph already touches the new class-node ids. We
re-audited the saved files: 0 held-out nodes carry a label edge, in all IMDb and Freebase
variants.

---

## 3. How MOTIF "reasons" its way to a class

🧒 **Plain language.** Think of gossip spreading through a social network. Start at the
test movie and pass messages along edges: to its director, from the director to the
director's other movies, and from those movies to the genre dots they are connected to.
After a few rounds, the genre dot that received the most (and most reliable) messages wins.
MOTIF does not just count. It learns *which kinds of paths* to trust. "Same director" may be
very informative for genre, "same release year" much less. That is where its two parts come
in.

🔧 **Technical.** MOTIF has two encoders:
1. **Relation encoder (HCNet over a relation-motif hypergraph).** MOTIF builds a graph *of
   relations*. Relations are connected when they appear together in small patterns (motifs,
   up to 3 relations) in the entity graph. It then computes a representation of each
   relation **conditioned on the query relation** (`node_to_class`). This is why MOTIF can
   handle a relation it has **never seen before**: it describes `node_to_class` by how it
   co-occurs with `movie_to_director`, `movie_to_actor` and so on, not by a stored embedding.
2. **Entity encoder (NBFNet-style conditional message passing).** Starting from the query
   node, it runs 6 layers of message passing over the whole graph, using the relation
   representations as edge messages. At the end, each candidate class node has a
   query-specific representation, which a small MLP turns into a score.

MOTIF has **no per-node embeddings**. It never memorizes "movie #812". It only learns how
relation patterns combine, which is what makes zero-shot transfer to a new graph possible.

---

## 4. Zero-shot: the frozen checkpoint

🧒 **Plain language.** We take MOTIF exactly as released. It was trained on other
knowledge graphs (Freebase-15k, WordNet, CoDEx) and has never seen IMDb, Freebase-BOOK or
genre dots. We ask it the question directly. If it does well, it is because its general
"follow the paths" skill transfers, not because it learned anything about movies.

🔧 **Technical.** `nc_motif.py eval` with `upstream/ckpts/motif_3g.pth`, no gradient updates, on
each variant's graph. There is no training seed, so the result is a single deterministic
run (the tables show it with std 0).

---

## 5. Fine-tuning: one model per variant

🧒 **Plain language.** We then give MOTIF some practice on each variant. We show it training
movies and their genres, and nudge it whenever it ranks the wrong genre dot higher. There is
one catch: when we quiz it on a training movie, we **temporarily hide that movie's own
genre line**. Otherwise the answer is sitting right there and it learns nothing.

🔧 **Technical** (`nc_motif.py finetune`):
* **Start point:** each run starts from the released checkpoint, and a separate model is
  trained for each variant and each lab seed (1566911444, 20241017, 20251017). This matches
  the lab's "original" arm.
* **Batches:** 8 training nodes per step. Each node's candidates are `[true class, the other C−1 classes]`.
* **Loss:** MOTIF's native binary cross-entropy with self-adversarial negative weighting
  (temperature 1): wrong classes the model currently ranks high get more weight.
* **Hiding the answer:** MOTIF's native `remove_easy_edges` removes each batch query's own
  `(v, node_to_class, class)` edge and its inverse before the forward pass, then rebuilds
  the relation-motif graph. The model can never read a training label directly.
* **Optimizer:** Adam, lr 5e-4, gradient-norm clip 1.0, 1,000 steps. This is the same untuned
  pilot recipe as our link-prediction runs. All trainable parameters are updated.
* **Checkpoint selection:** validation accuracy every 100 steps, on all validation nodes; on
  ties the earlier checkpoint is kept. Test labels are never used for training or selection.

---

## 6. Metrics

🧒 **Plain language.** **Accuracy** is "how often the top genre is right". **Macro**
precision, recall and F1 average over classes, so a rare class counts as much as a common
one. **Kendall-τ** asks: if I show the same movie to the model trained on v1 and the model
trained on v2, do they rank the three genres in the same order? 1 means always the same
order (fully invariant); 0 means unrelated orders.

🔧 **Technical** (`lp_metrics.classification_metrics`, matching the lab's `METRICS.md` §2 and §4.1):
* **Accuracy:** Micro-F1 equals accuracy for single-label tasks.
* **Macro precision/recall/F1:** averaged over all C classes, with `zero_division = 0`.
* **Kendall-τ_b:** computed per test node over its C class scores, then averaged over nodes.
  It is computed within each seed, then reported as mean ± population std over the 3 seeds.
* **Coarseness:** with C = 3 (IMDb), a per-node τ can only take the values {−1, −⅓, ⅓, 1},
  so IMDb's τ is a coarse statistic. Freebase (C = 7) is finer.

---

## 7. The data: the lab's own labels, splits and variants

### IMDb (genre; 4 variants)
🧒 Same movies, same genres and same train/val/test split as the lab's RGCN and SlotGAT. The
four variants store *who directed* and *who acted* either on the movie itself or on a
separate "link" node. The information is identical; only the shape of the graph changes.

🔧
* **Bundles:** built with the lab's own script
  (`../../GNN/models/RGCN/preprocess_IMDB_rgcn_structural.py --tasks nc`) from
  `movie_metadata.csv`.
* **Labels:** the first of Action, Comedy or Drama in a movie's genre list (4,180 movies:
  1,095 / 1,564 / 1,521).
* **Split:** fixed 2,926 / 418 / 836, seed 1566911444. All four variants share the same
  supervision hash.
* **Variants:** v1 = director and actor on the movie; v2 = both on the link node; v3 = director
  on the movie, actor on the link; v4 = the reverse. Year is always on the movie.
* **Edges:** the lab's bundles store both directions of every edge, so we keep one direction
  (`lp_data.forward_edges`, which checks that the mirror pairs match) and let MOTIF add its own
  inverses.

### Freebase (BOOK category; 3 variants)
🧒 A big general knowledge graph. We classify books into 7 categories. The variants add
"shortcut" edges that summarise 2-hop or 3-hop paths. They contain no new facts, just more
(and much larger) wiring.

🔧
* **Source:** `data_tierA/dataset_variant_3hops_filtered/{unchanged, exact_2, exact_3}`:

  | Variant | Directed edges | Relations |
  |---|---|---|
  | v1 `unchanged` | 1.06M | 36 |
  | v2 `exact_2` | 47.4M | 42 |
  | v3 `exact_3` | 447M | 58 |

* **Labels:** 7,954 BOOK nodes with **7 classes (0–6)**, although the lab's construction doc
  says 8.
* **Split:** the lab's stored `Freebase_unchanged/train_val_test_idx.npz`, which every lab
  model used. It holds **global BOOK node ids** (4,772 / 1,591 / 1,591), stratified. Rerunning
  the lab's `make_split` with the documented seed does *not* reproduce this file, so we use
  the stored file as authoritative.
* **Export:** `link.dat` is already directed, so MOTIF adds the inverses itself. The export
  runs as a CPU job (`slurm/nc_export.sbatch`) because parsing the 7.9 GB `exact_3` file
  needs about 40 GB of RAM.
* **GPU:** everything runs on one H200. Even v1/v2 run there, so that all Freebase runs share
  one GPU type. v3 has about 900M edges including inverses, and full-graph message passing
  over it is the open feasibility question (see section 10).

---

## 8. What the results can be compared with

🧒 **Plain language.** There is **no CMPNN** here: the lab never ran it on node
classification. We compare with the lab's RGCN and SlotGAT, but the two sides do not see
the same information.
* The baselines see a **bag of plot keywords** for each movie (IMDb only).
* MOTIF sees **the genres of training movies as graph edges**. It can reason "this
  director's other films are mostly Dramas", which the baselines cannot, because they never
  see their neighbours' labels.

Neither side strictly has more. They have *different* clues, so treat the comparison as
indicative, not a head-to-head.

🔧 **Reference points** (computed from the graphs; no model):

| | Prior (most common class) | Label-vote heuristic | RGCN | SlotGAT |
|---|---|---|---|---|
| IMDb | 0.354 | **0.584**: majority genre of co-director/co-actor training movies | 0.60–0.65 | 0.65–0.66 |
| Freebase | 0.488 | **0.625**: 2-hop vote over training books' labels | ≈ 0.68 | (see `Freebase_nc.tex`) |

These show how much of each task is "label propagation", the thing MOTIF's setup can do
naturally.

---

## 9. Results so far (IMDb)

| Accuracy | v1 | v2 | v3 | v4 | Avg | Avg \|Δ\| | Avg τ |
|---|---|---|---|---|---|---|---|
| RGCN | .652 | .603 | .649 | .618 | .631 | .029 | .784 |
| SlotGAT | .654 | .663 | .656 | .647 | .655 | .008 | .764 |
| MOTIF (ZS) | .610 | .357 | .505 | .548 | .505 | .134 | .557 |
| MOTIF (FT) | .609 | .592 | .605 | .603 | .602 | **.009** | **.830** |

🧒 **What it means.**
* **Fine-tuned MOTIF gets within about 5 points of the feature-using baselines without seeing
  any plot keywords.** It is also the *most consistent* across the four ways of drawing the
  graph: the smallest accuracy gap between variants and the highest agreement (τ).
* **Zero-shot is fragile.** It works when the director and actors hang directly off the movie
  (v1, 0.610). It falls to "always guess Comedy" (0.357) when they hang off the link node
  instead (v2): the useful paths get one hop longer, and the frozen model does not follow them.
* **Practice fixes this.** Fine-tuning teaches each model to reach through the link node.
  This is the opposite of link prediction, where fine-tuning made MOTIF *less* invariant.

🔧 The fine-tuned τ (.83) is close to the **seed floor**, i.e. the τ between two seeds on the
*same* variant (.85–.92). So most of the remaining cross-variant disagreement is training
noise, not the representation.


---

## 10. Limitations

1. **Scale.** MOTIF does full-graph message passing for every query, and rebuilds the
   relation-motif graph at every training step. That is cheap on IMDb (28k edges) but heavy
   on Freebase v3 (about 900M edges with inverses). If v3 does not fit on one GPU, the honest
   result is "MOTIF cannot run on this variant without subgraph sampling", which is itself a
   finding about GFM scalability.
2. **Label-edge advantage.** The setup lets MOTIF exploit neighbours' labels (label
   propagation), which the lab's baselines cannot. That is a property of how a link-prediction
   GFM must do classification, but it has to be stated whenever the numbers are compared.
3. **Untuned fine-tuning.** 1,000 steps at a fixed learning rate are not tuned for this task;
   the baselines trained to early stopping.
4. **τ is coarse with 3 classes** (IMDb).

---

## 11. Files and commands

| What | Where |
|---|---|
| Export, eval, fine-tune | `script/nc_motif.py` (`export` / `eval` / `finetune`) |
| Slurm jobs | `slurm/nc.sbatch` (GPU; array index → variant, or variant × seed), `slurm/nc_export.sbatch` (CPU) |
| Tables | `script/report_seeds.py --dataset nc_imdb` / `nc_freebase`; `script/report_overview.py` |
| Exported data (not in git) | `rai-datasets/preprocessed/NC/{imdb,freebase}/`: `manifest.json`, `labels.npz`, `v*.npy` |
| Results (not in git) | `results/motif_{zeroshot,finetuned}_nc_<dataset>_*`, `results/motif_finetune_nc_<dataset>_*` |
| Tables in git | `results/seed_summary_nc_imdb/`, overview in `results/seed_summary_overview/` |

```bash
# IMDb (V100)
python script/nc_motif.py export --dataset imdb --bundle-root rai-datasets/preprocessed/_build/IMDB_nc/IMDB/nc
sbatch --array=0-3  --export=ALL,DATASET=imdb,MODE=eval,RUN_TAG=full slurm/nc.sbatch
sbatch --array=0-11 --export=ALL,DATASET=imdb,MODE=finetune,RUN_TAG=seeds slurm/nc.sbatch

# Freebase (CPU export, then one H200)
sbatch --export=ALL,DATASET=freebase slurm/nc_export.sbatch
sbatch -p dgxh --constraint=h200 --mem=160G --array=0-2 --export=ALL,DATASET=freebase,MODE=eval,RUN_TAG=full slurm/nc.sbatch
sbatch -p dgxh --constraint=h200 --mem=160G --array=0-8 --export=ALL,DATASET=freebase,MODE=finetune,RUN_TAG=seeds slurm/nc.sbatch

# Tables
python script/report_seeds.py --dataset nc_imdb --zeroshot results/motif_zeroshot_nc_imdb_full \
  --finetune results/motif_finetuned_nc_imdb_seeds_s* \
  --baseline-tex $BASELINES/IMDB_nc.tex --baseline RGCN SlotGAT
python script/report_overview.py
```
