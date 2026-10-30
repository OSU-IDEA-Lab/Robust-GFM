from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_bias import (  # noqa: E402
    ACTOR,
    CHANNEL_NAMES,
    CONTEXT_DIRECT,
    CONTEXT_VIA_LINK,
    DIRECTOR,
    LINK,
    YEAR,
    NodeTypeLayout,
    RawTypedEdge,
    assert_context_plans_invariant,
    compile_conditional_context_plan,
)
from model.genuine_conditional_slotgat import GenuineConditionalSlotGAT  # noqa: E402


LAYOUT = NodeTypeLayout.from_counts([2, 2, 3, 2])
MOVIE_LINK = [(0, 7), (1, 8)]
MOVIE_DIRECTOR = [(0, 2), (1, 3)]
MOVIE_ACTOR = [(0, 4), (0, 5), (1, 6)]


def bidirectional(pairs, relation_start):
    edges = []
    for left, right in pairs:
        edges.append(RawTypedEdge(left, right, relation_start))
        edges.append(RawTypedEdge(right, left, relation_start + 1))
    return edges


def build_variant(actor_target: str, director_target: str, relation_offset: int):
    edges = bidirectional(MOVIE_LINK, relation_offset)
    movie_to_link = dict(MOVIE_LINK)
    director_pairs = (
        MOVIE_DIRECTOR
        if director_target == "movie"
        else [(movie_to_link[movie], director) for movie, director in MOVIE_DIRECTOR]
    )
    actor_pairs = (
        MOVIE_ACTOR
        if actor_target == "movie"
        else [(movie_to_link[movie], actor) for movie, actor in MOVIE_ACTOR]
    )
    edges += bidirectional(director_pairs, relation_offset + 2)
    edges += bidirectional(actor_pairs, relation_offset + 4)
    return edges


def four_plans():
    configs = {
        "v1": ("movie", "movie", 0),
        "v2": ("link", "link", 10),
        "v3": ("link", "movie", 20),
        "v4": ("movie", "link", 30),
    }
    return {
        name: compile_conditional_context_plan(
            LAYOUT, build_variant(actor, director, relation_offset)
        )
        for name, (actor, director, relation_offset) in configs.items()
    }


class ConditionalBiasTest(unittest.TestCase):
    def test_four_nonisomorphic_context_plans_remain_distinct(self):
        plans = four_plans()
        assert_context_plans_invariant(plans)
        self.assertEqual(len({p.raw_graph_hash for p in plans.values()}), 4)
        self.assertEqual(len({p.context_plan_hash for p in plans.values()}), 4)
        self.assertEqual(
            len({p.semantic_propagation_hash for p in plans.values()}), 1
        )
        self.assertEqual({len(p.emit_semantic_messages()) for p in plans.values()}, {33})
        self.assertTrue(
            all(not p.to_manifest()["materialized_shared_dgl_graph"] for p in plans.values())
        )

    def test_direct_and_via_link_witnesses_are_not_canonicalized(self):
        plans = four_plans()
        v1_kinds = {w.context_kind for w in plans["v1"].witnesses}
        v2_kinds = {w.context_kind for w in plans["v2"].witnesses}
        self.assertIn(CONTEXT_DIRECT, v1_kinds)
        self.assertIn(CONTEXT_VIA_LINK, v2_kinds)
        self.assertNotEqual(plans["v1"].context_plan_hash, plans["v2"].context_plan_hash)
        self.assertEqual(
            plans["v1"].semantic_propagation_hash,
            plans["v2"].semantic_propagation_hash,
        )

    def test_parallel_physical_witnesses_do_not_duplicate_semantics(self):
        base = build_variant("movie", "movie", 0)
        duplicate_relation = list(base) + [
            RawTypedEdge(0, 2, 99),
            RawTypedEdge(2, 0, 100),
            RawTypedEdge(0, 4, 101),
            RawTypedEdge(4, 0, 102),
        ]
        base_plan = compile_conditional_context_plan(LAYOUT, base)
        duplicate_plan = compile_conditional_context_plan(LAYOUT, duplicate_relation)
        self.assertEqual(
            base_plan.semantic_propagation_hash,
            duplicate_plan.semantic_propagation_hash,
        )
        self.assertNotEqual(base_plan.raw_graph_hash, duplicate_plan.raw_graph_hash)

    def test_nonbijective_movie_link_context_fails(self):
        edges = build_variant("movie", "movie", 0)
        edges += [RawTypedEdge(0, 8, 50), RawTypedEdge(8, 0, 51)]
        with self.assertRaises(ValueError):
            compile_conditional_context_plan(LAYOUT, edges)

    def test_fixed_channel_catalog(self):
        # 3 roles (director, actor, year) x 4 directed channels, plus movie<->link
        # and self.
        self.assertEqual(len(CHANNEL_NAMES), 15)
        self.assertEqual(CHANNEL_NAMES[-1], "self")
        for name in ("movie_to_year", "year_to_movie", "link_to_year", "year_to_link"):
            self.assertIn(name, CHANNEL_NAMES)

    def test_four_count_layout_still_supported(self):
        """The pre-year IMDB_var1..5 builds must still compile."""
        layout = NodeTypeLayout.from_counts([2, 2, 2, 2])
        self.assertEqual(layout.num_years, 0)
        self.assertEqual(layout.total, 8)
        plan = compile_conditional_context_plan(LAYOUT, build_variant("movie", "movie", 0))
        self.assertTrue(plan.semantic_propagation_hash)

    def test_year_role_is_projected_onto_both_endpoints(self):
        """movie->year exists in every variant, so link->movie->year is derivable."""
        layout = NodeTypeLayout.from_counts([2, 2, 2, 2, 2])
        m0, m1 = 0, 1
        l0, l1 = layout.offsets[LINK], layout.offsets[LINK] + 1
        y0 = layout.offsets[YEAR]
        edges = []
        for movie, link in ((m0, l0), (m1, l1)):
            edges += [RawTypedEdge(movie, link, 0), RawTypedEdge(link, movie, 1)]
            for peer, rel in ((layout.offsets[DIRECTOR], 2), (layout.offsets[ACTOR], 4), (y0, 6)):
                edges += [RawTypedEdge(movie, peer, rel), RawTypedEdge(peer, movie, rel + 1)]
        plan = compile_conditional_context_plan(layout, edges)
        counts = plan.channel_counts()
        self.assertEqual(counts["movie_to_year"], 2)
        self.assertEqual(counts["link_to_year"], 2)
        self.assertEqual(counts["year_to_movie"], 2)
        self.assertEqual(counts["year_to_link"], 2)


class GenuineConditionalModelTest(unittest.TestCase):
    def _features(self):
        generator = torch.Generator().manual_seed(7)
        return [
            torch.randn(count, 5, generator=generator)
            for count in LAYOUT.counts
        ]

    def _model(self):
        return GenuineConditionalSlotGAT(
            layout=LAYOUT,
            in_dims=[5, 5, 5, 5],
            num_classes=3,
            hidden_dim=8,
            num_heads=2,
            num_layers=2,
            edge_feats=4,
            dropout_feat=0.0,
            dropout_attn=0.0,
            slope=0.05,
            aggregator="SA",
            sa_attention_dim=3,
        )

    def test_distinct_raw_contexts_produce_exact_equal_forward(self):
        plans = four_plans()
        features = self._features()
        torch.manual_seed(123)
        model = self._model().eval()
        with torch.no_grad():
            reference, _ = model(features, plans["v1"])
            for name in ("v2", "v3", "v4"):
                current, _ = model(features, plans[name])
                self.assertTrue(torch.equal(reference, current), name)
        self.assertEqual(len(model._context_tensor_cache), 4)


    def test_fixed_dropout_stream_is_exact_across_distinct_contexts(self):
        plans = four_plans()
        features = self._features()
        torch.manual_seed(123)
        template = GenuineConditionalSlotGAT(
            layout=LAYOUT,
            in_dims=[5, 5, 5, 5],
            num_classes=3,
            hidden_dim=8,
            num_heads=2,
            num_layers=2,
            edge_feats=4,
            dropout_feat=0.3,
            dropout_attn=0.2,
            slope=0.05,
            aggregator="SA",
            sa_attention_dim=3,
        )
        state = {k: v.detach().clone() for k, v in template.state_dict().items()}
        outputs = []
        for name in ("v1", "v2", "v3", "v4"):
            model = GenuineConditionalSlotGAT(
                layout=LAYOUT,
                in_dims=[5, 5, 5, 5],
                num_classes=3,
                hidden_dim=8,
                num_heads=2,
                num_layers=2,
                edge_feats=4,
                dropout_feat=0.3,
                dropout_attn=0.2,
                slope=0.05,
                aggregator="SA",
                sa_attention_dim=3,
            )
            model.load_state_dict(state, strict=True)
            model.train()
            torch.manual_seed(999)
            current, _ = model(features, plans[name])
            outputs.append(current.detach())
        for current in outputs[1:]:
            self.assertTrue(torch.equal(outputs[0], current))


    def test_independent_optimizer_trajectories_match(self):
        plans = four_plans()
        features = self._features()
        labels = torch.tensor([0, 1], dtype=torch.long)
        train_idx = torch.tensor([0, 1], dtype=torch.long)
        torch.manual_seed(123)
        template = GenuineConditionalSlotGAT(
            layout=LAYOUT,
            in_dims=[5, 5, 5, 5],
            num_classes=3,
            hidden_dim=8,
            num_heads=2,
            num_layers=2,
            edge_feats=4,
            dropout_feat=0.3,
            dropout_attn=0.2,
            slope=0.05,
            aggregator="SA",
            sa_attention_dim=3,
        )
        initial = {k: v.detach().clone() for k, v in template.state_dict().items()}
        final_states = []
        final_logits = []
        for name in ("v1", "v2", "v3", "v4"):
            model = GenuineConditionalSlotGAT(
                layout=LAYOUT,
                in_dims=[5, 5, 5, 5],
                num_classes=3,
                hidden_dim=8,
                num_heads=2,
                num_layers=2,
                edge_feats=4,
                dropout_feat=0.3,
                dropout_attn=0.2,
                slope=0.05,
                aggregator="SA",
                sa_attention_dim=3,
            )
            model.load_state_dict(initial, strict=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
            torch.manual_seed(1001)
            for _ in range(3):
                model.train()
                logits, _ = model(features, plans[name])
                loss = F.cross_entropy(logits[train_idx], labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                logits, _ = model(features, plans[name])
            final_states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            final_logits.append(logits.detach().clone())
        for state in final_states[1:]:
            for key in final_states[0]:
                self.assertTrue(torch.equal(final_states[0][key], state[key]), key)
        for logits in final_logits[1:]:
            self.assertTrue(torch.equal(final_logits[0], logits))

    def test_distinct_raw_contexts_produce_exact_equal_gradients(self):
        plans = four_plans()
        features = self._features()
        labels = torch.tensor([0, 1], dtype=torch.long)
        train_idx = torch.tensor([0, 1], dtype=torch.long)
        torch.manual_seed(123)
        template = self._model()
        state = {k: v.detach().clone() for k, v in template.state_dict().items()}

        gradient_sets = []
        for name in ("v1", "v2", "v3", "v4"):
            model = self._model()
            model.load_state_dict(state, strict=True)
            model.train()
            logits, _ = model(features, plans[name])
            loss = F.cross_entropy(logits[train_idx], labels)
            model.zero_grad(set_to_none=True)
            loss.backward()
            gradient_sets.append(
                {
                    key: parameter.grad.detach().clone()
                    for key, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
            )
        reference = gradient_sets[0]
        for current in gradient_sets[1:]:
            self.assertEqual(set(reference), set(current))
            for key in reference:
                self.assertTrue(torch.equal(reference[key], current[key]), key)


if __name__ == "__main__":
    unittest.main()


class PhysicalPlanCacheKeyTest(unittest.TestCase):
    """A conditional and a physical plan for one variant must not collide.

    They share a ``context_plan_hash`` by construction (identical witnesses,
    different emission), so anything keyed on that hash alone would confuse them.
    """

    def _plans(self):
        from conditional_bias import compile_physical_context_plan
        raw_edges = build_variant("movie", "movie", 0)
        return (
            compile_conditional_context_plan(LAYOUT, raw_edges),
            compile_physical_context_plan(LAYOUT, raw_edges),
        )

    def test_context_plan_hashes_collide_but_semantic_hashes_do_not(self):
        conditional, physical = self._plans()
        self.assertEqual(conditional.context_plan_hash, physical.context_plan_hash)
        self.assertNotEqual(
            conditional.semantic_propagation_hash, physical.semantic_propagation_hash
        )
        self.assertEqual(conditional.mode, "semantic_closure")
        self.assertEqual(physical.mode, "physical")

    def test_physical_messages_are_a_strict_subset_of_the_closure(self):
        conditional, physical = self._plans()
        closure = set(conditional.emit_semantic_messages())
        raw = set(physical.emit_semantic_messages())
        self.assertTrue(raw < closure, "physical messages must be a strict subset")

    def test_model_cache_distinguishes_the_two_plans(self):
        import torch
        from model.genuine_conditional_slotgat import GenuineConditionalSlotGAT

        conditional, physical = self._plans()
        in_dims = [4] * len(conditional.layout.counts)
        model = GenuineConditionalSlotGAT(
            layout=conditional.layout,
            in_dims=in_dims,
            num_classes=3,
            hidden_dim=4,
            num_heads=2,
            num_layers=1,
            edge_feats=4,
            dropout_feat=0.0,
            dropout_attn=0.0,
            slope=0.05,
            aggregator="SA",
            sa_attention_dim=2,
        )
        device = torch.device("cpu")
        a_src = model._context_tensors(conditional, device)[0]
        b_src = model._context_tensors(physical, device)[0]
        self.assertNotEqual(
            int(a_src.numel()),
            int(b_src.numel()),
            "the physical plan received the conditional plan's cached tensors",
        )
