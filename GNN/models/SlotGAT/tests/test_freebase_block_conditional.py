from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SLOTGAT_DIR = Path(__file__).resolve().parents[1]
if str(SLOTGAT_DIR) not in sys.path:
    sys.path.insert(0, str(SLOTGAT_DIR))
if str(SLOTGAT_DIR / "model") not in sys.path:
    sys.path.insert(0, str(SLOTGAT_DIR / "model"))

from freebase_block_conditional import (
    BlockRelation,
    ChannelRecord,
    FreebaseBlockContextPlan,
    NodeTypeLayout,
    RelationInfo,
    build_base_message_arrays,
    boolean_compose,
    canonical_block_edge_set,
    local_relation_matrix,
    preprocess_pair,
)
from genuine_block_conditional_slotgat import GenuineBlockConditionalSlotGAT


def synthetic_fixture():
    # The endpoint relation has three distinct forward signatures and four
    # distinct backward signatures.  It is deliberately NOT endpoint-block
    # bijective, but its ordered composition remains an authoritative semantic
    # shortcut and is valid for the corrected conditional operator.
    layout = NodeTypeLayout(
        counts=(3, 4, 4), offsets=(0, 3, 7), names=("A", "B", "C")
    )
    relations = {
        "A-r-B": RelationInfo(0, 0, 1, "A-r-B"),
        "B-s-C": RelationInfo(1, 1, 2, "B-s-C"),
    }
    local_ab = np.asarray(
        [
            [0, 0, 1, 0],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    ab_rows, ab_cols = np.nonzero(local_ab)
    base_pairs = {
        "A-r-B": (
            ab_rows.astype(np.int64),
            (ab_cols + layout.offsets[1]).astype(np.int64),
        ),
        "B-s-C": (
            np.arange(layout.offsets[1], layout.offsets[1] + 4, dtype=np.int64),
            np.arange(layout.offsets[2], layout.offsets[2] + 4, dtype=np.int64),
        ),
    }
    channels = (
        ChannelRecord(0, "A-r-B", 1, "base_relation"),
        ChannelRecord(1, "A-r-B", -1, "base_relation"),
        ChannelRecord(2, "B-s-C", 1, "base_relation"),
        ChannelRecord(3, "B-s-C", -1, "base_relation"),
        ChannelRecord(4, "A-r-B-s-C", 1, "conditional_path"),
        ChannelRecord(5, "A-r-B-s-C", -1, "conditional_path"),
        ChannelRecord(6, "__self__", 0, "self"),
    )
    left = local_relation_matrix(
        pairs=base_pairs["A-r-B"], relation=relations["A-r-B"], layout=layout
    )
    right = local_relation_matrix(
        pairs=base_pairs["B-s-C"], relation=relations["B-s-C"], layout=layout
    )
    endpoint = boolean_compose(left, right)
    relation = BlockRelation(
        name="A-r-B-s-C",
        composition=("A-r-B", "B-s-C"),
        start_type=0,
        end_type=2,
        forward_channel=4,
        reverse_channel=5,
        forward_steps=(0, 2),
        reverse_steps=(3, 1),
        forward_support=layout.global_nodes(2),
        reverse_support=layout.global_nodes(0),
        endpoint_count=int(endpoint.nnz),
        endpoint_fingerprint={"count": int(endpoint.nnz), "xor64": 0, "sum64": 0},
        hop_recovery=(
            {"drop_index": 0, "dropped_relation": "A-r-B", "status": "registry_authoritative_not_recomputed"},
            {"drop_index": 1, "dropped_relation": "B-s-C", "status": "registry_authoritative_not_recomputed"},
        ),
    )
    base_src, base_dst, base_channel = build_base_message_arrays(
        base_pairs=base_pairs, channels=channels, total_nodes=layout.total
    )
    common = dict(
        layout=layout,
        channels=channels,
        base_src=base_src,
        base_dst=base_dst,
        base_channel=base_channel,
        block_relations=(relation,),
        base_graph_hash="base",
        semantic_program_hash="semantic-program",
        source_dir="synthetic",
    )
    unchanged = FreebaseBlockContextPlan(
        variant="unchanged",
        context_kind="base_composition",
        raw_graph_hash="raw-unchanged",
        raw_link_hash="link-unchanged",
        physical_shortcut_fingerprints={
            relation.name: {"count": 0, "xor64": 0, "sum64": 0}
        },
        context_plan_hash="context-unchanged",
        **common,
    )
    exact = FreebaseBlockContextPlan(
        variant="exact_2",
        context_kind="physical_shortcut",
        raw_graph_hash="raw-exact",
        raw_link_hash="link-exact",
        physical_shortcut_fingerprints={
            relation.name: {"count": int(endpoint.nnz), "xor64": 123, "sum64": 456}
        },
        context_plan_hash="context-exact",
        **common,
    )
    features = []
    for count in layout.counts:
        indices = torch.arange(count, dtype=torch.int64).repeat(2, 1)
        values = torch.ones(count)
        features.append(
            torch.sparse_coo_tensor(indices, values, (count, count)).coalesce()
        )
    return layout, endpoint, unchanged, exact, features, base_pairs


class MappingTests(unittest.TestCase):
    def test_endpoint_quotient_is_not_required(self):
        _, endpoint, unchanged, _, _, _ = synthetic_fixture()
        row_signatures = {
            tuple(endpoint.indices[endpoint.indptr[row] : endpoint.indptr[row + 1]])
            for row in range(endpoint.shape[0])
            if endpoint.indptr[row + 1] > endpoint.indptr[row]
        }
        csc = endpoint.tocsc()
        column_signatures = {
            tuple(csc.indices[csc.indptr[col] : csc.indptr[col + 1]])
            for col in range(endpoint.shape[1])
            if csc.indptr[col + 1] > csc.indptr[col]
        }
        self.assertEqual(len(row_signatures), 3)
        self.assertEqual(len(column_signatures), 4)
        self.assertNotEqual(len(row_signatures), len(column_signatures))
        self.assertEqual(unchanged.block_relations[0].endpoint_count, endpoint.nnz)
        self.assertGreater(len(canonical_block_edge_set(endpoint)), 0)

    def test_contexts_are_distinct_but_semantic_program_is_equal(self):
        _, _, unchanged, exact, _, _ = synthetic_fixture()
        self.assertNotEqual(unchanged.raw_graph_hash, exact.raw_graph_hash)
        self.assertNotEqual(unchanged.context_plan_hash, exact.context_plan_hash)
        self.assertEqual(unchanged.semantic_program_hash, exact.semantic_program_hash)
        unchanged.verify_runtime_context()
        exact.verify_runtime_context()


class PreprocessIntegrationTests(unittest.TestCase):
    def test_preprocessing_accepts_non_bijective_endpoint_relation(self):
        layout, endpoint, _, _, _, base_pairs = synthetic_fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            unchanged = root / "Freebase_unchanged"
            exact = root / "Freebase_exact_2"
            output = root / "out"
            unchanged.mkdir()
            exact.mkdir()

            node_types = [0] * 3 + [1] * 4 + [2] * 4
            node_lines = [
                f"{node}\tn{node}\t{node_type}\n"
                for node, node_type in enumerate(node_types)
            ]
            labels = "0\tn0\t0\t0\n1\tn1\t0\t1\n2\tn2\t0\t2\n"
            base_info = """node.dat
TYPE\tMEANING
0\tA
1\tB
2\tC
link.dat
LINK\tSTART\tEND\tMEANING
0\t0\t1\tA-r-B
1\t1\t2\tB-s-C
label.dat
TYPE\tCLASS\tMEANING
"""
            exact_info = base_info.replace(
                "label.dat\n", "2\t0\t2\tA-r-B-s-C\nlabel.dat\n"
            )
            base_edges = []
            for relation_id, relation_name in enumerate(("A-r-B", "B-s-C")):
                src, dst = base_pairs[relation_name]
                base_edges.extend(
                    (int(u), int(v), relation_id) for u, v in zip(src, dst)
                )
            shortcut = []
            for local_src in range(endpoint.shape[0]):
                columns = endpoint.indices[
                    endpoint.indptr[local_src] : endpoint.indptr[local_src + 1]
                ]
                shortcut.extend(
                    (local_src, int(layout.offsets[2] + col), 2) for col in columns
                )
            for directory, info, edges in (
                (unchanged, base_info, base_edges),
                (exact, exact_info, base_edges + shortcut),
            ):
                (directory / "node.dat").write_text("".join(node_lines))
                (directory / "label.dat").write_text(labels)
                (directory / "info.dat").write_text(info)
                (directory / "link.dat").write_text(
                    "".join(f"{u}\t{v}\t{r}\n" for u, v, r in edges)
                )
            registry = root / "exact_2.jsonl"
            registry.write_text(
                json.dumps(
                    {
                        "new_relation_id": 2,
                        "new_relation_name": "A-r-B-s-C",
                        "composition": "A-r-B ; B-s-C",
                        "hop_count": 2,
                        "start_type": "A",
                        "end_type": "C",
                        "node_type_path": "A ; B ; C",
                        "num_new_edges": int(endpoint.nnz),
                        "unique_endpoint_pairs": int(endpoint.nnz),
                        "skipped_reason": "",
                    }
                )
                + "\n"
            )
            audit = preprocess_pair(
                unchanged_dir=unchanged,
                exact_dir=exact,
                registry_path=registry,
                output_root=output,
            )
            self.assertTrue(audit["passed"])
            left = FreebaseBlockContextPlan.load(output / "unchanged")
            right = FreebaseBlockContextPlan.load(output / "exact_2")
            self.assertNotEqual(left.context_plan_hash, right.context_plan_hash)
            self.assertEqual(left.semantic_program_hash, right.semantic_program_hash)
            self.assertEqual(right.block_relations[0].endpoint_count, endpoint.nnz)
            self.assertEqual(
                right.block_relations[0].forward_steps,
                (0, 2),
            )


class ModelInvarianceTests(unittest.TestCase):
    def build_model(self, layout):
        return GenuineBlockConditionalSlotGAT(
            layout=layout,
            in_dims=layout.counts,
            num_channels=7,
            num_classes=3,
            hidden_dim=12,
            num_heads=3,
            num_layers=2,
            edge_feats=5,
            dropout_feat=0.25,
            dropout_attn=0.2,
            slope=0.05,
            aggregator="SA",
            sa_attention_dim=4,
        )

    def test_eval_logits_are_bitwise_equal_for_distinct_contexts(self):
        layout, _, unchanged, exact, features, _ = synthetic_fixture()
        torch.manual_seed(123)
        model = self.build_model(layout)
        model.eval()
        with torch.no_grad():
            left, _ = model(features, unchanged)
            right, _ = model(features, exact)
        self.assertTrue(torch.equal(left, right))
        self.assertEqual(len(model._runtime_cache), 2)

    def test_fixed_dropout_forward_and_gradients_are_bitwise_equal(self):
        layout, _, unchanged, exact, features, _ = synthetic_fixture()
        torch.manual_seed(321)
        template = self.build_model(layout)
        state = copy.deepcopy(template.state_dict())
        labels = torch.tensor([0, 1, 2], dtype=torch.long)

        results = []
        for plan in (unchanged, exact):
            model = self.build_model(layout)
            model.load_state_dict(state)
            model.train()
            torch.manual_seed(999)
            logits, _ = model(features, plan)
            loss = F.cross_entropy(logits[:3], labels)
            model.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [
                parameter.grad.detach().clone()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            results.append((logits.detach().clone(), gradients))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertEqual(len(results[0][1]), len(results[1][1]))
        for left, right in zip(results[0][1], results[1][1]):
            self.assertTrue(torch.equal(left, right))

    def test_three_adam_steps_remain_bitwise_equal(self):
        layout, _, unchanged, exact, features, _ = synthetic_fixture()
        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        torch.manual_seed(777)
        template = self.build_model(layout)
        initial = copy.deepcopy(template.state_dict())
        final_states = []
        for plan in (unchanged, exact):
            model = self.build_model(layout)
            model.load_state_dict(initial)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            model.train()
            for step in range(3):
                torch.manual_seed(10000 + step)
                logits, _ = model(features, plan)
                loss = F.cross_entropy(logits[:3], labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            final_states.append(copy.deepcopy(model.state_dict()))
        self.assertEqual(final_states[0].keys(), final_states[1].keys())
        for name in final_states[0]:
            self.assertTrue(
                torch.equal(final_states[0][name], final_states[1][name]),
                msg=name,
            )


if __name__ == "__main__":
    unittest.main()
