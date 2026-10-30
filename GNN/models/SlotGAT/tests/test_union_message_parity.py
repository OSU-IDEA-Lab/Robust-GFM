"""The Freebase union-message SlotGAT's ``parity`` switch.

Checks, on a synthetic schedule: the parity layer runs forward/backward with and without
chunk checkpointing and matches itself; the projected residual and the upstream edge path
are present; the parity model readout produces unit-norm logits (l2byslot + final L2
normalisation, the dominant factor in the IMDb attribution) and the parity flag changes the
parameter count exactly as expected (no slot bias, wider edge path, output-layer residual).
"""
from __future__ import annotations

import os
import sys
import unittest

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "model"))
from model.genuine_union_message_slotgat import ChunkedDirectUnionSlotLayer  # noqa: E402
from tests.test_union_message_checkpointing import build_runtime  # noqa: E402


def make_layer(parity: bool, *, input_dim=8, output_dim=8, heads=2, channels=4, edge_dim=4, residual=True):
    torch.manual_seed(1566911444)
    return ChunkedDirectUnionSlotLayer(num_slots=3, input_dim=input_dim, output_dim=output_dim, num_heads=heads,
                                       num_channels=channels, edge_dim=edge_dim, feature_dropout=0.0, attention_dropout=0.0,
                                       negative_slope=0.05, activation=True, residual=residual, checkpoint_chunks=True, parity=parity)


class UnionParityTest(unittest.TestCase):
    def test_parameter_accounting(self):
        T, H, D, K, E, C = 3, 2, 8, 4, 4, 8   # slots, heads, in, edge_dim, channels, out
        base = sum(p.numel() for p in make_layer(False).parameters())
        par = sum(p.numel() for p in make_layer(True).parameters())
        # parity: - slot bias (T*H*K_head) ; edge path Linear(edge, edge*H) + attn[H, edge] instead of
        # Linear(edge, H*head_dim) + attn[H, head_dim]
        head_dim = C // H
        expected = base - T * H * head_dim + (K * K * H + H * K) - (K * H * head_dim + H * head_dim)
        self.assertEqual(par, expected)
        # output-layer style: widths differ -> parity adds a projected residual [T, in, out]
        out_base = sum(p.numel() for p in make_layer(False, output_dim=4, heads=1).parameters())
        out_par = sum(p.numel() for p in make_layer(True, output_dim=4, heads=1).parameters())
        self.assertEqual(out_par - out_base, T * D * 4 + (K * K * 1 + 1 * K) - (K * 1 * 4 + 1 * 4) - T * 1 * 4)

    def test_forward_backward_and_checkpoint_equality(self):
        device = torch.device("cpu")
        runtime = build_runtime(6, [5, 3, 4, 2, 6, 1], 4, device)
        outs = {}
        for ckpt in (False, True):
            layer = make_layer(True); layer.checkpoint_chunks = ckpt; layer.train()
            slots = (torch.arange(6 * 3 * 8, dtype=torch.float32).reshape(6, 3, 8) / 97.0).requires_grad_(True)
            out = layer(slots, runtime); out.sum().backward()
            outs[ckpt] = (out.detach(), slots.grad.detach(), torch.cat([p.grad.flatten() for p in layer.parameters() if p.grad is not None]))
        for a, b in zip(outs[False], outs[True]):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertTrue(all(torch.isfinite(t).all() for t in outs[True]))

    def test_readout_is_unit_norm(self):
        from model.genuine_union_message_slotgat import ExactUnionMessageConditionalSlotGAT
        m = ExactUnionMessageConditionalSlotGAT.__new__(ExactUnionMessageConditionalSlotGAT)
        torch.nn.Module.__init__(m)
        m.parity = True; m.num_classes = 7
        m.macroLinear = torch.nn.Linear(7, 3); m.macroSemanticVec = torch.nn.Parameter(torch.randn(3, 1))
        m.register_buffer("epsilon", torch.tensor([1e-12]))
        slots = torch.randn(11, 4, 7) * 5
        out, w = m._readout(slots)
        self.assertEqual(tuple(out.shape), (11, 7)); self.assertEqual(tuple(w.shape), (4,))
        self.assertTrue(torch.allclose(out.norm(dim=1), torch.ones(11), atol=1e-5))
        self.assertTrue(torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
