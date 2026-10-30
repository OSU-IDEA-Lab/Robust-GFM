"""Gradient checkpointing over union-message blocks must be exact.

Regression test for a closure late-binding bug: ``compute``
was defined inside the destination-block loop and closed over the loop variables,
so ``torch.utils.checkpoint(..., use_reentrant=False)`` recomputed EVERY block
with the LAST block's ``source``/``channel``/``degrees``/``destination_start``
during the backward pass. That is silently wrong whenever the shapes agree and a
hard failure when they do not. The union-message operator had never been trained
correctly with checkpointing enabled.
"""
from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))

from model.genuine_union_message_slotgat import (  # noqa: E402
    ChunkedDirectUnionSlotLayer,
    _RuntimeUnionMessages,
)


def build_runtime(num_nodes: int, degrees_per_node: list[int], num_channels: int, device):
    """A runtime whose destination blocks have DIFFERENT edge counts."""
    sources, channels = [], []
    for node, degree in enumerate(degrees_per_node):
        for k in range(degree):
            sources.append((node + k) % num_nodes)
            channels.append((node + k) % num_channels)
    source = torch.tensor(sources, dtype=torch.int64, device=device)
    channel = torch.tensor(channels, dtype=torch.int64, device=device)

    # Three blocks of deliberately unequal size, so a late-bound closure cannot
    # accidentally produce the right shapes.
    boundaries = [(0, 1), (1, 3), (3, num_nodes)]
    blocks = []
    offsets = [0]
    for degree in degrees_per_node:
        offsets.append(offsets[-1] + degree)
    for start, stop in boundaries:
        edge_start, edge_stop = offsets[start], offsets[stop]
        degrees = torch.tensor(degrees_per_node[start:stop], dtype=torch.int64, device=device)
        blocks.append((start, stop, edge_start, edge_stop, degrees))
    return _RuntimeUnionMessages(
        source=source, channel=channel, blocks=tuple(blocks), message_program_hash="test"
    )


class UnionMessageCheckpointTest(unittest.TestCase):
    def _layer(self, num_channels: int) -> ChunkedDirectUnionSlotLayer:
        torch.manual_seed(1566911444)
        return ChunkedDirectUnionSlotLayer(
            num_slots=3,
            input_dim=8,
            output_dim=8,
            num_heads=2,
            num_channels=num_channels,
            edge_dim=4,
            feature_dropout=0.0,
            attention_dropout=0.0,
            negative_slope=0.05,
            activation=True,
            residual=True,
            checkpoint_chunks=True,
        )

    def test_checkpointed_forward_and_backward_match_uncheckpointed(self):
        device = torch.device("cpu")
        num_nodes, num_channels = 6, 4
        degrees = [5, 3, 4, 2, 6, 1]  # unequal per-node degrees -> unequal blocks
        runtime = build_runtime(num_nodes, degrees, num_channels, device)

        results = {}
        for use_checkpoint in (False, True):
            layer = self._layer(num_channels)
            layer.checkpoint_chunks = use_checkpoint
            layer.train()
            slots = torch.arange(
                num_nodes * 3 * 8, dtype=torch.float32, device=device
            ).reshape(num_nodes, 3, 8) / 97.0
            slots.requires_grad_(True)
            out = layer(slots, runtime)
            out.sum().backward()
            results[use_checkpoint] = (
                out.detach().clone(),
                slots.grad.detach().clone(),
                torch.cat([p.grad.flatten() for p in layer.parameters() if p.grad is not None]),
            )

        plain_out, plain_input_grad, plain_param_grad = results[False]
        ckpt_out, ckpt_input_grad, ckpt_param_grad = results[True]
        self.assertTrue(
            torch.allclose(plain_out, ckpt_out, atol=1e-6),
            "checkpointed forward differs from the plain forward",
        )
        self.assertTrue(
            torch.allclose(plain_input_grad, ckpt_input_grad, atol=1e-6),
            "checkpointing changed the input gradient (late-bound block closure?)",
        )
        self.assertTrue(
            torch.allclose(plain_param_grad, ckpt_param_grad, atol=1e-6),
            "checkpointing changed the parameter gradients",
        )


if __name__ == "__main__":
    unittest.main()
