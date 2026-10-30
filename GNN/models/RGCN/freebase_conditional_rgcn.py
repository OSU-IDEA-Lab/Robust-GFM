#!/usr/bin/env python3
"""Memory-safe sampled RGCN over compiled block-conditional Freebase channels."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from freebase_block_common import load_csr, load_json

MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    z = (int(value) + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def deterministic_positions(degree: int, fanout: int, key: int) -> np.ndarray:
    if fanout <= 0 or degree <= fanout:
        return np.arange(degree, dtype=np.int64)
    offset = splitmix64(key) % degree
    step = 1 + splitmix64(key ^ 0xD6E8FEB86659FD93) % (degree - 1)
    while math.gcd(step, degree) != 1:
        step += 1
        if step >= degree:
            step = 1
    positions = (offset + np.arange(fanout, dtype=np.int64) * step) % degree
    positions.sort()
    return positions


@dataclass(frozen=True)
class CSRChannel:
    channel_id: int
    name: str
    direction: int
    source_type: int
    destination_type: int
    indptr: np.ndarray
    indices: np.ndarray


class BlockConditionalGraph:
    def __init__(
        self,
        preprocessed_dir: Path,
        mmap: bool = True,
        relation_vocabulary: dict[tuple[str, int], int] | None = None,
    ) -> None:
        """Load one compiled channel store.

        ``relation_vocabulary`` maps ``(semantic_relation, direction)`` to a GLOBAL
        channel slot. It exists for the data-augmentation arm, where one shared
        model trains over several variants whose stores have different relation
        sets (physical unchanged 36 / exact_2 42 / exact_3 58 relations). Without a
        shared vocabulary the same relation would occupy different weight slots in
        different variants, so the "one shared model" contract would be violated.

        With a vocabulary supplied, each channel takes its global slot,
        ``num_relations`` becomes the global size, and relations absent from this
        variant simply contribute no edges (their ``relation_ptr`` slice is empty).
        Channels stay sorted by slot id, which is what ``build_sampled_block``
        relies on when it accumulates ``relation_counts`` and cumsums them.

        Default (None) preserves the per-variant behaviour exactly.
        """
        self.root = Path(preprocessed_dir).resolve()
        self.catalog = load_json(self.root / "relation_catalog.json")
        self.hashes = load_json(self.root / "model_input_hashes.json")
        self.overview = load_json(self.root / "preprocessing_overview.json")
        self.node_types = np.load(self.root / "node_types.npy", mmap_mode="r" if mmap else None)
        self.target_nodes = np.load(self.root / "target_nodes.npy", mmap_mode="r" if mmap else None)
        self.labels = np.load(self.root / "labels.npy", mmap_mode="r" if mmap else None)
        splits = np.load(self.root / "train_val_test_idx.npz")
        self.splits = {key: np.asarray(splits[key], dtype=np.int64) for key in splits.files}
        self.relation_vocabulary = relation_vocabulary
        self.channels: list[CSRChannel] = []
        for record in sorted(self.catalog["channels"], key=lambda x: int(x["channel_id"])):
            directory = self.root / record["directory"]
            indptr, indices = load_csr(directory, mmap=mmap)
            if relation_vocabulary is None:
                slot = int(record["channel_id"])
            else:
                key = (str(record["semantic_relation"]), int(record["direction"]))
                if key not in relation_vocabulary:
                    raise KeyError(
                        f"Channel {key} of {self.root} is absent from the shared relation "
                        f"vocabulary; the vocabulary must cover every variant's relations."
                    )
                slot = int(relation_vocabulary[key])
            self.channels.append(
                CSRChannel(
                    channel_id=slot,
                    name=str(record["semantic_relation"]),
                    direction=int(record["direction"]),
                    source_type=int(record["source_type"]),
                    destination_type=int(record["destination_type"]),
                    indptr=indptr,
                    indices=indices,
                )
            )
        if relation_vocabulary is None:
            if [c.channel_id for c in self.channels] != list(range(len(self.channels))):
                raise ValueError("Channel ids must be contiguous")
        else:
            # Sorted by global slot: build_sampled_block writes
            # relation_counts[channel_id] while iterating channels in order and
            # then cumsums, so ascending slot order is required for the
            # relation_ptr slices to line up with the concatenated edge arrays.
            self.channels.sort(key=lambda c: c.channel_id)
            slots = [c.channel_id for c in self.channels]
            if len(set(slots)) != len(slots):
                raise ValueError("Duplicate relation slots after vocabulary mapping")
        self.num_nodes = int(self.node_types.size)
        self.num_relations = (
            len(self.channels) if relation_vocabulary is None else len(relation_vocabulary)
        )
        self.num_node_types = int(np.max(self.node_types)) + 1
        self.num_classes = int(np.max(self.labels)) + 1
        self.center_type = int(self.catalog["center_type"])


@dataclass
class SampledBlock:
    src_nodes: np.ndarray
    dst_nodes: np.ndarray
    src_index: Tensor
    dst_index: Tensor
    edge_type: Tensor
    dst_in_src: Tensor
    relation_ptr: Tensor

    def to(self, device: torch.device | str) -> "SampledBlock":
        return SampledBlock(
            src_nodes=self.src_nodes,
            dst_nodes=self.dst_nodes,
            src_index=self.src_index.to(device),
            dst_index=self.dst_index.to(device),
            edge_type=self.edge_type.to(device),
            dst_in_src=self.dst_in_src.to(device),
            relation_ptr=self.relation_ptr.to(device),
        )


def sample_incoming(
    channel: CSRChannel,
    dst: int,
    fanout: int,
    key: int,
) -> np.ndarray:
    start, end = int(channel.indptr[dst]), int(channel.indptr[dst + 1])
    neighbors = channel.indices[start:end]
    degree = int(end - start)
    if degree == 0:
        return np.empty(0, dtype=np.int32)
    pos = deterministic_positions(degree, fanout, key)
    selected = np.asarray(neighbors[pos], dtype=np.int32)
    selected.sort()
    return selected


def build_sampled_block(
    graph: BlockConditionalGraph,
    dst_nodes: Sequence[int] | np.ndarray,
    *,
    fanout: int,
    sampling_seed: int,
    layer: int,
) -> SampledBlock:
    dst_nodes = np.asarray(dst_nodes, dtype=np.int64)
    if dst_nodes.ndim != 1 or dst_nodes.size == 0:
        raise ValueError("dst_nodes must be a nonempty vector")
    if len(np.unique(dst_nodes)) != len(dst_nodes):
        raise ValueError("dst_nodes must be unique")

    edge_src_global: list[np.ndarray] = []
    edge_dst_local: list[np.ndarray] = []
    edge_rel: list[np.ndarray] = []
    relation_counts = np.zeros(graph.num_relations, dtype=np.int64)
    all_sources: list[np.ndarray] = [dst_nodes.astype(np.int32, copy=False)]

    dst_types = np.asarray(graph.node_types[dst_nodes], dtype=np.int64)
    for channel in graph.channels:
        rel_sources: list[np.ndarray] = []
        rel_destinations: list[np.ndarray] = []
        eligible = np.where(dst_types == channel.destination_type)[0]
        for dst_local in eligible.tolist():
            dst_global = int(dst_nodes[dst_local])
            key = (
                int(sampling_seed)
                ^ (int(layer) * 0x9E3779B97F4A7C15)
                ^ (int(channel.channel_id) * 0xBF58476D1CE4E5B9)
                ^ (int(dst_global) * 0x94D049BB133111EB)
            ) & MASK64
            selected = sample_incoming(channel, int(dst_global), fanout, key)
            if selected.size:
                rel_sources.append(selected)
                rel_destinations.append(np.full(selected.size, dst_local, dtype=np.int64))
        if rel_sources:
            src = np.concatenate(rel_sources)
            dst = np.concatenate(rel_destinations)
            # Inputs are relation-major, destination-major, and source-sorted.
            edge_src_global.append(src)
            edge_dst_local.append(dst)
            edge_rel.append(np.full(src.size, channel.channel_id, dtype=np.int64))
            relation_counts[channel.channel_id] = src.size
            all_sources.append(src)

    src_nodes = np.unique(np.concatenate(all_sources).astype(np.int64, copy=False))
    dst_in_src = np.searchsorted(src_nodes, dst_nodes)
    if not np.array_equal(src_nodes[dst_in_src], dst_nodes):
        raise RuntimeError("Destination nodes missing from source-node closure")

    if edge_src_global:
        src_global = np.concatenate(edge_src_global).astype(np.int64, copy=False)
        src_index = np.searchsorted(src_nodes, src_global)
        dst_index = np.concatenate(edge_dst_local).astype(np.int64, copy=False)
        edge_type = np.concatenate(edge_rel).astype(np.int64, copy=False)
    else:
        src_index = np.empty(0, dtype=np.int64)
        dst_index = np.empty(0, dtype=np.int64)
        edge_type = np.empty(0, dtype=np.int64)

    relation_ptr = np.empty(graph.num_relations + 1, dtype=np.int64)
    relation_ptr[0] = 0
    np.cumsum(relation_counts, out=relation_ptr[1:])
    return SampledBlock(
        src_nodes=src_nodes,
        dst_nodes=dst_nodes,
        src_index=torch.from_numpy(src_index),
        dst_index=torch.from_numpy(dst_index),
        edge_type=torch.from_numpy(edge_type),
        dst_in_src=torch.from_numpy(dst_in_src.astype(np.int64, copy=False)),
        relation_ptr=torch.from_numpy(relation_ptr),
    )


def build_two_layer_blocks(
    graph: BlockConditionalGraph,
    target_nodes: Sequence[int] | np.ndarray,
    *,
    fanout_layer1: int,
    fanout_layer2: int,
    sampling_seed: int,
) -> tuple[SampledBlock, SampledBlock]:
    block2 = build_sampled_block(
        graph,
        target_nodes,
        fanout=fanout_layer2,
        sampling_seed=sampling_seed,
        layer=2,
    )
    block1 = build_sampled_block(
        graph,
        block2.src_nodes,
        fanout=fanout_layer1,
        sampling_seed=sampling_seed,
        layer=1,
    )
    if not np.array_equal(block1.dst_nodes, block2.src_nodes):
        raise RuntimeError("Two-layer block closure mismatch")
    return block1, block2


def deterministic_segment_mean(messages: Tensor, counts: Tensor) -> Tensor:
    if messages.numel() == 0:
        return messages.new_empty((0, messages.size(-1)))
    try:
        return torch.segment_reduce(messages, reduce="mean", lengths=counts, axis=0)
    except (AttributeError, RuntimeError):
        pieces = []
        start = 0
        for count in counts.detach().cpu().tolist():
            end = start + int(count)
            pieces.append(messages[start:end].mean(dim=0))
            start = end
        return torch.stack(pieces, dim=0)


class SampledRGCNConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        *,
        num_bases: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_relations = int(num_relations)
        self.num_bases = int(num_bases) if num_bases is not None else None
        if self.num_bases is None:
            self.weight = nn.Parameter(torch.empty(num_relations, in_channels, out_channels))
            self.register_parameter("comp", None)
        else:
            self.weight = nn.Parameter(torch.empty(self.num_bases, in_channels, out_channels))
            self.comp = nn.Parameter(torch.empty(num_relations, self.num_bases))
        self.root = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.comp is not None:
            nn.init.xavier_uniform_(self.comp)
        nn.init.xavier_uniform_(self.root)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def relation_weights(self) -> Tensor:
        if self.comp is None:
            return self.weight
        return (self.comp @ self.weight.reshape(self.num_bases, -1)).reshape(
            self.num_relations, self.in_channels, self.out_channels
        )

    def forward(self, x_src: Tensor, block: SampledBlock) -> Tensor:
        if x_src.shape != (len(block.src_nodes), self.in_channels):
            raise ValueError(
                f"Expected x_src {(len(block.src_nodes), self.in_channels)}, got {tuple(x_src.shape)}"
            )
        out = x_src[block.dst_in_src] @ self.root
        weights = self.relation_weights()
        for relation in range(self.num_relations):
            start = int(block.relation_ptr[relation].item())
            end = int(block.relation_ptr[relation + 1].item())
            if start == end:
                continue
            src = block.src_index[start:end]
            dst = block.dst_index[start:end]
            messages = x_src[src] @ weights[relation]
            unique_dst, counts = torch.unique_consecutive(dst, return_counts=True)
            reduced = deterministic_segment_mean(messages, counts)
            out[unique_dst] = out[unique_dst] + reduced
        if self.bias is not None:
            out = out + self.bias
        return out


class FreebaseBlockConditionalRGCN(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        num_node_types: int,
        num_relations: int,
        num_classes: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_bases: int = 8,
        dropout: float = 0.3,
        use_type_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.type_embedding = (
            nn.Embedding(num_node_types, embedding_dim) if use_type_embedding else None
        )
        self.conv1 = SampledRGCNConv(
            embedding_dim, hidden_dim, num_relations, num_bases=num_bases
        )
        self.conv2 = SampledRGCNConv(
            hidden_dim, num_classes, num_relations, num_bases=num_bases
        )
        self.dropout = float(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_embedding.weight)
        if self.type_embedding is not None:
            nn.init.xavier_uniform_(self.type_embedding.weight)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def input_features(self, global_nodes: np.ndarray, node_types: np.ndarray, device: torch.device) -> Tensor:
        ids = torch.as_tensor(global_nodes, dtype=torch.long, device=device)
        x = self.node_embedding(ids)
        if self.type_embedding is not None:
            types = torch.as_tensor(np.asarray(node_types[global_nodes]), dtype=torch.long, device=device)
            x = x + self.type_embedding(types)
        return x

    def forward(
        self,
        graph: BlockConditionalGraph,
        block1: SampledBlock,
        block2: SampledBlock,
        device: torch.device,
    ) -> Tensor:
        block1 = block1.to(device)
        block2 = block2.to(device)
        x0 = self.input_features(block1.src_nodes, graph.node_types, device)
        h1 = F.relu(self.conv1(x0, block1))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        return self.conv2(h1, block2)
