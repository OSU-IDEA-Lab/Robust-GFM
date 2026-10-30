#!/usr/bin/env python3
"""Shared time/space instrumentation for the runners.

Records, beside task metrics and Kendall-tau: train time, epochs, parameter size (MiB),
peak training and inference GPU memory (MiB) and inference time. Peak memory and
inference time are runtime measurements, so they are captured during the run.
"""
from __future__ import annotations

from typing import Any

import torch


def bytes_to_mib(value: int | float | None) -> float | None:
    return None if value is None else float(value) / (1024.0 ** 2)


def model_size(model: torch.nn.Module) -> dict[str, Any]:
    """Parameter/buffer counts and size in MiB, at the tensors' real dtype."""
    p_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    b_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameter_count": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "parameter_bytes": int(p_bytes),
        "buffer_bytes": int(b_bytes),
        "parameter_mib": bytes_to_mib(p_bytes),
        "static_model_mib": bytes_to_mib(p_bytes + b_bytes),
    }


def gpu_reset(device: torch.device | str | None) -> None:
    """Start a fresh peak-memory measurement window."""
    if device is None:
        return
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)


def gpu_peak(device: torch.device | str | None) -> dict[str, Any]:
    """Peak allocated/reserved GPU memory since the last reset, in MiB."""
    if device is None:
        return {"peak_gpu_mib": None, "peak_gpu_reserved_mib": None}
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return {"peak_gpu_mib": None, "peak_gpu_reserved_mib": None}
    torch.cuda.synchronize(dev)
    return {
        "peak_gpu_mib": bytes_to_mib(torch.cuda.max_memory_allocated(dev)),
        "peak_gpu_reserved_mib": bytes_to_mib(torch.cuda.max_memory_reserved(dev)),
    }


def resource_record(
    model: torch.nn.Module,
    device: torch.device | str | None,
    *,
    train_time_sec: float | None = None,
    inference_time_sec: float | None = None,
    peak_train_gpu_mib: float | None = None,
    peak_train_gpu_reserved_mib: float | None = None,
    peak_inference_gpu_mib: float | None = None,
    peak_inference_gpu_reserved_mib: float | None = None,
    epochs_run: int | None = None,
    best_epoch: int | None = None,
) -> dict[str, Any]:
    """One record with the field names the Freebase runners already emit."""
    record = dict(model_size(model))
    record.update(
        {
            "train_time_sec": train_time_sec,
            "inference_time_sec": inference_time_sec,
            "peak_training_gpu_mib": peak_train_gpu_mib,
            "peak_training_reserved_gpu_mib": peak_train_gpu_reserved_mib,
            "peak_inference_gpu_mib": peak_inference_gpu_mib,
            "peak_inference_reserved_gpu_mib": peak_inference_gpu_reserved_mib,
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "device": str(device) if device is not None else None,
        }
    )
    return record
