"""Canonical devices and optional CUDA allocator telemetry.

CUDA ordinals are logical indices AFTER CUDA_VISIBLE_DEVICES filtering. Never
interpret them as physical GPU IDs or silently replace an invalid device.
"""
import logging

import torch

logger = logging.getLogger(__name__)
DeviceSpec = str | int | torch.device


def normalize_device(device: DeviceSpec) -> torch.device:
    """Resolve strings, torch.device and integer CUDA ordinals consistently.

    Explicit CUDA ordinals do not initialize/query hardware here, which also
    permits offline tests. Bare 'cuda' resolves the current logical device once.
    Hardware validation and selection belong to select_device().
    """
    if isinstance(device, bool):
        raise ValueError("Boolean values are not device specifications")
    if isinstance(device, int):
        if device < 0:
            raise ValueError("CUDA device index must be nonnegative")
        device = torch.device("cuda", device)
    else:
        try:
            device = torch.device(device)
        except (TypeError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid device specification: {device!r}") from exc
    if device.type == "cpu":
        return torch.device("cpu")
    if device.type != "cuda":
        raise ValueError(f"Only CPU and CUDA are supported, got {device}")
    index = device.index
    if index is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --smoke-test --device cpu for local tests")
        index = torch.cuda.current_device()
    return torch.device("cuda", index)


def select_device(device: DeviceSpec) -> torch.device:
    """Validate visible CUDA ordinals and initialize/select exactly that device."""
    device = normalize_device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --smoke-test --device cpu for local tests")
        count = torch.cuda.device_count()
        if device.index >= count:
            raise ValueError(f"Invalid CUDA device {device}: {count} logical devices visible. "
                             "Indices are relative to CUDA_VISIBLE_DEVICES.")
        # set_device initializes CUDA even for a parameterless synthetic encoder.
        torch.cuda.set_device(device.index)
    return device


class CudaMemoryProfiler:
    """Best-effort peak allocation profiling; compute/transfer errors stay fatal.

    Only the optional telemetry APIs are guarded. A failed reset disables reads
    so a stale process-wide peak is never presented as a stage measurement.
    """
    def __init__(self, device: DeviceSpec):
        self.device = normalize_device(device)
        self.active = False
        self.errors: list[str] = []

    def _failed(self, operation, exc):
        self.active = False
        message = f"{operation} on {self.device}: {type(exc).__name__}: {exc}"
        self.errors.append(message)
        logger.warning("Optional CUDA memory profiling unavailable: %s", message)

    def start(self) -> bool:
        self.active = False
        if self.device.type != "cuda":
            return False
        try:
            # reset_peak_memory_stats itself need not trigger lazy initialization.
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(self.device.index)
        except Exception as exc:
            self._failed("reset_peak_memory_stats", exc)
        else:
            self.active = True
        return self.active

    def peak_bytes(self) -> int | None:
        if not self.active:
            return None
        try:
            peak = int(torch.cuda.max_memory_allocated(self.device.index))
            if peak < 0:
                raise ValueError("Negative peak allocation")
            return peak
        except Exception as exc:
            self._failed("max_memory_allocated", exc)
            return None

    def metadata(self) -> dict:
        return {"device": str(self.device), "available": self.active,
                "errors": list(self.errors)}
