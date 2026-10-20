"""Rank-local training timings; CUDA events are resolved only at log intervals."""
from collections import defaultdict
from contextlib import contextmanager
import time

import torch


class TrainingTiming:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.micro_batches = 0
        self.host = defaultdict(float)
        self.events = defaultdict(list)

    def add_wall(self, name, seconds):
        if self.enabled:
            self.host[name] += seconds

    @contextmanager
    def wall(self, name):
        start = time.monotonic()
        try:
            yield
        finally:
            self.add_wall(name, time.monotonic() - start)

    @contextmanager
    def cuda(self, name):
        if not self.enabled:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events[name].append((start, end))

    def result_and_reset(self):
        # One interval boundary synchronization, never one per measured phase.
        # Stream elapsed time includes launch gaps and communication, not just kernels.
        if self.events:
            torch.cuda.current_stream().synchronize()
        count = max(1, self.micro_batches)
        result = {f"timing/rank0/{name}_host_seconds": total / count
                  for name, total in self.host.items()}
        result.update({f"timing/rank0/{name}_cuda_seconds":
                       sum(start.elapsed_time(end) for start, end in events) / (1000 * count)
                       for name, events in self.events.items()})
        self.host.clear()
        self.events.clear()
        self.micro_batches = 0
        return result
