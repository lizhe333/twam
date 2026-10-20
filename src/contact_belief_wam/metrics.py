"""Metric aggregation helpers used by training and debug runs."""
from __future__ import annotations

import torch


class Metrics:
    """Accumulate per-sample metric sums/counts with one device-to-CPU copy.

    ``output['metrics']`` values contain ``sum`` and ``count`` tensors whose
    leading dimension indexes the batch. All metric values are packed together
    on their existing device, copied to CPU once, then reduced by source/task
    groups on CPU. The resulting labels and weighted means match the trainer's
    previous ``Metrics`` implementation.
    """

    def __init__(self):
        self.values: dict[str, list[float]] = {}

    @staticmethod
    def _reduce_per_sample(value, batch_size: int, key: str, field: str):
        value = torch.as_tensor(value).detach()
        if value.ndim == 0:
            if batch_size != 1:
                raise ValueError(
                    f"Metric {key!r} {field!r} is scalar for batch size {batch_size}"
                )
            return value.reshape(1)
        if value.shape[0] != batch_size:
            raise ValueError(
                f"Metric {key!r} {field!r} has leading size {value.shape[0]}, "
                f"expected batch size {batch_size}"
            )
        return value.reshape(batch_size, -1).sum(dim=1)

    @staticmethod
    def _groups(sources, tasks):
        batch_size = len(sources)
        groups = [
            ("all", list(range(batch_size))),
            ("original", [index for index, source in enumerate(sources) if source == "original"]),
            ("recovery", [index for index, source in enumerate(sources) if source == "recovery"]),
        ]
        if tasks is not None:
            for task in sorted(set(tasks)):
                task_indices = [index for index, value in enumerate(tasks) if value == task]
                groups.append((f"tasks/{task}", task_indices))
                paired = list(zip(tasks, sources))
                for source in ("original", "recovery"):
                    indices = [index for index, (value, current_source) in enumerate(paired)
                               if value == task and current_source == source]
                    groups.append((f"tasks/{task}/{source}", indices))
        return groups

    def add(self, output, sources, tasks=None):
        metrics = output.get("metrics", {})
        if not metrics:
            return
        sources = list(sources)
        batch_size = len(sources)

        keys = list(metrics)
        per_metric_sums, per_metric_counts = [], []
        device = None
        for key in keys:
            item = metrics[key]
            sums = self._reduce_per_sample(item["sum"], batch_size, key, "sum")
            counts = self._reduce_per_sample(item["count"], batch_size, key, "count")
            if sums.device != counts.device:
                counts = counts.to(device=sums.device)
            if device is None:
                device = sums.device
            elif sums.device != device:
                raise ValueError("All metrics in one batch must be on the same device")
            per_metric_sums.append(sums)
            per_metric_counts.append(counts)

        packed = torch.stack(
            (torch.stack(per_metric_sums), torch.stack(per_metric_counts)), dim=1
        )
        packed = packed.to(device="cpu")
        sum_values, count_values = packed[:, 0], packed[:, 1]

        group_values = self._groups(sources, tasks)
        selections = [torch.tensor(indices, dtype=torch.long) for _, indices in group_values]
        for metric_index, key in enumerate(keys):
            for (group, _), indices in zip(group_values, selections):
                label = f"{group}/{key}"
                pair = self.values.setdefault(label, [0.0, 0.0])
                if indices.numel() == 0:
                    continue
                pair[0] += float(sum_values[metric_index].index_select(0, indices).sum())
                pair[1] += float(count_values[metric_index].index_select(0, indices).sum())

    def result(self):
        return {key: total / count for key, (total, count) in self.values.items() if count > 0}
