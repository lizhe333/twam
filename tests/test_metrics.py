import torch

from contact_belief_wam.metrics import Metrics


class _ReferenceMetrics:
    """The trainer's previous per-group accumulation, kept as a test oracle."""

    def __init__(self):
        self.values = {}

    def add(self, output, sources, tasks=None):
        for key, item in output["metrics"].items():
            groups = {
                group: [group == "all" or source == group for source in sources]
                for group in ("all", "original", "recovery")
            }
            if tasks is not None:
                for task in sorted(set(tasks)):
                    groups[f"tasks/{task}"] = [value == task for value in tasks]
                    for source in ("original", "recovery"):
                        groups[f"tasks/{task}/{source}"] = [
                            task_value == task and current_source == source
                            for task_value, current_source in zip(tasks, sources)
                        ]
            for group, selection in groups.items():
                selected = torch.tensor(selection, device=item["sum"].device, dtype=torch.bool)
                label = f"{group}/{key}"
                pair = self.values.setdefault(label, [0.0, 0.0])
                pair[0] += float(item["sum"][selected].sum())
                pair[1] += float(item["count"][selected].sum())

    def result(self):
        return {key: total / count for key, (total, count) in self.values.items() if count > 0}


def _batch(sources, tasks, offset=0):
    size = len(sources)
    indices = torch.arange(size, dtype=torch.float32)
    return {
        "metrics": {
            "flow": {
                "sum": indices + 1 + offset,
                "count": torch.tensor([1, 1, 0, 2, 0][:size], dtype=torch.float32),
            },
            "future": {
                "sum": (indices + 2 + offset).reshape(size, 1),
                "count": torch.tensor([0, 1, 1, 0, 2][:size], dtype=torch.float32).reshape(size, 1),
            },
        },
        "sources": sources,
        "tasks": tasks,
    }


def _compare_with_reference(batches):
    actual, expected = Metrics(), _ReferenceMetrics()
    for batch in batches:
        actual.add(batch, batch["sources"], batch.get("tasks"))
        expected.add(batch, batch["sources"], batch.get("tasks"))
    assert actual.result() == expected.result()


def test_metrics_match_reference_for_sources_tasks_and_zero_counts():
    batch = _batch(
        ["original", "recovery", "other", "original", "recovery"],
        ["hole", "hdmi", "hole", "hdmi", "hole"],
    )
    _compare_with_reference([batch])


def test_metrics_match_reference_for_missing_sources_and_repeated_adds():
    first = _batch(
        ["original", "recovery", "original", "recovery", "other"],
        ["hole", "hole", "hdmi", "hdmi", "hole"],
    )
    # The second batch omits recovery entirely and has a task with zero valid
    # future count; result() should keep the same filtering behavior.
    second = _batch(["original", "other"], ["insert", "insert"], offset=10)
    _compare_with_reference([first, second])


def test_metrics_without_task_labels_match_reference():
    batch = _batch(["original", "recovery", "other"], None)
    _compare_with_reference([batch])

