import numpy as np
import torch

from contact_belief_wam.sequence import prepare_batch


class _RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, prompts, states):
        self.calls.append((list(prompts), np.asarray(states).copy()))
        count = len(prompts)
        ids = torch.arange(count * 3, dtype=torch.long).reshape(count, 3)
        mask = torch.ones(count, 3, dtype=torch.bool)
        return ids, mask


def _batch():
    batch, length, height, width = 2, 3, 12, 16
    image = np.arange(batch * length * height * width * 3, dtype=np.uint8).reshape(
        batch, length, height, width, 3
    )
    return {
        "image": image,
        "future_image": image[::-1].copy(),
        "state": np.arange(batch * length * 8, dtype=np.float32).reshape(batch, length, 8),
        "prompt": ["insert hole", "insert HDMI"],
        "source": ["original", "recovery"],
        "task": ["insert_hole", "insert_HDMI"],
    }


def test_prepare_batch_without_language_keeps_other_preprocessing_and_shapes():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        batch = _batch()
        tokenizer = _RecordingTokenizer()
        with_language = prepare_batch(batch, tokenizer, "cpu")
        calls_before = len(tokenizer.calls)

        class FailIfCalled:
            def __call__(self, *args, **kwargs):
                raise AssertionError("tokenizer should not be called when include_language=False")

        without_language = prepare_batch(batch, FailIfCalled(), "cpu", include_language=False)

        assert len(tokenizer.calls) == calls_before == 1
        assert without_language["tokens"].shape == (2, 3, 1)
        assert without_language["tokens"].dtype == torch.long
        assert torch.count_nonzero(without_language["tokens"]) == 0
        assert without_language["token_mask"].shape == (2, 3, 1)
        assert without_language["token_mask"].dtype == torch.bool
        assert not without_language["token_mask"].any()

        for key in batch:
            if key in {"tokens", "token_mask"}:
                continue
            first, second = with_language[key], without_language[key]
            if torch.is_tensor(first):
                torch.testing.assert_close(first, second)
            elif isinstance(first, np.ndarray):
                np.testing.assert_array_equal(first, second)
            else:
                assert first == second
    finally:
        torch.set_num_threads(previous_threads)
