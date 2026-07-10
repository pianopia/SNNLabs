from __future__ import annotations

from torch.utils.data import Dataset

from benchmarks.neuromorphic.datasets import dataset_targets
from benchmarks.neuromorphic.run_nmnist import _random_split_from_dataset, _stratified_split_from_dataset


class _IndexDataset(Dataset):
    def __len__(self) -> int:
        return 100

    def __getitem__(self, index: int) -> int:
        return index


class _TargetDataset(_IndexDataset):
    def __init__(self) -> None:
        self.targets = [0] * 20 + [1] * 20 + [2] * 20

    def __len__(self) -> int:
        return len(self.targets)


def _indices(subset) -> list[int]:
    return list(subset.indices)


def test_random_split_is_seeded_and_not_prefix_only():
    train_a, test_a = _random_split_from_dataset(_IndexDataset(), 8, 8, seed=123)
    train_b, test_b = _random_split_from_dataset(_IndexDataset(), 8, 8, seed=123)

    assert _indices(train_a) == _indices(train_b)
    assert _indices(test_a) == _indices(test_b)
    assert _indices(train_a) != list(range(8))
    assert set(_indices(train_a)).isdisjoint(_indices(test_a))


def test_dataset_targets_reads_cached_targets():
    assert dataset_targets(_TargetDataset())[:3] == [0, 0, 0]


def test_stratified_split_uses_multiple_classes_from_sorted_targets():
    dataset = _TargetDataset()
    train, test = _stratified_split_from_dataset(dataset, 12, 9, seed=123)
    train_labels = [dataset.targets[index] for index in _indices(train)]
    test_labels = [dataset.targets[index] for index in _indices(test)]

    assert set(train_labels) == {0, 1, 2}
    assert set(test_labels) == {0, 1, 2}
    assert set(_indices(train)).isdisjoint(_indices(test))
