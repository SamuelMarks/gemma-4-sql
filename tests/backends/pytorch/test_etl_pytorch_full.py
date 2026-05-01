
import pytest
@pytest.fixture(autouse=True)
def clean_sys_modules():
    import sys
    keys = list(sys.modules.keys())
    yield
    for k in list(sys.modules.keys()):
        if k not in keys and "gemma_4_sql" in k:
            del sys.modules[k]
"""
Tests for PyTorch-specific ETL pipeline (full implementation).
"""

import sys

import pytest


class MockDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MockDatasets:
    def load_dataset(self, name, split):
        return MockDataset(
            [
                {"question": "What is 1?", "query": "SELECT 1"},
                {"question": "What is 2?", "query": "SELECT 2"},
            ]
        )


class MockTensor:
    def __init__(self, data, dtype=None):
        self.data = data
        self.dtype = dtype

    def __repr__(self):
        return f"MockTensor({self.data})"

    def __len__(self):
        return len(self.data)


class MockNNUtilsRNN:
    @staticmethod
    def pad_sequence(sequences, batch_first=False):
        return sequences


class MockNNUtils:
    rnn = MockNNUtilsRNN()


class MockNN:
    utils = MockNNUtils()


class MockTorch:
    long = "long"
    nn = MockNN()

    @staticmethod
    def tensor(data, dtype=None):
        return MockTensor(data, dtype)


class MockDataLoader:
    def __init__(self, dataset, batch_size, shuffle, collate_fn):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn


class MockDatasetClass:
    pass


@pytest.fixture
def mock_pytorch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture to mock datasets and torch."""
    monkeypatch.setitem(sys.modules, "datasets", MockDatasets())
    monkeypatch.setitem(sys.modules, "torch", MockTorch())
    monkeypatch.setitem(
        sys.modules,
        "torch.utils.data",
        type("mock", (), {"DataLoader": MockDataLoader, "Dataset": MockDatasetClass}),
    )

    # Reload the module so it picks up the mocked sys.modules
    if "gemma_4_sql.backends.pytorch.etl" in sys.modules:
        del sys.modules["gemma_4_sql.backends.pytorch.etl"]


def test_build_dataloader_pytorch_loaded(mock_pytorch_env: None) -> None:
    """Test PyTorch build_dataloader when libraries are present."""
    from gemma_4_sql.backends.pytorch.etl import build_dataloader

    res = build_dataloader("dummy/data", "train", 16, False)
    assert res["backend"] == "pytorch"
    assert res["status"] == "loaded"
    assert res["batch_size"] == 16
    assert hasattr(res["loader"], "collate_fn")

    # Test __getitem__ of PyTorchDataset
    dataset = res["loader"].dataset
    assert len(dataset) == 2
    item = dataset[0]
    assert "inputs" in item
    assert "targets" in item

    # Test collate function manually
    batch = [
        {"inputs": MockTensor([1, 2]), "targets": MockTensor([3, 4])},
        {"inputs": MockTensor([5, 6, 7]), "targets": MockTensor([8])},
    ]
    collate_fn = res["loader"].collate_fn
    collated = collate_fn(batch)
    assert len(collated["inputs"]) == 2
    assert len(collated["targets"]) == 2
