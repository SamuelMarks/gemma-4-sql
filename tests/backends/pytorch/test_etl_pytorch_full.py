"""Tests for PyTorch-specific ETL pipeline (full implementation)."""

import sys

import pytest


@pytest.fixture(autouse=True)
def _clean_sys_modules() -> object:
    """Initialize function clean_sys_modules."""
    sys = __import__("sys")
    keys = list(sys.modules.keys())
    yield
    for k in list(sys.modules.keys()):
        if k not in keys and "gemma_4_sql" in k:
            del sys.modules[k]


class MockDataset:
    """Initialize class MockDataset."""

    def __init__(self, data: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        data: Description of data.

        """
        self.data = data

    def __len__(self) -> object:
        """Initialize function __len__."""
        return len(self.data)

    def __getitem__(self, idx: object) -> object:
        """Initialize function __getitem__.

        Args:
        ----
        idx: Description of idx.

        """
        return self.data[idx]


class MockDatasets:
    """Initialize class MockDatasets."""

    def load_dataset(*_args: object, **_kwargs: object) -> object:
        """Initialize function load_dataset.

        Args:
        ----
        name: Description of name.
        split: Description of split.

        """
        return MockDataset([{"question": "What is 1?", "query": "SELECT 1"}, {"question": "What is 2?", "query": "SELECT 2"}])


class MockTensor:
    """Initialize class MockTensor."""

    def __init__(self, data: object, dtype: object = None) -> None:
        """Initialize function __init__.

        Args:
        ----
        data: Description of data.
        dtype: Description of dtype.

        """
        self.data = data
        self.dtype = dtype

    def __repr__(self) -> object:  # type: ignore[override]
        """Initialize function __repr__."""
        return f"MockTensor({self.data})"

    def __len__(self) -> object:
        """Initialize function __len__."""
        return len(self.data)


class MockNNUtilsRNN:
    """Initialize class MockNNUtilsRNN."""

    @staticmethod
    def pad_sequence(sequences: object, *, _batch_first: object = False, **_kwargs: object) -> object:
        """Initialize function pad_sequence.

        Args:
        ----
        sequences: Description of sequences.
        batch_first: Description of batch_first.

        """
        return sequences


class MockNNUtils:
    """Initialize class MockNNUtils."""

    rnn = MockNNUtilsRNN()


class MockNN:
    """Initialize class MockNN."""

    utils = MockNNUtils()


class MockTorch:
    """Initialize class MockTorch."""

    long = "long"
    nn = MockNN()

    @staticmethod
    def tensor(_cls: object, *args: object, **kwargs: object) -> object:
        """Initialize function tensor."""
        return MockTensor(kwargs.get("data") if "data" in kwargs else (args[0] if args else getattr(_cls, "data", _cls)))


class MockDataLoader:
    """Initialize class MockDataLoader."""

    def __init__(self, dataset: object, batch_size: object, shuffle: object, collate_fn: object, sampler: object = None, **kwargs: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        dataset: Description of dataset.
        batch_size: Description of batch_size.
        shuffle: Description of shuffle.
        collate_fn: Description of collate_fn.
        sampler: Description of sampler.
        kwargs: Description of kwargs.

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn


class MockDatasetClass:
    """Initialize class MockDatasetClass."""


@pytest.fixture
def _mock_pytorch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture to mock datasets and torch."""
    monkeypatch.setitem(sys.modules, "datasets", MockDatasets())
    monkeypatch.setitem(sys.modules, "torch", MockTorch())
    monkeypatch.setitem(sys.modules, "torch.utils.data", type("mock", (), {"DataLoader": MockDataLoader, "Dataset": MockDatasetClass}))
    if "gemma_4_sql.backends.pytorch.etl" in sys.modules:
        del sys.modules["gemma_4_sql.backends.pytorch.etl"]


@pytest.mark.usefixtures("_mock_pytorch_env")
def test_build_dataloader_pytorch_loaded() -> None:
    """Test PyTorch build_dataloader when libraries are present."""
    build_dataloader = __import__("gemma_4_sql.backends.pytorch.etl", fromlist=["build_dataloader"]).build_dataloader
    res = build_dataloader("dummy/data", "train", 16, distributed=False)
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["status"] == "loaded":
        raise AssertionError
    if not res["batch_size"] == int("16"):
        raise AssertionError
    if not hasattr(res["loader"], "collate_fn"):
        raise AssertionError
    dataset = res["loader"].dataset
    if not len(dataset) == int("2"):
        raise AssertionError
    item = dataset[0]
    if "inputs" not in item:
        raise AssertionError
    if "targets" not in item:
        raise AssertionError
    batch = [{"inputs": MockTensor([1, 2]), "targets": MockTensor([3, 4])}, {"inputs": MockTensor([5, 6, 7]), "targets": MockTensor([8])}]
    collate_fn = res["loader"].collate_fn
    collated = collate_fn(batch)
    if not len(collated["inputs"]) == int("2"):
        raise AssertionError
    if not len(collated["targets"]) == int("2"):
        raise AssertionError
