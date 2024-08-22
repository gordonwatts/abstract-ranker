from unittest.mock import patch
import pytest


def pytest_configure(config):
    """Add custom markers to pytest.

    Args:
        config (_type_): _description_
    """
    config.addinivalue_line(
        "markers",
        "requires_pytorch: mark test to be skipped if pytorch is not installed",
    )


def is_torch_installed():
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def pytest_collection_modifyitems(config, items):
    """Skip tests that require pytorch if it is not installed.

    Args:
        config (_type_): _description_
        items (_type_): _description_
    """
    if not is_torch_installed():
        skip_pytorch = pytest.mark.skip(reason="pytorch is not installed")
        for item in items:
            if "requires_pytorch" in item.keywords:
                item.add_marker(skip_pytorch)


@pytest.fixture(autouse=True)
def cache_dir(tmp_path):
    # Create a temporary test directory
    cache_dir = tmp_path / "cache_dir"
    cache_dir.mkdir()

    with patch("abstract_ranker.config.CACHE_DIR", cache_dir):
        yield cache_dir


@pytest.fixture(autouse=True)
def setup_before_test():
    if is_torch_installed():
        "Reset state"
        from abstract_ranker.local_llms import reset

        reset()
