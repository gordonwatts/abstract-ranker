from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def cache_dir(tmp_path):
    # Create a temporary test directory
    cache_dir = tmp_path / "cache_dir"
    cache_dir.mkdir()

    with patch("abstract_ranker.config.CACHE_DIR", cache_dir):
        yield cache_dir


@pytest.fixture()
def setup_before_test():
    "Reset state"
    from abstract_ranker.local_llms import reset

    reset()
