import pytest


@pytest.fixture(autouse=True)
def setup_before_test():
    "Reset state"
    from abstract_ranker.local_llms import reset

    reset()
