import pytest
from array_api._2024_12 import ArrayNamespaceFull


@pytest.fixture(scope="session", params=["numpy"])
def xp(request: pytest.FixtureRequest) -> ArrayNamespaceFull:
    """Get the array namespace for the given backend."""
    backend = request.param
    if backend == "numpy":
        from array_api_compat import numpy as xp

        rng = xp.random.default_rng()

        def random_uniform(low=0, high=1, shape=None, dtype=None):
            return rng.random(shape, dtype) * (high - low) + low

        def integers(low, high=None, shape=None):
            return rng.integers(low, high, size=shape)

        xp.random.random_uniform = random_uniform
        xp.random.integers = integers
    elif backend == "torch":
        from array_api_compat import torch as xp

        def random_uniform(low=0, high=1, shape=None, dtype=None):
            return xp.rand(shape, dtype=dtype) * (high - low) + low

        def integers(low, high=None, shape=None):
            return xp.randint(low, high, size=shape)

        xp.random.random_uniform = random_uniform
        xp.random.integers = integers
        import torch

        if torch.cuda.is_available():
            xp.set_default_device("cuda")
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return xp
