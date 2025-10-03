from ultrasphere import create_from_branching_types
from biem_helmholtz_sphere.biem import biem, plane_wave
from array_api._2024_12 import ArrayNamespaceFull
import pytest

@pytest.mark.parametrize("branching_types", ["a", "ba"])
def test_biem(xp: ArrayNamespaceFull, branching_types: str) -> None:
    c = create_from_branching_types(branching_types)
    calc = biem(
        c,
        plane_wave(xp.asarray(1.0), xp.asarray((1,) + (0,) * (c.c_ndim - 1))),
        k=xp.asarray(1.0),
        n_end=3,
        eta=None,
        centers=xp.asarray(((1,) + (0,) * c.s_ndim, (-1,) + (0,) * c.s_ndim)),
        radii=xp.asarray((0.5, 0.5)),
        kind="outer",
    )