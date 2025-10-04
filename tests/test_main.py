from typing import Any

import pytest
from array_api._2024_12 import ArrayNamespaceFull
from ultrasphere import create_from_branching_types

from biem_helmholtz_sphere.biem import BIEMResultCalculator, biem, plane_wave


@pytest.mark.parametrize("branching_types", ["a", "ba"])
def test_biem(xp: ArrayNamespaceFull, branching_types: str) -> None:
    c = create_from_branching_types(branching_types)
    calc: BIEMResultCalculator[Any, Any] = biem(
        c,
        plane_wave(xp.asarray(1.0), xp.asarray((1,) + (0,) * (c.c_ndim - 1))),
        k=xp.asarray(1.0),
        n_end=3,
        eta=None,
        centers=xp.asarray(((1,) + (0,) * c.s_ndim, (-1,) + (0,) * c.s_ndim)),
        radii=xp.asarray((0.5, 0.5)),
        kind="outer",
    )
    calc.uscat(xp.asarray((4,) + (0,) * (c.c_ndim - 1)))
