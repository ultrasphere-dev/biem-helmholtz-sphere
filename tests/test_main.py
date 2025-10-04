from pathlib import Path
from typing import Any

import pytest
from array_api._2024_12 import ArrayNamespaceFull
from ultrasphere import create_from_branching_types

from biem_helmholtz_sphere.biem import BIEMResultCalculator, biem, plane_wave
from biem_helmholtz_sphere.plot import plot_biem


@pytest.mark.parametrize("branching_types", ["a", "ba"])
def test_biem(xp: ArrayNamespaceFull, branching_types: str) -> None:
    c = create_from_branching_types(branching_types)
    calc: BIEMResultCalculator[Any, Any] = biem(
        c,
        plane_wave(k=xp.asarray(1.0), direction=xp.asarray((0.5,) + (0,) * (c.c_ndim - 1))),
        k=xp.asarray(1.0),
        n_end=7,
        eta=None,
        centers=xp.asarray(((0.5,) + (0,) * c.s_ndim, (-0.5,) + (0,) * c.s_ndim)),
        radii=xp.asarray((0.25, 0.25)),
        kind="outer",
    )
    calc.uscat(xp.asarray((4,) + (0,) * (c.c_ndim - 1)))
    fig = plot_biem(calc, n_t=10, xspace=(-1, 1, 100), yspace=(-1, 1, 100))
    # save plotly figure to html
    Path("tests/.cache").mkdir(exist_ok=True)
    fig.write_html(f"tests/.cache/test_biem_{branching_types}.html")
    fig.write_image(f"tests/.cache/test_biem_{branching_types}.png", scale=3)
