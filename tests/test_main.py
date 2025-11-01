from os import environ
from pathlib import Path
from typing import Any

import array_api_extra as xpx
import numpy as np
import pytest
from array_api._2024_12 import ArrayNamespaceFull
from joblib import Memory
from ultrasphere import create_from_branching_types

from biem_helmholtz_sphere.bempp_cl_sphere import bempp_cl_sphere
from biem_helmholtz_sphere.biem import BIEMResultCalculator, biem, plane_wave
from biem_helmholtz_sphere.plot import plot_biem

memory = Memory("tests/.cache/joblib", verbose=0)
bempp_cl_sphere = memory.cache(bempp_cl_sphere)
IS_CI = environ.get("CI") in ("true", "1", "yes")


@pytest.mark.parametrize("branching_types", ["a", "ba"])
def test_biem(xp: ArrayNamespaceFull, branching_types: str, device: Any, dtype: Any) -> None:
    c = create_from_branching_types(branching_types)
    uin, uin_grad = plane_wave(
        k=xp.asarray(1.0),
        direction=xp.asarray((1.0,) + (0.0,) * (c.c_ndim - 1), device=device, dtype=dtype),
    )
    calc: BIEMResultCalculator[Any, Any] = biem(
        c,
        uin=uin,
        uin_grad=uin_grad,
        k=xp.asarray(1.0, device=device, dtype=dtype),
        n_end=6,
        eta=xp.asarray(1.0, device=device, dtype=dtype),
        centers=xp.asarray(
            (
                (
                    0.0,
                    2.0,
                )
                + (0.0,) * (c.c_ndim - 2),
                (
                    0.0,
                    -2.0,
                )
                + (0.0,) * (c.c_ndim - 2),
            ),
            device=device,
            dtype=dtype,
        ),
        radii=xp.asarray((1.0, 1.0), device=device, dtype=dtype),
        kind="outer",
    )
    calc.uscat(xp.asarray((4,) + (0,) * (c.c_ndim - 1), device=device, dtype=dtype))
    fig = plot_biem(calc, n_t=10, xspace=(-4, 4, 100), yspace=(-4, 4, 100))
    # save plotly figure to html
    Path("tests/.cache").mkdir(exist_ok=True)
    fig.write_html(f"tests/.cache/test_biem_{branching_types}.html")
    fig.write_image(f"tests/.cache/test_biem_{branching_types}.png", scale=3)


@pytest.mark.parametrize("h, rtol", [(0.3, 2e-1)])
@pytest.mark.parametrize("n_spheres", [1, 3])
@pytest.mark.parametrize("alpha, beta", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
def test_match(
    xp: ArrayNamespaceFull,
    n_spheres: int,
    h: float,
    rtol: float,
    alpha: complex,
    beta: complex,
    device: Any,
    dtype: Any,
) -> None:
    k = xp.random.random_uniform(0.5, 2.0, (), device=device, dtype=dtype)
    for _ in range(100):
        centers = xp.random.random_uniform(-1, 1, (n_spheres, 3), device=device, dtype=dtype)
        radii = xp.random.random_uniform(0.1, 0.2, (n_spheres,), device=device, dtype=dtype)
        no_touch = (
            xp.linalg.vector_norm(centers[:, None, :] - centers[None, :, :], axis=-1)
            >= (radii[:, None] + radii[None, :]) * 1.1
        )
        if xp.all(xp.eye(n_spheres, dtype=xp.bool, device=device) | no_touch):
            break
    else:
        raise RuntimeError("Failed to generate non-overlapping spheres")
    x = xp.random.random_uniform(-1, 1, (3, 100), device=device, dtype=dtype)
    x = x[
        :,
        (
            xp.linalg.vector_norm(x.T[:, None, :] - centers[None, :, :], axis=-1) > radii[None, :]
        ).all(axis=-1),
    ]
    uin, uin_grad = plane_wave(
        k=k, direction=xp.asarray((1.0, 0.0, 0.0), device=device, dtype=dtype)
    )
    calc: BIEMResultCalculator[Any, Any] = biem(
        create_from_branching_types("ba"),
        uin=uin,
        uin_grad=uin_grad,
        k=k,
        n_end=10,
        eta=xp.asarray(1.0),
        centers=centers,
        radii=radii,
        kind="outer",
        alpha=alpha,
        beta=beta,
    )
    uscat_actual = calc.uscat(x)
    calc_expected = bempp_cl_sphere(
        k=float(k),
        h=h,
        centers=np.asarray(centers),
        radii=np.asarray(radii),
        alpha=alpha,
        beta=beta,
    )
    uscat_expected = calc_expected(x[0, ...], x[1, ...], x[2, ...])
    uscat_expected = xp.asarray(uscat_expected, device=device, dtype=dtype)

    assert (
        xp.mean(xp.astype(xpx.isclose(uscat_actual, uscat_expected, rtol=rtol), xp.float64)) > 0.8
    )
