from collections.abc import Sequence
from typing import Any

import numpy as np
from bempp_cl.api import GridFunction, complex_callable, function_space
from bempp_cl.api.grid import union
from bempp_cl.api.linalg import gmres
from bempp_cl.api.operators.boundary.helmholtz import single_layer
from bempp_cl.api.operators.potential.helmholtz import single_layer as single_layer_potential
from bempp_cl.api.shapes import sphere
from numpy.typing import NDArray

centers = [
    [0.5, 0.0, 0.0],
    [-0.5, 0.0, 0.0],
]
radii = [0.25, 0.25]


def bempp_cl(
    *,
    k: float,
    centers: Sequence[Sequence[float]],
    radii: Sequence[float],
    x: NDArray[Any],
    y: NDArray[Any],
    z: NDArray[Any],
) -> NDArray[Any]:
    x, y, z = np.broadcast_arrays(x, y, z)
    centers_ = np.asarray(centers)
    radii_ = np.asarray(radii)
    if radii_.ndim != 1:
        raise ValueError("radii must be 1-dimensional")
    if centers_.ndim != 2:
        raise ValueError("centers must be 2-dimensional")
    if centers_.shape[0] != radii_.shape[0]:
        raise ValueError("centers and radii must have the same length")
    if centers_.shape[0] == 0:
        raise ValueError("centers and radii must have length > 0")
    if centers_.shape[1] != 3:
        raise ValueError("centers must have shape (N, 3)")

    grid = union(
        [
            sphere(h=0.2, origin=center, r=radius)
            for center, radius in zip(centers, radii, strict=False)
        ]
    )
    space = function_space(grid, "DP", 0)
    lhs = single_layer(space, space, space, k)

    @complex_callable
    def f(x, n, domain_index, result):
        result[0] = -np.exp(1j * k * x[0])

    rhs = GridFunction(space, fun=f)
    neumann_fun, _ = gmres(lhs, rhs, tol=1e-5)
    x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), indexing="ij")
    points = np.stack((x, y, z), axis=0)
    slp = single_layer_potential(space, points, k)
    val = slp * neumann_fun
    return val
