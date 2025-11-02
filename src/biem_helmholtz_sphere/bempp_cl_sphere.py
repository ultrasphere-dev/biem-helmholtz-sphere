from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from bempp_cl.api import GridFunction, complex_callable, function_space
from bempp_cl.api.grid import union
from bempp_cl.api.linalg import gmres
from bempp_cl.api.operators.boundary.helmholtz import adjoint_double_layer, single_layer
from bempp_cl.api.operators.boundary.sparse import identity
from bempp_cl.api.operators.potential.helmholtz import single_layer as single_layer_potential
from bempp_cl.api.shapes import sphere
from numpy.typing import NDArray


def bempp_cl_sphere(
    *,
    k: float,
    h: float,
    centers: Sequence[Sequence[float]],
    radii: Sequence[float],
    alpha: complex = 1.0,
    beta: complex = 0.0,
) -> Callable[[NDArray[Any], NDArray[Any], NDArray[Any]], NDArray[Any]]:
    """
    Calculate the scattered field by multiple spheres using bempp-cl.

    Uses OBIE, not Burton-Miller formulation.

    Parameters
    ----------
    k : float
        The wavenumber.
    h : float
        The element size for the mesh.
    centers : Sequence[Sequence[float]]
        The centers of the spheres of shape (B, 3).
    radii : Sequence[float]
        The radii of the spheres of shape (B,).
    alpha : complex, optional
        The coefficient for the dirichlet part, by default 1.0
    beta : complex, optional
        The coefficient for the neumann part, by default 0.0

    Returns
    -------
    Callable[[NDArray[Any], NDArray[Any], NDArray[Any]], NDArray[Any]]
        A function that takes x, y, z coordinates and returns the scattered field at those points.

    """
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
            sphere(h=h * radius, origin=center, r=radius)
            for center, radius in zip(centers, radii, strict=False)
        ]
    )
    space = function_space(grid, "DP", 0)
    lhs = alpha * single_layer(space, space, space, k) + beta * (
        -1 / 2 * identity(space, space, space) + adjoint_double_layer(space, space, space, k)
    )

    @complex_callable
    def f(x: Any, n: Any, domain_index: Any, result: Any) -> None:
        result[0] = -alpha * np.exp(1j * k * x[0]) - beta * 1j * k * np.exp(1j * k * x[0]) * n[0]

    rhs = GridFunction(space, fun=f)
    neumann_fun, _ = gmres(lhs, rhs, tol=1e-5)

    def inner(x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], /) -> NDArray[Any]:
        x, y, z = np.broadcast_arrays(x, y, z)
        points = np.stack((x, y, z), axis=0)
        slp = single_layer_potential(space, points, k)
        val = slp * neumann_fun
        val = val[0, ...]  # (...,)
        points_ = np.moveaxis(points, 0, -1)  # (..., 3)
        val[
            np.any(
                np.linalg.norm(points_[..., None, :] - centers_, axis=-1) < radii_,
                axis=-1,
            )
        ] = np.nan
        return val

    inner.grid = grid  #  type: ignore

    return inner
