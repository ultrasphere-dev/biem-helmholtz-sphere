import warnings
from collections.abc import Callable, Mapping
from typing import Any, Literal, NotRequired, Protocol, TypedDict, TypeVar

import array_api_extra as xpx
import attrs
import ultrasphere_harmonics as ush
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from batch_tensorsolve import btensorsolve
from ultrasphere import (
    SphericalCoordinates,
    potential_coef,
    shn1,
)

TCartesian = TypeVar("TCartesian")
TSpherical = TypeVar("TSpherical")


class BIEMKwargs(TypedDict):
    """The kwargs for the BIEM."""

    centers: Array
    """The centers of the spheres.
    The first dimension corresponds to the vector of the centers.
    The second dimension corresponds to the number of the spheres.
    [..., B, v]"""
    radii: Array
    """The radii of the spheres.
    The first dimension corresponds to the number of the spheres.
    [..., B]"""
    k: Array
    """The wavenumber.
    [...]"""
    n_end: int
    """The maximum degree of the spherical harmonics expansion."""
    eta: NotRequired[Array]
    """The decoupling parameter, by default 1.
    [...]"""
    kind: NotRequired[Literal["inner", "outer"]]
    """The kind of the scattering problem, by default "outer"."""
    force_matrix: NotRequired[bool]
    """Whether to use linear equation solver to compute the solution
    even if there is only one sphere, by default False."""


class UinCallable(Protocol):
    """Callable that computes the incident field at the given cartesian coordinates."""

    def __call__(self, x: Array, /, *, expand_x: bool = True) -> Array:
        """
        Return the incident field at the given cartesian coordinates.

        Parameters
        ----------
        x : Array
            The cartesian coordinates of shape (c.c_ndim, ...(x), ...(first))
            if expand_x is True,
            or of shape (c.c_ndim, ...(x))
            if expand_x is False.
        expand_x : bool, optional
            Whether the input x has the ...(first) dimensions, by default True.
            If False, the input x is assumed to have only ...(x) dimensions.

        Returns
        -------
        Array
            The incident field of shape (...(x), ...(first))

        """
        ...


class BIEMResultCalculatorProtocol[TSpherical, TCartesian](Protocol):
    """Callable that computes the BIEMResult at the given cartesian coordinates."""

    c: SphericalCoordinates[TSpherical, TCartesian]
    """The spherical coordinates system."""
    uin: UinCallable
    """The incident field."""
    centers: Array
    """The centers of the spheres."""
    radii: Array
    """The radii of the spheres."""
    k: Array
    """The wavenumber."""
    n_end: int
    """The maximum degree of the spherical harmonics expansion."""
    eta: Array
    """The decoupling parameter."""
    kind: Literal["inner", "outer"]
    """The kind of the scattering problem."""
    density: Array | None = None
    """The flattened density of the BIEM
    of shape [..., B, harm]."""
    matrix: Array | None = None
    """The flattened matrix of the BIEM
    of shape [..., B, harm, B', harm']."""

    def uscat(
        self,
        x: Array,
        /,
        far_field: bool = False,
        per_ball: bool = False,
        expand_x: bool = True,
    ) -> Array:
        """
        Return the scattered field at the given cartesian coordinates.

        Parameters
        ----------
        x : Array
            The cartesian coordinates of shape (c.c_ndim, ...(x), ...(first))
            if expand_x is True,
            or of shape (c.c_ndim, ...(x))
            if expand_x is False.
        far_field : bool, optional
            Whether to compute the far field, by default False.
        per_ball : bool, optional
            Whether to return the scattered field per ball, by default False.
            If False, the scattered field is summed over all balls.
        expand_x : bool, optional
            Whether the input x has the ...(first) dimensions, by default True.
            If False, the input x is assumed to have only ...(x) dimensions.

        Returns
        -------
        Array
            The scattered field of shape (...(x), ...(first))
            if per_ball is False,
            or of shape (...(x), ...(first), B)
            if per_ball is True.

        """
        ...


@attrs.frozen(kw_only=True)
class BIEMResultCalculator(BIEMResultCalculatorProtocol[TSpherical, TCartesian]):
    """Callable that computes the BIEMResult at the given cartesian coordinates."""

    c: SphericalCoordinates[TSpherical, TCartesian]
    """The spherical coordinates system."""
    uin: UinCallable
    """The incident field."""
    centers: Array
    """The centers of the spheres."""
    radii: Array
    """The radii of the spheres."""
    k: Array
    """The wavenumber."""
    n_end: int
    """The maximum degree of the spherical harmonics expansion."""
    eta: Array
    """The decoupling parameter."""
    kind: Literal["inner", "outer"]
    """The kind of the scattering problem."""
    density: Array | None = None
    """The flattened density of the BIEM
    of shape [..., B, harm]."""
    matrix: Array | None = None
    """The flattened matrix of the BIEM
    of shape [..., B, harm, B', harm']."""

    def uscat(  # noqa: D102
        self,
        x: Array,
        /,
        far_field: bool = False,
        per_ball: bool = False,
        expand_x: bool = True,
    ) -> Array:
        return biem_u(
            self,
            x,
            far_field=far_field,
            per_ball=per_ball,
            expand_x=expand_x,
        )


def _check_biem_inputs(
    c: SphericalCoordinates[TSpherical, TCartesian],
    centers: Array,
    radii: Array,
    k: Array,
    eta: Array | None = None,
    /,
) -> tuple[Array, Array, Array, Array]:
    xp = array_namespace(centers, radii, k, eta)

    # convert to array
    centers = xp.asarray(centers)
    radii = xp.asarray(radii)
    k = xp.asarray(k)
    if eta is None:
        eta = xp.asarray(1.0)[(None,) * k.ndim]
    else:
        eta = xp.asarray(eta)

    # check decoupling parameter
    if xp.any(xp.imag(eta) != 0):
        raise ValueError("The decoupling parameter must be real.")
    if xp.any(eta == 0):
        warnings.warn(
            "The solution may be incorrect"
            "if k is an eigenvalue for laplacian"
            "on the interior region with"
            "Neumann boundary condition.",
            UserWarning,
            stacklevel=2,
        )
    if xp.any((xp.imag(k) < 0) | (eta * xp.real(k) < 0)):
        warnings.warn(
            "The solution may be incorrectif not (Im k >= 0 and eta Re k >= 0).",
            UserWarning,
            stacklevel=2,
        )

    # check if broadcastable
    if len({k.ndim, eta.ndim, centers.ndim - 2, radii.ndim - 1}) != 1:
        raise ValueError(
            f"{k.ndim=}, {eta.ndim=}, {centers.ndim - 2=}, {radii.ndim -1=}are not the same."
        )
    try:
        xpx.broadcast_shapes(k.shape, eta.shape, centers.shape[:-2], radii.shape[:-1])
    except Exception as e:
        raise ValueError(
            "Shapes of k, eta and "
            "centers.shape[:-2], radii.shape[:-1] "
            "are not broadcastable\n"
            f"{tuple(k.shape)=}\n"
            f"{tuple(eta.shape)=}\n"
            f"{tuple(centers.shape)=}\n"
            f"{tuple(radii.shape)=}"
        ) from e

    try:
        xpx.broadcast_shapes(centers.shape[:-1], radii.shape)
    except Exception as e:
        raise ValueError(
            "centers.shape[:-1] and radii.shape "
            "are not broadcastable\n"
            f"{tuple(centers.shape)=}\n"
            f"{tuple(radii.shape)=}"
        ) from e

    if centers.shape[-1] != c.c_ndim:
        raise ValueError(
            f"The last dimension of centers must be {c.c_ndim=}, but got {centers.shape[-1]}"
        )

    return centers, radii, k, eta


# [..., B, harm1, ..., harmN]
def plane_wave(*, k: Array, direction: Array) -> Callable[[Array], Array]:
    r"""
    Plane wave.

    $$
    d := \frac{\text{direction}}{\|\text{direction}\|}
    $$
    $$
    u (x) := e^{i k d \cdot x}
    $$

    Parameters
    ----------
    k : Array
        The wavenumber of shape (...).
    direction : Array
        The direction of the plane wave of shape (c.c_ndim, ...).

        Will be normalized.

    Returns
    -------
    Callable[[Array], Array]
        Given cartesian coordinates of shape (c.c_ndim, ...(any), ...),
        returns the incident field of shape (...(any), ...)

    """
    xp = array_namespace(k, direction)
    try:
        xpx.broadcast_shapes(k.shape, direction.shape[1:])
    except Exception as e:
        raise ValueError(
            "Shapes of k and direction[1:] "
            "are not broadcastable\n"
            f"{tuple(k.shape)=}\n"
            f"{tuple(direction.shape)=}"
        ) from e
    if direction.ndim != k.ndim + 1:
        raise ValueError(f"{direction.ndim=} is not {k.ndim + 1=}")
    direction = direction / xp.linalg.vector_norm(direction, axis=0, keepdims=True)

    def inner(x: Array, /) -> Array:
        ip = xp.sum(direction[(slice(None),) + (None,) * (x.ndim - direction.ndim)] * x, axis=0)
        return xp.exp(1j * k * ip)

    return inner


def point_source(*, k: Array, source: Array, n: int) -> Callable[[Array], Array]:
    r"""
    Point source.

    $$
    u (x) := h^{(1)}_n (k \|x - \text{source}\|)
    $$

    Parameters
    ----------
    k : Array
        The wavenumber of shape (...).
    source : Array
        The position of the point source of shape (c.c_ndim, ...).
    n : int
        The order of the Hankel function.

    Returns
    -------
    Callable[[Array], Array]
        Given cartesian coordinates of shape (c.c_ndim, ...(any), ...),
        returns the incident field of shape (...(any), ...)

    """
    xp = array_namespace(k, source)
    try:
        xpx.broadcast_shapes(k.shape, source.shape[1:])
    except Exception as e:
        raise ValueError(
            "Shapes of k and source[1:] "
            "are not broadcastable\n"
            f"{tuple(k.shape)=}\n"
            f"{tuple(source.shape)=}"
        ) from e
    if source.ndim != k.ndim + 1:
        raise ValueError(f"{source.ndim=} is not {k.ndim + 1=}")
    n_ = xp.asarray(n)

    def inner(x: Array, /) -> Array:
        x = x - source[(slice(None),) + (None,) * (x.ndim - source.ndim)]
        d = int(x.shape[0])
        return shn1(n_, xp.asarray(d), k * xp.linalg.vector_norm(x, axis=0))

    return inner


def biem(
    c: SphericalCoordinates[TSpherical, TCartesian],
    uin: Callable[[Array], Array],
    *,
    centers: Array,
    radii: Array,
    k: Array,
    n_end: int,
    eta: Array | None = None,
    kind: Literal["inner", "outer"] = "outer",
    force_matrix: bool = False,
) -> BIEMResultCalculator[TSpherical, TCartesian]:
    r"""
    Boundary Integral Equation Method (BIEM) for the Helmholtz equation.

    Let $d \in \mathbb{N} \setminus \lbrace 1 \rbrace$ be the dimension of the space,
    $k$ be the wave number,
    and $\mathbb{S}^{d-1} = \lbrace x \in \mathbb{R}^d \mid \|x\| = 1 \rbrace$
    be a unit sphere in $\mathbb{R}^d$.

    Asuume that $u_\text{in}$ is an incident wave satisfying the Helmholtz equation

    $$
    \Delta u_\text{in} + k^2 u_\text{in} = 0
    $$

    and scattered wave $u$ satisfies the following:

    $$
    \begin{cases}
    \Delta u + k^2 u = 0 \quad &x \in \mathbb{R}^d \setminus \overline{\mathbb{S}^{d-1}} \\
    \alpha u + \beta \grad u \dot n_x = -u_\text{in} \quad
    &x \in \mathbb{S}^{d-1} \\
    \lim_{\|x\| \to \infty} \|x\|^{\frac{d-1}{2}}
    \left( \frac{\partial u}{\partial \|x\|} - i k u \right) = 0 \quad
    &\frac{x}{\|x\|} \in \mathbb{S}^{d-1}
    \end{cases}
    $$

    $$
    \newcommand\slc{\operatorname{slc}}
    \newcommand\dlc{\operatorname{dlc}}
    \newcommand\blc{\operatorname{blc}}
    \slc_n (\rho) := i k^{d-2} rho^{d-1} j_n (k rho) \\
    \dlc_n (\rho) := i k^{d-1} rho^{d-1} j_n' (k rho) \\
    \blc_n (\rho) := \slc_n (rho) - i \eta \dlc_n (rho) \\
    A_{bnpb'n'p'} := \blc_{n'} \times \begin{cases}
    \delta_{n,n'} \delta_{p,p'} (\alpha h^{(1)}_n (k rho_b) + \beta h^{(1)'}_n (k rho_b))
    & b = b' \\
    (S|R)_{n,p,n',p'} (c_b - c_b') (\alpha j_n (k rho_b) + \beta j'_n (k rho_b))
    & b \neq b'
    \end{cases} \\
    f_{bnp} := - \integral_{\partial B_b} u_{in} (x) Y_{n,p} (\hat{x - c_b}) dx
    = - \integral_{S^{d-1}} u_{in} (c_b + rho_b y) Y_{n,p} (y) rho_b^{d-1} dy \\
    \sum_{b',n',p'} A_{bnpb'n'p'} \phi_{b'n'p'} = f_{bnp}
    $$

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TCartesian]
        The spherical coordinates system.
    uin : Callable[[Array], Array]
        The incident field.

        Given cartesian coordinates of shape (c.c_ndim, ...(any), ...),
        should return the incident field of shape (...(any), ...)

        Must satisfy the Helmholtz equation with wavenumber k.
    centers : Array
        The centers of the spheres of shape (..., B, c.c_ndim).
    radii : Array
        The radii of the spheres of shape (..., B).
    k : Array
        The wavenumber of shape (...).
    n_end : int
        The maximum degree of the spherical harmonics expansion.
    eta : Array , optional
        The decoupling parameter of shape (...), by default 1.
    kind : Literal["inner", "outer"], optional
        The kind of the scattering problem, by default "outer".
    force_matrix : bool, optional
        Whether to use linear equation solver to compute the solution
        even if there is only one sphere, by default False.

    Returns
    -------
    BIEMResultCalculator
        The function that computes the incident and scattered fields
        at the given spherical coordinates.
        Each field has shape [...(x), ...(first), harm1, ..., harmN]

    """
    centers, radii, k, eta = _check_biem_inputs(c, centers, radii, k, eta)
    xp = array_namespace(centers, radii, k, eta)

    # [..., B, v] -> [v, ..., B]
    centers = xp.moveaxis(centers, -1, 0)

    # ...(x).ndim
    ndim_first = k.ndim

    d = xp.asarray(c.c_ndim)
    n_spheres = radii.shape[-1]

    # boundary condition
    def f(spherical: Mapping[TSpherical, Array]) -> Array:
        # (c_ndim, ...(f), B, ...)
        x = c.to_cartesian(spherical, as_array=True)[(...,) + (None,) * (1 + ndim_first)]
        x = (
            x + centers[(slice(None),) + (None,) * c.s_ndim + (slice(None),) + (None,) * ndim_first]
        )  # x - c_i
        return -uin(x)

    # (B, ..., harm)
    f_expansion = ush.expand(
        c,
        f,
        does_f_support_separation_of_variables=False,
        n_end=n_end,
        n=n_end,
        phase=ush.Phase(0),
        xp=xp,
    )
    # (..., B, harm)
    f_expansion = xp.moveaxis(f_expansion, 0, -2)

    # compute SL and DL, [..., B, harm1, ..., harmN]
    # (sizes except for B and harm_root are 1)
    use_matrix = n_spheres is None or n_spheres > 1 or force_matrix

    # compute a
    if not use_matrix:
        # simply divide by the potential coefficients
        # [harm1, ..., harmN] -> [..., B=1, harm1, ..., harmN]
        n = ush.index_array_harmonics(c, c.root, n_end=n_end, expand_dims=True, xp=xp)[
            (None,) * (ndim_first + 1) + (...,)
        ]
        S_coef = potential_coef(
            n,
            d,
            k[(...,) + (None,) * (c.s_ndim + 1)],
            y_abs=radii[(...,) + (None,) * c.s_ndim],
            x_abs=radii[(...,) + (None,) * c.s_ndim],
            derivative="S",
            for_func="harmonics",
        )
        D_coef = potential_coef(
            n,
            d,
            k[(...,) + (None,) * (c.s_ndim + 1)],
            y_abs=radii[(...,) + (None,) * c.s_ndim],
            x_abs=radii[(...,) + (None,) * c.s_ndim],
            derivative="D",
            limit=False,
            for_func="harmonics",
        )
        SD_coef = D_coef - 1j * eta * S_coef
        SD_coef = ush.flatten_harmonics(c, SD_coef)
        density = f_expansion / SD_coef
    else:
        # (e_ndim, ..., B, B')
        center_current = centers[:, ..., :, None]
        center_to_add = centers[:, ..., None, :]
        # [..., B, B', harm, harm']
        translation_coef = ush.harmonics_translation_coef(
            c,
            c.from_cartesian(center_current - center_to_add),  # ?
            n_end=n_end,
            n_end_add=n_end,
            phase=ush.Phase(0),
            k=k[..., None, None],
            is_type_same=False,
        )
        # (B,) -> (..., B, B', harm, harm')
        ball_current = xp.arange(n_spheres)[
            (None,) * (ndim_first) + (slice(None), None, None, None)
        ]
        # (B') -> (..., B, B', harm, harm')
        ball_to_add = xp.arange(n_spheres)[(None,) * (ndim_first) + (None, slice(None), None, None)]
        # (..., B) -> (..., B, B')
        radius_current = radii[..., :, None]
        # (..., B') -> (..., B, B')
        radius_to_add = radii[..., None, :]
        # Not flattened
        n_to_add = ush.index_array_harmonics(c, c.root, n_end=n_end, xp=xp)[
            (None,) * (ndim_first + 3) + (slice(None),) * c.s_ndim
        ]
        S_coef = potential_coef(
            n_to_add,
            d,
            k[(...,) + (None,) * (3 + c.s_ndim)],
            y_abs=radius_to_add[(...,) + (None,) * (1 + c.s_ndim)],
            x_abs=radius_to_add[(...,) + (None,) * (1 + c.s_ndim)],
            derivative="S",
            for_func="solution",
        )
        D_coef = potential_coef(
            n_to_add,
            d,
            k[(...,) + (None,) * (3 + c.s_ndim)],
            y_abs=radius_to_add[(...,) + (None,) * (1 + c.s_ndim)],
            x_abs=radius_to_add[(...,) + (None,) * (1 + c.s_ndim)],
            derivative="D",
            limit=False,
            for_func="solution",
        )
        SD_coef = D_coef - 1j * eta[(...,) + (None,) * (3 + c.s_ndim)] * S_coef
        SD_coef = ush.flatten_harmonics(c, SD_coef, n_end=n_end, include_negative_m=True)
        # ([..., B, B', harm, harm')
        matrix = SD_coef * xp.where(
            ball_current == ball_to_add,
            xpx.create_diagonal(
                ush.harmonics_regular_singular_component(
                    c,
                    {"r": radius_current},
                    n_end=n_end,
                    k=k[..., None, None],
                    type="singular",
                )
            ),
            xp.moveaxis(translation_coef, -1, -2)
            * ush.harmonics_regular_singular_component(
                c,
                {"r": radius_current},
                n_end=n_end,
                k=k[..., None, None],
                type="regular",
            )[..., :, None],
        )
        # [..., B, B', harm, harm'] -> [..., B, harm, B', harm']
        matrix = xp.moveaxis(matrix, -3, -2)
        # [..., B, harm, B', harm'] and [..., B, harm]
        density = btensorsolve(matrix, f_expansion, num_batch_axes=ndim_first)

    def uin_wrapped(x: Array, /, *, expand_x: bool = True) -> Array:
        if expand_x:
            x = x[(...,) + (None,) * ndim_first]
        return uin(x)

    return BIEMResultCalculator(
        c=c,
        centers=centers,
        radii=radii,
        k=k,
        n_end=n_end,
        eta=eta,
        kind=kind,
        uin=uin_wrapped,
        density=density,
        matrix=matrix,
    )


def biem_u(
    res: BIEMResultCalculatorProtocol[Any, Any],
    x: Array,
    /,
    far_field: bool = False,
    per_ball: bool = False,
    expand_x: bool = True,
) -> Array:
    """
    Return the scattered field at the given cartesian coordinates.

    Parameters
    ----------
    res : BIEMResultCalculatorProtocol
        The result of the BIEM.
    x : Array
        The cartesian coordinates of shape (c.c_ndim, ...(x), ...(first))
        if expand_x is True,
        or of shape (c.c_ndim, ...(x))
        if expand_x is False.
    far_field : bool, optional
        Whether to compute the far field, by default False.
    per_ball : bool, optional
        Whether to return the scattered field per ball, by default False.
        If False, the scattered field is summed over all balls.
    expand_x : bool, optional
        Whether the input x has the ...(first) dimensions, by default True.
        If False, the input x is assumed to have only ...(x) dimensions.

    Returns
    -------
    Array
        The scattered field of shape (...(x), ...(first))
        if per_ball is False,
        or of shape (...(x), ...(first), B)
        if per_ball is True.

    """
    # center: (v, ...(first), B)
    if res.density is None:
        raise ValueError("The BIEMResult does not have density.")
    c = res.c
    n_end, _ = ush.assume_n_end_and_include_negative_m_from_harmonics(c, res.density)
    centers = res.centers
    radii = res.radii
    k = res.k
    eta = res.eta
    xp = array_namespace(x, centers, radii, k, eta)
    kind = res.kind
    d = xp.asarray(c.c_ndim)
    ndim_first = k.ndim

    x = xp.stack([x[i] for i in range(c.c_ndim)], axis=0)
    ndim_x = x.ndim - 1
    if not expand_x:
        ndim_x -= ndim_first

    # (v, ...(x)) -> (v, ...(x), ...(first), B)
    x_ = x[(slice(None), ...) + (None,) * ((ndim_first if expand_x else 0) + 1)]
    # (v, ...(x), ...(first), B) -> (...(x), ...(first), B)
    spherical = c.from_cartesian(x_ - centers[(slice(None),) + (None,) * ndim_x + (...,)])
    # (...(x), ...(first), B)
    r = spherical["r"]
    # (harm1, ..., harmN) -> (...(x), ...(first), B, harm1, ..., harmN)
    n = ush.index_array_harmonics(c, c.root, n_end=n_end, expand_dims=True, xp=xp)[
        (None,) * r.ndim + (...,)
    ]
    # (...(x), ...(first), B, harm1, ..., harmN)
    k_harm = k[(None,) * ndim_x + (...,) + (None,) * (c.s_ndim + 1)]
    y_abs = radii[(None,) * ndim_x + (...,) + (None,) * c.s_ndim]
    x_abs = r[(...,) + (None,) * c.s_ndim]
    SL_coef_ = potential_coef(
        n,
        d,
        k_harm,
        y_abs=y_abs,
        x_abs=x_abs,
        derivative="S",
        for_func="solution" if far_field else "harmonics",
    )
    DL_coef_ = potential_coef(
        n,
        d,
        k_harm,
        y_abs=y_abs,
        x_abs=x_abs,
        derivative="D",
        for_func="solution" if far_field else "harmonics",
        limit=False,
    )
    SD_coef_ = DL_coef_ - xp.asarray(1j) * eta * SL_coef_
    # (...(x), ...(first), B, harm)
    SD_coef_ = ush.flatten_harmonics(c, SD_coef_, n_end=n_end, include_negative_m=True)
    # (...(first), B, harm)
    # -> (...(x), ...(first), B, harm)
    density_ = res.density[(None,) * ndim_x + (...,)]
    # (...(x), ...(first), B, harm)
    Y = ush.harmonics(
        c,
        spherical,
        n_end=n_end,
        phase=ush.Phase(0),
        expand_dims=True,
        concat=True,
    )
    if far_field:
        uscatfarcoef = (
            (-1j) ** n
            / (1j * k_harm) ** ((d - 1) / 2)
            * xp.exp(
                1j
                * k_harm
                * xp.sum(
                    # centers: [v, ...(first), B]
                    # x: [v, ...(x), ...(first), B]
                    x_[(...,) + (None,) * (c.s_ndim)]
                    * -centers[(slice(None),) + (None,) * ndim_x + (...,) + (None,) * c.s_ndim],
                    axis=0,
                )
            )
        )
        uscatfar = density_ * SD_coef_ * Y * uscatfarcoef
        uscatfar = xp.sum(uscatfar, axis=-1)
        if per_ball:
            return uscatfar
        return xp.sum(uscatfar, axis=-1)
    # (...(x), ...(first), B, harm)
    uscat = density_ * SD_coef_ * Y
    # (...(x), ...(first), B)
    uscat = xp.sum(uscat, axis=-1)
    if not per_ball:
        # (...(x), ...(first))
        uscat = xp.sum(uscat, axis=-1)
    # for 0d case
    uscat = xp.asarray(uscat)

    # fill invalid regions with nan
    if kind == "outer":
        uscat[xp.any(r < radii, axis=-1), ...] = xp.nan
    elif kind == "inner":
        uscat[xp.any(r > radii, axis=-1), ...] = xp.nan
    else:
        raise ValueError(f"Invalid kind: {kind}")
    return uscat
