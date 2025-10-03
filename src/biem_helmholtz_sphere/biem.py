import warnings
from collections.abc import Callable, Mapping
from typing import Literal, NotRequired, Protocol, TypedDict

import array_api_extra as xpx
import attrs
import ultrasphere_harmonics as ush
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from batch_tensorsolve import btensorsolve
from ultrasphere import (
    SphericalCoordinates,
    TCartesian,
    TSpherical,
    potential_coef,
    shn1,
    sjv,
)


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


class BIEMResultCalculatorProtocol(Protocol):
    """Callable that computes the BIEMResult at the given cartesian coordinates."""

    c: SphericalCoordinates[TSpherical, TCartesian]
    """The spherical coordinates system."""
    uin: Callable[[Array], Array] | Array
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


@attrs.frozen(kw_only=True)
class BIEMResultCalculator(BIEMResultCalculatorProtocol):
    """Callable that computes the BIEMResult at the given cartesian coordinates."""

    c: SphericalCoordinates[TSpherical, TCartesian]
    """The spherical coordinates system."""
    uin: Callable[[Array], Array] | Array
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


def _check_biem_inputs(
    c: SphericalCoordinates[TSpherical, TCartesian],
    centers: Array,
    radii: Array,
    k: Array,
    eta: Array | None = None,
    /,
):
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
            f"{k.ndim=}, {eta.ndim=}, {centers.ndim - 2=}, {radii.ndim -1=}"
            "are not the same."
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
            f"The last dimension of centers must be {c.c_ndim=}, "
            f"but got {centers.shape[-1]}"
        )

    return centers, radii, k, eta


# [..., B, harm1, ..., harmN]
def uin_(x: Array, with_ball: bool = False) -> tuple[Array, Array | None]:
    # x is [v, ...(points), ...(first), B] or [v, ...(x), ...(first)]
    # if uin is a function, call it
    if isinstance(uin, Callable):  # type: ignore
        return uin(x), None

    # else assume incident field is a plane wave
    # and uin is the direction of the plane wave

    # normalize direction
    uin_: Array = uin
    uin_ = xp.stack([uin_[e_node] for e_node in c.e_nodes], axis=0)
    d = uin_ / xp.vector_norm(uin_, axis=0)
    # [..., ...(first), B]
    ip = xp.einsum("v,v...->...", d, x)
    k_ = k[..., None] if with_ball else k
    uin_v = xp.exp(1j * k_ * ip)
    # ∂e^{ikd.x}/∂|x| = x/|x|.ikde^{ikd.x}
    duindr_v = 1j * k_ * ip * uin_v
    # in some definitions we need to apply `/ xp.vector_norm(x, axis=0)``
    # although this would not be well defined for x=0
    return uin_v, duindr_v


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
) -> BIEMResultCalculator:
    r"""
    Boundary Integral Equation Method (BIEM) for the Helmholtz equation.

    Let $d \in \mathbb{N} \setminus \lbrace 1 \rbrace$ be the dimension of the space, $k$ be the wave number, and $\mathbb{S}^{d-1} = \lbrace x \in \mathbb{R}^d \mid \|x\| = 1 \rbrace$ be a unit sphere in $\mathbb{R}^d$.

    Asuume that $u_\text{in}$ is an incident wave satisfying the Helmholtz equation

    $$
    \Delta u_\text{in} + k^2 u_\text{in} = 0
    $$

    and scattered wave $u$ satisfies the following:

    $$
    \begin{cases}
    \Delta u + k^2 u = 0 \quad &x \in \mathbb{R}^d \setminus \overline{\mathbb{S}^{d-1}} \\
    \alpha u + \beta \grad u \dot n_x = -u_\text{in} \quad &x \in \mathbb{S}^{d-1} \\
    \lim_{\|x\| \to \infty} \|x\|^{\frac{d-1}{2}} \left( \frac{\partial u}{\partial \|x\|} - i k u \right) = 0 \quad &\frac{x}{\|x\|} \in \mathbb{S}^{d-1}
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
    \delta_{n,n'} \delta_{p,p'} (\alpha h^{(1)}_n (k rho_b) + \beta h^{(1)'}_n (k rho_b)) & b = b' \\
    (S|R)_{n,p,n',p'} (c_b - c_b') (\alpha j_n (k rho_b) + \beta j'_n (k rho_b)) & b \neq b'
    \end{cases} \\
    f_{bnp} := - \integral_{\partial B_b} u_{in} (x) Y_{n,p} (\hat{x - c_b}) dx
    = - \integral_{S^{d-1}} u_{in} (c_b + rho_b \hat{y}) Y_{n,p} (\hat{y}) rho_b^{d-1} d\hat{y} \\
    \sum_{b',n',p'} A_{bnpb'n'p'} \phi_{b'n'p'} = f_{bnp}
    $$

    Parameters
    ----------
    c : SphericalCoordinates[TSpherical, TCartesian]
        The spherical coordinates system.
    uin : Callable[Array, Array ]
        The incident field / e^(-iωx). (not e^(iωx))

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
    centers = centers.moveaxis(-1, 0)

    # ...(x).ndim
    ndim_first = k.ndim

    # []
    d = xp.array(c.c_ndim)
    n_spheres = radii.shape[-1]

    # boundary condition
    def f(spherical: Mapping[TSpherical, Array]) -> Array:
        # [v, ..., ...(first), B]
        x = c.to_cartesian(spherical, as_array=True)[
            (...,) + (None,) * (ndim_first + 1)
        ]
        x += centers[
            (slice(None),) + (None,) * c.s_ndim + (slice(None),) * (ndim_first + 1)
        ]  # x - c_i
        return -1.0 * uin_(x, with_ball=True)[0]

    # [..., B, harm1, ..., harmN]
    f_expansion = ush.expand(
        c,
        f,
        does_f_support_separation_of_variables=False,
        n_end=n_end,
        n=n_end,
        condon_shortley_phase=False,
    )

    # compute SL and DL, [..., B, harm1, ..., harmN]
    # (sizes except for B and harm_root are 1)
    use_matrix = n_spheres > 1 or force_matrix

    # compute a
    if not use_matrix:
        # simply divide by the potential coefficients
        # [harm1, ..., harmN] -> [..., B=1, harm1, ..., harmN]
        n = ush.index_array_harmonics(c, c.root, n_end=n_end, expand_dims=True)[
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
            c.from_cartesian(center_current - center_to_add),  # ?
            n_end=n_end,
            n_end_add=n_end,
            condon_shortley_phase=False,
            k=k[..., None, None],
            is_type_same=False,
        )
        # (B,) -> (..., B, B', harm, harm')
        ball_current = xp.arange(n_spheres)[
            (None,) * (ndim_first) + (slice(None), None, None, None)
        ]
        # (B') -> (..., B, B', harm, harm')
        ball_to_add = xp.arange(n_spheres)[
            (None,) * (ndim_first) + (None, slice(None), None, None)
        ]
        # (..., B) -> (..., B, B', harm, harm')
        radius_current = radii[..., :, None, None, None]
        # (..., B') -> (..., B, B', harm, harm')
        radius_to_add = radii[..., None, :, None, None]
        # Not flattened
        n_to_add = ush.index_array_harmonics(c, c.root, n_end=n_end)[
            (None,) * (ndim_first + 2 + c.s_ndim) + (slice(None),) * c.s_ndim
        ]
        S_coef = potential_coef(
            n_to_add,
            d,
            k[(...,) + (None,) * (2 * c.s_ndim + 2)],
            y_abs=radius_to_add,
            x_abs=radius_to_add,
            derivative="S",
            for_func="solution",
        )
        D_coef = potential_coef(
            n_to_add,
            d,
            k[(...,) + (None,) * (2 * c.s_ndim + 2)],
            y_abs=radius_to_add,
            x_abs=radius_to_add,
            derivative="D",
            limit=False,
            for_func="solution",
        )
        SD_coef = D_coef - 1j * eta[(...,) + (None,) * (2 * c.s_ndim + 2)] * S_coef
        SD_coef = ush.flatten_harmonics(c, SD_coef)

        n_current_all = ush.index_array_harmonics_all(
            c, n_end=n_end, expand_dims=True, as_array=True
        )[
            (slice(None),)
            + (None,) * (ndim_first + 2)
            + (slice(None),) * c.s_ndim
            + (None,) * c.s_ndim
        ]
        n_to_add_all = ush.index_array_harmonics_all(
            c, n_end=n_end, expand_dims=True, as_array=True
        )[
            (slice(None),)
            + (None,) * (ndim_first + 2 + c.s_ndim)
            + (slice(None),) * c.s_ndim
        ]
        A = SD_coef * xp.where(
            ball_current == ball_to_add,
            xp.where(
                (n_current_all == n_to_add_all).all(axis=0),
                shn1(
                    n_to_add,
                    d,
                    k[(...,) + (None,) * (4)] * radius_current,
                ),
                xp.asarray(0),
            ),
            translation_coef
            * sjv(
                n_to_add, d, k[(...,) + (None,) * (2 * c.s_ndim + 2)] * radius_current
            ),
        )
        # [..., B, B', harm, harm'] -> [..., B, harm, B', harm']
        matrix = xp.moveaxis(A, -2 - c.s_ndim, -1 - c.s_ndim)
        # [..., B, harm, B', harm'] and [..., B, harm]
        density = btensorsolve(matrix, f_expansion, num_batch_axes=ndim_first)

    if density.ndim != ndim_first + 1 + c.s_ndim:
        raise AssertionError(f"{density.ndim=} is not {ndim_first=} + 1 + {c.s_ndim=}")

    return BIEMResultCalculator(
        c=c,
        centers=centers,
        radii=radii,
        k=k,
        n_end=n_end,
        eta=eta,
        kind=kind,
        uin=uin,
        density=density,
        matrix=matrix,
    )


def biem_u(
    res: BIEMResultCalculatorProtocol, x: Array, /, far_field: bool = False
) -> Array:
    """
    Return the scattered field at the given cartesian coordinates.

    Parameters
    ----------
    res : BIEMResultCalculatorProtocol
        The result of the BIEM.
    x : Array
        The cartesian coordinates.
    far_field : bool, optional
        Whether to compute the far field, by default False.

    Returns
    -------
    BIEMResult
        The scattered field and incident field.
        Each field has shape [...(x), ...(first), harm1, ..., harmN]

    """
    # center: [v, ...(first), B]
    # solution_expansion: [...(first), B, harm1, ..., harmN]
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
    # [v, ...(x)] -> [v, ...(x), ...(first), B]
    x_ = x[(slice(None), ...) + (None,) * (ndim_first + 1)]
    # [v, ...(x), ...(first), B] -> [...(x), ...(first), B]
    spherical = c.from_cartesian(
        x_ - centers[(slice(None),) + (None,) * ndim_x + (...,)]
    )
    # [...(x), ...(first), B]
    res = spherical["r"]
    # [harm1, ..., harmN] -> [...(x), ...(first), B, harm1, ..., harmN]
    n = c.index_array_harmonics(c.root, n_end=n_end, expand_dims=True)[
        (None,) * res.ndim + (...,)
    ]
    # [...(x), ...(first), B, harm1, ..., harmN]
    k_harm = k[(None,) * ndim_x + (...,) + (None,) * (c.s_ndim + 1)]
    y_abs = radii[(None,) * ndim_x + (...,) + (None,) * c.s_ndim]
    x_abs = res[(...,) + (None,) * c.s_ndim]
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
    # [...(first), B, harm1, ..., harmN]
    # -> [...(x), ...(first), B, harm1, ..., harmN]
    density_ = res.density[(None,) * ndim_x + (...,)]
    # [...(x), ...(first), B]
    spherical_no_r: dict[TSpherical, Array] = {
        k: v for k, v in spherical.items() if k != "r"
    }
    # [harm1, ..., harmN]
    Y = c.harmonics(
        spherical_no_r,
        n_end=n_end,
        condon_shortley_phase=False,
        expand_dims=True,
        concat=True,
    )
    # [...(x), ...(first), B, harm1, ..., harmN]
    uscat = density_ * SD_coef_ * Y
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
                * -centers[
                    (slice(None),) + (None,) * ndim_x + (...,) + (None,) * c.s_ndim
                ],
                axis=0,
            )
        )
    )
    uscatfar = density_ * SD_coef_far_ * Y * uscatfarcoef
    # [...(x), ...(first), B, harm1, ..., harmN] -> [...(x), ...(first)]
    uscateach = xp.sum(uscat, axis=tuple(range(-c.s_ndim, 0)))
    uscatfareach = xp.sum(uscatfar, axis=tuple(range(-c.s_ndim, 0)))
    uscat = xp.sum(uscateach, axis=-1)
    uscatfar = xp.sum(uscatfareach, axis=-1)

    # fill invalid regions with nan
    if kind == "outer":
        u[(res < radii).any(axis=-1), ...] = xp.nan
    elif kind == "inner":
        u[(res > radii).any(axis=-1), ...] = xp.nan
    else:
        raise ValueError(f"Invalid kind: {kind}")
