import warnings
from collections.abc import Mapping
from typing import Callable, Literal, NotRequired, Protocol, TypedDict

import attrs

from attrs import frozen
from batch_tensorsolve import btensorsolve
from array_api._2024_12 import Array,Array
from ultrasphere import SphericalCoordinates, TCartesian, TSpherical

from ultrasphere import potential_coef, shn1, sjv
import array_api_extra as xpx
from array_api_compat import array_namespace
import ultrasphere_harmonics as ush
@frozen(kw_only=True)
class BIEMResult:
    """The result of the Boundary Integral Equation Method (BIEM)."""

    uscat: Array
    """The scattered field."""
    uscateach: Array
    """The scattered field of each sphere.
    First dimension is for the spheres."""
    uscatd: Array
    """The directional derivative to x of the scattered field."""
    uin: Array
    """The incident field."""
    uind: Array
    """The directional derivative to x of the incident field."""
    uscatfar: Array
    """The far-field pattern of the scattered field."""
    uscatfareach: Array
    """The far-field pattern of the scattered field of each sphere.
    First dimension is for the spheres."""


class BIEMResultCalculatorProtocol(Protocol):
    """Callable that computes the BIEMResult at the given cartesian coordinates."""

    density: Array
    """The density of the BIEM
    of shape [..., B, harm1, ..., harmN]."""
    density_flatten: Array | None
    """The flattened density of the BIEM
    of shape [..., B, harm]."""
    density_flatten_flatten: Array | None
    """The flattened density of the BIEM
    of shape [..., B * harm]."""
    matrix: Array | None
    """The matrix of the BIEM
    of shape [..., B, B', harm1, ..., harmN, harm1', ..., harmN']."""
    matrix_flatten: Array | None
    """The flattened matrix of the BIEM
    of shape [..., B, harm, B', harm']."""
    matrix_flatten_flatten: Array | None
    """The flattened matrix of the BIEM
    of shape [..., B * harm, B' * harm']."""

    def __call__(self, x: Array ) -> BIEMResult:
        """
        Return the scattered field at the given cartesian coordinates.

        Parameters
        ----------
        x : Array 
            The cartesian coordinates.

        Returns
        -------
        BIEMResult
            The scattered field and incident field.
            Each field has shape [...(x), ...(first), harm1, ..., harmN]

        """
        ...


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
    eta: NotRequired[Array ]
    """The decoupling parameter, by default 1.
    [...]"""
    kind: NotRequired[Literal["inner", "outer"]]
    """The kind of the scattering problem, by default "outer"."""
    force_matrix: NotRequired[bool]
    """Whether to use linear equation solver to compute the solution
    even if there is only one sphere, by default False."""


@attrs.frozen(kw_only=True)
class BIEMResultCalculator(BIEMResultCalculatorProtocol):
    """Callable that computes the BIEMResult at the given cartesian coordinates."""

    c: SphericalCoordinates[TSpherical, TCartesian]
    """The spherical coordinates system."""
    uin: Callable[[Array], Array ] | Array
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

    def __call__(self, x: Array ) -> BIEMResult:
        """
        Return the scattered field at the given cartesian coordinates.

        Parameters
        ----------
        x : Array 
            The cartesian coordinates.

        Returns
        -------
        BIEMResult
            The scattered field and incident field.
            Each field has shape [...(x), ...(first), harm1, ..., harmN]

        """
        # center: [v, ...(first), B]
        # solution_expansion: [...(first), B, harm1, ..., harmN]
        c = self.c
        n_end, _ = c.get_n_end_and_include_negative_m_from_expansion(self.density)
        centers = self.centers
        radii = self.radii
        k = self.k
        eta = self.eta
        kind = self.kind
        d = xp.array(c.c_ndim)
        n_spheres = radii.shape[-1]
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
        r = spherical["r"]
        # [harm1, ..., harmN] -> [...(x), ...(first), B, harm1, ..., harmN]
        n = c.index_array_harmonics(c.root, n_end=n_end, expand_dims=True)[
            (None,) * r.ndim + (...,)
        ]
        # [...(x), ...(first), B, harm1, ..., harmN]
        k_harm = k[(None,) * ndim_x + (...,) + (None,) * (c.s_ndim + 1)]
        y_abs = radii[(None,) * ndim_x + (...,) + (None,) * c.s_ndim]
        x_abs = r[(...,) + (None,) * c.s_ndim]
        SL_coef_ = potential_coef(
            n, d, k_harm, y_abs=y_abs, x_abs=x_abs, derivative="S", for_func="harmonics"
        )
        SL_coef_far_ = potential_coef(
            n,
            d,
            k_harm,
            y_abs=y_abs,
            x_abs=x_abs,
            derivative="S",
            for_func="solution",
        )
        DL_coef_ = potential_coef(
            n,
            d,
            k_harm,
            y_abs=y_abs,
            x_abs=x_abs,
            derivative="D",
            for_func="harmonics",
            limit=False,
        )
        DL_coef_far_ = potential_coef(
            n,
            d,
            k_harm,
            y_abs=y_abs,
            x_abs=x_abs,
            derivative="D",
            for_func="solution",
            limit=False,
        )
        SD_coef_ = DL_coef_ - xp.array(1j) * eta * SL_coef_
        SD_coef_far_ = DL_coef_far_ - xp.array(1j) * eta * SL_coef_far_
        if n_spheres == 1:
            SLD_coef_ = potential_coef(
                n,
                d,
                k_harm,
                y_abs=y_abs,
                x_abs=x_abs,
                derivative="D*",
                for_func="harmonics",
                limit=False,
            )
            DLD_coef_ = potential_coef(
                n,
                d,
                k_harm,
                y_abs=y_abs,
                x_abs=x_abs,
                derivative="N",
                for_func="harmonics",
            )
            SDD_coef_ = DLD_coef_ - xp.array(1j) * eta * SLD_coef_
        # [...(first), B, harm1, ..., harmN]
        # -> [...(x), ...(first), B, harm1, ..., harmN]
        density_ = self.density[(None,) * ndim_x + (...,)]
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
        if n_spheres == 1:
            uscatd = density_ * SDD_coef_ * Y
            # [...(x), ...(first), B, harm1, ..., harmN] -> [...(x), ...(first)]
            uscatd = xp.sum(uscatd, axis=tuple(range(-c.s_ndim - 1, 0)))

        # get uin
        uin, uind = self.uin(x)
        if uin.shape != uscat.shape:
            raise AssertionError(f"{uin.shape=} != {uscat.shape=}")

        # fill invalid regions with nan
        for u in [uscat, uin, uscateach] + (
            ([uscatd] + ([uind] if uind is not None else [])) if n_spheres == 1 else []
        ):
            if kind == "outer":
                u[(r < radii).any(axis=-1)] = xp.nan
            elif kind == "inner":
                u[(r > radii).any(axis=-1)] = xp.nan
            else:
                raise ValueError(f"Invalid kind: {kind}")

        uscateach = xp.moveaxis(uscateach, -1, 0)
        uscatfareach = xp.moveaxis(uscatfareach, -1, 0)
        # return results
        return BIEMResult(
            uscat=uscat,
            uscateach=uscateach,
            uscatd=uscatd if n_spheres == 1 else None,
            uin=uin,
            uind=uind if n_spheres == 1 else None,
            uscatfar=uscatfar,
            uscatfareach=uscatfareach,
        )

def _check_biem_inputs(
    c: SphericalCoordinates[TSpherical, TCartesian],
    centers: Array ,
    radii: Array ,
    k: Array ,
    eta: Array | None = None,
    /):
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
            "The solution may be incorrect" "if not (Im k >= 0 and eta Re k >= 0).",
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
    centers: Array ,
    radii: Array ,
    k: Array ,
    n_end: int,
    eta: Array | None = None,
    kind: Literal["inner", "outer"] = "outer",
    force_matrix: bool = False,
) -> BIEMResultCalculator:
    """
    Boundary Integral Equation Method (BIEM) for the Helmholtz equation.

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
    centers, radii, k, eta = _check_biem_inputs(
        c, centers, radii, k, eta
    )
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
            (None,) * (ndim_first)
            + (
                slice(None),
                None, None, None)
        ]
        # (B') -> (..., B, B', harm, harm')
        ball_to_add = xp.arange(n_spheres)[
            (None,) * (ndim_first)
            + (
                None,
                slice(None), None, None)
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
        
        n_current_all = ush.index_array_harmonics_all(c,
            n_end=n_end, expand_dims=True, as_array=True
        )[
            (slice(None),)
            + (None,) * (ndim_first + 2)
            + (slice(None),) * c.s_ndim
            + (None,) * c.s_ndim
        ]
        n_to_add_all = ush.index_array_harmonics_all(c,
            n_end=n_end, expand_dims=True, as_array=True
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
        density = btensorsolve(
            matrix, f_expansion, num_batch_axes=ndim_first
        )

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
        matrix=matrix
    )
