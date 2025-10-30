from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import plotly.express as px
from array_api_compat import array_namespace
from plotly.graph_objects import Figure

from .biem import BIEMResultCalculator


def plot_biem(
    biem_res: BIEMResultCalculator[Any, Any],
    /,
    *,
    plot_uin: bool = True,
    plot_uscateach: bool | Sequence[bool] = True,
    xspace: tuple[float, float, int] | None = None,
    yspace: tuple[float, float, int] | None = None,
    n_t: int = 1,
    xaxis: int = 0,
    yaxis: int = 1,
    log: bool = False,
    **plot_kwargs: Any,
) -> Figure:
    """
    Plot the results of a BIEM calculation.

    Parameters
    ----------
    biem_res : BIEMResult
        The result of a BIEM calculation.
    plot_uin : bool, optional
        Whether to plot the input field, by default True
    plot_uscateach : bool | Sequence[bool], optional
        Whether to plot the scattered field for each frequency, by default True
    xspace : tuple[float, float, int], optional
        The linspace arguments for the x-axis, by default None
    yspace : tuple[float, float, int], optional
        The linspace arguments for the y-axis, by default None
    n_t : int, optional
        The number of time steps, by default 1
    xaxis : int, optional
        The x-axis index, by default 0
    yaxis : int, optional
        The y-axis index, by default 1
    log : bool, optional
        Whether to use logarithmic scaling for the color axis, by default False
        Useful for plotting resonance fields because they diverges as |x| -> ∞.
    plot_kwargs : Any, optional
        Additional arguments to pass to plotly express scatter, by default None
        See https://plotly.com/python-api-reference/generated/plotly.express.scatter.html
        for more information.

    Returns
    -------
    Figure
        The plotly figure.

    """
    xspace_ = xspace or (-1, 1, 100)
    yspace_ = yspace or (-1, 1, 100)
    xp = array_namespace(biem_res.k)
    plot_uscateach_ = xp.asarray(plot_uscateach)
    if plot_uscateach_.ndim == 0:
        plot_uscateach_ = plot_uscateach_[None]

    c = biem_res.c
    x = xp.linspace(*xspace_)[:, None]
    y = xp.linspace(*yspace_)[None, :]
    spherical = c.from_cartesian(
        defaultdict(lambda: xp.asarray(0)[None, None], {xaxis: x, yaxis: y})
    )
    cartesian = c.to_cartesian(spherical, as_array=True)
    if biem_res.uin is None:
        uin = xp.zeros_like(cartesian[0])
    else:
        uin = biem_res.uin(cartesian)
    uscateach = biem_res.uscat(cartesian, per_ball=True)

    # time
    t = xp.arange(n_t)[:, None, None] / n_t
    texp = xp.exp(-1j * t * xp.asarray(2 * xp.pi))
    uplot = plot_uin * uin + xp.sum(plot_uscateach_[None, None, :] * uscateach, axis=-1)
    uplot_re = xp.real(uplot * texp)
    if log:
        uplot_re = xp.sign(uplot_re) * xp.log1p(xp.abs(uplot_re))

    # title
    title = ""
    if plot_uin:
        title += "Incident Field"
    if xp.any(plot_uscateach_):
        if plot_uin:
            title += " + "
        title += "Scattered Field by Ball " + ", ".join(
            [str(x) for x in xp.nonzero(plot_uscateach_)[0]]
        )
    title += r"<br>"
    title += (
        f"{c.c_ndim:g}D, "
        f"type {c.branching_types_expression_str} coordinates, "
        f"Max Degree={biem_res.n_end - 1:g}, "
        f"k={complex(biem_res.k) if 'complex' in str(biem_res.k.dtype) else float(biem_res.k):g}, "
        f"η={float(biem_res.eta):g}"
    )

    plot_2d = px.imshow(
        xp.moveaxis(uplot_re, -1, -2),
        animation_frame=0,
        y=x[:, 0],
        x=y[0, :],
        title=title,
        labels={
            "x": f"x<sub>{xaxis}</sub>",
            "y": f"x<sub>{yaxis}</sub>",
        },
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        **plot_kwargs,
    )
    plot_2d.update_layout(plot_bgcolor="black", xaxis_visible=False, yaxis_visible=False)
    plot_2d.update_xaxes(showgrid=False)
    plot_2d.update_yaxes(showgrid=False)
    return plot_2d
