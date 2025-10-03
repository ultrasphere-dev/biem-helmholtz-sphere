from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import plotly.express as px
from pandas import DataFrame
from plotly.graph_objects import Figure

from .biem import BIEMResultCalculator


def plot_biem(
    biem_res: BIEMResultCalculator,
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
    xspace_ = xspace or (-1, 1, 10)
    yspace_ = yspace or (-1, 1, 10)
    plot_uscateach_ = xp.asarray(plot_uscateach)
    if plot_uscateach_.ndim == 0:
        plot_uscateach_ = plot_uscateach_[None]

    c = biem_res.c
    x = xp.linspace(*xspace_)[:, None]
    y = xp.linspace(*yspace_)[None, :]
    spherical = c.from_cartesian(
        defaultdict(lambda: xp.array(0)[None, None], {xaxis: x, yaxis: y})
    )
    cartesian = c.to_cartesian(spherical, as_array=True)
    ures = biem_res(cartesian)

    # time
    t = xp.arange(n_t)[:, None, None] / n_t
    texp = xp.exp(-1j * t * xp.array(2 * xp.pi))
    shape = (n_t, *xp.broadcast_shapes(x.shape, y.shape))
    uplot = plot_uin * ures.uin + xp.sum(
        plot_uscateach_[:, None, None] * ures.uscateach, axis=0
    )
    df = DataFrame(
        {
            "x": xp.broadcast_to(x[None, ...], shape).flatten(),
            "y": xp.broadcast_to(y[None, ...], shape).flatten(),
            "t": xp.broadcast_to(t, shape).flatten(),
            "uin": (ures.uin * texp).real.flatten(),
            "uscat": (ures.uscat * texp).real.flatten(),
            "uall": ((ures.uin + ures.uscat) * texp).real.flatten(),
            "uplot": (uplot * texp).real.flatten(),
        }
        | {
            f"uscat{i}": (uscateach_item * texp).real.flatten()
            for i, uscateach_item in enumerate(ures.uscateach)
        }
    )
    if log:
        df["uplot"] = xp.sign(df["uplot"]) * xp.log1p(xp.abs(df["uplot"]))

    # title
    title = ""
    if plot_uin:
        title += "Incident Field"
    if xp.any(plot_uscateach_):
        if plot_uin:
            title += " + "
        title += "Scattered Field by Ball " + ", ".join(
            plot_uscateach_.nonzero()[0].astype(str).to_numpy().tolist()
        )
    title += r"<br>"
    title += (
        f"{c.c_ndim:g}D, Type {c.branching_types_expression_str} coordinates, "
        f"Max Degree={biem_res.n_end - 1:g}, k={biem_res.k.item():g}, "
        f"η={biem_res.eta.item():g}"
    )

    plot_2d = px.scatter(
        df,
        x="x",
        y="y",
        color="uplot",
        range_color=[-df["uplot"].abs().max(), df["uplot"].abs().max()],
        range_x=[xspace_[0], xspace_[1]],
        range_y=[yspace_[0], yspace_[1]],
        custom_data=["t", "uin", "uscat", "uall"]
        + [f"uscat{i}" for i in range(ures.uscateach.shape[0])],
        hover_data={"t": False, "uin": True, "uscat": True, "uall": True}
        | {f"uscat{i}": True for i in range(ures.uscateach.shape[0])},
        animation_frame="t",
        title=title,
        labels={
            "x": f"x{xaxis}",
            "y": f"x{yaxis}",
        },
        color_continuous_scale="RdBu_r",
        **plot_kwargs,
    )
    return plot_2d
