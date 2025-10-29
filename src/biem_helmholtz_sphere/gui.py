from typing import Any, Literal

import pandas as pd
import panel as pn
from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import numpy as np
from array_api_compat import torch
from ultrasphere import (
    create_from_branching_types,
    create_hopf,
    create_random,
    create_standard,
    create_standard_prime,
)

from .biem import BIEMResultCalculator, biem, max_n_end, plane_wave
from .plot import plot_biem


def serve() -> None:
    """Serve panel app."""
    xp: ArrayNamespaceFull = np
    res: BIEMResultCalculator[Any, Any] | None = None
    rescountw = pn.widgets.IntInput(name="Result count", value=0)
    # coordinates
    backendw = pn.widgets.ToggleGroup(
        name="Backend", options=["numpy", "torch"], behavior="radio", value="numpy"
    )
    dw = pn.widgets.IntSlider(name="Number of dimensions", value=2, start=2, end=7)
    ctypew = pn.widgets.ToggleGroup(
        name="Coordinates",
        options=["standard", "standard_prime", "hopf", "random", "custom"],
        behavior="radio",
    )
    ccustomw = pn.widgets.TextAreaInput(name="Custom coordinates", value="ba")
    g_coordinates = pn.WidgetBox(
        "## Coordinates",
        dw,
        ctypew,
        ccustomw,
    )

    # calculation parameters
    kindw = pn.widgets.Select(name="Outer/Inner", options=["outer", "inner"])
    k_rew = pn.widgets.FloatInput(name="Wavenumber k (Re)", value=1)
    k_imw = pn.widgets.FloatInput(name="Wavenumber k (Im)", value=0)
    etaw = pn.widgets.FloatInput(name="Decoupling parameter eta", value=1)
    force_matrixw = pn.widgets.Checkbox(name="Force matrix", value=False)
    n_endw = pn.widgets.IntSlider(name="Maximum degree", value=5, start=1, end=40)
    radiuscenter_addw = pn.widgets.Button(name="Add sphere", button_type="primary")
    radiuscenter_removew = pn.widgets.Button(name="Remove sphere", button_type="danger")
    radiuscenterw = pn.widgets.Tabulator(
        pd.DataFrame(
            {
                "alpha": [1.0, 1.0],
                "beta": [0.0, 0.0],
                "radius": [1.0, 1.0],
                0: [0.0, 0.0],
                1: [2.0, -2.0],
            }
        ),
        show_index=False,
    )
    g_calculation = pn.WidgetBox(
        "## Calculation",
        kindw,
        k_rew,
        k_imw,
        etaw,
        force_matrixw,
        n_endw,
        pn.Row(radiuscenter_addw, radiuscenter_removew),
        radiuscenterw,
    )

    # plot
    plot_whichw = pn.widgets.ToggleGroup(
        name="Plot", options=["uin", "uscat0"], value=["uin", "uscat0"]
    )
    r_plotw = pn.widgets.FloatInput(name="Plot radius", value=4)
    n_plotw = pn.widgets.IntSlider(name="Points to plot", value=60, start=1, end=200)
    n_tw = pn.widgets.IntSlider(name="Time count", value=4, start=1, end=50)
    axisxw = pn.widgets.IntSlider(name="Axis x", value=0, start=0, end=1)
    axisyw = pn.widgets.IntSlider(name="Axis y", value=1, start=0, end=1)
    g_plot = pn.WidgetBox(
        "## Plot",
        plot_whichw,
        r_plotw,
        n_plotw,
        n_tw,
        axisxw,
        axisyw,
    )

    downloadsvgw = pn.widgets.FileDownload(file="plot.svg")
    downloadpngw = pn.widgets.FileDownload(file="plot.png")
    downloadjpgw = pn.widgets.FileDownload(file="plot.jpg")
    downloaddataw = pn.widgets.FileDownload(file="data.csv")
    g_download = pn.WidgetBox(
        "## Download", pn.Row(downloadsvgw, downloadpngw, downloadjpgw, downloaddataw)
    )

    progressw = pn.widgets.Progress(name="Progress", value=0, max=100)

    @pn.depends(backendw.param.value)
    def update_backend(backend: str) -> None:
        nonlocal xp
        if backend == "numpy":
            xp = np
        elif backend == "torch":
            xp = torch
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @pn.depends(dw.param.value, ctypew.param.value)
    def update_custom(d: int, ctype: str) -> None:
        if ctype == "custom":
            if ccustomw.disabled:
                ccustomw.disabled = False
        else:
            ccustomw.disabled = True
            if ctype == "standard":
                ccustomw.value = create_standard(d - 1).branching_types_expression_str
            elif ctype == "standard_prime":
                ccustomw.value = create_standard_prime(d - 1).branching_types_expression_str
            elif ctype == "hopf":
                if xp.pow(int(xp.log2(d)), 2) == d:
                    ccustomw.value = create_hopf(int(xp.log2(d))).branching_types_expression_str
                else:
                    raise ValueError(f"d must be a power of 2, but {d}")
            elif ctype == "random":
                ccustomw.value = create_random(d - 1).branching_types_expression_str
            else:
                raise ValueError(f"Invalid cstr: {ctype}")

    @pn.depends(ccustomw.param.value)
    def update_d_from_custom(cstr: str) -> None:
        if ctypew.value == "custom":
            d = create_from_branching_types(cstr).c_ndim
            if dw.value != d:
                dw.value = d

    @pn.depends(dw.param.value, radiuscenterw.param.value)
    def update_n_end(d: int, radiuscenter: pd.DataFrame) -> None:
        from psutil import virtual_memory

        n_endw.end = min(
            max_n_end(
                c_ndim=d, memory_limit=virtual_memory().available // 16, n_balls=len(radiuscenter)
            ),
            20,
        )
        n_endw.value = min(n_endw.value, n_endw.end)

    @pn.depends(dw.param.value)
    def update_axis(d: int) -> None:
        axisxw.end = d - 1
        axisyw.end = d - 1
        radiuscenterw_int_columns = [
            col for col in radiuscenterw.value.columns if isinstance(col, int)
        ]
        if len(radiuscenterw_int_columns) != d:
            if len(radiuscenterw_int_columns) > d:
                new = radiuscenterw.value.drop(columns=radiuscenterw_int_columns[d:])
            else:
                new = pd.concat(
                    (
                        radiuscenterw.value,
                        pd.DataFrame(dtype=float, columns=range(len(radiuscenterw_int_columns), d)),
                    ),
                    axis=1,
                ).fillna(0)
            radiuscenterw.value = new

    @pn.depends(radiuscenterw.param.value)
    def update_plot_which(radiuscenter: pd.DataFrame) -> None:
        old_options = plot_whichw.options
        new_options = ["uin"] + [f"uscat{i}" for i in range(len(radiuscenter))]
        plot_whichw.options = new_options
        if len(old_options) < len(new_options):
            plot_whichw.value = plot_whichw.value + new_options[len(old_options) :]

    def add_sphere(_: Any) -> None:
        radiuscenterw.value = pd.concat(
            (
                radiuscenterw.value,
                pd.Series(
                    {  # type: ignore
                        "radius": 1.0,
                        "alpha": 1.0,
                        "beta": 0.0,
                        **dict.fromkeys(range(dw.value), 0.0),
                    },
                    name=len(radiuscenterw.value),
                )
                .to_frame()
                .T,
            )
        )

    def remove_sphere(_: Any) -> None:
        if len(radiuscenterw.value) > 0:
            radiuscenterw.value = radiuscenterw.value.iloc[:-1]

    radiuscenter_addw.on_click(add_sphere)
    radiuscenter_removew.on_click(remove_sphere)

    @pn.depends(
        ccustomw.param.value,
        k_rew.param.value,
        k_imw.param.value,
        etaw.param.value,
        force_matrixw.param.value,
        radiuscenterw.param.value,
        n_endw.param.value,
        kindw.param.value,
    )
    def update_sol(
        cstr: str,
        k_re: float,
        k_im: float,
        eta: float,
        force_matrix: bool,
        radiuscenter: Array,
        n_end: int,
        kind: Literal["inner", "outer"],
    ) -> None:
        nonlocal res
        if k_im != 0:
            k = complex(k_re, k_im)
        else:
            k = k_re
        progressw.value = 0
        progressw.active = True
        progressw.bar_color = "primary"
        c = create_from_branching_types(cstr)
        d = c.c_ndim
        if (d - 1) not in radiuscenter.columns:
            progressw.active = False
            progressw.bar_color = "danger"
            progressw.value = 100
            return
        res = biem(
            c,
            uin=plane_wave(k=xp.asarray(k), direction=xp.asarray((1,) + (0,) * (d - 1))),
            k=k,
            n_end=n_end,
            eta=eta,
            centers=xp.asarray(radiuscenter[list(range(d))]),
            radii=xp.asarray(radiuscenter["radius"]),
            alpha=xp.asarray(radiuscenter["alpha"]),
            beta=xp.asarray(radiuscenter["beta"]),
            kind=kind,
            force_matrix=force_matrix,
        )
        rescountw.value = rescountw.value + 1

    @pn.depends(
        plot_whichw.param.value,
        n_plotw.param.value,
        n_tw.param.value,
        r_plotw.param.value,
        axisxw.param.value,
        axisyw.param.value,
        rescountw.param.value,
    )
    def update_plot(
        plot_which: list[str],
        n_plot: int,
        n_t: int,
        r_plot: float,
        xaxis: int,
        yaxis: int,
        _: int,
    ) -> pn.pane.Pane | None:
        nonlocal res
        if res is None:
            return None
        progressw.value = 50
        progressw.active = True
        progressw.bar_color = "secondary"

        # plot
        plot_2d = plot_biem(
            res,
            plot_uin="uin" in plot_which,
            plot_uscateach=xp.asarray(
                [f"uscat{i}" in plot_which for i in range(res.radii.shape[-1])]
            ),
            xspace=(-r_plot, r_plot, n_plot),
            yspace=(-r_plot, r_plot, n_plot),
            xaxis=xaxis,
            yaxis=yaxis,
            n_t=n_t,
            width=600,
            height=600,
        )
        plot_2d.update_yaxes(scaleanchor="x", scaleratio=1)
        plot_2d.update_layout(title_x=0.5)
        plot_2d.write_image("plot.svg")
        plot_2d.write_image("plot.png", scale=3)
        plot_2d.write_image("plot.jpg", scale=3)
        progressw.value = 100
        progressw.active = False
        progressw.bar_color = "success"
        return plot_2d

    def exception_handler(ex: Any) -> None:
        if pn.state.notifications is not None:
            pn.state.notifications.error(f"{ex}")

    layout = pn.Row(
        pn.Column(
            progressw,
            g_coordinates,
            g_calculation,
            g_plot,
            g_download,
        ),
        update_backend,
        update_custom,
        update_d_from_custom,
        update_n_end,
        update_axis,
        update_plot_which,
        update_sol,
        update_plot,
    )
    pn.serve(layout, exception_handler=exception_handler)
