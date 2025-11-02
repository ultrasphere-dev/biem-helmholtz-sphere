from logging import getLogger
from typing import Any, Literal

import pandas as pd
import panel as pn
import panel_material_ui as pmui
from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import numpy, torch
from array_api_compat import numpy as np
from ultrasphere import (
    create_from_branching_types,
    create_hopf,
    create_random,
    create_standard,
    create_standard_prime,
)

from .biem import BIEMResultCalculator, biem, max_n_end, plane_wave
from .plot import plot_biem

LOG = getLogger(__name__)


def serve() -> None:
    """Serve panel app."""
    pn.extension("katex", "mathjax", "plotly")
    res: BIEMResultCalculator[Any, Any] | None = None
    rescountw = pn.widgets.IntInput(name="Result count", value=0)
    # coordinates
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
        sizing_mode="stretch_width",
    )

    # calculation parameters
    backendw = pn.widgets.ToggleGroup(
        name="Backend",
        options={
            "numpy": numpy,
            "torch": torch,
        },
        behavior="radio",
    )
    devicew = pn.widgets.ToggleGroup(name="Device", behavior="radio")
    dtypew = pn.widgets.ToggleGroup(name="Dtype", behavior="radio")
    kindw = pn.widgets.Select(name="Outer/Inner", options=["outer", "inner"])
    k_rew = pn.widgets.FloatInput(name="Wavenumber k (Re)", value=1)
    k_imw = pn.widgets.FloatInput(name="Wavenumber k (Im)", value=0)
    etaw = pn.widgets.FloatInput(name="Decoupling parameter eta", value=1)
    force_matrixw = pn.widgets.Checkbox(name="Force matrix", value=False)
    n_endw = pn.widgets.IntSlider(name="Maximum degree + 1", value=5, start=1, end=40)
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
        backendw,
        dtypew,
        devicew,
        kindw,
        k_rew,
        k_imw,
        etaw,
        force_matrixw,
        n_endw,
        pn.Row(radiuscenter_addw, radiuscenter_removew),
        radiuscenterw,
        pn.pane.LaTeX(
            r"""$
        \begin{aligned}
        \Delta u + k^2 u = 0 \quad &x \in \mathbb{R}^d \setminus \overline{\mathbb{S}^{d-1}} \\
        \alpha u + \beta \nabla u \cdot n_x
        = -\alpha u_\text{in} -\beta \nabla u_\text{in} \cdot n_x \quad
        &x \in \mathbb{S}^{d-1} \\
        \lim_{\|x\| \to \infty} \|x\|^{\frac{d-1}{2}}
        \left( \frac{\partial u}{\partial \|x\|} - i k u \right) = 0 \quad
        &\frac{x}{\|x\|} \in \mathbb{S}^{d-1}
        \end{aligned}
        $""",
            renderer="mathjax",
            styles={"font-size": "10pt"},
        ),
        sizing_mode="stretch_width",
    )

    # plot
    plot_whichw = pn.widgets.ToggleGroup(
        name="Plot", options=["uin", "uscat0"], value=["uin", "uscat0"]
    )
    r_plotw = pn.widgets.FloatInput(name="Plot radius", value=4)
    n_plotw = pn.widgets.IntSlider(name="Points to plot", value=60, start=1, end=200)
    n_tw = pn.widgets.IntSlider(name="Time count", value=1, start=1, end=50)
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
        sizing_mode="stretch_width",
    )

    downloadsvgw = pn.widgets.FileDownload(file="plot.svg")
    downloadpngw = pn.widgets.FileDownload(file="plot.png")
    downloadjpgw = pn.widgets.FileDownload(file="plot.jpg")
    downloaddataw = pn.widgets.FileDownload(file="data.csv")
    g_download = pn.WidgetBox(
        "## Download",
        downloadsvgw,
        downloadpngw,
        downloadjpgw,
        downloaddataw,
        sizing_mode="stretch_width",
    )

    progressw = pn.widgets.Progress(name="Progress", value=0, max=100)

    @pn.depends(backendw.param.value, on_init=True)
    def update_device(xp: ArrayNamespaceFull) -> None:
        devices = xp.__array_namespace_info__().devices()
        devicew.options = devices
        if devicew.value is None or devicew.value not in devices:
            devicew.value = next(iter(devices))

    @pn.depends(backendw.param.value, devicew.param.value, on_init=True)
    def update_dtype(xp: ArrayNamespaceFull, device: Any) -> None:
        if device is None or device not in xp.__array_namespace_info__().devices():
            LOG.debug(f"update_dtype: {device=} is not available")
            return
        dtypes = xp.__array_namespace_info__().dtypes(device=device, kind="real floating")
        dtypew.options = dtypes
        if dtypew.value is None or dtypew.value not in dtypes.values():
            dtypew.value = next(iter(dtypes.values()))

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
                if np.pow(int(np.log2(d)), 2) == d:
                    ccustomw.value = create_hopf(int(np.log2(d))).branching_types_expression_str
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
            30,
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
        backendw.param.value,
        devicew.param.value,
        dtypew.param.value,
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
        xp: ArrayNamespaceFull,
        device: Any,
        dtype: Any,
    ) -> None:
        if device is None or device not in xp.__array_namespace_info__().devices():
            LOG.debug(f"{device=} is not available, skip calculation")
            return
        if (
            dtype is None
            or dtype
            not in xp.__array_namespace_info__()
            .dtypes(device=device, kind="real floating")
            .values()
        ):
            LOG.debug(f"{dtype=} is not available, skip calculation")
            return
        dtype_complex = xp.result_type(xp.complex64, dtype)
        nonlocal res
        if k_im != 0:
            k = complex(k_re, k_im)
            dtype_k = dtype_complex
        else:
            k = k_re
            dtype_k = dtype
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
        uin, uin_grad = plane_wave(
            k=xp.asarray(k, device=device, dtype=dtype_k),
            direction=xp.asarray((1.0,) + (0.0,) * (d - 1), device=device, dtype=dtype),
        )
        res = biem(
            c,
            uin=uin,
            uin_grad=uin_grad,
            k=xp.asarray(k, device=device, dtype=dtype_k),
            n_end=n_end,
            eta=xp.asarray(eta, device=device, dtype=dtype),
            centers=xp.asarray(radiuscenter[list(range(d))].values, device=device, dtype=dtype),
            radii=xp.asarray(radiuscenter["radius"], device=device, dtype=dtype),
            alpha=xp.asarray(radiuscenter["alpha"], device=device, dtype=dtype_complex),
            beta=xp.asarray(radiuscenter["beta"], device=device, dtype=dtype_complex),
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
        xp = backendw.value
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
                [f"uscat{i}" in plot_which for i in range(res.radii.shape[-1])],
                device=devicew.value,
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
        return pn.pane.Plotly(plot_2d, sizing_mode="stretch_both")

    def exception_handler(ex: Any) -> None:
        if pn.state.notifications is not None:
            pn.state.notifications.error(f"{ex}")

    pmui.Page(
        dark_theme=True,
        title="Acoustic Scattering by Multiple Spheres",
        main=[update_plot],
        sidebar=[
            pn.Row(
                pn.Column(
                    progressw,
                    g_coordinates,
                    g_calculation,
                    g_plot,
                    g_download,
                    update_custom,
                    update_d_from_custom,
                    update_n_end,
                    update_axis,
                    update_plot_which,
                    update_sol,
                    update_dtype,
                    update_device,
                ),
            )
        ],
        sidebar_width=510,
    ).show()
