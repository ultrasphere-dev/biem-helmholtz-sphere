from pathlib import Path
from typing import Any, Literal

import typer
from array_api._2024_12 import ArrayNamespaceFull
from ultrasphere import create_from_branching_types

from biem_helmholtz_sphere.bempp_cl_sphere import bempp_cl_sphere
from biem_helmholtz_sphere.biem import BIEMResultCalculator, biem, plane_wave

from .gui import serve as serve_plot

app = typer.Typer()


@app.command()
def serve() -> None:
    """Serve panel app."""
    serve_plot()


@app.command()
def jascome(
    backend: Literal["numpy", "torch"] = "numpy",
    device: str = "cpu",
    dtype: str = "float32",
    btanching_types: str = "a,ba,bpa,bba,bpbpa,caa",
) -> None:
    """Numerical examples for JASCOME."""
    branchin_types = btanching_types.split(",")
    xp: ArrayNamespaceFull
    if backend == "numpy":
        from array_api_compat import numpy as xp  # type: ignore
    elif backend == "torch":
        from array_api_compat import torch as xp  # type: ignore
    with Path("jascome_output.csv").open("w") as f:
        f.write("branching_types,n_end,uscat_norm\n")
        for btype in reversed(branchin_types):
            try:
                for n_end in range(10):
                    c = create_from_branching_types(btype)
                    calc: BIEMResultCalculator[Any, Any] = biem(
                        c,
                        uin=plane_wave(
                            k=xp.asarray(1.0, device=device, dtype=dtype),
                            direction=xp.asarray(
                                (1,) + (0.0,) * (c.c_ndim - 1), device=device, dtype=dtype
                            ),
                        ),
                        k=xp.asarray(1.0, device=device, dtype=dtype),
                        n_end=n_end,
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
                    uscat = calc.uscat(xp.asarray((0.0,) * c.c_ndim, device=device, dtype=dtype))
                    f.write(f"{btype},{n_end},{complex(uscat)},{device},{dtype}\n")
            except Exception as e:
                print(e)
                continue


@app.command()
def jascome_bempp(
    min_h: float = 0.05,
) -> None:
    """Numerical examples for JASCOME using Bempp-cl."""
    import numpy as np

    with Path("jascome_bempp_output.csv").open("w") as f:
        f.write("h,n_elements,uscat_norm\n")
        for h in np.logspace(np.log10(0.5), np.log10(min_h), 5):
            calc = bempp_cl_sphere(
                k=1.0,
                h=h,
                centers=(
                    (0.0, 2.0, 0.0),
                    (0.0, -2.0, 0.0),
                ),
                radii=(1.0, 1.0),
            )
            uscat = calc(0.0, 0.0, 0.0)
            f.write(f"{h},{calc.grid.number_of_elements},{complex(uscat)}\n")  # type: ignore
