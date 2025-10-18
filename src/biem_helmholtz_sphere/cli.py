from logging import DEBUG, INFO, basicConfig, getLogger
from pathlib import Path
from typing import Any, Literal

import typer
from array_api._2024_12 import ArrayNamespaceFull
from rich.logging import RichHandler
from tqdm.rich import tqdm_rich
from ultrasphere import create_from_branching_types

from biem_helmholtz_sphere.bempp_cl_sphere import bempp_cl_sphere
from biem_helmholtz_sphere.biem import BIEMResultCalculator, biem, plane_wave

from .gui import serve as serve_plot

app = typer.Typer()
LOG = getLogger(__name__)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    basicConfig(handlers=[RichHandler(rich_tracebacks=True)], level=DEBUG if verbose else INFO)


@app.command()
def serve() -> None:
    """Serve panel app."""
    serve_plot()


@app.command()
def jascome(
    backend: Literal["numpy", "torch"] = "numpy",
    device: str = "cpu",
    dtype: str = "float64",
    branching_types: str = "a,ba,bpa,bba,bpbpa,caa",
) -> None:
    """Numerical examples for JASCOME."""
    branchin_types = branching_types.split(",")
    xp: ArrayNamespaceFull
    if backend == "numpy":
        from array_api_compat import numpy as xp  # type: ignore
    elif backend == "torch":
        from array_api_compat import torch as xp  # type: ignore
    with Path("jascome_output.csv").open("w") as f:
        f.write("branching_types,n_end,uscat_norm\n")
    for btype in tqdm_rich(list(reversed(branchin_types)), position=0):
        try:
            for n_end in tqdm_rich(list(range(1, 10)), position=1, leave=False):
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
                with Path("jascome_output.csv").open("a") as f:
                    f.write(
                        f"{btype},{n_end},{complex(uscat)},"
                        f"{device},{dtype},"
                        f"{calc.density.dtype},{calc.density.device},"  # type: ignore
                        f"{uscat.dtype},{uscat.device}\n"
                    )
        except Exception as e:
            LOG.error(e)
            continue


@app.command()
def jascome_bempp(
    min_h: float = 0.05,
) -> None:
    """Numerical examples for JASCOME using Bempp-cl."""
    import numpy as np

    with Path("jascome_bempp_output.csv").open("w") as f:
        f.write("h,n_elements,uscat_norm\n")
    for h in tqdm_rich((2.0 ** -np.arange(1, int(-np.log2(min_h)) + 1)), position=0):
        calc = bempp_cl_sphere(
            k=1.0,
            h=h,
            centers=(
                (0.0, 2.0, 0.0),
                (0.0, -2.0, 0.0),
            ),
            radii=(1.0, 1.0),
        )
        uscat = calc(np.asarray((0.0,)), np.asarray((0.0,)), np.asarray((0.0,)))
        with Path("jascome_bempp_output.csv").open("a") as f:
            f.write(f"{h},{calc.grid.number_of_elements},{complex(uscat)}\n")  # type: ignore
