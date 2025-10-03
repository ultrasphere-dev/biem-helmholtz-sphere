import typer

from .gui import serve as serve_plot

app = typer.Typer()


@app.command()
def serve() -> None:
    """Serve panel app."""
    serve_plot()
