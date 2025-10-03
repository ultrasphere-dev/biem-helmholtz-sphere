from ultrasphere import create_from_branching_types
from biem_helmholtz_sphere.biem import biem
from array_api._2024_12 import ArrayNamespaceFull
def test_biem_batch(xp: ArrayNamespaceFull) -> None:
    c = create_from_branching_types("a")
    # number of dimensions: 2
    # batch shape: (3,)
    # number of spheres: 4
    biem(
        c,
        xp.asarray([1.0, 0.0]),
        k=xp.asarray([1, 0.5, 0.3]),
        n_end=3,
        eta=xp.asarray([1, 0.1, 0.3]),
        centers=xp.asarray([[[0.0, 2.0], [0.0, -2.0], [4.0, 3.0], [5.0, 0.0]]]),
        radii=xp.asarray([[1.0, 1.0, 0.3, 0.2]]),
    )