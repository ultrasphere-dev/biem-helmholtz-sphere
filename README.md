# biem-helmholtz-sphere

<p align="center">
  <a href="https://github.com/ultrasphere-dev/biem-helmholtz-sphere/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/ultrasphere-dev/biem-helmholtz-sphere/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://biem-helmholtz-sphere.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/biem-helmholtz-sphere.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/ultrasphere-dev/biem-helmholtz-sphere">
    <img src="https://img.shields.io/codecov/c/github/ultrasphere-dev/biem-helmholtz-sphere.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/biem-helmholtz-sphere/">
    <img src="https://img.shields.io/pypi/v/biem-helmholtz-sphere.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/biem-helmholtz-sphere.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/biem-helmholtz-sphere.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://biem-helmholtz-sphere.readthedocs.io" target="_blank">https://biem-helmholtz-sphere.readthedocs.io </a>

**Source Code**: <a href="https://github.com/ultrasphere-dev/biem-helmholtz-sphere" target="_blank">https://github.com/ultrasphere-dev/biem-helmholtz-sphere </a>

---

Acoustic scattering from multiple n-spheres in NumPy / PyTorch

## Installation

Install this via pip (or your favourite package manager):

`pip install biem-helmholtz-sphere`

## Usage (GUI)

![GUI](https://raw.githubusercontent.com/ultrasphere-dev/biem-helmholtz-sphere/main/gui.webp)

```shell
uvx biem-helmholtz-sphere serve
```

## Usage

```python
>>> from array_api_compat import numpy as xp
>>> from biem_helmholtz_sphere import BIEMResultCalculator, biem, plane_wave
>>> from ultrasphere import create_from_branching_types
>>> c = create_from_branching_types("ba")
>>> uin, uin_grad = plane_wave(k=xp.asarray(1.0), direction=xp.asarray((1.0, 0.0, 0.0)))
>>> calc = biem(c, uin=uin, uin_grad=uin_grad, k=xp.asarray(1.0), n_end=3, eta=xp.asarray(1.0), centers=xp.asarray(((0.0, 2.0, 0.0), (0.0, -2.0, 0.0))), radii=xp.asarray((1.0, 1.0)), kind="outer")
>>> calc.uscat(xp.asarray((0.0, 0.0, 0.0)))
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
