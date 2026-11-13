# AGENTS.md — `stokes` Solver Architecture & Agent Guide

## Project Overview
`stokes` is a 2D Updated Lagrangian finite element solver for glacier mechanics and continuum damage evolution. It pairs a high-level Python simulation driver (`src/stokes/`) with performance-critical C99 + NumPy C-API numerical kernels (`csrc/`).

## Finite Element Discretization & Invariants
- **Taylor-Hood ($Q_8\text{--}Q_4$) Velocity-Pressure Pair:**
  - **Velocity ($\mathbf{u}$):** 8-node biquadratic serendipity elements (`cells` shape `(num_cells, 8)`). Nodes `0..3` are counter-clockwise (CCW) corners: `(-1,-1), (+1,-1), (+1,+1), (-1,+1)`. Nodes `4..7` are CCW mid-side edge nodes: `(0,-1), (+1,0), (0,+1), (-1,0)`.
  - **Pressure ($p$):** 4-node bilinear elements using the corner vertices (`cells[:, :4]`).
- **Element DOF Ordering (`20` DOFs per cell):**
  - Indices `0..15`: Interleaved velocity DOFs `[u0, v0, u1, v1, ..., u7, v7]`.
  - Indices `16..19`: Corner pressure DOFs `[p0, p1, p2, p3]`.
- **Quadrature:** 9-point ($3 \times 3$) tensor-product Gauss-Legendre rule (`Q9`).
  - Coordinates: $\xi, \eta \in \{-\sqrt{0.6}, 0, +\sqrt{0.6}\}$.
  - Weights: $25/81$ (corners), $40/81$ (edges), $64/81$ (center, index 4). Sum of weights is $4.0$.
- **Strain Rate Convention:**
  - 3-component engineering Voigt vector $\mathbf{d} = [\dot{\varepsilon}_{xx}, \dot{\varepsilon}_{yy}, \dot{\gamma}_{xy}]$ where $\dot{\gamma}_{xy} = \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} = 2\dot{\varepsilon}_{xy}$.
  - Second invariant squared: $\dot{\varepsilon}_e^2 = \frac{1}{2}(d_0^2 + d_1^2 + \frac{1}{2}d_2^2)$.

## Physical Units & Sign Conventions
- **Coordinates & Mesh:** meters ($\text{m}$)
- **Velocity ($\mathbf{u}$):** meters per year ($\text{m/yr}$)
- **Pressure ($p$) & Cauchy/Deviatoric Stress ($\boldsymbol{\sigma}, \boldsymbol{\tau}$):** Pascals ($\text{Pa}$). Pressure $p$ is positive in compression ($\boldsymbol{\sigma} = \boldsymbol{\tau} - p\mathbf{I}$).
- **Glen's Flow Law Softness ($B$, `b_glen`):** $\text{Pa} \cdot \text{yr}^{1/n}$ (with $n = 3$).
- **Damage Law Rate ($\hat{B}$, `b_hat`):** $\text{MPa}^{-r} \cdot \text{s}^{-1}$ (stresses converted to $\\text{MPa}$ via $10^{-6}$ factor; time steps converted to seconds for damage integration).

## Build, Lint & Test Commands
This repository is managed with `uv`. Always run commands via `uv`:
- **Compile / Recompile C Extension (`stokes._core`):**
  ```bash
  uv pip install -e . --no-build-isolation
  ```
  *(Must be run after any change to `csrc/*.c`, `csrc/*.h`, or `setup.py`.)*
- **Verify C Extension Bindings:**
  ```bash
  uv run python -c "import stokes._core; print('stokes._core loaded successfully')"
  ```
- **Lint & Format Python Code:**
  ```bash
  uv run ruff check --fix
  uv run ruff format
  ```
- **Run Tests:**
  ```bash
  uv run pytest
  ```

## Repository Structure
- `csrc/`: Pure C99 numerical kernels and Python C-API wrapper.
- `src/stokes/`: Python package.
- `setup.py` & `pyproject.toml`: Build configuration compiling `csrc/` into `stokes._core`.
