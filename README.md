# Stokes Flow

`stokes` is a 2D Updated Lagrangian finite element solver for glacier mechanics and variational phase-field fracture modeling.

The solver pairs a high-level Python simulation driver with high-performance C99 numerical kernels (`stokes._core` via NumPy C-API). It uses mixed Taylor-Hood ($Q_8\text{--}Q_4$) serendipity elements for incompressible non-linear creeping flow (Glen's flow law) coupled to a variational phase-field formulation for crevasse nucleation and crack propagation.

## Features

- **Mixed $Q_8\text{--}Q_4$ Taylor-Hood Discretization:** 8-node biquadratic serendipity velocity elements paired with 4-node bilinear pressure elements, integrated via $3 \times 3$ Gauss-Legendre quadrature (`Q9`).
- **Variational Phase-Field Fracture:** Solved on the $Q_8$ velocity nodes (`mesh.num_points` DOFs) using an irreversibility history field $\mathcal{H}^+$ driven by the 3D plane-strain virtual tensile elastic strain energy.
- **Degraded Ice Viscosity:** Continuous deviatoric viscosity degradation $\mu = g(\phi)\mu_{\text{Glen}} = ((1 - \phi)^2 + \eta_{\text{res}})\mu_{\text{Glen}}$ with unscaled incompressibility constraints.
- **Topological Erosion:** Optional cell-averaged phase-field threshold ($\bar{\phi}_e \ge \text{erosion\_threshold}$) for removing fully fractured elements from the active mesh.
- **Zero-Allocation Sparse Assembly:** Precomputed COO-to-CSC sparsity mappings for $O(\text{nnz})$ system assembly.
- **ParaView Export:** Time-series VTU and PVD export with velocity, pressure, phase-field, Cauchy stress, and deviatoric stress tensors.

## Installation

This repository is managed with [`uv`](https://docs.astral.sh/uv/).

To install dependencies and compile the C extension:

```bash
uv sync
uv pip install -e . --no-build-isolation
```

Recompile after any modification to C source files in `csrc/` or `setup.py`:

```bash
uv pip install -e . --no-build-isolation
```

## Running Simulations

Run the simulation driver using `uv run` and a TOML input file:

```bash
uv run stokes simulation.toml
```

Alternatively, activate the virtual environment and execute `stokes` directly:

```bash
source .venv/bin/activate
stokes simulation.toml
```

## Configuration (TOML)

Simulations are configured using TOML files:

```toml
[mesh]
shape = [100, 25]
size = [500.0, 125.0]

[rheology]
apply_gravity = true
ice_density = 917.0        # kg/m^3
ice_temperature = 263.15    # K

[phase_field]
gc = 50.0                  # Fracture energy release rate (J/m^2)
length_scale = 10.0        # Regularization length scale ell (m)
youngs_modulus = 9.0e9     # Virtual elastic Young's modulus E (Pa)
poissons_ratio = 0.33      # Poisson's ratio
critical_stress = 0.0      # Tensile initiation threshold sigma_c (Pa)
residual_stiffness = 1.0e-4 # Degraded stiffness floor eta_res

[boundary_conditions]
left = "free slip"
bottom = "free slip"

[solver]
convergence_tolerance = 1.0e-6
max_iterations = 50
num_steps = 480
time_step = 1.0            # hours
strict_convergence = false

[plotting]
output_interval = 6
plot_cauchy_stress = true
plot_deviatoric_stress = true
plot_phase_field = true
plot_strain_rate = false
```

## References

[^jimenez2017]: Stephen Jiménez, Ravindra Duddu, Jeremy Bassis, *An updated-Lagrangian damage mechanics formulation for modeling the creeping flow and fracture of ice sheets*, Computer Methods in Applied Mechanics and Engineering, Volume 313, 2017, Pages 406–432, [https://doi.org/10.1016/j.cma.2016.09.034](https://doi.org/10.1016/j.cma.2016.09.034).
