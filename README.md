# Stokes Flow

This is a code for running 2D glacier dynamics simulations.

The finite element method (FEM) is used to discretize the Stokes equations of motion following the updated-Lagrangian approach in [^jimenez2017]. Mixed Q8-Q4 (Taylor-Hood) elements are used to model the incompressible flow. Creep damage mechanics is used to simulate crevasse nucleation and propagation.

## Installation

The *Stokes Flow* code requires the [`uv`](https://docs.astral.sh/uv/) package manage to install and execute. Download the required packages and installing supporting C code can be achieved by navigating into the repo and executing the following command:

```bash
uv sync
```

Should you choose to modify the supporting C code for system assembly, you can reinstall it by executing the following from the root level of the repo:

```bash
uv pip install -e . --no-build-isolation
```

## Executing the Code

You can execute the *Stokes Flow* code within the repo by executing the following command:

```bash
uv run stokes simulation.toml
```

If you prefer to execute the code outside the repo (which is recommended), you can do so by first activating the Python virtual environment:

```bash
source /path/to/stokes/.venv/bin/activate
```

Once activated, `stokes` is available directly on your `PATH`:

```bash
stokes simulation.toml
```

## Examples

More documentation to come.

For now, see the examples in the `examples/` directory to learn how to set up TOML input files.

[^jimenez2017]: Stephen Jiménez, Ravindra Duddu, Jeremy Bassis, *An updated-Lagrangian damage mechanics formulation for modeling the creeping flow and fracture of ice sheets*, Computer Methods in Applied Mechanics and Engineering, Volume 313, 2017, Pages 406–432, [https://doi.org/10.1016/j.cma.2016.09.034](https://doi.org/10.1016/j.cma.2016.09.034).
