"""Command line interface for Stokes flow."""

from stokes.config import SimulationConfig
from stokes.simulation import execute


def main() -> None:
    """Entry point to the code.."""
    config = SimulationConfig.from_toml("simulation.toml")
    execute(config)
