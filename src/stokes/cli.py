"""Command line interface for Stokes flow."""

import sys

from stokes.config import SimulationConfig
from stokes.simulation import execute


def main() -> None:
    """Entry point to the code."""
    file = "simulation.toml" if len(sys.argv) == 1 else sys.argv[1]
    config = SimulationConfig.from_toml(file)
    execute(config)
