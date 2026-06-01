"""Pydantic models for the simulation configuration .yml. See configs/sim_config.yml."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class SimSceneConfig(BaseModel):
    """Bounds and sampling of the 2D (x, z) simulation grid. All units in meters."""
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    spacing: float = Field(gt=0, description="Grid spacing in meters")

    @model_validator(mode="after")
    def _check_bounds(self) -> "SimSceneConfig":
        if self.x_max <= self.x_min:
            raise ValueError(f"x_max ({self.x_max}) must be greater than x_min ({self.x_min})")
        if self.z_max <= self.z_min:
            raise ValueError(f"z_max ({self.z_max}) must be greater than z_min ({self.z_min})")
        return self


class SimConfig(BaseModel):
    """Top-level simulation config."""
    sim_scene: SimSceneConfig
    plot_path: Path = Field(description="Path that plot_scene() writes the output figure to")


def load_config(path: Path) -> SimConfig:
    """Loads and validates a simulation config .yml into a SimConfig model."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return SimConfig(**raw)
