"""Pydantic models for the simulation configuration .yml. See configs/sim_config.yml."""

from pathlib import Path
from typing import List, Optional

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


class TxApertureConfig(BaseModel):
    """TX aperture geometry, sampling and emitted caustic trajectory, defined
    independently of the scene grid. All units in meters.

    The aperture sits at height `z` and projects toward -Z (down) onto the RX at
    z=0; move it around by editing `z` (height) and `x_min`/`x_max` (lateral span)."""
    x_min: float = -0.35
    x_max: float = 0.35
    z: float = Field(default=2.0, description="TX plane height (beam travels -Z to the RX at z=0)")
    dx: float = Field(default=0.25e-3, gt=0, description="Aperture sampling spacing in meters")
    # caustic trajectory x(d) = a*d^2 + b*d + c, where d is the distance from the TX
    trajectory: List[float] = Field(default_factory=lambda: [0.075, 0.0, 0.0],
        description="Caustic coefficients [a, b, c] for x(d) = a*d^2 + b*d + c")

    @model_validator(mode="after")
    def _check(self) -> "TxApertureConfig":
        if self.x_max <= self.x_min:
            raise ValueError(f"x_max ({self.x_max}) must be greater than x_min ({self.x_min})")
        if len(self.trajectory) != 3:
            raise ValueError(f"trajectory must be [a, b, c] (3 values), got {self.trajectory}")
        return self


class GerchbergSaxtonConfig(BaseModel):
    """Hyperparameters and start/stop conditions for the modified Gerchberg-Saxton solver."""
    max_iters: int = Field(default=10000, gt=0, description="Max iterations before stopping")
    convergence_count: int = Field(default=10, gt=0,
        description="Window of recent losses checked for flatness")
    convergence_threshold: float = Field(default=-1e-10,
        description="Converged when mean(diff(recent loss)) exceeds this value")
    lr0: float = Field(default=0.05, gt=0, description="Initial backtracking step size")
    bt_shrink: float = Field(default=0.5, gt=0, lt=1, description="Backtracking shrink factor")
    bt_tries: int = Field(default=8, gt=0, description="Max backtracking reductions per iteration")
    seed: Optional[int] = Field(default=None,
        description="RNG seed for the initial phase. If null, one is drawn and recorded.")
    history_stride: int = Field(default=50, gt=0,
        description="Capture phase/RX-field every Nth iteration (1 = every iteration)")


class OutputConfig(BaseModel):
    """Where and what to persist for each run."""
    output_dir: Path = Field(default=Path("results"),
        description="Base dir; each run is saved to <output_dir>/<run_name>")
    run_name: Optional[str] = Field(default="data_dump",
        description="Run directory name under output_dir (existing contents are cleared)")
    save_run: bool = Field(default=True, description="Master toggle for persistence")
    save_scene_fields: bool = Field(default=False,
        description="Persist the large 2D complex scene + reconstruction fields")
    save_gs_history: bool = Field(default=True,
        description="Persist strided per-iteration GS history arrays")


class SimConfig(BaseModel):
    """Top-level simulation config."""
    sim_scene: SimSceneConfig
    tx_aperture: TxApertureConfig = Field(default_factory=TxApertureConfig)
    plot_path: Path = Field(description="Path that plot_scene() writes the output figure to")
    gerchberg_saxton: GerchbergSaxtonConfig = Field(default_factory=GerchbergSaxtonConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: Path) -> SimConfig:
    """Loads and validates a simulation config .yml into a SimConfig model."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return SimConfig(**raw)
