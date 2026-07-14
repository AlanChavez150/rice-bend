"""Pydantic models for the simulation configuration .yml.
See configs/caustic_config.yml and configs/directional_config.yml."""

from pathlib import Path
from typing import List, Literal, Optional

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


class BeamConfig(BaseModel):
    """The beam emitted by the TX aperture and its type-specific parameters.

    - `caustic`: an accelerating beam following x(d) = a*d^2 + b*d + c, where d is
      the distance travelled from the TX aperture along the beam. Requires
      `trajectory = [a, b, c]`.
    - `directional`: a steered plane wave at `steer_angle_deg` off the -Z axis
      (phase ramp -k*x*sin(theta)). Requires `steer_angle_deg`."""
    type: Literal["caustic", "directional"] = "caustic"
    trajectory: Optional[List[float]] = Field(default=None,
        description="Caustic coefficients [a, b, c] for x(d)=a*d^2+b*d+c (caustic only)")
    steer_angle_deg: Optional[float] = Field(default=None,
        description="Steering angle in degrees off the -Z axis (directional only)")

    @model_validator(mode="after")
    def _check(self) -> "BeamConfig":
        if self.type == "caustic":
            if self.trajectory is None or len(self.trajectory) != 3:
                raise ValueError(
                    f"caustic beam requires trajectory = [a, b, c] (3 values), got {self.trajectory}")
        elif self.type == "directional":
            if self.steer_angle_deg is None:
                raise ValueError("directional beam requires steer_angle_deg")
        return self


class TxApertureConfig(BaseModel):
    """TX aperture geometry, sampling and emitted beam, defined independently of
    the scene grid. All units in meters.

    The aperture sits at height `z` and projects toward -Z (down) onto the RX at
    z=0; move it around by editing `z` (height) and `x_min`/`x_max` (lateral span).
    The emitted beam (caustic or directional) is configured in the `beam` block."""
    x_min: float = -0.35
    x_max: float = 0.35
    z: float = Field(default=2.0, description="TX plane height (beam travels -Z to the RX at z=0)")
    dx: float = Field(default=0.25e-3, gt=0, description="Aperture sampling spacing in meters")
    beam: BeamConfig = Field(default_factory=BeamConfig,
        description="The emitted beam (type: caustic | directional) and its parameters")

    @model_validator(mode="after")
    def _check(self) -> "TxApertureConfig":
        if self.x_max <= self.x_min:
            raise ValueError(f"x_max ({self.x_max}) must be greater than x_min ({self.x_min})")
        return self


class RxApertureConfig(BaseModel):
    """RX aperture geometry: a window of the given lateral `width` centered at
    `x_center`, sitting at the scene origin (z=0). All units in meters.

    Shrink `width` to model a smaller receiver. `dx` defaults to wavelength/20 at
    runtime when left null (matching the historical RX sampling)."""
    x_center: float = Field(default=0.0, description="Lateral center of the RX window (m)")
    width: float = Field(default=0.05, gt=0, description="Lateral span of the RX window (m)")
    dx: Optional[float] = Field(default=None, gt=0,
        description="RX sampling spacing (m); null -> wavelength/20 at runtime")


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


class AxisSweep(BaseModel):
    """A linear sweep from `min` to `max` over `num` inclusive points."""
    min: float
    max: float
    num: int = Field(gt=0, description="Number of points (1 yields just `min`)")

    @model_validator(mode="after")
    def _check(self) -> "AxisSweep":
        if self.max < self.min:
            raise ValueError(f"max ({self.max}) must be >= min ({self.min})")
        return self

    def values(self) -> List[float]:
        """Inclusive linspace from min..max (matches numpy.linspace endpoints)."""
        if self.num == 1:
            return [float(self.min)]
        step = (self.max - self.min) / (self.num - 1)
        return [float(self.min + i * step) for i in range(self.num)]


class GridApertureConfig(BaseModel):
    """The fixed speculative aperture placed at each grid point: a window of the
    given lateral `width` and sampling `dx`, with amplitude assumed uniform across
    it (the unknown real amplitude is not used by the search)."""
    width: float = Field(gt=0, description="Lateral span of the assumed aperture (m)")
    dx: float = Field(gt=0, description="Aperture sampling spacing (m)")


class GridGSOverrides(BaseModel):
    """GS hyperparameter overrides applied only during the grid sweep (e.g. fewer
    iterations for a cheaper search). Unset fields fall back to gerchberg_saxton."""
    max_iters: Optional[int] = Field(default=None, gt=0,
        description="Override gerchberg_saxton.max_iters during the sweep")


class GridSearchConfig(BaseModel):
    """Speculative (z, x_center) sweep for `grid-search-mgs`.

    The TX location is treated as unknown. At each grid point a fixed-width
    aperture (uniform assumed amplitude) is centered at `x_center` and placed at
    height `z`; MGS then reconstructs its phase against the single shared measured
    RX field. Each result is saved as a candidate beam."""
    z: AxisSweep
    x_center: AxisSweep
    aperture: GridApertureConfig
    seed: Optional[int] = Field(default=0,
        description="Fixed GS seed reused across every candidate so residuals are comparable")
    gs_overrides: GridGSOverrides = Field(default_factory=GridGSOverrides)


class SimConfig(BaseModel):
    """Top-level simulation config."""
    sim_scene: SimSceneConfig
    tx_aperture: TxApertureConfig = Field(default_factory=TxApertureConfig)
    rx_aperture: RxApertureConfig = Field(default_factory=RxApertureConfig)
    plot_path: Path = Field(description="Path that plot_scene() writes the output figure to")
    gerchberg_saxton: GerchbergSaxtonConfig = Field(default_factory=GerchbergSaxtonConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    grid_search: Optional[GridSearchConfig] = Field(default=None,
        description="Optional speculative TX-location sweep (used by grid-search-mgs)")
    frequencies: Optional[List[float]] = Field(default=None,
        description="Frequencies (Hz) to run the grid search at; null -> single --freq/150e9. "
                    "More than one runs the whole sweep independently per frequency.")

    @model_validator(mode="after")
    def _check_frequencies(self) -> "SimConfig":
        if self.frequencies is not None:
            if len(self.frequencies) == 0:
                raise ValueError("frequencies, if set, must be a non-empty list")
            if any(f <= 0 for f in self.frequencies):
                raise ValueError(f"all frequencies must be positive (got {self.frequencies})")
            if len(set(self.frequencies)) != len(self.frequencies):
                raise ValueError(f"frequencies must be unique (got {self.frequencies})")
        return self


def load_config(path: Path) -> SimConfig:
    """Loads and validates a simulation config .yml into a SimConfig model."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return SimConfig(**raw)
