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
    phase_model: Literal["achromatic", "delay"] = Field(default="delay",
        description="What one profile is shared across frequencies in a joint solve: "
                    "'delay' models a physical plate (per-frequency phase = (f/f_ref) "
                    "times one shared profile — phase scales with wavenumber, like both "
                    "simulated beams); 'achromatic' models a mask imposing the identical "
                    "phase at every frequency. Identical at a single frequency.")
    init: Literal["random", "warm_start"] = Field(default="warm_start",
        description="Initial phase for a multi-frequency solve: 'warm_start' (default) "
                    "runs the multi-wavelength initialization — solve the reference "
                    "frequency alone, unwrap, scan the absolute offset over one "
                    "synthetic-wavelength period — before the joint descent (see "
                    "docs/delay_model_warm_start.md); 'random' starts from the seeded "
                    "random draw. A single-frequency solve always uses the random path "
                    "(the two are the same problem there).")
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
    dx: float = Field(gt=0,
        description="PROVENANCE ONLY: recorded in the manifest, but no longer used. "
                    "Candidate apertures are stored on the scene grid, so the scene's "
                    "`spacing` is what samples them.")


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


class ExperimentalConfig(BaseModel):
    """Bench constants for the experimental (.mat) path, used only by
    MGS.from_experiment. All lengths in meters.

    These were thirteen magic numbers spread across two modules. They describe one
    physical rig: how the capture is down-converted, where the TX aperture's edges
    sit in rig coordinates, and how rig coordinates map into the scene.
    """
    lo_freq: float = Field(default=25e9, gt=0,
        description="Local-oscillator frequency of the receive chain (Hz)")
    trx_n: float = Field(default=6, gt=0,
        description="LO multiplication factor; the carrier is down-mixed to "
                    "freq - lo_freq*trx_n, which must be >= 0")
    rig_x_origin: float = Field(default=0.3,
        description="Rig x coordinates are mirrored about this point to get scene x")
    rig_z_origin: float = Field(default=0.3,
        description="Rig z coordinates are mirrored about this point to get scene z "
                    "(heatmap capture)")
    rx_z_origin: float = Field(default=0.35,
        description="RX plane height: rx.z = rx_z_origin - (captured z), measured "
                    "during experiment setup")
    tx_left_edge: float = Field(default=0.2558,
        description="TX aperture left edge in rig x coordinates")
    tx_right_edge: float = Field(default=0.1573,
        description="TX aperture right edge in rig x coordinates")
    rx_amplitude_scale: float = Field(default=6.0,
        description="The captured RX profile is renormalised to this peak amplitude, "
                    "so the solver's error weighting and step size behave the same "
                    "across captures")
    scene_x_margin: float = Field(default=0.2,
        description="Scene x extends this far either side of the TX aperture")
    scene_z_min: float = Field(default=0.0, description="Scene floor (m)")
    scene_z_max: float = Field(default=0.4, description="Scene ceiling (m)")


class SimConfig(BaseModel):
    """Top-level simulation config."""
    sim_scene: SimSceneConfig
    tx_aperture: TxApertureConfig = Field(default_factory=TxApertureConfig)
    rx_aperture: RxApertureConfig = Field(default_factory=RxApertureConfig)
    plot_path: Path = Field(description="Path that plot_scene() writes the output figure to")
    gerchberg_saxton: GerchbergSaxtonConfig = Field(default_factory=GerchbergSaxtonConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    experimental: ExperimentalConfig = Field(default_factory=ExperimentalConfig,
        description="Bench constants for the experimental .mat path (ignored by "
                    "the simulated workflows)")
    grid_search: Optional[GridSearchConfig] = Field(default=None,
        description="Optional speculative TX-location sweep (used by grid-search-mgs)")
    frequencies: Optional[List[float]] = Field(default=None,
        description="Frequencies (Hz) both entry points solve at; null -> --freq, else 150e9. "
                    "More than one means ONE JOINT solve: a single phase mask fitted "
                    "against every frequency at once, the loss being the mean of the "
                    "per-frequency losses.")

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


def center_freq_index(freqs: List[float]) -> int:
    """Index of the centre-by-value frequency: the median of the sorted list.

    One convention, three consumers: the delay solver's reference frequency, the
    default display frequency for scene renders, and the scatter-3d-diff baseline.
    Lives here (not in a consumer module) so mgs.py can use it without an import
    cycle."""
    order = sorted(range(len(freqs)), key=lambda i: freqs[i])
    return order[len(order) // 2]


def resolve_frequencies(config: "SimConfig",
                        cli_freqs: Optional[List[float]]) -> List[float]:
    """Resolve the frequency list: CLI override -> config.frequencies -> [150e9].

    Shared by both entry points (`mgs` and `grid-search-mgs`), so it lives here on
    neutral ground rather than in either of them. Duplicates are dropped
    (order-preserving): a repeated frequency adds no information and would be
    double-counted wherever the frequencies are combined.
    """
    if cli_freqs:
        freqs = [float(f) for f in cli_freqs]
    elif config.frequencies:
        freqs = [float(f) for f in config.frequencies]
    else:
        return [150e9]
    return list(dict.fromkeys(freqs))


# Default config shipped in the repo's configs/ folder, used by both entry points
# when --config is omitted.
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "caustic_config.yml"


def load_config(path: Path) -> SimConfig:
    """Loads and validates a simulation config .yml into a SimConfig model."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return SimConfig(**raw)
