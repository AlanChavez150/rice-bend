"""Pydantic models for the simulation configuration .yml.
See configs/caustic_config.yml, directional_config.yml, bessel_config.yml and
focus_config.yml."""

import argparse
from pathlib import Path
from typing import List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


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
    """The beam emitted by the TX aperture and its type-specific parameters
    (see rice_bend/beam_aperatures.py for the constructions).

    - `caustic`: an accelerating beam following x(d) = a*d^2 + b*d + c, where d is
      the distance travelled from the TX aperture along the beam. Requires
      `trajectory = [a, b, c]`.
    - `directional`: a steered plane wave at `steer_angle_deg` off the -Z axis
      (phase ramp -k*x*sin(theta)). Requires `steer_angle_deg`.
    - `bessel`: an axicon, two plane-wave halves crossing at `cone_angle_deg` about
      the aperture centre. They interfere into non-spreading fringes of period
      wavelength/(2*sin(cone)) down to (width/2)/tan(cone) below the plate. Requires
      `cone_angle_deg`; `steer_angle_deg` optionally tilts the axis.
    - `focus`: a lens converging on the point `focal_length` from the aperture
      centre, along the ray `steer_angle_deg` off the -Z axis. Requires
      `focal_length`; `steer_angle_deg` is optional.

    A positive `steer_angle_deg` steers toward -x for every type that takes one."""
    # steer_angle_deg is optional for bessel and focus, so a typo (steer_angle) must
    # not silently leave the beam pointing straight down
    model_config = ConfigDict(extra="forbid")

    type: Literal["caustic", "directional", "bessel", "focus"] = "caustic"
    trajectory: Optional[List[float]] = Field(default=None,
        description="Caustic coefficients [a, b, c] for x(d)=a*d^2+b*d+c (caustic only)")
    steer_angle_deg: Optional[float] = Field(default=None,
        description="Steering angle in degrees off the -Z axis, positive toward -x "
                    "(required for directional; optional for bessel and focus, "
                    "null -> 0)")
    cone_angle_deg: Optional[float] = Field(default=None, gt=0, lt=90,
        description="Half-angle in degrees between the two crossing plane waves "
                    "(bessel only)")
    focal_length: Optional[float] = Field(default=None, gt=0,
        description="Distance in meters from the aperture centre to the focal point "
                    "(focus only)")

    @model_validator(mode="after")
    def _check(self) -> "BeamConfig":
        if self.type == "caustic":
            if self.trajectory is None or len(self.trajectory) != 3:
                raise ValueError(
                    f"caustic beam requires trajectory = [a, b, c] (3 values), got {self.trajectory}")
        elif self.type == "directional":
            if self.steer_angle_deg is None:
                raise ValueError("directional beam requires steer_angle_deg")
        elif self.type == "bessel":
            if self.cone_angle_deg is None:
                raise ValueError("bessel beam requires cone_angle_deg")
            # both halves must still leave the plate downward
            steer = abs(self.steer_angle_deg or 0.0)
            if steer + self.cone_angle_deg >= 90:
                raise ValueError(
                    f"bessel beam requires |steer_angle_deg| + cone_angle_deg < 90, "
                    f"got {steer} + {self.cone_angle_deg}")
        elif self.type == "focus":
            if self.focal_length is None:
                raise ValueError("focus beam requires focal_length")
            # the focal point must sit below the plate
            if abs(self.steer_angle_deg or 0.0) >= 90:
                raise ValueError(
                    f"focus beam requires |steer_angle_deg| < 90, got {self.steer_angle_deg}")
        return self


class TxApertureConfig(BaseModel):
    """TX aperture geometry, sampling and emitted beam, defined independently of
    the scene grid. All units in meters.

    The aperture sits at height `z` and projects toward -Z (down) onto the RX at
    z=0; move it around by editing `z` (height) and `x_min`/`x_max` (lateral span).
    The emitted beam (caustic, directional, bessel or focus) is configured in the
    `beam` block."""
    x_min: float = -0.35
    x_max: float = 0.35
    z: float = Field(default=2.0, description="TX plane height (beam travels -Z to the RX at z=0)")
    dx: float = Field(default=0.25e-3, gt=0, description="Aperture sampling spacing in meters")
    beam: BeamConfig = Field(default_factory=BeamConfig,
        description="The emitted beam (type: caustic | directional | bessel | focus) "
                    "and its parameters")

    @model_validator(mode="after")
    def _check(self) -> "TxApertureConfig":
        if self.x_max <= self.x_min:
            raise ValueError(f"x_max ({self.x_max}) must be greater than x_min ({self.x_min})")
        return self


class RxApertureConfig(BaseModel):
    """RX aperture geometry: a window of the given lateral `width` centered at
    `x_center`, sitting at the scene origin (z=0). All units in meters.

    Shrink `width` to model a smaller receiver. Where the elements sit inside the
    window is one of two receivers, and the two are mutually exclusive:

    - `dx` (the default path): elements at a pitch in metres, on an axis of their
      own that does not line up with the solver grid. The RX-plane field is linearly
      interpolated onto the elements, and the element samples are interpolated back
      onto the solver grid, where the loss is scored at every grid point in the
      window. `dx: null` means wavelength/20 at runtime (the historical RX
      sampling), which on the shipped scenes is denser than the grid, so the round
      trip costs little (the target is about -42 to -51 dB off the true field at
      300 GHz, depending on the beam); at a pitch coarser than the grid it costs a
      lot (about -18 dB at lambda/2), because the field's phase turns between
      elements and the straight-line fill between them dips.
    - `grid_stride`: elements ON the solver grid, every `grid_stride`-th scene x
      sample inside the window (see sim_scene.grid_element_indices). Nothing is
      interpolated: the field is read at those samples and the loss is scored at
      those samples alone. The pitch is `grid_stride` scene x steps, so it is set by
      `sim_scene.spacing` rather than typed in metres.

    Leaving both null is the `dx` path at wavelength/20."""
    # grid_stride and dx are both optional, so a typo (grid_strid) must not silently
    # run the interpolated receiver
    model_config = ConfigDict(extra="forbid")

    x_center: float = Field(default=0.0, description="Lateral center of the RX window (m)")
    width: float = Field(default=0.05, gt=0, description="Lateral span of the RX window (m)")
    dx: Optional[float] = Field(default=None, gt=0,
        description="RX sampling spacing (m) of the interpolated receiver; null -> "
                    "wavelength/20 at runtime (unless grid_stride is set)")
    # StrictInt, not int: pydantic's lax int takes a YAML boolean, so
    # `grid_stride: true` would load as stride 1 -- a different receiver from the one
    # the file appears to ask for, with no error -- and it rounds 2.0 and parses "2"
    # the same quiet way. A stride is a count of grid samples: only an integer is one.
    grid_stride: Optional[StrictInt] = Field(default=None, ge=1,
        description="Put the RX elements ON the solver grid: every grid_stride-th "
                    "scene x sample inside the window, read directly and scored only "
                    "there (no interpolation). An integer >= 1, written as one (true, "
                    "2.0 and \"2\" are rejected). Mutually exclusive with dx; null -> "
                    "the interpolated receiver")

    @model_validator(mode="after")
    def _check(self) -> "RxApertureConfig":
        if self.dx is not None and self.grid_stride is not None:
            raise ValueError(
                f"rx_aperture.dx ({self.dx}) and rx_aperture.grid_stride "
                f"({self.grid_stride}) are mutually exclusive: dx is the pitch of the "
                "interpolated receiver, grid_stride puts the elements on the solver "
                "grid. Set one and leave the other null")
        return self


class GerchbergSaxtonConfig(BaseModel):
    """Hyperparameters and start/stop conditions for the modified Gerchberg-Saxton solver."""
    # init has a default and names the start of every solve, so a typo (int: backprop)
    # must not silently run the legacy warm start
    model_config = ConfigDict(extra="forbid")

    init: Literal["random", "warm_start", "backprop"] = Field(default="warm_start",
        description="Where the descent starts. 'warm_start' (default, the legacy "
                    "start) runs the multi-wavelength initialization on a "
                    "multi-frequency solve — solve the reference frequency alone "
                    "from the seeded random draw, unwrap, scan the absolute offset "
                    "over 2π·f_ref/Δf_min — before the joint descent (see "
                    "docs/delay_model_warm_start.md); at a single frequency it is the "
                    "random start (the two are the same problem there). 'random' "
                    "starts from the seeded random draw at any number of frequencies. "
                    "'backprop' starts from the DATA at any number of frequencies: the "
                    "reference tone's weighted measurement back-propagated to the "
                    "candidate plane, its phase unwrapped where the plate illuminates "
                    "the RX window and continued at constant slope where it does not, "
                    "then (several tones) the absolute offset scanned over the comb's "
                    "TRUE period, and the plain descent from there with no "
                    "reference-frequency solve. The warm start's scan range is a "
                    "period of the joint loss only when f_ref/Δf is an integer, and "
                    "its random-start reference solve is not unique where part of the "
                    "beam misses the RX window; 'backprop' has neither limit (see "
                    "mgs.comb_period and mgs._backprop_start).")
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
    engine: Literal["python", "rust"] = Field(default="python",
        description="Solver implementation: 'python' (the reference) or 'rust' (the "
                    "rice_bend_core extension, ~5x faster, parity-checked against the "
                    "reference; see rust/README.md). Solves that capture GS history "
                    "(single-run mgs, --true-mgs) always run on python.")


class NoiseConfig(BaseModel):
    """Receiver noise added to the SIMULATED measurement: complex AWGN on the RX
    elements, one floor for every antenna and every frequency, referenced to the
    peak RX-element power across all tones (see rice_bend/noise.py for the model).

    Ignored on the experimental path, whose capture carries its own receiver noise.
    """
    # a typo (snr_dB) must not silently run noiseless
    model_config = ConfigDict(extra="forbid")

    snr_db: Optional[float] = Field(default=None, allow_inf_nan=False,
        description="Peak-referenced SNR in dB: sigma^2 = P_peak / 10^(snr_db/10), "
                    "P_peak the max |field|^2 over RX elements and tones. "
                    "null -> noiseless (no random draw at all)")
    seed: Optional[int] = Field(default=0, ge=0, lt=2**32,
        description="Seed of the noise draw (each tone's stream is keyed by its "
                    "frequency). If null, one is drawn and recorded.")


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
    # every field is optional, so a typo (max_iter) must not silently sweep at the
    # single-run iteration cap
    model_config = ConfigDict(extra="forbid")

    max_iters: Optional[int] = Field(default=None, gt=0,
        description="Override gerchberg_saxton.max_iters during the sweep")


class GridRefineConfig(BaseModel):
    """Second pass of the grid sweep: re-solve the candidates the sweep left near
    its minimum, each from two starts (mgs.gs_reconstruct_multistart), and keep
    the lower loss. Defined and validated here; the sweep is what consumes it.

    The sweep ranks every candidate under limits that suit the thousands far from
    the TX and not the few that decide the answer:

    - its iteration cap (grid_search.gs_overrides.max_iters) is below what the
      candidates near the truth need, so the ones that matter are the ones it
      cuts off;
    - gerchberg_saxton.convergence_threshold is an ABSOLUTE loss slope. At high
      SNR the loss near the truth is small, and a threshold that does not scale
      with it stops the descent short of the floor it could reach;
    - no single start wins at every wrong location, so a ranking made from one
      start mixes the loss of a location with the luck of its start.

    The re-solve therefore has its own cap and its own stop threshold, and runs
    both starts.

    `max_candidates` bounds the cost, and it SKIPS the refinement rather than
    truncating it. A set cut to the best N by sweep loss holds the sweep's best
    candidate by construction and leaves out others that qualified; re-solving
    lowers the loss of the ones kept (more iterations, the better of two starts)
    while the ones left out keep their sweep loss — so the answer would be forced
    into the truncated set. When more than this qualify, no candidate is
    re-solved and the sweep's result stands."""
    # every field has a default, so a typo (withn) must not silently refine at the
    # default
    model_config = ConfigDict(extra="forbid")

    within: float = Field(default=10.0, gt=1.0,
        description="Re-solve every candidate whose sweep loss is <= within x the "
                    "sweep's minimum loss (> 1: at 1 only the best candidate itself "
                    "would qualify)")
    max_iters: int = Field(default=6000, gt=0,
        description="Iteration cap of the re-solve (the sweep's own cap is "
                    "grid_search.gs_overrides.max_iters)")
    convergence_threshold: Optional[float] = Field(default=-1e-10,
        description="Stop threshold of the re-solve, on mean(diff(recent loss)); "
                    "null -> gerchberg_saxton.convergence_threshold")
    max_candidates: int = Field(default=12000, gt=0,
        description="If more candidates than this qualify, the refinement is SKIPPED "
                    "for the run (never truncated to the best N)")


class GridSearchConfig(BaseModel):
    """Speculative (z, x_center) sweep for `grid-search-mgs`.

    The TX location is treated as unknown. At each grid point a fixed-width
    aperture (uniform assumed amplitude) is centered at `x_center` and placed at
    height `z`; MGS then reconstructs its phase against the single shared measured
    RX field. Each result is saved as a candidate beam."""
    # refine and gs_overrides are optional, so a typo (refne) must not silently run
    # the sweep without the block it was meant to carry
    model_config = ConfigDict(extra="forbid")

    z: AxisSweep
    x_center: AxisSweep
    aperture: GridApertureConfig
    seed: Optional[int] = Field(default=0,
        description="Fixed GS seed reused across every candidate so residuals are comparable")
    gs_overrides: GridGSOverrides = Field(default_factory=GridGSOverrides)
    refine: Optional[GridRefineConfig] = Field(default=None,
        description="Re-solve the near-best candidates after the sweep (see "
                    "GridRefineConfig); null -> the sweep's result stands")


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


# the carrier every entry point falls back to when neither --freq nor the
# config's `frequencies` names one
DEFAULT_FREQ_HZ = 300e9


class SimConfig(BaseModel):
    """Top-level simulation config."""
    sim_scene: SimSceneConfig
    tx_aperture: TxApertureConfig = Field(default_factory=TxApertureConfig)
    rx_aperture: RxApertureConfig = Field(default_factory=RxApertureConfig)
    plot_path: Path = Field(description="Path that plot_scene() writes the output figure to")
    gerchberg_saxton: GerchbergSaxtonConfig = Field(default_factory=GerchbergSaxtonConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig,
        description="Receiver noise on the simulated RX measurement (default: none)")
    experimental: ExperimentalConfig = Field(default_factory=ExperimentalConfig,
        description="Bench constants for the experimental .mat path (ignored by "
                    "the simulated workflows)")
    grid_search: Optional[GridSearchConfig] = Field(default=None,
        description="Optional speculative TX-location sweep (used by grid-search-mgs)")
    frequencies: Optional[List[float]] = Field(default=None,
        description="Frequencies (Hz) both entry points solve at; null -> --freq, else "
                    f"{DEFAULT_FREQ_HZ / 1e9:g}e9. "
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
    """Resolve the frequency list: CLI override -> config.frequencies -> [DEFAULT_FREQ_HZ].

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
        return [DEFAULT_FREQ_HZ]
    return list(dict.fromkeys(freqs))


# `--snr-db off`: force a noiseless run even when the config's noise block enables noise
SNR_OFF = "off"


def parse_snr_db_arg(text: str) -> Union[float, str]:
    """argparse `type=` for --snr-db: a number of dB, or `off` (any case) -> SNR_OFF.
    Finiteness is NOT checked here; apply_noise_overrides validates through the
    model, so the CLI and the YAML reject the same values."""
    if text.strip().lower() == SNR_OFF:
        return SNR_OFF
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number of dB or '{SNR_OFF}', got {text!r}")


def apply_noise_overrides(config: "SimConfig", snr_db: Union[float, str, None],
                          noise_seed: Optional[int]) -> None:
    """CLI overrides onto config.noise. None keeps the config's value; snr_db ==
    SNR_OFF forces noiseless.

    REBUILDS config.noise through the NoiseConfig constructor so validation runs
    (plain attribute assignment bypasses pydantic: the models have no
    validate_assignment) — argparse happily parses `nan`, `inf` and negative seeds.
    Raises pydantic.ValidationError on an invalid value."""
    cur = config.noise
    new_snr = cur.snr_db
    if snr_db == SNR_OFF:
        new_snr = None
    elif snr_db is not None:
        new_snr = snr_db
    new_seed = cur.seed if noise_seed is None else noise_seed
    config.noise = NoiseConfig(snr_db=new_snr, seed=new_seed)


# Default config shipped in the repo's configs/ folder, used by both entry points
# when --config is omitted.
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "caustic_config.yml"


def load_config(path: Path) -> SimConfig:
    """Loads and validates a simulation config .yml into a SimConfig model."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return SimConfig(**raw)
