"""Pydantic models for the simulation configuration .yml.
See configs/caustic_config.yml (the `mgs` default) for a commented general config,
and configs/scenario_caustic_hit_1f_grid2l.yml for a campaign config with the
on-grid receiver and the data-driven start. The general
configs of the other three beam types are archived and not maintained:
configs/archive/directional_config.yml, configs/archive/bessel_config.yml and
configs/archive/focus_config.yml. configs/measured/ holds the measured THz-TDS
data configs: a shared base.yml and one thin file per acquisition laid over it
(see load_config's base_config and MeasuredConfig)."""

import argparse
import math
from pathlib import Path
from typing import ClassVar, List, Literal, Optional, Union

import numpy as np
import yaml
from pydantic import (BaseModel, ConfigDict, Field, StrictInt, field_validator,
                      model_serializer, model_validator)

from rice_bend.sim_scene import whole_steps


class SimSceneConfig(BaseModel):
    """Bounds and sampling of the 2D (x, z) simulation grid. All units in meters."""
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    spacing: float = Field(gt=0, description="Grid spacing in meters")
    exact_step: bool = Field(default=False,
        description="Step the scene axes by exactly `spacing` (sim_scene.scene_axis) "
                    "instead of the historical linspace, whose step is not `spacing`. "
                    "Needs x_min == -x_max and whole numbers of steps in x_max and "
                    "z_max - z_min. Required by a `measured` config, whose elements "
                    "sit at whole multiples of the stage pitch; false for every "
                    "simulated config, whose saved numbers depend on the linspace")

    @model_validator(mode="after")
    def _check_bounds(self) -> "SimSceneConfig":
        if self.x_max <= self.x_min:
            raise ValueError(f"x_max ({self.x_max}) must be greater than x_min ({self.x_min})")
        if self.z_max <= self.z_min:
            raise ValueError(f"z_max ({self.z_max}) must be greater than z_min ({self.z_min})")
        return self

    @model_validator(mode="after")
    def _check_exact_step(self) -> "SimSceneConfig":
        if not self.exact_step:
            return self
        # exactly, not to a tolerance: the RS kernel is evaluated on the scene x axis
        # itself, so it is centred only when the axis has its middle sample at 0.0
        if self.x_min != -self.x_max:
            raise ValueError(
                f"sim_scene.exact_step needs an x axis centred on 0: x_min "
                f"({self.x_min}) must be exactly -x_max ({-self.x_max})")
        if whole_steps(self.x_max, self.spacing) is None:
            raise ValueError(
                f"sim_scene.exact_step needs x_max / spacing to be a whole number, got "
                f"{self.x_max} / {self.spacing} = {self.x_max / self.spacing:.12g}; "
                "pick a spacing that divides x_max")
        span = self.z_max - self.z_min
        if whole_steps(span, self.spacing) is None:
            raise ValueError(
                f"sim_scene.exact_step needs (z_max - z_min) / spacing to be a whole "
                f"number, got ({self.z_max} - {self.z_min}) / {self.spacing} = "
                f"{span / self.spacing:.12g}; pick z bounds a whole number of "
                "spacings apart")
        return self

    @model_serializer(mode="wrap")
    def _dump(self, handler):
        # exact_step is left out of a dump while it is false, so every simulated
        # config dumps exactly as it did before the field existed: config_snapshot.json
        # is unchanged, and mgs-study's resume check (which compares sim_scene with a
        # saved snapshot, a missing key counting as null) still accepts the roots
        # saved before. Loading such a dump gives false again
        data = handler(self)
        if not self.exact_step:
            data.pop("exact_step", None)
        return data


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
    # The setting that asks the plate's power to be found from the clean measurement
    # (amplitude: rx_peak); any other string is refused
    AMPLITUDE_RX_PEAK: ClassVar[str] = "rx_peak"

    x_min: float = -0.35
    x_max: float = 0.35
    z: float = Field(default=2.0, description="TX plane height (beam travels -Z to the RX at z=0)")
    dx: float = Field(default=0.25e-3, gt=0, description="Aperture sampling spacing in meters")
    beam: BeamConfig = Field(default_factory=BeamConfig,
        description="The emitted beam (type: caustic | directional | bessel | focus) "
                    "and its parameters")
    amplitude: Union[Literal["rx_peak"], float] = Field(default=1.0,
        description="One real factor s > 0 on the SIMULATED plate of every tone: the "
                    "TX power, unknown to the solver (a jammer). A number is s itself, "
                    "finite and > 0. 'rx_peak' sets s = 1 / P, P the largest |clean RX "
                    "element| over every tone from the unscaled plate, so the clean RX "
                    "peak element is 1: the simulated counterpart of "
                    "measured.amplitude_scale (1/max|V|). Applied once to the plate "
                    "when the simulated MGS is built, before measure(); the noise is "
                    "then drawn on the scaled field. Not used on the experimental and "
                    "measured paths, which have no simulated plate. Default 1.0 (the "
                    "beam generators' unit amplitude)")

    @field_validator("amplitude", mode="before")
    @classmethod
    def _check_amplitude(cls, v):
        # Checked BEFORE pydantic's union, which would quietly take a YAML boolean
        # (`amplitude: true`) as 1.0: a file that asks for something other than what
        # it gets. A string is 'rx_peak' exactly or a number: PyYAML loads `5e-1`,
        # `1e0` and `2.0e3` (no dot, or no signed exponent) as strings, which every
        # other float field takes as numbers, so this one does too (and so a quoted
        # "0.5"). numpy real scalars (np.float32, np.int64) are numbers. One message
        # for 0, negatives, nan and inf alike (as measured.amplitude_scale)
        if isinstance(v, (bool, np.bool_)):
            raise ValueError(
                f"tx_aperture.amplitude must be a finite number > 0 or "
                f"'{cls.AMPLITUDE_RX_PEAK}', got the boolean {v}")
        if isinstance(v, str):
            if v == cls.AMPLITUDE_RX_PEAK:
                return v
            try:
                v = float(v)
            except ValueError:
                raise ValueError(
                    f"tx_aperture.amplitude must be a finite number > 0 or "
                    f"'{cls.AMPLITUDE_RX_PEAK}' (the clean RX peak element -> 1), got "
                    f"the string {v!r}") from None
        if isinstance(v, (int, float, np.integer, np.floating)):
            if not (math.isfinite(v) and v > 0):
                raise ValueError(
                    f"tx_aperture.amplitude must be a finite number > 0 or "
                    f"'{cls.AMPLITUDE_RX_PEAK}', got {v}: it multiplies the simulated "
                    "plate of every tone")
            return float(v)
        raise ValueError(
            f"tx_aperture.amplitude must be a finite number > 0 or "
            f"'{cls.AMPLITUDE_RX_PEAK}', got {v!r} ({type(v).__name__})")

    @model_validator(mode="after")
    def _check(self) -> "TxApertureConfig":
        if self.x_max <= self.x_min:
            raise ValueError(f"x_max ({self.x_max}) must be greater than x_min ({self.x_min})")
        return self

    @model_serializer(mode="wrap")
    def _dump(self, handler):
        # amplitude is left out of a dump while it is the default 1.0, for the reason
        # given at SimSceneConfig._dump: every config that does not set it dumps
        # exactly as it did before the field existed (config_snapshot.json, and
        # mgs-study's resume identity, which compares the whole tx_aperture block),
        # and loading such a dump gives 1.0 again
        data = handler(self)
        if self.amplitude == 1.0:
            data.pop("amplitude", None)
        return data


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
    amplitude: Literal["fixed", "descend", "level"] = Field(default="fixed",
        description="What the solve does with the plate's amplitude. 'fixed' (default) "
                    "holds it at the aperture's own amplitude (the candidates' uniform "
                    "box). 'descend' fits it too: one amplitude a(x) on the support, "
                    "shared by every tone, started from the raw magnitude of the "
                    "reference tone's back-propagated measurement and moved in the same "
                    "backtracking step as the phase, clipped to [0, amp_max] after "
                    "every step. Both engines (python and rust). 'level' fits ONE "
                    "uniform level A >= 0 over the support, shared by every tone: it "
                    "starts where the start phase's predicted RX power matches the "
                    "measured one, and moves in the same backtracking step as the "
                    "phase, with no cap and no assumed brightness. Both engines; with "
                    "several tones it needs init: backprop (the legacy warm start would "
                    "match the level to a random phase). Neither 'descend' nor 'level' "
                    "runs on a solve that captures GS history (single-run mgs); "
                    "grid-search-mgs --true-mgs then solves at the true (or measured) "
                    "TX as a candidate (true_mgs.json) instead of the plain-mgs "
                    "baseline")
    amp_max: Optional[float] = Field(default=None,
        description="The cap on the descended amplitude, in the units of the fixed "
                    "solver's uniform amplitude 1 (so relative to the data as "
                    "measured.amplitude_scale scales it). Finite and > 0; required with "
                    "amplitude 'descend', refused with 'fixed' and 'level' (no cap)")
    power_penalty: Optional[float] = Field(default=None,
        description="The excess-power penalty's exponent gamma (null = off). With it "
                    "set, the descent minimizes J = L * max(1, P_plate/P_rx)**gamma "
                    "instead of the plain joint loss L: P_plate is the power leaving "
                    "the candidate plate (sum of a^2 dx over the support), P_rx the "
                    "power measured on the RX elements (mean over the tones). A plate "
                    "that radiates more than the receiver caught is charged for it; one "
                    "that radiates less is not (the floor at 1: unfloored, the product "
                    "would favour dark plates). Finite and > 0; needs a fitted "
                    "amplitude (amplitude 'level' or 'descend'), refused with 'fixed', "
                    "where P_plate is the same for every solve. Both engines")

    @field_validator("amp_max", mode="after")
    @classmethod
    def _check_amp_max(cls, v: Optional[float]) -> Optional[float]:
        # one message for 0, negatives, nan and inf alike (as amplitude_scale)
        if v is not None and not (math.isfinite(v) and v > 0):
            raise ValueError(f"amp_max must be a finite number > 0, got {v}")
        return v

    @field_validator("power_penalty", mode="after")
    @classmethod
    def _check_power_penalty(cls, v: Optional[float]) -> Optional[float]:
        # one message for 0, negatives, nan and inf alike (as amp_max)
        if v is not None and not (math.isfinite(v) and v > 0):
            raise ValueError(f"power_penalty must be a finite number > 0, got {v}")
        return v

    @model_validator(mode="after")
    def _check_amplitude(self) -> "GerchbergSaxtonConfig":
        if self.amplitude == "descend" and self.amp_max is None:
            raise ValueError(
                "amplitude 'descend' needs amp_max: the cap on the descended "
                "amplitude, in the units of the fixed solver's uniform amplitude 1")
        if self.amplitude == "fixed" and self.amp_max is not None:
            raise ValueError(
                f"amp_max ({self.amp_max}) is set but amplitude is 'fixed', which "
                "holds the amplitude and caps nothing: set amplitude: descend, or "
                "remove amp_max")
        if self.amplitude == "level" and self.amp_max is not None:
            raise ValueError(
                f"amp_max ({self.amp_max}) is set but amplitude is 'level', which has "
                "no cap: its one level is fitted from the data alone. Remove amp_max, "
                "or set amplitude: descend for a capped per-sample amplitude")
        if self.amplitude == "fixed" and self.power_penalty is not None:
            raise ValueError(
                f"power_penalty ({self.power_penalty}) is set but amplitude is 'fixed': "
                "the held amplitude radiates the same power from every solve, so the "
                "penalty would only rescale the loss. It needs a fitted amplitude: set "
                "amplitude: level or amplitude: descend, or remove power_penalty")
        return self

    @model_serializer(mode="wrap")
    def _dump(self, handler):
        # amplitude and amp_max are left out of a dump while the amplitude is fixed,
        # for the reason given at SimSceneConfig._dump: every fixed-amplitude config
        # dumps exactly as it did before the fields existed (config_snapshot.json,
        # mgs-study's saved configs), and loading such a dump gives 'fixed' again
        data = handler(self)
        if self.amplitude == "fixed":
            data.pop("amplitude", None)
            data.pop("amp_max", None)
        # power_penalty likewise, while it is off: a config without the penalty
        # dumps exactly as it did before the field existed
        if self.power_penalty is None:
            data.pop("power_penalty", None)
        return data


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


class GridSearchConfig(BaseModel):
    """Speculative (z, x_center) sweep for `grid-search-mgs`.

    The TX location is treated as unknown. At each grid point a fixed-width
    aperture (uniform assumed amplitude) is centered at `x_center` and placed at
    height `z`; MGS then reconstructs its phase against the single shared measured
    RX field. Each result is saved as a candidate beam."""
    # gs_overrides is optional, so a typo (gs_override) must not silently run the
    # sweep without the block it was meant to carry. A `refine` block (the removed
    # post-sweep re-solve pass) is refused the same way: such a config was written
    # for a sweep that no longer exists
    model_config = ConfigDict(extra="forbid")

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


class StagePoint(BaseModel):
    """A point in the measurement STAGE frame, in millimetres. Named keys, not a
    pair: the stage's (x, z) order is easy to write the wrong way round, and a
    swapped pair would load without complaint."""
    model_config = ConfigDict(extra="forbid")

    x: float = Field(allow_inf_nan=False, description="Stage x (mm)")
    z: float = Field(allow_inf_nan=False, description="Stage z (mm)")


class StageSegment(BaseModel):
    """A horizontal aperture in the stage frame: its two edge points, in either
    order along x (the smaller x is the left edge). Both points must share one z:
    the scene models every aperture as a constant-z line."""
    model_config = ConfigDict(extra="forbid")

    left: StagePoint
    right: StagePoint

    @model_validator(mode="after")
    def _check(self) -> "StageSegment":
        if self.left.z != self.right.z:
            raise ValueError(
                f"left.z ({self.left.z} mm) and right.z ({self.right.z} mm) differ: a "
                "tilted aperture is not supported: both points need the same z")
        if self.left.x == self.right.x:
            raise ValueError(
                f"left.x and right.x are both {self.left.x} mm: the aperture has no "
                "width; give its two edges")
        return self


class MeasuredConfig(BaseModel):
    """A measured THz-TDS raster used in place of the simulated RX measurement
    (grid-search-mgs only; see rice_bend/measured.py).

    The data is the spectra CSV that topt-parser writes from the folder of pulse
    traces (rice_bend/topt_parser.py): one row per stage position, X and Z in mm,
    one complex column per frequency. The apertures are given in the stage frame,
    in mm, as on the bench; the TX and RX blocks of the simulation (tx_aperture,
    rx_aperture, the grid's aperture width) are derived from them at run time, so
    the config must not set them. `amplitude_scale` multiplies the CSV's raw
    spectra: there is no automatic calibration, so the value written here is the
    one used."""
    # every field here changes what is read or where it is placed, so a typo
    # (amplitude_scal) must not silently fall back to a default or be dropped
    model_config = ConfigDict(extra="forbid")

    # the keys of the retired pulse-folder input, refused with directions to the
    # spectra CSV (_refuse_pulse_folder_keys) rather than as unknown keys
    RETIRED_KEYS: ClassVar[tuple] = ("data_dir", "file_unit_mm")

    data_file: Path = Field(
        description="The spectra CSV written by `topt-parser <pulse folder>` "
                    "(<pulse folder>/topt_spectra.csv by default); '~' is expanded. "
                    "Its existence is checked when the data is read")
    tx_aperture: StageSegment = Field(
        description="TX plate edges in the stage frame (mm); the ground truth "
                    "the search is scored against")
    rx_aperture: StageSegment = Field(
        description="RX edges in the stage frame (mm). Its z must be a row of the "
                    "data file and both edges X values of that row; only the "
                    "positions between them are used")
    amplitude_scale: float = Field(
        description="Multiplies the raw spectra (Signal 1, nA, summed over the "
                    "samples) into the solver's units; finite and > 0")

    @model_validator(mode="before")
    @classmethod
    def _refuse_pulse_folder_keys(cls, data):
        """A config written for the pulse-folder input (data_dir, file_unit_mm) is
        refused with what to do instead: the grid search no longer reads pulse
        files, and the file-name unit is topt-parser's --unit-mm."""
        if not isinstance(data, dict):
            return data
        retired = [k for k in cls.RETIRED_KEYS if k in data]
        if not retired:
            return data
        unit = data.get("file_unit_mm")
        unit_flag = f" --unit-mm {unit}" if unit not in (None, 0.1) else ""
        if "data_dir" not in data:
            rewrite = (f", and write the CSV with `topt-parser <the pulse folder>"
                       f"{unit_flag}`" if unit_flag else "")
            raise ValueError(
                "measured.file_unit_mm is no longer read: the file-name unit is "
                "topt-parser's --unit-mm (default 0.1), applied when it writes the "
                f"spectra CSV. Remove file_unit_mm{rewrite}")
        folder = str(data["data_dir"] or "").rstrip("/") or "<the pulse folder>"
        names = " and ".join(f"measured.{k}" for k in retired)
        raise ValueError(
            f"{names} {'is' if len(retired) == 1 else 'are'} no longer read: the grid "
            "search reads the one spectra CSV that topt-parser writes from the pulse "
            f"traces. Run `topt-parser {folder}{unit_flag}`, then set "
            f"measured.data_file to its output ({folder}/topt_spectra.csv by "
            f"default) and remove {' and '.join(retired)}")

    @field_validator("data_file", mode="after")
    @classmethod
    def _expand_data_file(cls, v: Path) -> Path:
        return v.expanduser()

    @field_validator("amplitude_scale", mode="after")
    @classmethod
    def _check_amplitude_scale(cls, v: float) -> float:
        # one message for 0, negatives, nan and inf alike (pydantic's gt and
        # allow_inf_nan would give two different ones, and gt alone lets inf through)
        if not (math.isfinite(v) and v > 0):
            raise ValueError(
                f"amplitude_scale must be a finite number > 0, got {v}: it multiplies "
                "the raw spectra, and the run logs the raw max |V| of the RX row "
                "(1 / that max puts the RX peak at 1.0)")
        return v


class PlotWindowConfig(BaseModel):
    """The rectangle of the (x, z) plane that grid-search-mgs's plots show, in metres
    in the SOLVER frame (RX centre at (0, 0), TX above it at +z). A view only: the
    search, the manifest's numbers and analysis.json are the same with or without
    it (see residual_plots and candidate_scenes for what it crops)."""
    model_config = ConfigDict(extra="forbid")

    x_min: float = Field(allow_inf_nan=False)
    x_max: float = Field(allow_inf_nan=False)
    z_min: float = Field(allow_inf_nan=False)
    z_max: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _check(self) -> "PlotWindowConfig":
        problems = [f"{a}_max ({hi}) must be greater than {a}_min ({lo})"
                    for a, lo, hi in (("x", self.x_min, self.x_max),
                                      ("z", self.z_min, self.z_max)) if not hi > lo]
        if problems:
            raise ValueError("; ".join(problems))
        return self

    # A point exactly on an edge is inside: grid values are float sums (0.15 + 125 *
    # 0.002 is not exactly 0.4), so the test carries a tolerance far below any step.
    EDGE_TOL: ClassVar[float] = 1e-9

    def contains(self, x: float, z: float) -> bool:
        """True when the point (x, z) lies inside the window, edges included."""
        tol = self.EDGE_TOL
        return (self.x_min - tol <= x <= self.x_max + tol
                and self.z_min - tol <= z <= self.z_max + tol)


# the carrier every entry point falls back to when neither --freq nor the
# config's `frequencies` names one
DEFAULT_FREQ_HZ = 300e9


def _default_tx_aperture(data: dict) -> TxApertureConfig:
    """SimConfig.tx_aperture when the file has no tx_aperture block.

    A simulated config gets TxApertureConfig(), which raises (the default caustic
    beam has no trajectory), so such a file is refused exactly as it always was. A
    measured config must not have the block, and gets a stand-in instead: its
    window is overwritten by measured.apply_measured, which keeps its dx and beam,
    and the beam is never synthesised on the measured path (the plate's profile is
    the unknown). The beam is a flat plate, steered nowhere, only because a
    BeamConfig has to validate. `measured` is missing from `data` when it failed
    validation; the stand-in then lets that error be the one reported.

    Takes the fields validated so far, which is why `measured` is declared before
    tx_aperture (pydantic >= 2.10 passes them to a one-argument factory)."""
    if "measured" in data and data["measured"] is None:
        return TxApertureConfig()
    return TxApertureConfig(beam=BeamConfig(type="directional", steer_angle_deg=0.0))


class SimConfig(BaseModel):
    """Top-level simulation config."""
    sim_scene: SimSceneConfig
    # before tx_aperture: _default_tx_aperture reads it
    measured: Optional[MeasuredConfig] = Field(default=None,
        description="Measured THz-TDS data in place of the simulated measurement "
                    "(grid-search-mgs only). tx_aperture and rx_aperture are then "
                    "derived from it and must not be set; null -> simulate")
    tx_aperture: TxApertureConfig = Field(default_factory=_default_tx_aperture)
    rx_aperture: RxApertureConfig = Field(default_factory=RxApertureConfig)
    plot_path: Path = Field(description="Path that plot_scene() writes the output figure to")
    plot_window: Optional[PlotWindowConfig] = Field(default=None,
        description="grid-search-mgs only: the (x, z) rectangle its residual plots and "
                    "candidate scenes show, in metres in the solver frame; "
                    "--plot-window overrides it. null -> the whole grid / scene")
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
                raise ValueError(f"all frequencies must be positive (got {self.frequencies}): "
                                 "0 Hz or below has no wavelength to propagate at")
            if len(set(self.frequencies)) != len(self.frequencies):
                raise ValueError(f"frequencies must be unique (got {self.frequencies})")
        return self

    # The two measured checks run at construction only: the models have no
    # validate_assignment, and measured.apply_measured writes the derived
    # tx_aperture / rx_aperture into a config that has passed them, so they must not
    # (and do not) run again on those assignments.

    @model_validator(mode="before")
    @classmethod
    def _check_measured_blocks(cls, data):
        """A measured config writes no tx_aperture / rx_aperture block (in the file
        or its base_config): both are derived. Checked on the input keys, which are
        the keys model_fields_set would hold, BEFORE the fields are validated: a
        partial block (a tx_aperture without its beam) is then refused for being
        there rather than for what it lacks."""
        if isinstance(data, dict) and data.get("measured") is not None:
            written = [k for k in ("tx_aperture", "rx_aperture") if k in data]
            if written:
                raise ValueError(
                    f"a `measured` config: remove {' and '.join(written)} (from this "
                    "file and its base_config): with a measured block, tx_aperture and "
                    "rx_aperture are derived from measured.tx_aperture / "
                    "measured.rx_aperture (stage mm) and the data")
        return data

    @model_validator(mode="after")
    def _check_measured(self) -> "SimConfig":
        """A measured config's other cross-block rules, all reported at once."""
        if self.measured is None:
            return self
        problems = []
        if not self.sim_scene.exact_step:
            problems.append(
                "sim_scene.exact_step must be true: the measured elements sit at whole "
                "multiples of the stage pitch, which only the exact-step axis samples")
        if self.noise.snr_db is not None:
            problems.append(
                f"noise.snr_db must be null, got {self.noise.snr_db}: measured data "
                "carries its own receiver noise")
        if self.grid_search is None:
            problems.append(
                "grid_search must be set: measured data is searched with "
                "grid-search-mgs, which needs the candidate grid")
        if problems:
            raise ValueError("a `measured` config: " + "; ".join(problems))
        return self

    @model_serializer(mode="wrap")
    def _dump(self, handler):
        # measured and plot_window are left out of a dump while they are null, for
        # the reason given at SimSceneConfig._dump: every simulated config dumps
        # exactly as before
        data = handler(self)
        if self.measured is None:
            data.pop("measured", None)
        if self.plot_window is None:
            data.pop("plot_window", None)
        return data


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


# The top-level key naming the config a file is laid over (see load_config).
BASE_CONFIG_KEY = "base_config"


def _deep_merge(base: dict, over: dict) -> dict:
    """`over` laid over `base`, neither one modified. Where both hold a mapping
    under a key, the two are merged the same way one level down; anything else
    `over` holds (a list, a scalar, a null) replaces the base's value whole."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> SimConfig:
    """Loads and validates a simulation config .yml into a SimConfig model.

    A file with a top-level `base_config: <path>` is laid over that base before
    validation (_deep_merge: mappings merge key by key at every depth; a list, a
    scalar or a null replaces the base's value whole). The base path is relative
    to the file's own directory, '~' expanded; every other path in either file
    (plot_path, output.output_dir, measured.data_file) keeps its usual meaning.
    One level only: a base that names a base of its own is refused, so what a
    file contains is never more than two files away. A file without the key
    loads exactly as before."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if not (isinstance(raw, dict) and BASE_CONFIG_KEY in raw):
        return SimConfig(**raw)

    path = Path(path)
    raw = dict(raw)
    ref = raw.pop(BASE_CONFIG_KEY)
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError(
            f"{path}: base_config must be the path of a config .yml (relative to "
            f"{path.parent}), got {ref!r}")
    base_path = Path(ref).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    if not base_path.is_file():
        raise FileNotFoundError(
            f"{path}: base_config {ref!r} resolves to {base_path}, which is not a file")
    with open(base_path, "r") as f:
        base = yaml.safe_load(f)
    if not isinstance(base, dict):
        raise ValueError(
            f"{base_path} (the base_config of {path}) must hold a mapping of config "
            f"blocks, got {type(base).__name__}")
    if BASE_CONFIG_KEY in base:
        raise ValueError(
            f"{base_path} (the base_config of {path}) sets base_config "
            f"{base[BASE_CONFIG_KEY]!r} itself, and base_config is one level only: "
            f"copy what it needs from that file into {base_path.name}, or point "
            f"{path.name} at a base that names none")
    return SimConfig(**_deep_merge(base, raw))
