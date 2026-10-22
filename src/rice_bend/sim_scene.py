"""Aperture and scene geometry. No I/O, no plotting -- the .mat readers live in
exp_data.py, and the beams an aperture emits are built in
beam_aperatures.py."""

from typing import Optional

import numpy as np


def sampled_axis(lo: float, hi: float, spacing: float) -> np.ndarray:
    """The project's scene sampling: linspace over [lo, hi] with as many points as
    `spacing` fits into the span.

    Note linspace's actual step is (hi - lo) / (n - 1), NOT `spacing` -- the
    configured spacing sets the point COUNT and is never the step used. The "dx N
    wavelengths" log lines are slightly wrong for the same reason. Changing this is
    a real physics improvement that shifts every saved number in the 2nd-3rd
    significant digit, so it is a decision to make between result sets, not during
    a refactor.
    """
    return np.linspace(lo, hi, int((hi - lo) / spacing))


# How close span / spacing must come to an integer for an exact-step axis. Relative,
# because 0.3 / 0.00025 is 1199.9999999999998 in floats, not 1200.
EXACT_STEP_RTOL = 1e-9


def whole_steps(span: float, spacing: float) -> Optional[int]:
    """`span / spacing` as an int when it is a whole number >= 1 (to EXACT_STEP_RTOL,
    relative), else None. The one integrality rule shared by scene_axis and the
    sim_scene.exact_step config check, so the config never accepts a scene the
    axis builder would refuse."""
    ratio = span / spacing
    n = round(ratio)
    if n >= 1 and abs(ratio - n) <= EXACT_STEP_RTOL * abs(ratio):
        return int(n)
    return None


def scene_axis(lo: float, hi: float, spacing: float,
               exact_step: bool = False) -> np.ndarray:
    """A scene axis over [lo, hi]. Every axis built from a SimSceneConfig goes
    through here, so the scene, the candidate grid and the receiver agree on it.

    `exact_step` False is sampled_axis, unchanged: the linspace whose step is not
    `spacing`. Every simulated config uses it and its saved numbers depend on it.

    `exact_step` True steps by exactly `spacing`, for measured data: the receiver
    elements sit at whole multiples of the stage pitch, and only an axis whose
    samples are whole multiples of `spacing` has samples at those positions. A
    centred axis (lo == -hi) is built as spacing * arange(-N, N + 1), so its middle
    sample is exactly 0.0 and the axis is exactly antisymmetric: the RS kernel is
    evaluated on the scene x axis itself, so it is centred only when the axis is.
    Any other axis is lo + spacing * arange(n). The last sample matches `hi` to
    rounding, not bit for bit. Raises ValueError unless (hi - lo) / spacing (and
    hi / spacing for a centred axis) is a whole number."""
    if not exact_step:
        return sampled_axis(lo, hi, spacing)
    if lo == -hi:
        half = whole_steps(hi, spacing)
        if half is None:
            raise ValueError(
                f"an exact-step axis needs hi / spacing to be a whole number, got "
                f"{hi} / {spacing} = {hi / spacing:.12g}")
        return spacing * np.arange(-half, half + 1, dtype=np.float64)
    steps = whole_steps(hi - lo, spacing)
    if steps is None:
        raise ValueError(
            f"an exact-step axis needs (hi - lo) / spacing to be a whole number, got "
            f"({hi} - {lo}) / {spacing} = {(hi - lo) / spacing:.12g}")
    return lo + spacing * np.arange(steps + 1, dtype=np.float64)


def grid_element_indices(scene_x_axis: np.ndarray, x_min: float, x_max: float,
                         stride: int) -> np.ndarray:
    """Indices into `scene_x_axis` of an ON-GRID receiver's elements: every
    `stride`-th scene x sample inside the closed window [x_min, x_max].

    The window is CLOSED, (x >= x_min) & (x <= x_max): the same mask as
    MGS.measure()'s weighting and landing.landing_metrics, so the elements are
    drawn from exactly the samples the interpolated receiver is scored on.

    The elements are centred in the window. From the first element to the last is a
    whole number of strides, which in general falls short of the window's n - 1
    scene steps by (n - 1) % stride of them; that leftover is split between the two
    edges (the odd step going to the upper one) instead of being left as one gap at
    the top. On the shipped scenes -- 160 window samples at stride 2 -- the leftover
    is a single step: the elements are window samples 0, 2, ..., 158, and the last
    window sample carries none.

    Pure: the scene axis and three scalars in, an int index array out. Raises
    ValueError when fewer than two elements result (a window narrower than the
    stride, or one that holds no scene sample at all): one sample is not an array,
    and a pitch could not be read off it.
    """
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"grid stride must be >= 1, got {stride}")
    scene_x_axis = np.asarray(scene_x_axis)
    in_win = np.flatnonzero((scene_x_axis >= x_min) & (scene_x_axis <= x_max))
    offset = ((len(in_win) - 1) % stride) // 2
    idx = in_win[offset::stride]
    if len(idx) < 2:
        raise ValueError(
            f"an on-grid receiver needs at least 2 elements, but stride {stride} "
            f"leaves {len(idx)} in the window [{x_min}, {x_max}], which holds "
            f"{len(in_win)} scene x samples; widen the window, lower the stride or "
            "use a finer scene spacing")
    return idx


class SimAperature():
    """A 1D aperture at height `z`: its window [x_min, x_max], the axis its samples
    sit on, and the complex profile across them.

    Two ways to lay out the axis. The constructor samples the window at `dx` on an
    axis of the aperture's own, which in general does not line up with the scene
    grid: fields reach it and leave it by interpolation. `on_grid` takes the samples
    FROM the scene x-axis instead, so nothing has to be interpolated either way.

    `grid_idx` / `grid_stride` say which one an instance is, and are set on every
    instance: None for an aperture on its own axis, else the indices of its samples
    in the scene x-axis and the stride between them.
    """
    def __init__(self, x_min: float, x_max: float, z: float, dx: float):
        assert x_max > x_min
        self.x_min = x_min
        self.x_max = x_max
        self.z = z
        self.dx = dx
        # Same sampling rule as the scene. This used to floor-divide, which disagrees
        # with `/` wherever the span is an exact multiple of dx: 0.11/0.00025 is
        # 440.0, but // gives 439.0.
        self.aper_axis = sampled_axis(x_min, x_max, dx)
        self.num_points = len(self.aper_axis)
        self.aper_profile = np.zeros(len(self.aper_axis), dtype=np.complex128)
        # not on the scene grid (see on_grid)
        self.grid_idx = None
        self.grid_stride = None

    @classmethod
    def on_grid(cls, x_min: float, x_max: float, z: float, scene_x_axis: np.ndarray,
                stride: int) -> "SimAperature":
        """An aperture whose samples ARE scene x samples: every `stride`-th one
        inside the closed window [x_min, x_max] (grid_element_indices picks them).

        `aper_axis` is scene_x_axis[grid_idx], the same floats, so a field on the
        scene axis is read at the elements by indexing and put back the same way.
        `dx` is the REALIZED pitch, `stride` scene x steps -- the scene's own step,
        which is not the configured spacing (see sampled_axis). `x_min` / `x_max`
        stay the configured window edges, not the outermost elements: the window is
        what every in-window mask is built from, and the elements need not reach
        its edges.

        `scene_x_axis` must be the axis of the scene the aperture is used in; the
        indices mean nothing against any other.
        """
        assert x_max > x_min
        scene_x_axis = np.asarray(scene_x_axis)
        # an alternate constructor: __init__ lays the axis out from dx, and here
        # the axis comes from the scene
        self = cls.__new__(cls)
        self.x_min = x_min
        self.x_max = x_max
        self.z = z
        self.grid_idx = grid_element_indices(scene_x_axis, x_min, x_max, stride)
        self.grid_stride = int(stride)
        self.dx = float(self.grid_stride * (scene_x_axis[1] - scene_x_axis[0]))
        # a copy (fancy indexing), so the aperture never aliases the scene axis
        self.aper_axis = np.array(scene_x_axis[self.grid_idx], dtype=np.float64)
        self.num_points = len(self.aper_axis)
        self.aper_profile = np.zeros(len(self.aper_axis), dtype=np.complex128)
        return self


class SimScene():
    """
    Simple class used to keep track of the parameters of the simulation.
    All units are in meters. `exact_step` picks the axis rule (see scene_axis);
    it is False for every simulated config.
    """
    def __init__(self, x_min: float, x_max: float, z_min: float, z_max: float, spacing: float, tx_ap: SimAperature, rx_ap: SimAperature,
                 exact_step: bool = False):
        assert x_max > x_min
        assert z_max > z_min
        self.x_min = x_min
        self.x_max = x_max
        self.z_min = z_min
        self.z_max = z_max
        self.spacing = spacing
        self.exact_step = exact_step

        self.x_axis = scene_axis(self.x_min, self.x_max, spacing, exact_step)
        self.z_axis = scene_axis(self.z_min, self.z_max, spacing, exact_step)

        self.tx_ap = tx_ap
        self.rx_ap = rx_ap

        # Lazily filled by MGS.illuminate_real(). Allocating it here cost 130.6 MB
        # of complex128 that the first propagation immediately overwrote with
        # complex64 -- and in the grid-search path was never written at all.
        self.data = None
