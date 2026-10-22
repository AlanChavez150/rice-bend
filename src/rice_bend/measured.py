"""From a `measured:` config block to the simulation's geometry and measurement.

The config gives the TX plate and the RX as they sit on the bench: two edge points
each, in the STAGE frame, in mm. The data is the spectra CSV that topt-parser writes
from the pulse traces (topt_parser.py): one row per stage position, X and Z in stage
mm, one complex column V(f) per frequency. The solver works in the SIMULATION frame,
in metres, with the RX at z = 0, the TX above it, and propagation toward -z. This
module is the one place that maps one onto the other:

    z_sim = (z_rx - z_stage) * 1e-3        the RX row at 0, the TX at +z
    x_sim = (x_stage - x_c) * 1e-3         x_c = the RX centre; the scene's x axis
                                           must be centred on 0 for the RS kernel

z is the only coordinate inverted; x is NOT mirrored (a 2D propagation does not
depend on the mirror, so mirroring would only make the stage and the plots
disagree).

apply_measured rewrites a loaded SimConfig into an ordinary one (tx_aperture,
rx_aperture and the grid's aperture width derived from the stage geometry), so
everything downstream of the measurement runs the simulated on-grid code path
unchanged. place_elements and measured_rx_profile give MGS.from_measured the RX
elements' scene indices and their measurement.

Only the RX row's values are taken from the file (read_spectra_csv returns every
row's Z, to check the RX z against, and values for that one row), and only the
elements between the RX edges enter the solve. The file runs to hundreds of MB and
a run resolves the geometry twice (the CLI, for its early errors and the dry run,
then MGS.from_measured), so a read is kept for the process (_read_rx_row).
"""

import functools
import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

from rice_bend.config import (GridApertureConfig, RxApertureConfig, SimConfig,
                              SimSceneConfig, TxApertureConfig)
from rice_bend.tds_data import MeasuredDataError
from rice_bend.topt_parser import SpectraRows, freq_header, read_spectra_csv

if TYPE_CHECKING:
    from rice_bend.config import MeasuredConfig

# How close a ratio must come to a whole number to be one (relative): the RX pitch
# over the scene spacing. The rule sim_scene's exact-step axis uses.
_WHOLE_RTOL = 1e-9
# How far an element may sit from its scene sample (m), and the slack on the
# inclusive "inside the scene" checks: float round-off only, far below any pitch.
_POS_TOL = 1e-9
# How close a configured stage position must come to one in the data file to be it
# (mm): the reader's own row rule, |Z - z| < 1e-6 mm. Steps and gaps are compared
# on an integer lattice of that size (_lattice: _PER_MM units to the mm), so exactly.
_POS_MM_TOL = 1e-6
_PER_MM = 1_000_000


@dataclass(frozen=True)
class MeasuredGeometry:
    """The stage geometry of a `measured:` block, resolved against the data and
    put in the simulation frame (see the module docstring)."""
    data_file: Path                # the spectra CSV, absolute
    z_rx_mm: float                 # RX row, stage mm
    x_center_mm: float             # RX centre (x_left+x_right)/2, stage mm
    rx_x_mm: np.ndarray            # (n,) float64, ascending: every X of the RX row in
    #                                [x_left, x_right], as the file has it (stage mm)
    rx_x_sim: np.ndarray           # (n,) metres, sim frame
    pitch_m: float                 # uniform element pitch (m)
    rx_width_m: float              # x_right - x_left (m)
    tx_x_min: float                # sim frame (m)
    tx_x_max: float
    tx_z: float                    # sim frame (m), > 0
    tx_width_m: float
    tx_stage_mm: dict              # {"left": {"x","z"}, "right": {...}} as configured
    rx_stage_mm: dict


def stage_to_sim(x_mm, z_mm, z_rx_mm, x_center_mm):
    """Stage mm -> simulation m: (x_sim, z_sim). Scalars or arrays.

    z_sim = (z_rx_mm - z_mm) * 1e-3 puts the RX row at 0 and the TX (smaller stage
    z, upstream) at +z; x_sim = (x_mm - x_center_mm) * 1e-3 centres the RX on 0.
    No x mirror (see the module docstring)."""
    return (x_mm - x_center_mm) * 1e-3, (z_rx_mm - z_mm) * 1e-3


def _mm(v: float) -> str:
    """A length in mm for a message: 174.0 -> '174', 0.30000000000000004 -> '0.3'."""
    s = f"{round(float(v), 6):.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _whole(v: float) -> Optional[int]:
    """`v` as an int when it is a whole number (to _WHOLE_RTOL), else None."""
    n = round(v)
    if abs(v - n) <= _WHOLE_RTOL * max(1.0, abs(v)):
        return int(n)
    return None


def _lattice(values_mm) -> np.ndarray:
    """Stage positions (mm) as whole multiples of _POS_MM_TOL (int64), so steps and
    gaps compare exactly: in floats, 175.1 - 175.0 is 0.09999999999999432."""
    return np.round(np.asarray(values_mm, dtype=np.float64) * _PER_MM).astype(np.int64)


def _nearest_text(values_mm: np.ndarray, target_mm: float) -> str:
    """'174 or 177 mm': the positions in the file (sorted, mm) either side of
    `target_mm`. One side only at either end of the range."""
    below = values_mm[values_mm < target_mm]
    above = values_mm[values_mm > target_mm]
    names = ([_mm(below[-1])] if len(below) else []) + \
            ([_mm(above[0])] if len(above) else [])
    return " or ".join(names) + " mm"


def _row_step(lattice: np.ndarray) -> int:
    """The step of a sorted run of positions (lattice units): its most common one
    (the smallest on a tie), so a skipped position or two does not change what the
    row's step is."""
    vals, counts = np.unique(np.diff(lattice), return_counts=True)
    return int(vals[np.argmax(counts)])


def _range_text(values_mm: np.ndarray) -> str:
    """'0..300 mm every 3 mm' (or '..., mostly every 3 mm') for sorted positions."""
    text = f"{_mm(values_mm[0])}..{_mm(values_mm[-1])} mm"
    if len(values_mm) > 1:
        lat = _lattice(values_mm)
        uniform = len(set(np.diff(lat).tolist())) == 1
        step_mm = _row_step(lat) / _PER_MM
        text += f"{' ' if uniform else ', mostly '}every {_mm(step_mm)} mm"
    return text


def _stage_dict(seg) -> dict:
    return {"left": {"x": float(seg.left.x), "z": float(seg.left.z)},
            "right": {"x": float(seg.right.x), "z": float(seg.right.z)}}


def _check_segment(name: str, seg) -> None:
    """The config validates this already; repeated here because derive_geometry
    takes any object with the MeasuredConfig fields, and a tilted or zero-width
    aperture would derive nonsense rather than fail."""
    if seg.left.z != seg.right.z:
        raise MeasuredDataError(
            f"measured.{name}: left.z ({_mm(seg.left.z)} mm) and right.z "
            f"({_mm(seg.right.z)} mm) differ: a tilted aperture is not supported: "
            "both points need the same z")
    if seg.left.x == seg.right.x:
        raise MeasuredDataError(
            f"measured.{name}: left.x and right.x are both {_mm(seg.left.x)} mm; "
            "give the aperture's two edges")


def _check_freqs(freqs) -> List[float]:
    """The run's frequencies as a list of floats, each finite and > 0.

    The one check of the measured path (the config's `frequencies` are validated at
    load, but --freq is not): the data file has a 0GHz column, the DC sum of each
    trace, and nothing propagates at 0 Hz, which has no wavelength."""
    freqs = [float(f) for f in np.atleast_1d(np.asarray(freqs, dtype=np.float64))]
    if not freqs:
        raise MeasuredDataError("no frequency given: a measured run needs at least one")
    bad = [f for f in freqs if not (math.isfinite(f) and f > 0)]
    if bad:
        raise MeasuredDataError(
            f"cannot solve at {', '.join(f'{f:g}' for f in bad)} Hz: every frequency of "
            "a measured run must be finite and > 0, because 0 Hz (or less) has no "
            "wavelength to propagate at. The data file's 0GHz column is the DC sum "
            "of each trace, kept as a record; pick other columns (e.g. --freq 300e9)")
    return freqs


def _data_file(path) -> Path:
    """measured.data_file, checked to be a file, as an absolute path."""
    p = Path(path).expanduser()
    if p.is_dir():
        raise MeasuredDataError(
            f"measured.data_file {p} is a directory: set it to the spectra CSV that "
            f"`topt-parser {p}` writes ({p / 'topt_spectra.csv'} by default)")
    if not p.is_file():
        raise MeasuredDataError(
            f"measured.data_file {p} does not exist: write it with `topt-parser "
            "<the folder of *.pulse.csv traces>` (default output "
            "<folder>/topt_spectra.csv) or fix the path")
    return p.resolve()


@functools.lru_cache(maxsize=4)
def _read_rx_row_cached(path: str, size: int, mtime_ns: int, z_mm: float,
                        freqs: Tuple[float, ...]) -> SpectraRows:
    # size and mtime are in the key only so a rewritten file is read afresh
    return read_spectra_csv(Path(path), z_mm, list(freqs))


def _read_rx_row(data_file: Path, z_mm: float, freqs: Sequence[float]) -> SpectraRows:
    """read_spectra_csv(data_file, z_mm, freqs), read once per process for the same
    file contents, row and frequencies (see the module docstring). The result is
    shared: callers copy what they keep. Raises MeasuredDataError (the reader's)
    when a frequency is not a column or the header is not the spectra format."""
    st = Path(data_file).stat()
    return _read_rx_row_cached(str(data_file), int(st.st_size), int(st.st_mtime_ns),
                               float(z_mm), tuple(float(f) for f in freqs))


def data_file_digest(path: Path) -> Tuple[int, str]:
    """(size in bytes, sha256 hex digest) of the data file, for the run manifest:
    topt-parser regenerates the file under the same name, so the name alone does
    not say which data a run used."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def derive_geometry(mcfg: "MeasuredConfig", scene_cfg: SimSceneConfig,
                    freqs: Sequence[float]) -> MeasuredGeometry:
    """Resolve a `measured:` block against the data file and the scene.

    The RX row must be a row of the file exactly (|Z - z| < 1e-6 mm), and both RX
    edges X values of it: the elements are the measured positions themselves, and
    a position the stage never visited has no spectrum to read (interpolating
    between rows is exactly what the on-grid receiver exists to avoid). Every X of
    the row between the edges is an element, at one uniform pitch that is a whole
    number of scene steps, so each element is a scene sample.

    `freqs` are the run's frequencies (Hz): the file must have a column for each,
    which the read checks (read_spectra_csv), and each must be > 0 (_check_freqs).
    Raises MeasuredDataError, with positions in stage mm, when a frequency is <= 0
    or not a column, the file is missing, the RX z is not a row of it (naming the
    nearest rows), an RX edge is not an X of that row (naming the nearest X), a
    position between the edges is missing, the pitch is not uniform or not a whole
    number of scene spacings, the TX is not upstream of the RX, or either aperture
    falls outside the scene.
    """
    freqs = _check_freqs(freqs)
    tx, rx = mcfg.tx_aperture, mcfg.rx_aperture
    _check_segment("tx_aperture", tx)
    _check_segment("rx_aperture", rx)

    # the frame: everything below is in the simulation frame unless it says mm
    z_rx_mm = float(rx.left.z)
    rx_left_mm, rx_right_mm = sorted((float(rx.left.x), float(rx.right.x)))
    x_center_mm = (rx_left_mm + rx_right_mm) / 2
    tx_left_mm, tx_right_mm = sorted((float(tx.left.x), float(tx.right.x)))
    tx_z_mm = float(tx.left.z)
    (tx_x_min, tx_x_max), tx_z = stage_to_sim(
        np.array([tx_left_mm, tx_right_mm]), tx_z_mm, z_rx_mm, x_center_mm)
    tx_x_min, tx_x_max, tx_z = float(tx_x_min), float(tx_x_max), float(tx_z)
    rx_x_lo, _ = stage_to_sim(rx_left_mm, z_rx_mm, z_rx_mm, x_center_mm)
    rx_x_hi, _ = stage_to_sim(rx_right_mm, z_rx_mm, z_rx_mm, x_center_mm)
    frame = (f"z_sim = ({_mm(z_rx_mm)} mm - z) * 1e-3, "
             f"x_sim = (x - {_mm(x_center_mm)} mm) * 1e-3")
    sc = scene_cfg

    # the TX against the RX and the scene: no data needed, so these fail first
    if not tx_z > 0:
        raise MeasuredDataError(
            f"measured.tx_aperture z = {_mm(tx_z_mm)} mm is not upstream of the RX "
            f"row at {_mm(z_rx_mm)} mm: the TX must have the SMALLER stage z (the "
            "beam travels toward +z on the stage, -z in the simulation); swap or fix "
            "the z values")
    if not (sc.z_min < tx_z <= sc.z_max + _POS_TOL):
        raise MeasuredDataError(
            f"the TX sits at z = {tx_z:.6g} m in the simulation frame ({frame}), "
            f"outside the scene's z range ({sc.z_min}, {sc.z_max}] m; raise "
            "sim_scene.z_max (keeping z_max - z_min a whole number of spacings) or "
            "move the TX closer")
    if tx_x_min < sc.x_min - _POS_TOL or tx_x_max > sc.x_max + _POS_TOL:
        raise MeasuredDataError(
            f"the TX at x = {_mm(tx_left_mm)}..{_mm(tx_right_mm)} mm maps to "
            f"x = {tx_x_min:.6g}..{tx_x_max:.6g} m in the simulation frame "
            f"({frame}), outside the scene's x range [{sc.x_min}, {sc.x_max}] m; "
            "widen sim_scene.x_min / x_max (keeping x_min == -x_max)")
    if rx_x_lo < sc.x_min - _POS_TOL or rx_x_hi > sc.x_max + _POS_TOL:
        raise MeasuredDataError(
            f"the RX window, {rx_right_mm - rx_left_mm:g} mm wide, does not fit in "
            f"the scene's x range [{sc.x_min}, {sc.x_max}] m; widen sim_scene.x_min "
            "/ x_max (keeping x_min == -x_max) or narrow the RX")

    # the RX row: a row of the file, exactly
    data_file = _data_file(mcfg.data_file)
    rows = _read_rx_row(data_file, z_rx_mm, freqs)
    all_z = np.asarray(rows.all_z_mm, dtype=np.float64)
    if all_z.size == 0:
        raise MeasuredDataError(
            f"{data_file} has a header but no data rows; re-run topt-parser on the "
            "pulse folder")
    xs = np.asarray(rows.x_mm, dtype=np.float64)
    if xs.size == 0:
        raise MeasuredDataError(
            f"measured.rx_aperture z = {_mm(z_rx_mm)} mm is not a measured row of "
            f"{data_file} (rows run {_range_text(all_z)}); set both RX z values to "
            f"{_nearest_text(all_z, z_rx_mm)}")
    lat = _lattice(xs)
    twice = np.unique(lat[1:][np.diff(lat) == 0])
    if twice.size:
        raise MeasuredDataError(
            f"{data_file} has more than one row at z = {_mm(z_rx_mm)} mm, x = "
            f"{', '.join(_mm(v / _PER_MM) for v in twice[:10])} mm; which one is "
            "the measurement is not ours to guess: re-run topt-parser on a folder "
            "with one trace per position")

    # the RX edges: X values of that row
    row_text = f"row {_mm(z_rx_mm)} mm (runs {_range_text(xs)})"
    edge = {}
    for key in ("left", "right"):
        x_mm = float(getattr(rx, key).x)
        hit = np.flatnonzero(np.abs(xs - x_mm) < _POS_MM_TOL)
        if hit.size == 0:
            raise MeasuredDataError(
                f"measured.rx_aperture.{key} x = {_mm(x_mm)} mm is not a measured x "
                f"of {row_text}; put the edge on {_nearest_text(xs, x_mm)}")
        edge[key] = int(hit[0])
    lo_i, hi_i = sorted(edge.values())
    if lo_i == hi_i:
        raise MeasuredDataError(
            f"measured.rx_aperture: both edges are the position x = {_mm(xs[lo_i])} "
            "mm; give the RX two distinct edges")

    # the elements: every position between the edges, at the row's own step. The
    # row's step is its most common one, so a stretch of the window at a multiple
    # of it is positions the stage skipped (or lost), not a coarser receiver
    win = lat[lo_i:hi_i + 1]
    steps = np.diff(win)
    row_step = _row_step(lat)
    if not np.all(steps == row_step):
        if np.all(steps % row_step == 0):
            missing = sorted(set(range(int(win[0]), int(win[-1]) + 1, row_step))
                             - set(win.tolist()))
            shown = ", ".join(_mm(x / _PER_MM) for x in missing[:10])
            more = f" and {len(missing) - 10} more" if len(missing) > 10 else ""
            raise MeasuredDataError(
                f"{row_text} has no value at x = {shown}{more} mm, inside the RX "
                f"window {_mm(rx_left_mm)}..{_mm(rx_right_mm)} mm; every element "
                "between the edges must be measured: move the edges to avoid the "
                "gap or re-measure those positions")
        raise MeasuredDataError(
            f"the positions of {row_text} inside the RX window "
            f"{_mm(rx_left_mm)}..{_mm(rx_right_mm)} mm are not evenly spaced "
            f"(steps of {', '.join(_mm(s / _PER_MM) for s in sorted(set(steps.tolist())))} "
            "mm); the receiver needs one pitch: move the edges to an evenly measured "
            "stretch")
    pitch_m = int(steps[0]) / (_PER_MM * 1000)      # lattice -> m, correctly rounded
    stride = _whole(pitch_m / sc.spacing)
    if stride is None or stride < 1:
        raise MeasuredDataError(
            f"the RX pitch {_mm(pitch_m * 1e3)} mm is not a whole number of scene "
            f"steps: {pitch_m:g} / sim_scene.spacing {sc.spacing:g} = "
            f"{pitch_m / sc.spacing:.6g}; set sim_scene.spacing to the pitch divided "
            f"by a whole number (e.g. {pitch_m / 2:g} or {pitch_m:g}) so every "
            "element is a scene sample")

    rx_x_mm = xs[lo_i:hi_i + 1].copy()
    rx_x_sim, _ = stage_to_sim(rx_x_mm, z_rx_mm, z_rx_mm, x_center_mm)
    return MeasuredGeometry(
        data_file=data_file,
        z_rx_mm=z_rx_mm,
        x_center_mm=x_center_mm,
        rx_x_mm=rx_x_mm,
        rx_x_sim=np.asarray(rx_x_sim, dtype=np.float64),
        pitch_m=pitch_m,
        rx_width_m=(rx_right_mm - rx_left_mm) * 1e-3,
        tx_x_min=tx_x_min,
        tx_x_max=tx_x_max,
        tx_z=tx_z,
        tx_width_m=tx_x_max - tx_x_min,
        tx_stage_mm=_stage_dict(tx),
        rx_stage_mm=_stage_dict(rx),
    )


def _derived_dump(config: SimConfig) -> tuple:
    """The three blocks apply_measured writes, as plain data (to tell whether a
    call changed anything)."""
    gs = config.grid_search
    return tuple(None if m is None else m.model_dump()
                 for m in (config.tx_aperture, config.rx_aperture,
                           None if gs is None else gs.aperture))


def apply_measured(config: SimConfig, freqs: Sequence[float]) -> MeasuredGeometry:
    """Derive the geometry of `config.measured` at the run's frequencies `freqs`
    (Hz; derive_geometry checks the data file has a column for each) and write it
    into `config` in place, so the rest of the pipeline sees an ordinary on-grid
    SimConfig:

      tx_aperture            the measured TX window (x_min, x_max, z); dx and beam
                             kept from the current block (for a measured config,
                             config.py's stand-in: the beam is never synthesized on
                             the measured path, the plate's profile being the
                             unknown, but the block needs one to validate).
      rx_aperture            centred on 0, the measured RX width, grid_stride =
                             pitch / spacing, dx null: the on-grid receiver.
      grid_search.aperture   width = the measured TX width (dx kept): the candidate
                             plate is as wide as the real one.

    The blocks are built through their constructors so their validation runs (the
    models have no validate_assignment). Idempotent: a second call derives the
    same values and leaves the config as it was. The geometry is logged at INFO by
    the call that writes it, at DEBUG by one that finds it already applied.
    Raises ValueError when the config has no measured block, MeasuredDataError
    (derive_geometry) when it cannot be resolved.
    """
    mcfg = getattr(config, "measured", None)
    if mcfg is None:
        raise ValueError("apply_measured needs a config with a `measured:` block")
    scene = config.sim_scene
    g = derive_geometry(mcfg, scene, freqs)
    stride = int(round(g.pitch_m / scene.spacing))
    before = _derived_dump(config)

    cur = config.tx_aperture
    config.tx_aperture = TxApertureConfig(x_min=g.tx_x_min, x_max=g.tx_x_max,
                                          z=g.tx_z, dx=cur.dx, beam=cur.beam)
    config.rx_aperture = RxApertureConfig(x_center=0.0, width=g.rx_width_m,
                                          grid_stride=stride, dx=None)
    if config.grid_search is not None:
        config.grid_search.aperture = GridApertureConfig(
            width=g.tx_width_m, dx=config.grid_search.aperture.dx)

    log = logging.getLogger()
    level = logging.INFO if _derived_dump(config) != before else logging.DEBUG
    log.log(level,
            f"measured geometry ({g.data_file}): RX row {_mm(g.z_rx_mm)} mm, "
            f"{len(g.rx_x_mm)} elements at "
            f"x = {_mm(g.rx_x_mm[0])}..{_mm(g.rx_x_mm[-1])} mm, "
            f"pitch {_mm(g.pitch_m * 1e3)} mm = {stride} scene steps -> "
            f"x_sim {g.rx_x_sim[0]:.6g}..{g.rx_x_sim[-1]:.6g} m; "
            f"TX x_sim {g.tx_x_min:.6g}..{g.tx_x_max:.6g} m (width {g.tx_width_m:.6g} m), "
            f"z_sim {g.tx_z:.6g} m; frame z_sim = ({_mm(g.z_rx_mm)} mm - z) * 1e-3, "
            f"x_sim = (x - {_mm(g.x_center_mm)} mm) * 1e-3")
    return g


def place_elements(scene_x_axis: np.ndarray, rx_x_sim: np.ndarray) -> np.ndarray:
    """Indices of the measured RX elements in `scene_x_axis`, one per element.

    Each element is placed at the NEAREST sample, idx = round((x - axis[0]) /
    step), and must coincide with it to 1e-9 m. Not grid_element_indices: that
    picks every stride-th sample of a closed window by float comparison, which
    decides the edge samples by round-off and knows nothing of where the elements
    actually are. Here the elements' positions are the measurement and the scene
    must have a sample at each (an exact-step axis whose spacing divides the
    pitch, which derive_geometry checks).

    Returns an ascending int array of unique indices. Raises MeasuredDataError
    when an element is off the axis's range, between samples, or two elements
    share a sample.
    """
    axis = np.asarray(scene_x_axis, dtype=np.float64)
    x = np.asarray(rx_x_sim, dtype=np.float64)
    if axis.ndim != 1 or len(axis) < 2:
        raise MeasuredDataError("the scene x axis needs at least two samples")
    step = axis[1] - axis[0]
    idx = np.round((x - axis[0]) / step).astype(np.intp)
    outside = (idx < 0) | (idx >= len(axis))
    if np.any(outside):
        bad = x[outside]
        raise MeasuredDataError(
            f"{int(outside.sum())} RX element(s) (x = {bad.min():.6g}..{bad.max():.6g} "
            f"m) lie outside the scene x axis [{axis[0]:.6g}, {axis[-1]:.6g}] m; widen "
            "sim_scene.x_min / x_max")
    off = np.abs(axis[idx] - x)
    if np.any(off >= _POS_TOL):
        worst = int(np.argmax(off))
        raise MeasuredDataError(
            f"{int((off >= _POS_TOL).sum())} RX element(s) fall between scene "
            f"samples (worst: x = {x[worst]:.9g} m, {off[worst]:.3g} m from the "
            f"nearest sample {axis[idx[worst]]:.9g} m); the scene needs "
            "sim_scene.exact_step: true and a spacing that divides the RX pitch")
    if not np.all(np.diff(idx) > 0):
        raise MeasuredDataError(
            "the RX elements do not map to strictly ascending, distinct scene samples; "
            "the scene spacing is coarser than the RX pitch, or the elements are not "
            "in ascending x")
    return idx


def measured_rx_profile(geom: MeasuredGeometry, freqs: Sequence[float],
                        amplitude_scale: float, log=None) -> Tuple[np.ndarray, List[dict]]:
    """The RX elements' measurement at each of `freqs` (Hz): amplitude_scale times
    the data file's V(f) cell of each element, from the column of that frequency.

    Returns (profiles, reports). profiles is (len(freqs), n) complex128, one row
    per frequency in the given order, each in ascending x (geom's order). reports
    is one dict per frequency, {"freq", "column", "amplitude_scale",
    "raw_abs_max", "raw_abs_median"}, all plain JSON values; "column" is the
    header of the column read.

    Only the RX row's values are read, and only the elements between the edges
    are kept: nothing outside the chosen aperture enters the solve.
    `amplitude_scale` is the config's fixed scale (no calibration happens here);
    the raw |V| max and median logged per frequency are what to choose it from.
    Raises MeasuredDataError on a frequency <= 0 or not a column, an element the
    file no longer has, or a non-finite value. Logs one INFO line per frequency.
    """
    log = log or logging.getLogger()
    freqs = _check_freqs(freqs)
    amplitude_scale = float(amplitude_scale)
    if not (math.isfinite(amplitude_scale) and amplitude_scale > 0):
        raise ValueError(
            f"amplitude_scale must be a finite number > 0, got {amplitude_scale}")

    rows = _read_rx_row(geom.data_file, geom.z_rx_mm, freqs)
    row = f"RX row {_mm(geom.z_rx_mm)} mm"
    lat_row = _lattice(rows.x_mm)
    lat_el = _lattice(geom.rx_x_mm)
    idx = np.searchsorted(lat_row, lat_el)
    found = (idx < len(lat_row)) & (lat_row[np.minimum(idx, len(lat_row) - 1)] == lat_el)
    if not np.all(found):
        gone = ", ".join(_mm(x) for x in geom.rx_x_mm[~found][:10])
        raise MeasuredDataError(
            f"{geom.data_file} has no value at {row}, x = {gone} mm, which the "
            "geometry was derived with: the file changed during the run; run again")
    raw = np.asarray(rows.values, dtype=np.complex128)[idx].T.copy()   # (F, n)
    columns = [freq_header(float(f)) for f in rows.freqs_hz]
    bad = ~np.isfinite(raw)
    if np.any(bad):
        f_i, x_i = np.nonzero(bad)
        where = ", ".join(f"x={_mm(geom.rx_x_mm[j])} mm in {columns[i]}"
                          for i, j in list(zip(f_i, x_i))[:5])
        raise MeasuredDataError(
            f"{geom.data_file}: {int(bad.sum())} non-finite value(s) on {row} "
            f"({where}); the solve needs a finite field at every element")

    profiles = amplitude_scale * raw
    reports = []
    for i, freq in enumerate(freqs):
        raw_abs = np.abs(raw[i])
        report = {
            "freq": freq,
            "column": columns[i],
            "amplitude_scale": amplitude_scale,
            "raw_abs_max": float(raw_abs.max()),
            "raw_abs_median": float(np.median(raw_abs)),
        }
        reports.append(report)
        log.info(f"measured {row} at {freq / 1e9:g} GHz (column {columns[i]} of "
                 f"{geom.data_file.name}): {raw.shape[1]} elements, raw |V| max "
                 f"{report['raw_abs_max']:.6g} median {report['raw_abs_median']:.6g}; "
                 f"amplitude_scale {amplitude_scale:g} -> peak |y| "
                 f"{amplitude_scale * report['raw_abs_max']:.6g}")
    return profiles, reports
