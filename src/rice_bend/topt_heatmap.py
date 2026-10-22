"""`topt-heatmap`: one frequency of a topt-parser spectra CSV as a log-magnitude map.

    topt-heatmap ~/Desktop/experiments/topt_spectra.csv --freq 300GHz
    topt-heatmap CSV --freq 300GHz --config configs/measured/measured_rx177.yml -o rx.png

THE SCALE. dB re the map's peak, 20*log10(|V| / max|V|), the max over every cell
of the map: the peak cell is 0 dB, a cell at half its amplitude -6.02 dB. The
colour floor is -range dB (default 60): a cell below it, or at |V| = 0, gets the
floor colour. A (Z, X) with no row in the CSV is drawn grey.

THE AXES. The CSV's stage X and Z in mm, X across and Z up, 1 mm of X the length
of 1 mm of Z. Each cell spans the midpoints to its neighbours (half a step past
the outer samples). With --config, the measured config's TX and RX apertures
(measured.tx_aperture / rx_aperture, stage mm) are drawn on top, and the axes grow
to take in any that lie outside the scan.

The output defaults to <csv stem>_<freq header>.png next to the CSV (e.g.
topt_spectra_300GHz.png); an existing PNG is overwritten.
"""

import argparse
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from rice_bend.cli import setup_logging
from rice_bend.tds_data import MeasuredDataError
from rice_bend.topt_parser import SpectraMap, freq_header, read_spectra_map

DEFAULT_DB_RANGE = 60.0
CMAP = "inferno"                 # the repo's scene plots use it too
MISSING_COLOR = "0.6"            # cells with no row in the CSV
TX_COLOR, RX_COLOR = "cyan", "red"
# Overlays outside the scan widen the axes by this fraction of the span, beyond them.
OVERLAY_MARGIN = 0.04
DPI = 150

# --freq: a number, then an optional unit (none = Hz)
_FREQ_ARG = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)\s*$")
_FREQ_UNITS = {"": 1.0, "hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12}


def parse_freq(text: str) -> float:
    """A --freq value in Hz: "300e9", "300GHz", "0.3THz", "3e11Hz" (units case
    blind, a space before them allowed). ValueError on anything else or < 0."""
    m = _FREQ_ARG.match(text)
    unit = m.group(2).lower() if m else None
    if unit not in _FREQ_UNITS:
        raise ValueError(
            f"--freq {text!r}: give a frequency in Hz (300e9) or with a unit "
            "(300GHz, 0.3THz, 3e11Hz)")
    hz = float(m.group(1)) * _FREQ_UNITS[unit]
    if not (math.isfinite(hz) and hz >= 0):
        raise ValueError(f"--freq {text!r}: give a frequency >= 0")
    return hz


@dataclass(frozen=True)
class ApertureOverlay:
    """A horizontal aperture to draw over the map: x0..x1 at z, stage mm."""
    label: str
    x0_mm: float
    x1_mm: float
    z_mm: float
    color: str
    linestyle: str


def measured_overlays(config_path: Path) -> Tuple[List[ApertureOverlay], Path]:
    """The TX and RX apertures of a measured config (load_config, so a base_config
    is laid under it), and its measured.data_file. apply_measured is not called:
    the stage points are drawn as written. ValueError (a pydantic ValidationError
    among them) on a config that does not load or has no `measured:` block."""
    from rice_bend.config import load_config   # pydantic: only when asked for

    cfg = load_config(Path(config_path).expanduser())
    m = cfg.measured
    if m is None:
        raise ValueError(
            f"{config_path} has no `measured:` block; --config takes a measured "
            "config (configs/measured/<name>.yml), whose measured.tx_aperture and "
            "measured.rx_aperture are drawn")
    overlays = [
        ApertureOverlay("TX (config)", m.tx_aperture.left.x, m.tx_aperture.right.x,
                        m.tx_aperture.left.z, TX_COLOR, "--"),
        ApertureOverlay("RX (config)", m.rx_aperture.left.x, m.rx_aperture.right.x,
                        m.rx_aperture.left.z, RX_COLOR, "-"),
    ]
    return overlays, Path(m.data_file)


def db_re_peak(values: np.ndarray, db_range: float = DEFAULT_DB_RANGE
               ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """(db, peak, (iz, ix)): 20*log10(|V| / peak) per cell with peak = max|V| over
    the finite cells, at values[iz, ix]. Clipped below at -db_range, so |V| = 0
    is -db_range too; NaN where |V| is not finite (no row). MeasuredDataError when
    no cell is finite or every finite cell is 0 (no peak to refer to)."""
    if not (math.isfinite(db_range) and db_range > 0):
        raise ValueError(f"db_range {db_range:g}: give a number of dB > 0")
    mag = np.abs(values)
    finite = np.isfinite(mag)
    if not finite.any():
        raise MeasuredDataError("no finite cell in the map: nothing to plot")
    k = int(np.argmax(np.where(finite, mag, -1.0)))
    peak = float(mag.flat[k])
    if not peak > 0:
        raise MeasuredDataError("every cell of the map is 0: there is no peak to "
                                "refer the dB scale to")
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.where(finite, mag, np.nan) / peak)
    db = np.maximum(db, -db_range)              # NaN stays NaN; -inf -> the floor
    iz, ix = np.unravel_index(k, values.shape)
    return db, peak, (int(iz), int(ix))


def cell_edges(v: np.ndarray) -> np.ndarray:
    """(n+1,) edges of the cells around ascending samples v: the midpoints, and
    half a step past each outer sample (a lone sample gets a 1 mm cell)."""
    v = np.asarray(v, dtype=np.float64)
    if v.size == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    mid = (v[:-1] + v[1:]) / 2
    return np.concatenate([[2 * v[0] - mid[0]], mid, [2 * v[-1] - mid[-1]]])


def _limits(lo: float, hi: float, points: Sequence[float]) -> Tuple[float, float]:
    """[lo, hi] widened to take in `points`, with a margin on a side they pass."""
    if not points:
        return lo, hi
    p_lo, p_hi = min(points), max(points)
    margin = OVERLAY_MARGIN * (max(hi, p_hi) - min(lo, p_lo))
    return (min(lo, p_lo - margin) if p_lo < lo else lo,
            max(hi, p_hi + margin) if p_hi > hi else hi)


def default_out_path(csv: Path, freq_hz: float) -> Path:
    """<csv stem>_<freq header>.png beside the CSV: topt_spectra_300GHz.png."""
    csv = Path(csv).expanduser()
    return csv.with_name(f"{csv.stem}_{freq_header(freq_hz)}.png")


def plot_heatmap(smap: SpectraMap, out: Path, *, db_range: float = DEFAULT_DB_RANGE,
                 overlays: Optional[Sequence[ApertureOverlay]] = None,
                 show: bool = False) -> Path:
    """Draw |smap.values| in dB re its peak (db_re_peak) on the stage X/Z grid,
    equal aspect, with `overlays` on top, save it to `out` (PNG) and return the
    path. `show` also opens a window (pyplot, the current backend); otherwise
    the figure is drawn off-screen and pyplot is not touched."""
    import matplotlib
    from matplotlib import patheffects
    from matplotlib.patches import Patch

    out = Path(out).expanduser()
    overlays = list(overlays or [])
    db, peak, (iz, ix) = db_re_peak(smap.values, db_range)
    xe, ze = cell_edges(smap.x_mm), cell_edges(smap.z_mm)
    xlim = _limits(xe[0], xe[-1], [x for o in overlays for x in (o.x0_mm, o.x1_mm)])
    zlim = _limits(ze[0], ze[-1], [o.z_mm for o in overlays])

    # the axes box follows the data aspect, within a sane page
    ratio = (zlim[1] - zlim[0]) / (xlim[1] - xlim[0])
    w_ax, h_ax = 6.0, 6.0 * ratio
    if h_ax > 10.0:
        w_ax, h_ax = 10.0 / ratio, 10.0
    figsize = (w_ax + 2.6, max(h_ax, 2.0) + 2.4)
    if show:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=figsize, layout="constrained")
    else:
        from matplotlib.figure import Figure
        fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot()

    cmap = matplotlib.colormaps[CMAP].with_extremes(bad=MISSING_COLOR)
    mesh = ax.pcolormesh(xe, ze, np.ma.masked_invalid(db), cmap=cmap,
                         vmin=-db_range, vmax=0.0, shading="flat")
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_xlabel("X (mm, stage)")
    ax.set_ylabel("Z (mm, stage)")
    f_txt = f"{smap.freq_hz / 1e9:g} GHz" + (" (DC)" if smap.freq_hz == 0 else "")
    ax.set_title(f"{smap.path.name}: |V| at {f_txt}\n"
                 f"20·log10(|V| / max|V|), floor −{db_range:g} dB", fontsize=11)

    cb = fig.colorbar(mesh, ax=ax, extend="min", fraction=0.06, pad=0.03)
    cb.set_label(f"dB re peak  (peak |V| = {peak:.4g} at X = "
                 f"{smap.x_mm[ix]:g} mm, Z = {smap.z_mm[iz]:g} mm)")

    handles = []
    for o in overlays:
        halo = "black" if o.color == TX_COLOR else "white"
        line, = ax.plot([o.x0_mm, o.x1_mm], [o.z_mm, o.z_mm], color=o.color,
                        linestyle=o.linestyle, linewidth=2.5, label=o.label,
                        solid_capstyle="butt", dash_capstyle="butt",
                        path_effects=[patheffects.Stroke(linewidth=4.5, foreground=halo),
                                      patheffects.Normal()])
        handles.append(line)
    n_missing = int(np.isnan(db).sum())
    if n_missing:
        handles.append(Patch(facecolor=MISSING_COLOR, edgecolor="none",
                             label=f"no row in the CSV ({n_missing} cell"
                                   f"{'s' if n_missing != 1 else ''})"))
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
                   frameon=False, fontsize=9)

    fig.savefig(out, dpi=DPI)
    if show:
        plt.show()
        plt.close(fig)
    return out


def _config_error_text(e: Exception) -> str:
    """One line for a config that does not load: pydantic's errors as
    `loc: msg; ...` (as grid-search-mgs words them), anything else as str(e)."""
    from pydantic import ValidationError
    if not isinstance(e, ValidationError):
        return str(e)
    problems = []
    for err in e.errors():
        msg = err["msg"]
        if msg.startswith("Value error, "):          # str.removeprefix is 3.9+
            msg = msg[len("Value error, "):]
        where = ".".join(str(p) for p in err["loc"])
        problems.append(f"{where}: {msg}" if where else msg)
    return "; ".join(problems)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="topt-heatmap",
        description="Plot one frequency column of a topt-parser spectra CSV as a "
                    "heatmap of |V| in dB re the map's peak (20*log10(|V|/max|V|)), "
                    "on the stage X/Z grid in mm at equal aspect")
    parser.add_argument("csv", type=Path, metavar="CSV",
                        help="The spectra CSV topt-parser wrote")
    parser.add_argument("--freq", required=True, metavar="F",
                        help="The frequency to plot, a column of the CSV (within 1 "
                             "Hz): Hz (300e9) or with a unit (300GHz, 0.3THz, 3e11Hz)")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="Output PNG (default: <csv stem>_<freq>.png beside the "
                             "CSV, e.g. topt_spectra_300GHz.png; overwritten if it "
                             "exists)")
    parser.add_argument("--range", type=float, default=DEFAULT_DB_RANGE, metavar="DB",
                        dest="db_range",
                        help=f"Colour range below the peak, dB (default: "
                             f"{DEFAULT_DB_RANGE:g}); cells below -DB get the floor "
                             "colour")
    parser.add_argument("--config", type=Path, default=None, metavar="YML",
                        help="A measured config (configs/measured/<name>.yml): draw "
                             "its measured TX and RX apertures on the map")
    parser.add_argument("--show", action="store_true",
                        help="Also open the figure in a window")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    args = parser.parse_args(argv)
    try:
        args.freq_hz = parse_freq(args.freq)
    except ValueError as e:
        parser.error(str(e))
    if not (math.isfinite(args.db_range) and args.db_range > 0):
        parser.error(f"--range {args.db_range:g}: give a number of dB > 0")
    return args


def main(argv=None):
    args = _parse_args(argv)
    log = setup_logging(args.debug)
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    csv = args.csv.expanduser()
    if not csv.is_file():
        log.error(f"{csv}: no such file; give the spectra CSV topt-parser wrote "
                  "(e.g. ~/Desktop/experiments/topt_spectra.csv)")
        raise SystemExit(2)
    out = None if args.out is None else args.out.expanduser()
    if out is not None and (out.is_dir() or not out.parent.is_dir()):
        log.error(f"-o {out}: " + ("is a directory; give a file path" if out.is_dir()
                                   else f"the directory {out.parent} does not exist"))
        raise SystemExit(2)

    overlays = []
    if args.config is not None:
        cfg_path = args.config.expanduser()
        if not cfg_path.is_file():
            log.error(f"--config {cfg_path}: no such file")
            raise SystemExit(2)
        import yaml
        try:
            overlays, data_file = measured_overlays(cfg_path)
        except (ValueError, TypeError, OSError, yaml.YAMLError) as e:
            log.error(f"--config {cfg_path}: {_config_error_text(e)}")
            raise SystemExit(2)
        if data_file.resolve() != csv.resolve():
            log.info(f"{cfg_path}: its measured.data_file is {data_file}, not the CSV "
                     f"plotted ({csv}); its apertures are drawn all the same")
        for o in overlays:
            log.info(f"overlay {o.label}: x {min(o.x0_mm, o.x1_mm):g}.."
                     f"{max(o.x0_mm, o.x1_mm):g} mm at z = {o.z_mm:g} mm")

    started = time.perf_counter()
    try:
        smap = read_spectra_map(csv, args.freq_hz)
    except MeasuredDataError as e:
        hint = ""
        m = _FREQ_ARG.match(args.freq)
        if m and not m.group(2) and 0 < args.freq_hz < 1e6:
            hint = (f" (--freq without a unit is in Hz: {args.freq} Hz; did you mean "
                    f"{args.freq}GHz?)")
        log.error(f"{e}{hint}")
        raise SystemExit(2)
    except OSError as e:
        log.error(f"{csv}: unreadable ({e})")
        raise SystemExit(2)
    nz, nx = smap.values.shape
    n_absent = int(np.isnan(smap.values.real).sum())
    log.info(f"read {csv} at {freq_header(smap.freq_hz)}: {nz} Z x {nx} X "
             f"(Z {smap.z_mm[0]:g}..{smap.z_mm[-1]:g} mm, X {smap.x_mm[0]:g}.."
             f"{smap.x_mm[-1]:g} mm), {nz * nx - n_absent} rows, in "
             f"{time.perf_counter() - started:.1f} s")
    if n_absent:
        log.info(f"{n_absent} of {nz * nx} (Z, X) cells have no row in the CSV: "
                 "drawn grey")

    out = out if out is not None else default_out_path(csv, smap.freq_hz)
    try:
        db, peak, (iz, ix) = db_re_peak(smap.values, args.db_range)
        log.info(f"peak |V| = {peak:.6g} at X = {smap.x_mm[ix]:g} mm, Z = "
                 f"{smap.z_mm[iz]:g} mm; {int((db <= -args.db_range).sum())} cells at "
                 f"or below the -{args.db_range:g} dB floor")
        plot_heatmap(smap, out, db_range=args.db_range, overlays=overlays,
                     show=args.show)
    except MeasuredDataError as e:
        log.error(f"{csv} at {freq_header(smap.freq_hz)}: {e}")
        raise SystemExit(2)
    except OSError as e:
        log.error(f"{out}: cannot write ({e})")
        raise SystemExit(2)
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
