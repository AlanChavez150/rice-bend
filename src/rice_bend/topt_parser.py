"""`topt-parser`: a folder of Toptica THz-TDS pulse traces -> one spectra CSV.

This is the only place a measured pulse is turned into numbers the solver uses.
The grid search reads the CSV (read_spectra_csv, below), never the traces, so the
format lives here twice over: the writer and the reader, side by side. (A second
reader, read_spectra_map, takes one column of every row, for topt-heatmap.)

THE FRAME. Every trace of the first raster is 4001 samples 0.05 ps apart, from the
window start to its closing endpoint included (950.000 ... 1150.000 ps). Dropping
that last sample leaves an N = 4000-sample frame exactly N*dt = 200 ps long, whose
FFT bins are k / (N*dt) = k * 5 GHz exactly: 300 GHz is bin 60, 1 THz bin 200. With
all 4001 samples the bins would be 4.99875 GHz apart and no carrier would land on
one. Bins k = 0 .. K, K = floor(fmax * N*dt), are kept: the DC bin too, as a column
like any other (the grid search refuses to solve at 0 Hz, where there is no
wavelength, not this file).

THE CELL. For one trace with absolute sample times t_n = t0 + n*dt and Signal 1
samples s_n, n < N:

    V(f_k) = conj( exp(-2j*pi*f_k*t0) * rfft(s[:N])[k] ) = conj( sum_n s_n exp(-2j*pi*f_k*t_n) )

  * The exp(-2j*pi*f_k*t0) factor references the phase to ABSOLUTE time. The window
    start moves from row to row; without it two rows' phases would be measured from
    different zeros and could not be compared.
  * The conjugate gives the exp(+ikr) (exp(-i w t)) convention of the RS/Hankel
    kernel in rs.py, which the simulated RX field is built with. A field amplitude
    a means s(t) = Re{a exp(-i w t)} = |a| cos(w t - arg a); the FFT sum picks out
    the exp(+i w t) half of that real signal, conj(a) * N/2, and conjugating returns
    a * N/2. So a tone A cos(w t - phi) gives A*N/2 * exp(+i phi), and a pulse
    delayed by tau gains phase +2*pi*f*tau, as a field carried a further c*tau
    through the exp(+ikr) kernel does.
  * No dt factor, gate, taper or amplitude scale: the measured config's
    amplitude_scale absorbs the units.
  * At k = 0 there is no time correction (f = 0): the cell is the plain sum of the
    N samples, real.

THE CSV. Plain comma-separated, "\\n" line ends, no comment lines:

    X,Z,0GHz,5GHz,...,1000GHz
    <X mm>,<Z mm>,<V(0)>,<V(5 GHz)>,...

X and Z are the stage position in mm (the file name's units times --unit-mm); rows
are sorted by Z, then X. Each cell is a Python complex literal at 17 significant
digits, e.g. 12.3-0.45j, which complex() reads back to the identical float64 pair.
A 4000-sample raster of 40,501 traces comes to about 380 MB.

The checks are format checks, not pulse QA: every trace must have the same number
of samples and the same step, on a uniform time grid, because one table of columns
needs one frequency per column. Whether a pulse is cut off by its window is not
judged here (the user re-captures such rows).
"""

import argparse
import logging
import math
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rice_bend.cli import setup_logging
from rice_bend.parallel import map_workers, round_robin_chunks, worker_shared
from rice_bend.tds_data import (MeasuredDataError, PulseTrace, index_pulse_dir,
                                read_pulse)

DEFAULT_FMAX_HZ = 1e12
DEFAULT_UNIT_MM = 0.1
DEFAULT_OUT_NAME = "topt_spectra.csv"

# Every trace's step must match the first trace's to this (relative): one column is
# one frequency only if every row's bin k sits at the same k / (N*dt).
DT_RTOL = 1e-9
# Every sample time must sit this close to t0 + n*dt (ps). The files print times to
# 1e-3 ps, so a uniform trace is off by float round-off only; a skipped or repeated
# sample is off by a whole step.
UNIFORM_TOL_PS = 1e-3
# A requested frequency picks the column whose header is within this (Hz).
FREQ_MATCH_HZ = 1.0
# A row belongs to the requested Z when within this (mm): the CSV prints Z to 1e-9.
Z_MATCH_MM = 1e-6
# How many offending traces an error names.
MAX_NAMED = 5

# One cell, both parts at 17 significant digits: the shortest width at which every
# float64 survives str -> float unchanged. Shared by format_complex and the writer's
# row loop (which skips the per-cell function call: 8 million cells).
_CELL = "{:.17g}{:+.17g}j".format
_HEADER_FREQ = re.compile(r"^(.+)GHz$")


def format_complex(z: complex) -> str:
    """One CSV cell: `f"{z.real:.17g}{z.imag:+.17g}j"`, e.g. "12.3-0.45j", which
    complex() reads back to exactly z. A negative-zero imaginary part is written as
    +0 (the DC cell's conjugate makes one), so a real value prints "123.4+0j"."""
    z = complex(z)
    return _CELL(z.real, z.imag + 0.0)       # -0.0 + 0.0 == +0.0


def freq_header(f_hz: float) -> str:
    """A frequency column's header, in GHz: 300e9 -> "300GHz", 0 -> "0GHz"."""
    return f"{f_hz / 1e9:g}GHz"


def _mm(v_file: int, unit_mm: float) -> str:
    """A file-unit position as the CSV's mm: 1770 * 0.1 -> "177" (not
    177.00000000000003), 2005 * 0.1 -> "200.5"."""
    return f"{round(v_file * unit_mm, 9):.12g}"


def _frame(trace: PulseTrace) -> Tuple[int, float, float]:
    """(n_samples, dt_ps, t0_ps) of a trace whose samples sit on a uniform grid.

    dt is (t[-1] - t[0]) / (n - 1) from the endpoints, not a mean of the diffs, and
    every time must lie within UNIFORM_TOL_PS of t0 + n*dt: the FFT assumes the
    samples are evenly spaced, and a dropped or doubled sample would shift every
    later one by a step without changing the count by much."""
    t = trace.t_ps
    n = int(t.size)
    if n < 2:
        raise MeasuredDataError(
            f"{trace.path}: {n} sample(s); a trace needs at least two for a time step")
    t0 = float(t[0])
    dt = (float(t[-1]) - t0) / (n - 1)
    if not dt > 0:
        raise MeasuredDataError(f"{trace.path}: the sample times do not increase")
    dev = np.abs(t - (t0 + np.arange(n) * dt))
    worst = int(np.argmax(dev))
    if not dev[worst] < UNIFORM_TOL_PS:
        raise MeasuredDataError(
            f"{trace.path}: the samples are not evenly spaced: sample {worst} is at "
            f"{t[worst]:.6f} ps, {dev[worst]:.6g} ps from t0 + n*dt with dt = "
            f"{dt:.9g} ps (tolerance {UNIFORM_TOL_PS:g} ps); re-export the trace")
    return n, dt, t0


def _df_hz(n_frame: int, dt_ps: float) -> float:
    """The bin spacing 1 / (N*dt) in Hz, as 1e12 / (N*dt_ps) so that a 200 ps frame
    gives exactly 5e9 and k * df lands on k * 5 GHz with no round-off."""
    return 1e12 / (n_frame * dt_ps)


def _n_bins(n_frame: int, dt_ps: float, fmax_hz: float, where) -> int:
    """K + 1, the bins kept: k = 0 .. K with K = floor(fmax * N*dt + 1e-9). The 1e-9
    keeps a bin that lands on fmax up to round-off (1 THz is 200.00000000000003 or
    199.99999999999997 bins, depending on how N*dt rounds)."""
    if not (math.isfinite(fmax_hz) and fmax_hz >= 0):
        raise MeasuredDataError(f"--fmax {fmax_hz:g} Hz: give a frequency >= 0")
    k_max = int(math.floor(fmax_hz / _df_hz(n_frame, dt_ps) + 1e-9))
    if k_max > n_frame // 2:
        nyquist = (n_frame // 2) * _df_hz(n_frame, dt_ps)
        raise MeasuredDataError(
            f"--fmax {fmax_hz / 1e9:g} GHz is above the {nyquist / 1e9:g} GHz the "
            f"{n_frame}-sample, {dt_ps:g} ps frame of {where} resolves (its "
            "Nyquist frequency); lower --fmax")
    return k_max + 1


def _bins(signal: np.ndarray, t0_ps: float, dt_ps: float, n_bins: int) -> np.ndarray:
    """The cells V(f_k), k < n_bins, of one trace (module docstring), last sample
    dropped.

    f_k * t0 is formed as k * t0 / (N*dt) in ps and reduced mod 1 before the exp:
    the same number as f_k[Hz] * t0[s], but with no 1e-12 conversions in it, so for
    this raster (t0 a multiple of 0.05 ps, N*dt = 200 ps) the correction is exact."""
    n_frame = signal.size - 1
    spec = np.fft.rfft(signal[:n_frame])[:n_bins]
    k = np.arange(n_bins)
    cycles = np.mod(k * t0_ps / (n_frame * dt_ps), 1.0)
    values = np.conj(np.exp(-2j * np.pi * cycles) * spec)
    # f = 0: no correction; the plain sum, real (and a +0, not -0, imaginary part)
    values[0] = complex(spec[0].real, 0.0)
    return values


def pulse_spectrum(trace: PulseTrace,
                   fmax_hz: float = DEFAULT_FMAX_HZ) -> Tuple[np.ndarray, np.ndarray]:
    """(freqs_hz, values) of one trace: bins k = 0 .. K at f_k = k / (N*dt), N the
    sample count less the window's closing endpoint, and V(f_k) as the module
    docstring defines it (complex128, no scaling). The cells parse_folder writes for
    a trace are exactly these values.

    Raises MeasuredDataError on fewer than two samples, uneven spacing, or an fmax
    above the frame's Nyquist frequency."""
    n, dt_ps, t0_ps = _frame(trace)
    n_bins = _n_bins(n - 1, dt_ps, fmax_hz, trace.path)
    freqs = np.arange(n_bins) * _df_hz(n - 1, dt_ps)
    return freqs, _bins(trace.signal_na, t0_ps, dt_ps, n_bins)


def _read_chunk(chunk: Sequence[Tuple[int, int, str]]):
    """Worker: read and transform one chunk of (z_file, x_file, path).

    Returns (rows, problems): rows (z_file, x_file, values) for the traces that fit
    the reference frame, and one message per trace that does not. A bad trace is
    recorded, not raised, so the error the user sees lists every offender at once
    instead of one per run."""
    n_ref, dt_ref, n_bins = worker_shared()
    rows, problems = [], []
    for z, x, path in chunk:
        try:
            trace = read_pulse(Path(path))
            n, dt_ps, t0_ps = _frame(trace)
            if n != n_ref:
                raise MeasuredDataError(f"{path}: {n} samples, not {n_ref}")
            if abs(dt_ps - dt_ref) > DT_RTOL * dt_ref:
                raise MeasuredDataError(
                    f"{path}: a {dt_ps:.12g} ps step, not {dt_ref:.12g} ps")
            # the trace's own dt: the exact DTFT at its own bins, which agree with
            # the header's to DT_RTOL
            values = _bins(trace.signal_na, t0_ps, dt_ps, n_bins)
        except MeasuredDataError as e:
            problems.append(str(e))
            continue
        except (OSError, UnicodeDecodeError) as e:
            problems.append(f"{path}: unreadable ({e})")
            continue
        rows.append((z, x, values))
    return rows, problems


def _check_out(out: Path, force: bool) -> None:
    """Refuse an output we must not or cannot write, before an hour of reading."""
    if out.is_dir():
        raise IsADirectoryError(f"output {out} is a directory; give -o a file path")
    if not out.parent.is_dir():
        raise FileNotFoundError(
            f"the output's directory {out.parent} does not exist; create it or "
            "point -o elsewhere")
    if out.exists() and not force:
        raise FileExistsError(
            f"{out} already exists; pass --force to overwrite it, or -o to write "
            "somewhere else")


def _problem_message(problems: List[str], total: int, folder: Path, ref: Path,
                     n_ref: int, dt_ref: float) -> str:
    problems = sorted(problems)
    shown = "; ".join(problems[:MAX_NAMED])
    more = f"; and {len(problems) - MAX_NAMED} more" if len(problems) > MAX_NAMED else ""
    msg = (f"{len(problems)} of {total} traces in {folder} do not fit one spectra "
           f"table: {shown}{more}. Every trace must be readable and share the frame "
           f"of the first one, {ref.name}: {n_ref} samples {dt_ref:g} ps apart, "
           "evenly spaced. Fix the offending files or move them out of the folder")
    if len(problems) > total / 2:
        msg += (f" (most traces disagree with {ref.name}: if it is the odd one out, "
                "move it instead)")
    return msg


def parse_folder(folder: Path, out: Optional[Path] = None, *,
                 fmax_hz: float = DEFAULT_FMAX_HZ, unit_mm: float = DEFAULT_UNIT_MM,
                 jobs: int = 1, force: bool = False,
                 log: Optional[logging.Logger] = None) -> dict:
    """Write the spectra CSV of every `{idx}-z{z}-x{x}.pulse.csv` in `folder` to
    `out` (default FOLDER/topt_spectra.csv) and return a summary dict (plain values).

    The first trace in (z, x) order fixes the frame: its sample count, step and
    hence the bins and the header. Every other trace must match it (DT_RTOL,
    UNIFORM_TOL_PS); the traces that do not are all named in one MeasuredDataError,
    and nothing is written. The CSV is written to a temporary file next to `out`
    and renamed into place at the end, so an interrupted run leaves no partial CSV
    behind. An existing `out` is a FileExistsError unless `force`.

    `jobs` worker processes read the traces (parallel.map_workers); the cells do
    not depend on it.
    """
    log = log or logging.getLogger()
    started = time.perf_counter()
    folder = Path(folder).expanduser()
    out = folder / DEFAULT_OUT_NAME if out is None else Path(out).expanduser()
    if not (math.isfinite(unit_mm) and unit_mm > 0):
        raise MeasuredDataError(
            f"--unit-mm {unit_mm:g}: give the length of one file-name unit in mm, > 0")
    if not folder.is_dir():
        raise MeasuredDataError(
            f"{folder} is not a directory; give the folder holding the "
            "{idx}-z{z}-x{x}.pulse.csv traces")
    _check_out(out, force)

    index = index_pulse_dir(folder)
    keys = sorted(index.files)
    grid_names = {p.name for p in index.files.values()}
    skipped = sorted(set(os.listdir(folder)) - grid_names)
    log.info(f"{folder}: {len(keys)} pulse traces"
             + (f"; skipped {len(skipped)} other entries ({', '.join(skipped[:8])}"
                f"{', ...' if len(skipped) > 8 else ''})" if skipped else ""))

    ref_path = index.files[keys[0]]
    n_ref, dt_ref, _ = _frame(read_pulse(ref_path))
    n_frame = n_ref - 1
    df = _df_hz(n_frame, dt_ref)
    n_bins = _n_bins(n_frame, dt_ref, fmax_hz, ref_path)
    freqs = np.arange(n_bins) * df
    header = ["X", "Z"] + [freq_header(f) for f in freqs]
    if len(set(header)) != len(header):
        raise MeasuredDataError(
            f"the {df / 1e9:g} GHz bins of a {n_frame * dt_ref:g} ps frame are too "
            "close to tell apart in a 6-significant-digit header; lower --fmax")
    mislabel = max(abs(float(h[:-3]) * 1e9 - f) for h, f in zip(header[2:], freqs))
    if mislabel > FREQ_MATCH_HZ:
        log.warning(f"the header rounds the bin frequencies by up to {mislabel:.3g} "
                    "Hz (6 significant digits): a column holds the bin nearest its "
                    "header, not one exactly at it")
    log.info(f"frame: {n_ref} samples per trace ({ref_path.name}), the closing "
             f"endpoint dropped -> N = {n_frame}, dt = {dt_ref:.9g} ps, "
             f"N*dt = {n_frame * dt_ref:.9g} ps, df = {df / 1e9:.9g} GHz; "
             f"{n_bins} bins, {header[2]} .. {header[-1]}")

    items = [(z, x, str(index.files[(z, x)])) for z, x in keys]
    # more chunks than workers, so the progress line moves (and a serial run shows
    # one too); round-robin, because every trace costs the same
    chunks = round_robin_chunks(items, max(20, 8 * max(1, int(jobs))))
    state = {"traces": 0, "decile": 0}

    def on_done(done, total, result):
        state["traces"] += len(result[0]) + len(result[1])
        decile = 10 * done // total
        if decile > state["decile"]:
            state["decile"] = decile
            log.info(f"  read {state['traces']}/{len(items)} traces "
                     f"({time.perf_counter() - started:.1f} s)")

    results = map_workers(_read_chunk, chunks, jobs=jobs,
                          shared=(n_ref, dt_ref, n_bins), on_done=on_done, log=log,
                          desc=f"Reading {len(items)} traces")
    rows, problems = [], []
    for chunk_rows, chunk_problems in results:
        rows.extend(chunk_rows)
        problems.extend(chunk_problems)
    if problems:
        raise MeasuredDataError(_problem_message(problems, len(items), folder,
                                                 ref_path, n_ref, dt_ref))
    rows.sort(key=lambda r: (r[0], r[1]))

    _write_csv(out, header, rows, unit_mm, force)
    size = out.stat().st_size
    elapsed = time.perf_counter() - started
    log.info(f"wrote {out}: {len(rows)} rows x {len(header)} columns, "
             f"{size / 1e6:.1f} MB, in {elapsed:.1f} s")
    return {
        "folder": str(folder),
        "out": str(out),
        "n_files": len(items),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "n_samples": n_ref,
        "frame_samples": n_frame,
        "dt_s": dt_ref * 1e-12,
        "df_hz": df,
        "n_bins": n_bins,
        "fmax_hz": float(freqs[-1]),
        "unit_mm": unit_mm,
        "bytes": size,
        "elapsed_s": elapsed,
    }


def _write_csv(out: Path, header: List[str], rows, unit_mm: float,
               force: bool) -> None:
    """Write to a hidden temporary file beside `out`, then rename it over `out`.
    The rename is atomic on one filesystem, so readers see the old file or the
    whole new one, and a run killed mid-write leaves only the temporary file, which
    the except clause removes when it can. Opened with "x" rather than mkstemp so
    the CSV gets the umask's permissions, not 0600."""
    tmp = out.with_name(f".{out.name}.{os.getpid()}-{secrets.token_hex(4)}.tmp")
    try:
        with open(tmp, "x", encoding="ascii", newline="\n") as f:
            f.write(",".join(header) + "\n")
            for z, x, values in rows:
                cells = map(_CELL, values.real.tolist(), (values.imag + 0.0).tolist())
                f.write(f"{_mm(x, unit_mm)},{_mm(z, unit_mm)}," + ",".join(cells)
                        + "\n")
            f.flush()
            os.fsync(f.fileno())
        if out.exists() and not force:     # appeared while we were reading
            raise FileExistsError(
                f"{out} appeared while the traces were read; pass --force to "
                "overwrite it, or -o to write somewhere else")
        os.replace(tmp, out)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class SpectraRows:
    """What the grid search needs from a spectra CSV: one measured row.

    all_z_mm   every row's Z, sorted and unique, so a caller can word "no row at
               z" with the rows there are
    x_mm       (n,) ascending X of the rows at the requested Z
    freqs_hz   (m,) the header frequencies selected, in the order requested
    values     (n, m) complex128, values[i, j] = V(freqs_hz[j]) at x_mm[i]
    """
    path: Path
    all_z_mm: np.ndarray
    x_mm: np.ndarray
    freqs_hz: np.ndarray
    values: np.ndarray


def _header_freqs(path: Path, line: str) -> np.ndarray:
    fields = [field.strip() for field in line.rstrip("\r\n").split(",")]
    if fields[:2] != ["X", "Z"]:
        raise MeasuredDataError(
            f"{path}: the header starts {','.join(fields[:3])!r}, not 'X,Z,...'; this "
            "is not a topt-parser spectra CSV (run topt-parser on the pulse folder "
            "to make one)")
    freqs = []
    for col, name in enumerate(fields[2:], start=3):
        m = _HEADER_FREQ.match(name)
        try:
            ghz = float(m.group(1)) if m else math.nan
        except ValueError:
            ghz = math.nan
        if not math.isfinite(ghz):
            raise MeasuredDataError(
                f"{path}: header column {col} is {name!r}, not a frequency like "
                "'300GHz'; this is not a topt-parser spectra CSV")
        freqs.append(ghz * 1e9)
    if not freqs:
        raise MeasuredDataError(f"{path}: the header has no frequency column")
    freqs = np.asarray(freqs)
    order = np.sort(freqs)
    if np.any(np.diff(order) <= FREQ_MATCH_HZ):
        dup = order[1:][np.diff(order) <= FREQ_MATCH_HZ][0]
        raise MeasuredDataError(
            f"{path}: two header columns at {dup / 1e9:g} GHz (within "
            f"{FREQ_MATCH_HZ:g} Hz); a frequency must name one column")
    return freqs


def read_spectra_csv(path: Path, z_mm: float,
                     freqs_hz: Sequence[float]) -> SpectraRows:
    """The row at Z = `z_mm` (within Z_MATCH_MM) of a topt-parser CSV, at
    `freqs_hz` (each must match a header column within FREQ_MATCH_HZ).

    The file is streamed once. Every line's Z is read, so all_z_mm can name the
    rows there are; X and the cells are parsed only on the lines of the requested
    row, and only the requested columns go through complex(), so pulling one
    80-element row out of a 380 MB file costs about one pass over its bytes.
    No row at z_mm gives empty x_mm / values: the caller words that error, with
    all_z_mm.

    Raises MeasuredDataError on a header that is not X,Z,<f>GHz..., a frequency
    with no column (naming the nearest ones), a malformed line, or two rows at the
    same (Z, X).
    """
    path = Path(path).expanduser()
    with open(path, "rb") as f:
        freqs = _header_freqs(path, f.readline().decode("utf-8-sig"))
        cols = []
        for want in freqs_hz:
            gap = np.abs(freqs - float(want))
            j = int(np.argmin(gap))
            if not gap[j] <= FREQ_MATCH_HZ:
                near = sorted(np.argsort(gap)[:2])
                raise MeasuredDataError(
                    f"{path}: no column at {float(want) / 1e9:.12g} GHz (within "
                    f"{FREQ_MATCH_HZ:g} Hz); the nearest are "
                    f"{' and '.join(freq_header(freqs[i]) for i in near)}. Ask for "
                    "one of the CSV's frequencies, or re-run topt-parser with a "
                    "frame whose bins include it")
            cols.append(j + 2)
        n_fields = len(freqs) + 2

        z_of: Dict[bytes, float] = {}
        xs, cells = [], []
        for lineno, line in enumerate(f, start=2):
            i = line.find(b",")
            j = line.find(b",", i + 1) if i >= 0 else -1
            if j < 0:
                if not line.strip():
                    continue
                raise MeasuredDataError(f"{path}:{lineno}: not an X,Z,... row")
            zb = line[i + 1:j]
            z = z_of.get(zb)
            if z is None:
                try:
                    z = z_of[zb] = float(zb)
                except ValueError:
                    raise MeasuredDataError(
                        f"{path}:{lineno}: Z {zb.decode(errors='replace')!r} is not "
                        "a number") from None
            if abs(z - z_mm) >= Z_MATCH_MM:
                continue
            fields = line.rstrip(b"\r\n").split(b",")
            if len(fields) != n_fields:
                raise MeasuredDataError(
                    f"{path}:{lineno}: {len(fields)} fields, the header has {n_fields}")
            try:
                xs.append(float(fields[0]))
                cells.append([complex(fields[c].decode()) for c in cols])
            except ValueError as e:
                raise MeasuredDataError(f"{path}:{lineno}: unreadable cell ({e})") from None

    all_z = np.unique(np.fromiter(z_of.values(), dtype=np.float64, count=len(z_of)))
    x = np.asarray(xs, dtype=np.float64)
    values = np.asarray(cells, dtype=np.complex128).reshape(len(xs), len(cols))
    order = np.argsort(x, kind="stable")
    x, values = x[order], values[order]
    if np.any(np.diff(x) == 0):
        raise MeasuredDataError(
            f"{path}: two rows at Z = {z_mm:g} mm, X = {x[1:][np.diff(x) == 0][0]:g} "
            "mm; re-run topt-parser rather than editing the CSV by hand")
    return SpectraRows(path=path, all_z_mm=all_z, x_mm=x,
                       freqs_hz=np.asarray([freqs[c - 2] for c in cols], dtype=np.float64),
                       values=values)


@dataclass(frozen=True)
class SpectraMap:
    """One frequency column of a spectra CSV, every row, as a (Z, X) map: what
    topt-heatmap draws (rice_bend/topt_heatmap.py).

    freq_hz    the header frequency of the column read (the one within
               FREQ_MATCH_HZ of the request)
    x_mm       (nx,) every X in the file, ascending and unique
    z_mm       (nz,) every Z in the file, ascending and unique
    values     (nz, nx) complex128, values[i, j] = V(freq_hz) at (z_mm[i], x_mm[j]);
               NaN+NaNj where the file has no row at that (Z, X)
    """
    path: Path
    freq_hz: float
    x_mm: np.ndarray
    z_mm: np.ndarray
    values: np.ndarray


def _freq_column(path: Path, freqs: np.ndarray, want_hz: float) -> int:
    """The index into the header's `freqs` of the column at `want_hz`, within
    FREQ_MATCH_HZ, or the MeasuredDataError read_spectra_csv raises for a frequency
    with no column (same lookup, same message; read_spectra_csv inlines it)."""
    gap = np.abs(freqs - float(want_hz))
    j = int(np.argmin(gap))
    if not gap[j] <= FREQ_MATCH_HZ:
        near = sorted(np.argsort(gap)[:2])
        raise MeasuredDataError(
            f"{path}: no column at {float(want_hz) / 1e9:.12g} GHz (within "
            f"{FREQ_MATCH_HZ:g} Hz); the nearest are "
            f"{' and '.join(freq_header(freqs[i]) for i in near)}. Ask for "
            "one of the CSV's frequencies, or re-run topt-parser with a "
            "frame whose bins include it")
    return j


def read_spectra_map(path: Path, freq_hz: float) -> SpectraMap:
    """Every row of a topt-parser CSV at one frequency (a header column within
    FREQ_MATCH_HZ of `freq_hz`), as a (Z, X) map.

    The file is streamed once; each line's X and Z are read and only the one
    column goes through complex(). The rows may come in any order. The grid is
    every distinct Z by every distinct X in the file, so a (Z, X) with no row is
    NaN+NaNj, not an error.

    Raises MeasuredDataError on a header that is not X,Z,<f>GHz..., a frequency
    with no column (naming the nearest ones), a line whose field count is not the
    header's, an unreadable X, Z or cell, no data rows, or two rows at the same
    (Z, X).
    """
    path = Path(path).expanduser()
    with open(path, "rb") as f:
        freqs = _header_freqs(path, f.readline().decode("utf-8-sig"))
        j = _freq_column(path, freqs, freq_hz)
        col = j + 2
        n_fields = len(freqs) + 2
        num: Dict[bytes, float] = {}      # X / Z text -> mm: a few hundred distinct
        xs, zs, cells, linenos = [], [], [], []
        for lineno, line in enumerate(f, start=2):
            n = line.count(b",") + 1
            if n != n_fields:
                if not line.strip():
                    continue
                raise MeasuredDataError(
                    f"{path}:{lineno}: {n} fields, the header has {n_fields}")
            fields = line.split(b",", col + 1)
            pos = []
            for name, text in (("X", fields[0]), ("Z", fields[1])):
                v = num.get(text)
                if v is None:
                    try:
                        v = num[text] = float(text)
                    except ValueError:
                        raise MeasuredDataError(
                            f"{path}:{lineno}: {name} "
                            f"{text.decode(errors='replace')!r} is not a number"
                        ) from None
                pos.append(v)
            try:
                cells.append(complex(fields[col].decode().strip()))
            except (ValueError, UnicodeDecodeError) as e:
                raise MeasuredDataError(f"{path}:{lineno}: unreadable cell ({e})") from None
            xs.append(pos[0])
            zs.append(pos[1])
            linenos.append(lineno)
    if not cells:
        raise MeasuredDataError(
            f"{path} has a header but no data rows; re-run topt-parser on the pulse "
            "folder")

    x = np.asarray(xs, dtype=np.float64)
    z = np.asarray(zs, dtype=np.float64)
    x_mm, ix = np.unique(x, return_inverse=True)
    z_mm, iz = np.unique(z, return_inverse=True)
    flat = iz * x_mm.size + ix
    order = np.argsort(flat, kind="stable")
    same = np.flatnonzero(np.diff(flat[order]) == 0)
    if same.size:
        a, b = order[same[0]], order[same[0] + 1]
        raise MeasuredDataError(
            f"{path}: two rows at Z = {z[a]:g} mm, X = {x[a]:g} mm (lines "
            f"{linenos[a]} and {linenos[b]}); re-run topt-parser rather than editing "
            "the CSV by hand")
    values = np.full((z_mm.size, x_mm.size), complex(math.nan, math.nan),
                     dtype=np.complex128)
    values[iz, ix] = np.asarray(cells, dtype=np.complex128)
    return SpectraMap(path=path, freq_hz=float(freqs[j]), x_mm=x_mm, z_mm=z_mm,
                      values=values)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="topt-parser",
        description="Turn a folder of Toptica THz-TDS {idx}-z{z}-x{x}.pulse.csv "
                    "traces into one CSV: X, Z (mm), then one complex column per "
                    "frequency bin, 0 Hz .. --fmax (see the rice_bend.topt_parser "
                    "docstring for the convention)")
    parser.add_argument("folder", type=Path, metavar="FOLDER",
                        help="Folder holding the pulse traces (other files in it "
                             "are skipped)")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help=f"Output CSV (default: FOLDER/{DEFAULT_OUT_NAME})")
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="Worker processes reading the traces (default: 1); the "
                             "output does not depend on it")
    parser.add_argument("--fmax", type=float, default=DEFAULT_FMAX_HZ, metavar="HZ",
                        help="Highest frequency column, in Hz (default: "
                             f"{DEFAULT_FMAX_HZ:g}); every bin k/(N*dt) up to it "
                             "is written")
    parser.add_argument("--unit-mm", type=float, default=DEFAULT_UNIT_MM, metavar="MM",
                        help="Length of one file-name unit in mm (default: "
                             f"{DEFAULT_UNIT_MM:g}, so z1770 is Z = 177 mm)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the output CSV if it exists")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error(f"--jobs {args.jobs}: give at least 1")
    return args


def main(argv=None):
    args = _parse_args(argv)
    log = setup_logging(args.debug)
    try:
        parse_folder(args.folder, args.out, fmax_hz=args.fmax, unit_mm=args.unit_mm,
                     jobs=args.jobs, force=args.force, log=log)
    except (MeasuredDataError, OSError) as e:
        log.error(str(e))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
