"""Readers for a measured THz-TDS raster: one pulse trace per (z, x) stage position,
stored as `{idx}-z{z}-x{x}.pulse.csv` files in one directory.

topt-parser (topt_parser.py) is their one user: it indexes a folder, reads every
trace and writes the spectra CSV the grid search reads (measured.py never opens a
pulse file). Kept out of exp_data.py, which reads the old 150 GHz oscilloscope
.mat captures. Nothing here knows the config or the simulation frame: file names
stay in FILE units (integers, whatever length one unit is), times in ps, signals
in nA. numpy and the standard library only.

The directory holds tens of thousands of traces (6.2 GB for the first raster), so
it is LISTED once (index_pulse_dir), and each trace is READ on its own (read_pulse).
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# `{idx}-z{z}-x{x}.pulse.csv`: idx is the acquisition counter (ignored), z and x the
# stage position in file units. Anything else in the directory (the probe-*, asdf*
# and af captures of the first raster) is not part of the grid.
PULSE_NAME = re.compile(r"^(\d+)-z(-?\d+)-x(-?\d+)\.pulse\.csv$")
# the first line of every trace, compared field by field with the whitespace
# around each field stripped
PULSE_HEADER = ("Time_abs/ps", "Signal 1/nA", "Signal 2/nA")


class MeasuredDataError(ValueError):
    """The measured data cannot be used as asked: a missing or malformed file, a
    position the data does not have, or a frequency it has no column for. A
    ValueError, so callers that already report config errors report these too."""


@dataclass(frozen=True)
class PulseIndex:
    """Every grid trace in a directory, keyed by its (z, x) stage position in file
    units. Built from the file names alone: no trace is opened."""
    data_dir: Path
    files: Dict[Tuple[int, int], Path]       # (z_file, x_file) -> path

    def rows(self) -> List[int]:
        """The measured rows: the distinct z positions, ascending (file units)."""
        return sorted({z for z, _ in self.files})

    def row_xs(self, z_file: int) -> List[int]:
        """The x positions measured on row `z_file`, ascending (file units). Empty
        when the raster has no such row."""
        return sorted(x for z, x in self.files if z == z_file)


def index_pulse_dir(data_dir: Path) -> PulseIndex:
    """Index the grid traces in `data_dir` from their file names.

    The directory is listed once and nothing is opened. Files that do not match
    PULSE_NAME are skipped (counted in a debug line); two files at the same (z, x)
    are an error, because which one is "the" measurement is not ours to guess.

    Raises MeasuredDataError when the directory is missing or holds no grid trace.
    """
    data_dir = Path(data_dir).expanduser()
    if not data_dir.is_dir():
        raise MeasuredDataError(
            f"{data_dir} is not a directory; give the folder holding the "
            "{idx}-z{z}-x{x}.pulse.csv traces")
    files: Dict[Tuple[int, int], Path] = {}
    skipped = []
    for name in os.listdir(data_dir):
        m = PULSE_NAME.match(name)
        if m is None:
            skipped.append(name)
            continue
        key = (int(m.group(2)), int(m.group(3)))
        if key in files:
            raise MeasuredDataError(
                f"two traces at the same stage position z={key[0]}, x={key[1]} "
                f"(file units) in {data_dir}: {files[key].name} and {name}; move "
                "the one that is not the measurement out of the directory")
        files[key] = data_dir / name
    if not files:
        raise MeasuredDataError(
            f"{data_dir} holds no trace named {{idx}}-z{{z}}-x{{x}}.pulse.csv "
            f"({len(skipped)} other entries); give the raster's folder")
    if skipped:
        logging.getLogger().debug(
            f"{data_dir}: indexed {len(files)} grid traces, skipped {len(skipped)} "
            f"non-grid entries ({', '.join(sorted(skipped)[:6])}"
            f"{', ...' if len(skipped) > 6 else ''})")
    return PulseIndex(data_dir=data_dir, files=files)


@dataclass(frozen=True)
class PulseTrace:
    """One measured pulse: the absolute sample times and Signal 1.

    The time axis is ABSOLUTE: the window start moves from row to row, and the
    phase a trace carries is only comparable to another's because both are
    referenced to the same t = 0. Signal 2 is noise only and is not kept."""
    path: Path
    t_ps: np.ndarray        # (T,) float64, absolute
    signal_na: np.ndarray   # (T,) float64, Signal 1


def read_pulse(path: Path) -> PulseTrace:
    """Read one trace. The header must be PULSE_HEADER (fields compared with the
    surrounding whitespace stripped): a file whose columns are something else would
    otherwise be read as times and signals without complaint.

    Text mode, so the traces' CRLF line ends and trailing blank line are handled
    by the reader. Raises MeasuredDataError on a wrong header, an unparsable row,
    fewer than two samples, a non-finite value or times that do not increase.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig") as f:
        header = tuple(field.strip() for field in f.readline().split(","))
        if header != PULSE_HEADER:
            raise MeasuredDataError(
                f"{path}: header {', '.join(header)!r} is not the THz-TDS pulse "
                f"header {', '.join(PULSE_HEADER)!r}; this reader takes only "
                "Time_abs/ps, Signal 1/nA, Signal 2/nA traces")
        try:
            data = np.loadtxt(f, delimiter=",", usecols=(0, 1), ndmin=2,
                              dtype=np.float64)
        except ValueError as e:
            raise MeasuredDataError(f"{path}: unreadable sample row ({e})") from e
    if data.shape[0] < 2:
        raise MeasuredDataError(
            f"{path}: {data.shape[0]} sample(s); a trace needs at least two")
    t_ps = np.ascontiguousarray(data[:, 0])
    signal = np.ascontiguousarray(data[:, 1])
    if not (np.all(np.isfinite(t_ps)) and np.all(np.isfinite(signal))):
        raise MeasuredDataError(f"{path}: non-finite time or Signal 1 value")
    if not np.all(np.diff(t_ps) > 0):
        raise MeasuredDataError(f"{path}: the sample times do not strictly increase")
    return PulseTrace(path=path, t_ps=t_ps, signal_na=signal)
