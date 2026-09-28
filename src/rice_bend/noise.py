"""Receiver noise for the SIMULATED RX measurement: complex AWGN on the RX elements.

Every result before this module was at infinite SNR. The model is deliberately the
simplest one that makes SNR a single, comparable number:

    P_pk    = max over tones f and RX elements n of |s_f[n]|^2      (clean field)
    sigma^2 = P_pk / 10^(snr_db / 10)                               (same for all f, n)
    n_f[n]  ~ CN(0, sigma^2)   real, imag each N(0, sigma^2 / 2), independent per (f, n)
    y_f[n]  = s_f[n] + n_f[n]

ONE floor for every antenna and every tone: a receiver's noise does not know which
tone it is on. The reference is the PEAK over the RX elements only (not the scene,
not the TX), so `snr_db` is what the brightest antenna sees; the mean per-antenna
SNR is lower by the beam's peak-to-average ratio, and MGS.measure() logs and records
it rather than compensating.

Noise lands on the RX ELEMENT axis, before the interpolation onto the solver grid,
because that is where a real receiver adds it — the interpolation then averages it
exactly as it would average a real capture.

Each tone's stream is keyed by its frequency VALUE, not its position in the list:
candidate_scenes rotates the frequency list for --scene-freq, and a replay must give
every tone the same draw; it also means a tone gets the same normalized draw whatever
other tones share the comb (common random numbers across runs with one seed).

numpy only, pure functions: nothing here touches MGS or the config.
"""

from typing import List, Sequence, Tuple

import numpy as np


def freq_key(freq_hz: float) -> int:
    """The integer key a tone's noise stream is spawned from: its frequency in Hz,
    rounded, so 150e9 and 150000000000.0000001 name the same stream."""
    return int(round(freq_hz))


def freq_rng(seed: int, freq_hz: float) -> np.random.Generator:
    """The independent noise stream for one tone. The (seed, frequency) pair alone
    fixes it — never the tone's list position — see the module docstring."""
    return np.random.default_rng(
        np.random.SeedSequence(int(seed), spawn_key=(freq_key(freq_hz),)))


def add_awgn(clean_profiles: Sequence[np.ndarray], freqs: Sequence[float],
             snr_db: float, seed: int) -> Tuple[List[np.ndarray], float, float]:
    """Add peak-referenced complex AWGN to each tone's RX element profile.

    `clean_profiles[i]` is the complex field on the RX element axis at `freqs[i]`.
    Returns (noisy_profiles, peak_power, sigma2) with noisy_profiles in input order.
    The inputs are not modified.

    Raises ValueError when the peak power is zero or non-finite (no signal to
    reference the SNR to) and when two tones share a freq_key (they would share a
    stream, i.e. perfectly correlated noise).
    """
    if len(clean_profiles) != len(freqs):
        raise ValueError(f"{len(clean_profiles)} profiles for {len(freqs)} frequencies")
    keys = [freq_key(f) for f in freqs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"frequencies must round to distinct Hz values for independent "
                         f"noise streams (got {list(freqs)})")

    clean = [np.asarray(p, dtype=np.complex128) for p in clean_profiles]
    peak_power = float(max(np.max(np.abs(p) ** 2) for p in clean))
    if not np.isfinite(peak_power) or peak_power <= 0.0:
        raise ValueError(f"cannot reference an SNR to peak RX power {peak_power}")
    sigma2 = peak_power / 10 ** (snr_db / 10)

    noisy = []
    for p, f in zip(clean, freqs):
        # one (n, 2) draw per tone: column 0 real, column 1 imag. The shape is part
        # of the reproducibility contract — change it and every recorded seed
        # replays a different realization.
        z = freq_rng(seed, f).standard_normal((len(p), 2))
        noise = np.sqrt(sigma2 / 2) * (z[:, 0] + 1j * z[:, 1])
        noisy.append(p + noise)
    return noisy, peak_power, sigma2
