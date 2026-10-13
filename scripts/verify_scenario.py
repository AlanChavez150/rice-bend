#!/usr/bin/env python
"""Verify a test-scenario config: propagate the real beam to the RX plane (z=0) and
measure where/how it lands, classifying HIT vs MISS and the beam size vs the RX window.

Forward only: no Gerchberg-Saxton, and no receiver noise even when the config's noise
block asks for some (this describes the beam, not a measurement of it). So it is fast
and works even for "miss" scenarios where the RX sees ~no energy. Only the RX-plane row
is propagated, not every scene plane: it is the row MGS.measure() synthesizes the
measurement from, and the bottom row of the illuminated scene when the scene floor is
the RX plane (z_min = 0, every shipped config).

The frequency is --freq, else the centre-by-value of the config's `frequencies`, else
rice_bend.config.DEFAULT_FREQ_HZ; the one used is printed.

Usage:
    python scripts/verify_scenario.py CONFIG.yml [--freq HZ] [--spacing S] \
        [--expect-hit | --expect-miss] [--expect-size larger|smaller|na]

Prints metrics and, when --expect-* is given, a final PASS/FAIL line. The numbers come
from rice_bend.landing.landing_metrics, the ones `mgs` logs in its LANDING block:
  beam_FWHM          the OUTERMOST extent where |field| >= half the peak. A true FWHM
                     for a single-lobed beam; for a bessel beam, the whole fringe zone
  energy_frac_in_rx  sum(|field|^2 inside the RX window) / sum(|field|^2 on the scene
                     x-axis). A plain sum, not a trapezoid-rule integral, so that it
                     is the number analysis.json records as fraction_in_window
  target_x           the beam's geometric aim on the RX plane (landing.beam_target).
                     The measured positions carry the propagator's lateral offset
                     (landing.propagator_offset: half a scene step toward +x at the
                     shipped scene spacings), which the target does not
"""
import argparse
from pathlib import Path

import numpy as np

from rice_bend.config import center_freq_index, load_config, resolve_frequencies
from rice_bend.landing import beam_target, landing_metrics
from rice_bend.mgs import MGS


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--freq", type=float, default=None,
                   help="Frequency in Hz (default: the centre-by-value of the config's "
                        "`frequencies`, else the project default carrier)")
    p.add_argument("--spacing", type=float, default=None,
                   help="Override sim_scene.spacing (coarser = faster design iteration)")
    p.add_argument("--expect-hit", action="store_true")
    p.add_argument("--expect-miss", action="store_true")
    p.add_argument("--expect-size", choices=["larger", "smaller", "na"], default="na")
    p.add_argument("--hit-thresh", type=float, default=0.5,
                   help="rx_peak/global_peak above this = HIT")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.spacing is not None:
        cfg.sim_scene.spacing = args.spacing
    # forward-only: the clean beam, whatever receiver noise the scenario configures
    cfg.noise.snr_db = None

    if args.freq is not None:
        freq, freq_source = args.freq, "--freq"
    else:
        freqs = resolve_frequencies(cfg, None)
        freq = freqs[center_freq_index(freqs)]
        freq_source = ("centre of config frequencies" if cfg.frequencies
                       else "project default")

    mgs = MGS(freq, cfg)
    mgs.measure()                             # propagates to the RX plane only
    x = mgs.scene.x_axis
    row = mgs.freq_states[0].rx_plane_row     # complex field on the RX plane (z = 0)

    # RX window geometry + how much of the beam lands inside it
    rxc, rxw = cfg.rx_aperture.x_center, cfg.rx_aperture.width
    rxlo, rxhi = rxc - rxw / 2.0, rxc + rxw / 2.0
    m = landing_metrics(x, row, rxlo, rxhi)
    if m is None:
        print("global_peak_amp=0  (no field reached the RX plane)")
        return 1
    gmax = float(np.abs(row).max())
    xpeak = m["peak_x_m"]
    width = m["half_amp_extent_m"]
    peak_ratio = m["rx_peak_over_global_peak"]
    frac = m["frac_in_window"]
    target = beam_target(cfg.tx_aperture, rx_z=mgs.scene.rx_ap.z)

    hit = peak_ratio >= args.hit_thresh
    size = "LARGER" if width > rxw else "SMALLER"

    print(f"config: {args.config}")
    print(f"freq: {freq / 1e9:g} GHz ({freq_source})")
    print(f"scene: x[{x.min():.3f},{x.max():.3f}] z[{mgs.scene.z_min:.3f},{mgs.scene.z_max:.3f}] "
          f"spacing={cfg.sim_scene.spacing:g}  tx_z={cfg.tx_aperture.z:.3f} "
          f"tx_w={cfg.tx_aperture.x_max - cfg.tx_aperture.x_min:.4f}")
    print(f"rx_window: center={rxc:+.4f} width={rxw:.4f} -> [{rxlo:+.4f},{rxhi:+.4f}]")
    print(f"beam_peak_x={xpeak:+.4f}  global_peak_amp={gmax:.4g}")
    print(f"beam_centroid_x={m['centroid_x_m']:+.4f}  "
          f"energy_5_95=[{m['e05_x_m']:+.4f},{m['e95_x_m']:+.4f}]")
    print(f"beam_FWHM={width:.4f}  -> beam_is_{size}_than_rx")
    print(f"rx_peak/global_peak={peak_ratio:.3f}  energy_frac_in_rx={frac:.3f}")
    if target is None:
        print("target_x=n/a")
    else:
        print(f"target_x={target['x_m']:+.4f}  ({target['kind']}: {target['note']})")
    print(f"classification: {'HIT' if hit else 'MISS'}")

    if args.expect_hit or args.expect_miss or args.expect_size != "na":
        ok = True
        reasons = []
        if args.expect_hit and not hit:
            ok = False; reasons.append("expected HIT, got MISS")
        if args.expect_miss and hit:
            ok = False; reasons.append("expected MISS, got HIT")
        if args.expect_size == "larger" and width <= rxw:
            ok = False; reasons.append(f"expected LARGER (FWHM {width:.4f} <= rx {rxw:.4f})")
        if args.expect_size == "smaller" and width >= rxw:
            ok = False; reasons.append(f"expected SMALLER (FWHM {width:.4f} >= rx {rxw:.4f})")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}" + ("" if ok else "  (" + "; ".join(reasons) + ")"))
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
