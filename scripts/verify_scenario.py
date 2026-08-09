#!/usr/bin/env python
"""Verify a test-scenario config: simulate the scene and measure where/how the beam
lands on the RX plane (z=0), classifying HIT vs MISS and the beam size vs the RX window.

Only the forward scene simulation is run (no Gerchberg-Saxton), so it is fast and works
even for "miss" scenarios where the RX sees ~no energy.

Usage:
    python scripts/verify_scenario.py CONFIG.yml [--spacing S] \
        [--expect-hit | --expect-miss] [--expect-size larger|smaller|na]

Prints metrics and, when --expect-* is given, a final PASS/FAIL line.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from rice_bend.config import load_config
from rice_bend.mgs import MGS


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--freq", type=float, default=150e9)
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

    mgs = MGS(args.freq, cfg)
    mgs.illuminate_real()

    x = mgs.scene.x_axis
    amp0 = np.abs(mgs.scene.data[0])          # |field| on the RX plane (z = z_min = 0)
    gmax = float(amp0.max())
    if gmax <= 0:
        print("global_peak_amp=0  (no field reached the RX plane)")
        return 1
    xpeak = float(x[np.argmax(amp0)])

    # beam FWHM at the RX plane
    above = amp0 >= 0.5 * gmax
    width = float(x[above].max() - x[above].min()) if above.any() else 0.0

    # RX window geometry + how much of the beam lands inside it
    rxc, rxw = cfg.rx_aperture.x_center, cfg.rx_aperture.width
    rxlo, rxhi = rxc - rxw / 2.0, rxc + rxw / 2.0
    inrx = (x >= rxlo) & (x <= rxhi)
    rx_peak = float(amp0[inrx].max()) if inrx.any() else 0.0
    etot = float(trapezoid(amp0 ** 2, x))
    erx = float(trapezoid((amp0 ** 2)[inrx], x[inrx])) if inrx.any() else 0.0
    frac = erx / etot if etot > 0 else 0.0

    hit = (rx_peak / gmax) >= args.hit_thresh
    size = "LARGER" if width > rxw else "SMALLER"

    print(f"config: {args.config}")
    print(f"scene: x[{x.min():.3f},{x.max():.3f}] z[{mgs.scene.z_min:.3f},{mgs.scene.z_max:.3f}] "
          f"spacing={cfg.sim_scene.spacing:g}  tx_z={cfg.tx_aperture.z:.3f} "
          f"tx_w={cfg.tx_aperture.x_max - cfg.tx_aperture.x_min:.4f}")
    print(f"rx_window: center={rxc:+.4f} width={rxw:.4f} -> [{rxlo:+.4f},{rxhi:+.4f}]")
    print(f"beam_peak_x={xpeak:+.4f}  global_peak_amp={gmax:.4g}")
    print(f"beam_FWHM={width:.4f}  -> beam_is_{size}_than_rx")
    print(f"rx_peak/global_peak={rx_peak / gmax:.3f}  energy_frac_in_rx={frac:.3f}")
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
