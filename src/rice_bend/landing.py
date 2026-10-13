"""Where a beam lands on the RX plane, as numbers: the peak, the centroid, the
span that holds the energy, how much of it the RX window catches -- and how
closely one RX-plane field matches another inside that window.

Pure measurement on one complex RX-plane row sampled on the scene x-axis. No I/O,
no plotting, no MGS: MGS.landing_report() feeds these the true and the
reconstructed rows, and scripts/verify_scenario.py feeds them the true row alone.

Every position is read off the scene grid, so it is quantised to the scene x step
and carries the propagator's lateral offset (propagator_offset below): half a scene
step toward +x on every shipped scene. That offset is common to everything
propagated the same way (the true field, the reconstruction and every candidate),
so differences between rows are unaffected; only a comparison against the
geometric beam_target() sees it.
"""

from typing import Optional

import numpy as np


def propagator_offset(x_axis) -> float:
    """How far rs.rs_apply displaces a field along x, in metres (+ is toward +x).

    The RS kernel is sampled on the scene x-axis itself and applied with
    fftconvolve(mode="same"), which centres the result on kernel sample (N-1)//2.
    The kernel's own centre is x = 0, so a propagated field comes out displaced by
    minus that sample's x. Every shipped scene axis is even-length and symmetric
    about x = 0: it has no sample AT 0, the centring sample sits at -dx/2, and the
    offset is +dx/2. An odd-length symmetric axis has a sample at 0 and no offset.
    An axis not centred on x = 0 displaces the field by minus its centre, which is
    not a bias to correct for but a scene the propagator does not support.
    """
    x = np.asarray(x_axis, dtype=float)
    # 0.0 - x, not -x: an odd-length symmetric axis must report 0.0, never -0.0
    return 0.0 - float(x[(len(x) - 1) // 2])


def landing_metrics(x_axis, row, rx_x_min: float, rx_x_max: float) -> Optional[dict]:
    """Landing numbers for one complex RX-plane `row` on `x_axis`. Positions in
    metres, every value a plain float (the two in-window positions can be None,
    see the last paragraph). Returns None when the row carries no power.

    The window is CLOSED, (x >= rx_x_min) & (x <= rx_x_max): the same mask as
    analysis._energy_dict and the exact complement of MGS.measure()'s weighting
    zero-mask, so `frac_in_window` is the number analysis.json records -- built
    from the same operations in the same order, hence equal to the last bit.

      peak_x_m                  x of max |row|
      peak_x_in_window_m        x of max |row| among the window samples
      centroid_x_m              sum(x*p) / sum(p), p = |row|^2
      centroid_x_in_window_m    the same over the window samples alone
      e05_x_m, e95_x_m          first x where the running sum of p reaches 5 % /
                                95 % of the total: the span holding the middle 90 %
      frac_in_window            p[win].sum() / p.sum()
      half_amp_extent_m         the OUTERMOST extent where |row| >= 0.5 * max. Not a
                                FWHM unless the beam has a single lobe: for a bessel
                                beam it is the width of the whole fringe zone
      rx_peak_over_global_peak  max |row| in the window over max |row|

    The peak is one sample and jumps between near-equal lobes (a bessel beam's
    fringes); the centroid and the 5-95 % span are sums and do not. The two
    in-window positions are None when the window holds no power (or no scene
    sample at all): there is no landing point to report inside it.
    """
    x = np.asarray(x_axis, dtype=float)
    amp = np.abs(np.asarray(row))
    # the row's own dtype (float32 for a propagated complex64 field), exactly as
    # analysis._energy_dict forms it: frac_in_window must reproduce that number
    p = amp ** 2
    tot = float(p.sum())
    if not np.isfinite(tot) or tot <= 0:
        return None
    win = (x >= rx_x_min) & (x <= rx_x_max)
    frac = float(p[win].sum() / tot)

    # positions are accumulated in float64: a float32 running sum over thousands
    # of samples loses the digits a 5 % threshold is read at
    p64 = p.astype(np.float64)
    tot64 = p64.sum()
    cum = np.cumsum(p64)
    last = len(x) - 1
    i05 = min(int(np.searchsorted(cum, 0.05 * tot64)), last)
    i95 = min(int(np.searchsorted(cum, 0.95 * tot64)), last)

    gmax = float(amp.max())
    above = amp >= 0.5 * gmax
    out = {
        "peak_x_m": float(x[np.argmax(amp)]),
        "peak_x_in_window_m": None,
        "centroid_x_m": float((x * p64).sum() / tot64),
        "centroid_x_in_window_m": None,
        "e05_x_m": float(x[i05]),
        "e95_x_m": float(x[i95]),
        "frac_in_window": frac,
        "half_amp_extent_m": float(x[above].max() - x[above].min()),
        "rx_peak_over_global_peak": 0.0,
    }
    if win.any():
        x_win, amp_win, p_win = x[win], amp[win], p64[win]
        out["rx_peak_over_global_peak"] = float(amp_win.max()) / gmax
        tot_win = p_win.sum()
        if tot_win > 0:
            out["peak_x_in_window_m"] = float(x_win[np.argmax(amp_win)])
            out["centroid_x_in_window_m"] = float((x_win * p_win).sum() / tot_win)
    return out


def beam_target(tx_cfg, rx_z: float = 0.0) -> Optional[dict]:
    """The GEOMETRIC landing point of the configured beam on the RX plane:
    {"x_m", "kind", "note"}, from the config alone (a config.TxApertureConfig), with
    d = tx_cfg.z - rx_z the drop from the plate to the RX plane. None for a beam
    type this module does not know.

    Ray optics, so it is where the beam is AIMED, not where a finite-wavelength
    field peaks -- the note says how the two differ for each beam:

      caustic      a*d^2 + b*d + c, the designed trajectory at the RX plane. Every
                   ray is a tangent to that curve and so stays on one side of it
                   (the -x side for a > 0, the plate's side); the main lobe sits on
                   that lit side, short of the caustic by an offset that shrinks
                   with wavelength
      bessel,      the axis from the aperture centre:
      directional  x_center - d * tan(steer)
      focus        the chief ray (aperture centre -> focal point) where it crosses
                   the RX plane, x_center - d * tan(steer). That is the focal point
                   itself only when the focal point lies ON the RX plane, so the
                   note carries the focal point
                   (x_center - F * sin(steer), tx.z - F * cos(steer))

    A positive steer_angle_deg steers toward -x (beam_aperatures.steered_profile's
    sign, shared by the bessel and focus plates); a null steer is 0.
    """
    beam = tx_cfg.beam
    d = float(tx_cfg.z) - float(rx_z)
    if beam.type == "caustic":
        a, b, c = (float(v) for v in beam.trajectory)
        side = " (-x)" if a > 0 else " (+x)" if a < 0 else ""
        return {
            "x_m": a * d**2 + b * d + c,
            "kind": "caustic",
            "note": (f"geometric caustic a*d^2+b*d+c at d={d * 1e3:.2f} mm; the main "
                     f"lobe sits on its lit side{side}, by an offset that shrinks "
                     "with wavelength"),
        }
    if beam.type not in ("bessel", "directional", "focus"):
        return None
    # beam_profile lays these plates out about the aperture centre (the steered
    # plane wave has no centre of its own; its axis is taken from the same point)
    x_center = 0.5 * (float(tx_cfg.x_min) + float(tx_cfg.x_max))
    steer_deg = float(beam.steer_angle_deg or 0.0)
    steer = np.deg2rad(steer_deg)
    x = float(x_center - d * np.tan(steer))
    origin = f"aperture centre x={x_center * 1e3:+.2f} mm, steer {steer_deg:+g} deg"
    if beam.type == "focus":
        focal = float(beam.focal_length)
        x_focal = x_center - focal * np.sin(steer)
        z_focal = float(tx_cfg.z) - focal * np.cos(steer)
        note = (f"chief ray from the {origin}; focal point at "
                f"x={x_focal * 1e3:+.2f} mm, z={z_focal * 1e3:+.2f} mm")
    elif beam.type == "bessel":
        note = f"axis from the {origin}; the fringes straddle it"
    else:
        note = f"axis from the {origin}"
    return {"x_m": x, "kind": beam.type, "note": note}


def field_match(ref_row, row, x_axis, rx_x_min: float, rx_x_max: float) -> dict:
    """How closely `row` matches `ref_row` INSIDE the closed RX window (the only
    region the solver is held to; outside it a reconstruction is unconstrained).

      corr         |<ref, row>| / (||ref|| * ||row||): shape agreement, blind to a
                   global phase and to scale
      rel_l2       ||row - ref|| / ||ref||, with NO phase alignment and no
                   rescaling -- the solver fits the complex field, global phase and
                   amplitude included, so both count as error here
      power_ratio  ||row||^2 / ||ref||^2

    A value is None where its denominator is zero (an empty window, a dark row).
    """
    x = np.asarray(x_axis, dtype=float)
    win = (x >= rx_x_min) & (x <= rx_x_max)
    ref = np.asarray(ref_row, dtype=np.complex128)[win]
    u = np.asarray(row, dtype=np.complex128)[win]
    n_ref = float(np.linalg.norm(ref))
    n_u = float(np.linalg.norm(u))
    out = {"corr": None, "rel_l2": None, "power_ratio": None}
    if n_ref > 0:
        out["rel_l2"] = float(np.linalg.norm(u - ref)) / n_ref
        out["power_ratio"] = (n_u / n_ref) ** 2
        if n_u > 0:
            out["corr"] = float(np.abs(np.vdot(ref, u))) / (n_ref * n_u)
    return out
