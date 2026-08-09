"""The two complex-interpolation conventions this project uses, named for what
they do — because they are not interchangeable and the choice is physics.

`interp_amp_phase` interpolates |f| and angle(f) separately. That is right for a
slowly varying phase and wrong across the ±π branch cut: a target sample landing
inside a source interval that wraps gets a phase averaged *through* the cut, so it
comes out wrong by up to π — at full amplitude, where nothing downstream flags it.
`interp_real_imag` interpolates the complex values themselves and has no branch cut.

Measured on the real caustic TX aperture of scenario_caustic_hit (528 samples,
16 phase wraps over 0.132 m) resampled to 400 points: the two conventions differ by
24% relative L2; the polar one round-trips at 27.6% error against the cartesian
one's 16.7%, while inventing 3.7% extra |aperture| energy.

Naming them is what makes the convention a one-line, reviewable decision at each
call site instead of nine lines of interp1d boilerplate. The names are
load-bearing in the other direction too: these are same-signature siblings, so
swapping one for the other is a one-keystroke way to silently change the physics.

Deliberately NOT absorbed here: the two np.interp sites (grid_search's
`_reilluminate` and animate's `_aligned_reference_phase`). np.interp and
scipy.interp1d disagree at ULP level — up to 1.3e-7 on complex64 input, because
one of them keeps float32 through the arithmetic — so folding them in would move
numbers. Both call sites carry a comment saying so.
"""

import numpy as np
import scipy.interpolate


def interp_amp_phase(src_x: np.ndarray, src_f: np.ndarray, target_x: np.ndarray, *,
                     assume_sorted: bool = True) -> np.ndarray:
    """Interpolate complex `src_f` onto `target_x`, amplitude and phase separately.

    Zero-filled outside [src_x[0], src_x[-1]]. Use where the quantity really is an
    amplitude and a phase — a fixed aperture magnitude, or a profile whose phase is
    smooth on the source grid. Across wrapped phase, prefer interp_real_imag.

    `assume_sorted=False` is needed for the caustic construction, whose x axis comes
    out of an argsort-then-flip and is descending.
    """
    amp = scipy.interpolate.interp1d(
        src_x, np.abs(src_f), kind="linear", fill_value=0,
        bounds_error=False, assume_sorted=assume_sorted)(target_x)
    phs = scipy.interpolate.interp1d(
        src_x, np.angle(src_f), kind="linear", fill_value=0,
        bounds_error=False, assume_sorted=assume_sorted)(target_x)
    return amp * np.exp(1j * phs)


def interp_real_imag(src_x: np.ndarray, src_f: np.ndarray,
                     target_x: np.ndarray) -> np.ndarray:
    """Interpolate complex `src_f` onto `target_x` as a complex array (no branch cut).

    Zero-filled outside [src_x[0], src_x[-1]] — deliberately, not clamped to the edge
    value. Clamping would smear a spurious ~unit-amplitude source across the whole
    scene, radiating energy from outside the aperture.
    """
    return scipy.interpolate.interp1d(
        src_x, src_f, kind="linear", fill_value=complex(0, 0),
        bounds_error=False, assume_sorted=True)(target_x)
