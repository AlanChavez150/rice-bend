"""The interpolation conventions this project uses, named for what they do —
because they are not interchangeable and the choice is physics.

`interp_real_imag` interpolates the complex values themselves, and it exists
because the obvious alternative is a trap. Interpolating |f| and angle(f)
separately is right for a slowly varying phase and wrong across the ±π branch cut:
a target sample landing inside a source interval that wraps gets a phase averaged
*through* the cut, so it comes out wrong by up to π — at full amplitude, where
nothing downstream flags it. No polar interpolation of a field remains anywhere.
The one construction that does work in amplitude and phase, the caustic plate
(beam_aperatures.caustic_profile), resamples its UNWRAPPED phase, which has no cut
to cross, and only then takes exp(1j * phase).

`interp_amplitude` interpolates |f| alone and returns a real array, for the sites
that want a magnitude rather than a field.

Naming them is what makes the convention a one-line, reviewable decision at each
call site instead of nine lines of interp1d boilerplate. The names are
load-bearing in the other direction too: both take (src_x, src_f, target_x), so
swapping one for the other is a one-keystroke way to silently change the physics.
|interp_real_imag(...)| is not interp_amplitude(...): the straight line between two
samples of different phase cuts inside the arc, so it dips wherever the phase turns
between source samples.

Which to use, as the code now stands:
  interp_real_imag  — every FIELD: the RX measurement, and the converged aperture on
                      both the single-shot and grid paths
  interp_amplitude  — the solver's fixed amplitude constraint and support mask, and
                      the scene plot's amplitude panel. Magnitudes, not fields

Deliberately NOT absorbed here: the two np.interp sites that resample a complex
field (candidate_scenes' `_reilluminate` and animate's
`_aligned_reference_phase`). np.interp and scipy.interp1d disagree at ULP level —
up to 1.3e-7 on complex64 input, because one of them keeps float32 through the
arithmetic — so folding them in would move numbers. Both call sites carry a
comment saying so.
"""

import numpy as np
import scipy.interpolate


def interp_amplitude(src_x: np.ndarray, src_f: np.ndarray, target_x: np.ndarray, *,
                     assume_sorted: bool = True) -> np.ndarray:
    """Interpolate |src_f| onto `target_x`, discarding the phase. Real-valued, so no
    branch cut can arise.

    For the places that genuinely want a magnitude -- the solver's fixed amplitude
    constraint and its support mask, and the amplitude panel of the scene plot.
    Spelled np.abs(interp_real_imag(...)) those sites would read as a field
    interpolation whose phase was thrown away, and would be a different number: the
    complex interpolant dips wherever the phase turns between source samples.
    """
    return scipy.interpolate.interp1d(
        src_x, np.abs(src_f), kind="linear", fill_value=0,
        bounds_error=False, assume_sorted=assume_sorted)(target_x)


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
