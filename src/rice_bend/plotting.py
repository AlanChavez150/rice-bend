"""The figure panels both entry points draw.

`mgs.plot_scene`'s two scene panels (83 lines) and grid_search's `_draw_scene`
(15 lines) render the same picture: a |field| heatmap with the RX aperture marked in
red and the TX aperture in blue. Two independent bodies of code for one picture is
how they came to disagree about where the receiver sits.
"""

from typing import Optional, Sequence, Tuple

import numpy as np


def draw_scene(fig, ax, field: np.ndarray, *,
               bounds: Tuple[float, float, float, float],
               rx_axis: np.ndarray, rx_z: float,
               tx_axis: np.ndarray, tx_z: float,
               title: str, vmax: Optional[float] = None,
               colorbar_label: str = "|field| (V/m)",
               legend_loc: str = "best") -> None:
    """imshow a |field| over the scene, marking the RX aperture (red) and TX (blue).

    `rx_z` is a parameter and not derived from bounds[2] on purpose: those coincide
    for every simulated config but not for the experimental path, where
    rx.z = 0.35 - zvec[0]*1e-3. Assuming the scene floor would silently relocate the
    markers with no error raised.

    `vmax=None` autoscales this panel independently. Pass a shared value to make two
    panels directly comparable.
    """
    x_min, x_max, z_min, z_max = bounds
    im = ax.imshow(field, extent=[x_min, x_max, z_min, z_max], origin="lower",
                   aspect="auto", cmap="inferno", vmin=0.0, vmax=vmax)
    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.scatter(rx_axis, np.full(len(rx_axis), rx_z), s=10, c="red",
               label="RX aperture", zorder=5)
    ax.scatter(tx_axis, np.full(len(tx_axis), tx_z), s=10, c="blue",
               label="TX aperture", zorder=5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    # Gridlines draw ABOVE the heatmap (matplotlib z-order: image 0, scatter 1,
    # gridlines 2). That is deliberate here -- these panels exist to read x/z
    # positions off, and the overlay is what makes that possible.
    ax.grid(True)
    ax.legend(loc=legend_loc, framealpha=0.9, markerscale=2)


def draw_line_panel(ax, series: Sequence, *, title: str, xlabel: str, ylabel: str,
                    xlim: Optional[Tuple[float, float]] = None,
                    ylim: Optional[Tuple[float, float]] = None) -> None:
    """Plot `series` -- an iterable of (label, x, y), or (label, x, y, color) -- on one
    labelled, gridded axis. `label=None` omits that curve from the legend."""
    any_labelled = False
    for entry in series:
        label, x, y = entry[0], entry[1], entry[2]
        color = entry[3] if len(entry) > 3 else None
        ax.plot(x, y, label=label, color=color)
        any_labelled = any_labelled or label is not None
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True)
    if any_labelled:
        ax.legend()
