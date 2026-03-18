"""
SRWLIB_URI_plot_reference_tracking.py
======================================

Plot the beam trajectory (X, Y) and along-path magnetic field (By, Bz)
for the reference condition:  beam_y = 0,  beam_yp = 0.

Trajectories and fields are computed at both representative Lgap values
(Lgap_min and Lgap_max) defined in the tracking script.

Output
------
  Oscillation/yy_yp_tracking/reference_tracking.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from srw_uri_yy_yp_common import MAG_DIR, MagFieldManager
from SRWLIB_URI_integrated_scan_yy_yp_tracking import (
    TrajectoryEngine,
    LGAP_TRACK, LGAP_LABELS, LGAP_COLORS,
    SAVE_DIR,
)


# =====================================================================
# Compute reference trajectories
# =====================================================================

def compute_reference(beam_y: float = 0.0, beam_yp: float = 0.0) -> list:
    """Return traj_list for (beam_y, beam_yp) at all LGAP_TRACK values."""
    mag_mgr = MagFieldManager(MAG_DIR)
    engine  = TrajectoryEngine(mag_mgr)
    return engine.run_tracking_driving(beam_y, beam_yp)


# =====================================================================
# Plot
# =====================================================================

def plot_reference(traj_list: list, save_dir: str) -> None:
    """
    4-panel figure:
      top-left  : X(ct) — horizontal position [mm]
      top-right : By(ct) — vertical magnetic field [T]
      bot-left  : Y(ct) — vertical position [mm]
      bot-right : Bz(ct) — longitudinal magnetic field [T]
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    fig.suptitle(
        "Reference trajectory: BeamY = 0 mm,  BeamYP = 0 mrad",
        fontsize=12,
    )

    ax_X, ax_By = axes[0]
    ax_Y, ax_Bz = axes[1]

    for (ct, arX, arY, arBy, arBz), label, color in zip(
        traj_list, LGAP_LABELS, LGAP_COLORS
    ):
        kw = dict(color=color, label=label, linewidth=0.8, linestyle="-", marker='')
        ax_X.plot(ct, arX,  **kw)
        ax_Y.plot(ct, arY,  **kw)
        ax_By.plot(ct, arBy, **kw)
        ax_Bz.plot(ct, arBz, **kw)

    ax_X.set_ylabel("X [mm]")
    ax_X.set_title("Horizontal position")
    ax_X.legend(fontsize=8)
    ax_X.axhline(0, color="gray", linewidth=0.4, linestyle="--")

    ax_Y.set_ylabel("Y [mm]")
    ax_Y.set_xlabel("ct [m]")
    ax_Y.set_title("Vertical position")
    ax_Y.legend(fontsize=8)
    ax_Y.axhline(0, color="gray", linewidth=0.4, linestyle="--")

    ax_By.set_ylabel("By [T]")
    ax_By.set_title("Vertical magnetic field (along trajectory)")
    ax_By.legend(fontsize=8)
    ax_By.axhline(0, color="gray", linewidth=0.4, linestyle="--")

    ax_Bz.set_ylabel("Bz [T]")
    ax_Bz.set_xlabel("ct [m]")
    ax_Bz.set_title("Longitudinal magnetic field (along trajectory)")
    ax_Bz.legend(fontsize=8)
    ax_Bz.axhline(0, color="gray", linewidth=0.4, linestyle="--")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fpath = os.path.join(save_dir, "reference_tracking.png")
    fig.savefig(fpath, dpi=600)
    plt.close(fig)
    print(f"Saved → {fpath}")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    print("Computing reference trajectories (beam_y=0, beam_yp=0) ...")
    traj_list = compute_reference(beam_y=0.0, beam_yp=0.0)
    plot_reference(traj_list, SAVE_DIR)
    print("Done.")
