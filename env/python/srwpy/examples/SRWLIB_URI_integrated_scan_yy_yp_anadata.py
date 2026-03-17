import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
from scipy.optimize import leastsq
import os
import re

# =====================================================================
# Configuration
# =====================================================================
DATA_DIR  = "/home/nishi/SRW/env/python/srwpy/examples/data_URI/"
SAVE_DIR  = "/home/nishi/SRW/env/python/srwpy/examples/Oscillation/yy_yp_ana/"
os.makedirs(SAVE_DIR, exist_ok=True)

LGAP_ARRAY     = np.arange(0.860, 1.685, 0.001)   # m, 825 points
SCREEN_Y_MINS  = np.arange(-20., 19., 2.) * 1e-3  # m, -20 to 18 mm (20 windows)
SCREEN_Y_PTS   = 51                                # points per window

# Sine function: A*(1 + sin(2*pi*d/T + phi))
def sin_fnc(prm, d):
    return prm[0] * (1.0 + np.sin(d * 2.0 * np.pi / prm[1] + prm[2]))

def fit_fnc(prm, d, osc):
    return sin_fnc(prm, d) - osc

# =====================================================================
# File discovery
# =====================================================================
def parse_filename(fname):
    """Return (beam_y_mm, beam_yp_mrad, screen_ymin_mm) from filename."""
    m = re.match(
        r"BeamY_([-\d.]+)mm_BeamYP_([-\d.]+)mrad_ScreenYmin_([-\d.]+)mm\.txt",
        os.path.basename(fname)
    )
    if m is None:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))

def get_all_conditions(data_dir):
    """Return sorted unique (beam_y_mm, beam_yp_mrad) pairs."""
    conditions = set()
    for fname in os.listdir(data_dir):
        result = parse_filename(fname)
        if result is None:
            continue
        by, byp, _ = result
        conditions.add((by, byp))
    return sorted(conditions)

# =====================================================================
# Data loading
# =====================================================================
def load_condition(data_dir, beam_y_mm, beam_yp_mrad):
    """
    Load and stitch all ScreenYmin files for a given (beam_y, beam_yp) condition.
    Returns:
        intensity : ndarray, shape (n_lgap, n_y)
        screen_y  : ndarray, shape (n_y,)  [m]
    """
    intensity_windows = []
    screen_y_windows  = []

    for ymin in SCREEN_Y_MINS:
        fname = (
            f"BeamY_{beam_y_mm:.3f}mm_BeamYP_{beam_yp_mrad:.3f}mrad"
            f"_ScreenYmin_{ymin*1e3:.3f}mm.txt"
        )
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
        data = np.loadtxt(fpath)   # shape (n_lgap, n_y_pts)
        y_arr = np.linspace(ymin, ymin + 2.e-3, SCREEN_Y_PTS)
        intensity_windows.append(data)
        screen_y_windows.append(y_arr)

    if len(intensity_windows) == 0:
        return None, None

    # Stitch: use all points from first window, then skip first point of each
    # subsequent window (they overlap at boundary).
    intensity_stitched = intensity_windows[0]
    screen_y_stitched  = screen_y_windows[0]
    for i in range(1, len(intensity_windows)):
        intensity_stitched = np.concatenate(
            [intensity_stitched, intensity_windows[i][:, 1:]], axis=1
        )
        screen_y_stitched = np.concatenate(
            [screen_y_stitched, screen_y_windows[i][1:]]
        )

    return intensity_stitched, screen_y_stitched   # (n_lgap, n_y), (n_y,)

# =====================================================================
# Sine fitting
# =====================================================================
def fit_sine_vs_lgap(intensity, lgap_mm, init_val=None):
    """
    For each screen y position, fit a sine to intensity vs Lgap.
    intensity : (n_lgap, n_y)
    lgap_mm   : (n_lgap,) [mm]
    Returns prm_arr : (n_y, 3)  columns = [amplitude, period_mm, phase]
            err_arr : (n_y, 3)
    """
    if init_val is None:
        # Rough initial guess from the data
        amp0    = np.nanmean(intensity)
        period0 = 115.0   # mm, typical undulator gap oscillation
        phase0  = 2.5
        init_val = [amp0, period0, phase0]

    n_y = intensity.shape[1]
    prm_arr = np.zeros((n_y, 3))
    err_arr = np.zeros((n_y, 3))

    for iy in range(n_y):
        data_y = intensity[:, iy]
        try:
            result = leastsq(
                fit_fnc, init_val,
                args=(lgap_mm, data_y),
                full_output=True
            )
            prm, cov, *_ = result
            prm_arr[iy] = prm
            if cov is not None:
                err_arr[iy] = np.sqrt(np.abs(np.diag(cov)))
            else:
                err_arr[iy] = np.nan
        except Exception:
            prm_arr[iy] = np.nan
            err_arr[iy] = np.nan

    return prm_arr, err_arr   # amplitude, period[mm], phase

# =====================================================================
# Plotting helpers
# =====================================================================
def plot_period_phase(screen_y_m, prm_arr, err_arr, beam_y_mm, beam_yp_mrad, save_dir):
    """Plot period and phase vs screen y for one (beam_y, beam_yp) condition."""
    y_mm   = screen_y_m * 1e3
    period = prm_arr[:, 1]
    phase  = prm_arr[:, 2]
    e_per  = err_arr[:, 1]
    e_phi  = err_arr[:, 2]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle(
        f"BeamY={beam_y_mm:.1f} mm, BeamYP={beam_yp_mrad:.1f} mrad", fontsize=12
    )

    axes[0].errorbar(y_mm, period, yerr=e_per, fmt='-o', markersize=2, linewidth=0.8)
    axes[0].set_ylabel("Fitted Period [mm]")
    axes[0].set_title("Oscillation Period vs Screen Y")

    axes[1].errorbar(y_mm, phase,  yerr=e_phi, fmt='-o', markersize=2, linewidth=0.8, color='tab:orange')
    axes[1].set_ylabel("Fitted Phase [rad]")
    axes[1].set_xlabel("Screen Y [mm]")
    axes[1].set_title("Oscillation Phase vs Screen Y")

    plt.tight_layout()
    fname = (
        f"PeriodPhase_BeamY{beam_y_mm:.1f}_BeamYP{beam_yp_mrad:.1f}.png"
        .replace("-", "m")
    )
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)

# =====================================================================
# Main analysis loop
# =====================================================================
def run_analysis():
    lgap_mm = LGAP_ARRAY * 1e3   # convert to mm for fitting

    conditions = get_all_conditions(DATA_DIR)
    print(f"Found {len(conditions)} (beam_y, beam_yp) conditions.")

    # Results: for each condition store (beam_y, beam_yp, y_maxperiod, y_minphase)
    summary = []

    for beam_y_mm, beam_yp_mrad in conditions:
        print(f"  Processing BeamY={beam_y_mm:.1f} mm, BeamYP={beam_yp_mrad:.1f} mrad ...",
              end=" ", flush=True)

        intensity, screen_y = load_condition(DATA_DIR, beam_y_mm, beam_yp_mrad)
        if intensity is None:
            print("  [SKIP: no data]")
            continue

        prm_arr, err_arr = fit_sine_vs_lgap(intensity, lgap_mm)

        # Extract extreme positions
        period = prm_arr[:, 1]
        phase  = prm_arr[:, 2]

        idx_max_period = np.nanargmax(period)
        idx_min_phase  = np.nanargmin(phase)
        y_max_period   = screen_y[idx_max_period] * 1e3   # mm
        y_min_phase    = screen_y[idx_min_phase]  * 1e3   # mm

        print(
            f"max_period at y={y_max_period:.2f} mm, "
            f"min_phase  at y={y_min_phase:.2f} mm"
        )

        # Per-condition plot
        plot_period_phase(screen_y, prm_arr, err_arr, beam_y_mm, beam_yp_mrad, SAVE_DIR)

        summary.append([beam_y_mm, beam_yp_mrad, y_max_period, y_min_phase,
                        period[idx_max_period], phase[idx_min_phase]])

    summary = np.array(summary)
    # columns: beam_y, beam_yp, y_max_period_mm, y_min_phase_mm, period_max, phase_min
    np.savetxt(
        os.path.join(SAVE_DIR, "summary.txt"),
        summary,
        header="beam_y_mm  beam_yp_mrad  y_maxPeriod_mm  y_minPhase_mm  period_max_mm  phase_min_rad",
        fmt="%.4f"
    )
    print(f"\nSummary saved to {SAVE_DIR}summary.txt")
    return summary

# =====================================================================
# Summary plots
# =====================================================================
def plot_summary(summary):
    """
    For each unique beam_y, plot y_maxperiod and y_minphase vs beam_yp.
    For each unique beam_yp, plot vs beam_y.
    """
    beam_y_vals  = np.unique(summary[:, 0])
    beam_yp_vals = np.unique(summary[:, 1])

    # --- Fixed beam_y: scan over beam_yp ---
    for by in beam_y_vals:
        mask = summary[:, 0] == by
        sub  = summary[mask]
        sub  = sub[np.argsort(sub[:, 1])]   # sort by beam_yp

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle(f"BeamY = {by:.1f} mm", fontsize=12)

        axes[0].plot(sub[:, 1], sub[:, 2], '-o', markersize=4)
        axes[0].set_ylabel("y at max Period [mm]")
        axes[0].set_title("Screen Y of max Oscillation Period")

        axes[1].plot(sub[:, 1], sub[:, 3], '-o', markersize=4, color='tab:orange')
        axes[1].set_ylabel("y at min Phase [mm]")
        axes[1].set_xlabel("BeamYP [mrad]")
        axes[1].set_title("Screen Y of min Oscillation Phase")

        plt.tight_layout()
        fname = f"Summary_vs_BeamYP_BeamY{by:.1f}.png".replace("-", "m")
        fig.savefig(os.path.join(SAVE_DIR, fname), dpi=150)
        plt.close(fig)

    # --- Fixed beam_yp: scan over beam_y ---
    for byp in beam_yp_vals:
        mask = summary[:, 1] == byp
        sub  = summary[mask]
        sub  = sub[np.argsort(sub[:, 0])]   # sort by beam_y

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle(f"BeamYP = {byp:.1f} mrad", fontsize=12)

        axes[0].plot(sub[:, 0], sub[:, 2], '-o', markersize=4)
        axes[0].set_ylabel("y at max Period [mm]")
        axes[0].set_title("Screen Y of max Oscillation Period")

        axes[1].plot(sub[:, 0], sub[:, 3], '-o', markersize=4, color='tab:orange')
        axes[1].set_ylabel("y at min Phase [mm]")
        axes[1].set_xlabel("BeamY [mm]")
        axes[1].set_title("Screen Y of min Oscillation Phase")

        plt.tight_layout()
        fname = f"Summary_vs_BeamY_BeamYP{byp:.1f}.png".replace("-", "m").replace(".", "p")
        fig.savefig(os.path.join(SAVE_DIR, fname), dpi=150)
        plt.close(fig)

    # --- 2D colormaps: y_maxperiod and y_minphase on (beam_y, beam_yp) grid ---
    grid_ymp = np.full((len(beam_y_vals), len(beam_yp_vals)), np.nan)
    grid_phi = np.full((len(beam_y_vals), len(beam_yp_vals)), np.nan)

    for row in summary:
        iy  = np.where(beam_y_vals  == row[0])[0][0]
        iyp = np.where(beam_yp_vals == row[1])[0][0]
        grid_ymp[iy, iyp] = row[2]
        grid_phi[iy, iyp] = row[3]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].imshow(
        grid_ymp, aspect='auto', origin='lower',
        extent=[beam_yp_vals[0], beam_yp_vals[-1], beam_y_vals[0], beam_y_vals[-1]]
    )
    axes[0].set_xlabel("BeamYP [mrad]")
    axes[0].set_ylabel("BeamY [mm]")
    axes[0].set_title("Screen Y at max Period [mm]")
    plt.colorbar(im0, ax=axes[0], label="y [mm]")

    im1 = axes[1].imshow(
        grid_phi, aspect='auto', origin='lower',
        extent=[beam_yp_vals[0], beam_yp_vals[-1], beam_y_vals[0], beam_y_vals[-1]]
    )
    axes[1].set_xlabel("BeamYP [mrad]")
    axes[1].set_ylabel("BeamY [mm]")
    axes[1].set_title("Screen Y at min Phase [mm]")
    plt.colorbar(im1, ax=axes[1], label="y [mm]")

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "Summary_2D_colormap.png"), dpi=150)
    plt.close(fig)
    print("Summary plots saved.")

# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    summary = run_analysis()
    if len(summary) > 0:
        plot_summary(summary)
