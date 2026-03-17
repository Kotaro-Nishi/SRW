"""
SRWLIB_URI_integrated_scan_yy_yp_preana.py

Pre-analysis: generate 2D intensity maps  intensity(Lgap, screen_y)
for every (beam_y, beam_yp) condition in parallel.

Output: SAVE_DIR/preview/Intensity2D_BeamY*_BeamYP*.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional
import os
import re

# =====================================================================
# Configuration
# =====================================================================

DATA_DIR      = "/home/nishi/SRW/env/python/srwpy/examples/data_URI/"
SAVE_DIR      = "/home/nishi/SRW/env/python/srwpy/examples/Oscillation/yy_yp_ana/preview/"
LGAP_ARRAY    = np.linspace(0.860, 1.684, 825)      # metres
SCREEN_Y_MINS = np.arange(-20.0, 19.0, 2.0) * 1e-3  # metres
SCREEN_Y_PTS  = 51
N_WORKERS     = 20

# =====================================================================
# Filename utilities
# =====================================================================

def safe_num(v: float, decimals: int = 1) -> str:
    """Float → filename-safe string  (e.g. -1.5 → 'm1p5')."""
    return f"{v:.{decimals}f}".replace("-", "m").replace(".", "p")


def parse_filename(fname: str) -> Optional[Tuple[float, float, float]]:
    """Return (beam_y_mm, beam_yp_mrad, screen_ymin_mm) or None."""
    m = re.match(
        r"BeamY_([-\d.]+)mm_BeamYP_([-\d.]+)mrad_ScreenYmin_([-\d.]+)mm\.txt",
        os.path.basename(fname),
    )
    if m is None:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def get_all_conditions(data_dir: str) -> List[Tuple[float, float]]:
    """Return sorted unique (beam_y_mm, beam_yp_mrad) pairs."""
    conditions: set = set()
    for fname in os.listdir(data_dir):
        result = parse_filename(fname)
        if result is not None:
            by, byp, _ = result
            conditions.add((by, byp))
    return sorted(conditions)

# =====================================================================
# Data loading
# =====================================================================

def load_condition(
    data_dir:      str,
    beam_y_mm:     float,
    beam_yp_mrad:  float,
    screen_y_mins: np.ndarray,
    screen_y_pts:  int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load and stitch all ScreenYmin files for one condition.

    Returns
    -------
    intensity : (n_lgap, n_y)  or None
    screen_y  : (n_y,) [m]    or None
    """
    intensity_windows: List[np.ndarray] = []
    screen_y_windows:  List[np.ndarray] = []

    for ymin in screen_y_mins:
        fname = (
            f"BeamY_{beam_y_mm:.3f}mm_BeamYP_{beam_yp_mrad:.3f}mrad"
            f"_ScreenYmin_{ymin * 1e3:.3f}mm.txt"
        )
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
        data = np.loadtxt(fpath, ndmin=2)   # always at least 2D

        # Shape validation: expected (n_lgap, n_y_pts).
        # If the file was written transposed, flip it back.
        # Use the first successfully loaded window as the reference.
        if intensity_windows:
            ref = intensity_windows[0]
            if data.shape == ref.shape:
                pass                           # correct
            elif data.shape == ref.shape[::-1]:
                data = data.T                  # transposed → flip
            else:
                print(f"    [warn] shape {data.shape} ≠ ref {ref.shape} "
                      f"in {os.path.basename(fpath)}, skipped")
                continue
        elif data.ndim == 2 and data.shape[0] == 1:
            # First window arrived as (1, N) — probably a single-row file;
            # treat N columns as lgap points and this as 1 screen-y point.
            data = data.T                      # → (N, 1)

        # Warn if the actual number of screen-y columns differs from expectation.
        if data.shape[1] != screen_y_pts:
            print(f"    [info] n_y_pts={data.shape[1]} (expected {screen_y_pts}) "
                  f"in {os.path.basename(fpath)}")

        y_arr = np.linspace(ymin, ymin + 2.0e-3, data.shape[1])
        intensity_windows.append(data)
        screen_y_windows.append(y_arr)

    if not intensity_windows:
        return None, None

    # Stitch: first window in full; later windows skip the overlapping boundary point.
    intensity = intensity_windows[0]
    screen_y  = screen_y_windows[0]
    for i in range(1, len(intensity_windows)):
        intensity = np.concatenate([intensity, intensity_windows[i][:, 1:]], axis=1)
        screen_y  = np.concatenate([screen_y,  screen_y_windows[i][1:]])

    return intensity, screen_y

# =====================================================================
# 2D intensity plot
# =====================================================================

def plot_intensity_2d(
    intensity:    np.ndarray,
    lgap_mm:      np.ndarray,
    screen_y_m:   np.ndarray,
    beam_y_mm:    float,
    beam_yp_mrad: float,
    save_dir:     str,
) -> None:
    """
    2D colormap:  x = Lgap [mm],  y = screen Y [mm],  color = intensity.

    Parameters
    ----------
    intensity  : (n_lgap, n_y)
    lgap_mm    : (n_lgap,) [mm]
    screen_y_m : (n_y,)   [m]
    """
    y_mm = screen_y_m * 1e3

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        f"BeamY={beam_y_mm:.1f} mm, BeamYP={beam_yp_mrad:.1f} mrad",
        fontsize=12,
    )

    im = ax.imshow(
        intensity.T,            # (n_lgap, n_y).T → (n_y, n_lgap) for imshow
        aspect="auto",
        origin="lower",
        extent=[lgap_mm[0], lgap_mm[-1], y_mm[0], y_mm[-1]],
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="Intensity [arb.]")
    ax.set_xlabel("Lgap [mm]")
    ax.set_ylabel("Screen Y [mm]")
    ax.set_title("Intensity(Lgap, Screen Y)")

    plt.tight_layout()
    fname = (
        f"Intensity2D"
        f"_BeamY{safe_num(beam_y_mm)}"
        f"_BeamYP{safe_num(beam_yp_mrad)}.png"
    )
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)

# =====================================================================
# Parallel workers  (top-level functions required for pickling)
# =====================================================================

def _plot_worker(args: tuple) -> Tuple[float, float, bool]:
    """Load data for one condition and save an individual 2D intensity plot."""
    beam_y_mm, beam_yp_mrad, data_dir, save_dir, \
        screen_y_mins, screen_y_pts, lgap_mm = args

    intensity, screen_y = load_condition(
        data_dir, beam_y_mm, beam_yp_mrad, screen_y_mins, screen_y_pts
    )
    if intensity is None:
        return beam_y_mm, beam_yp_mrad, False

    plot_intensity_2d(
        intensity, lgap_mm, screen_y,
        beam_y_mm, beam_yp_mrad, save_dir,
    )
    return beam_y_mm, beam_yp_mrad, True


def _load_worker(args: tuple) -> Tuple[float, float,
                                       Optional[np.ndarray],
                                       Optional[np.ndarray]]:
    """Load and stitch data for one condition; return arrays (no plotting)."""
    beam_y_mm, beam_yp_mrad, data_dir, screen_y_mins, screen_y_pts = args
    intensity, screen_y = load_condition(
        data_dir, beam_y_mm, beam_yp_mrad, screen_y_mins, screen_y_pts
    )
    return beam_y_mm, beam_yp_mrad, intensity, screen_y

# =====================================================================
# Individual plots (one file per condition)
# =====================================================================

def run(data_dir, save_dir, lgap_array, screen_y_mins, screen_y_pts, n_workers):
    """Save one Intensity2D PNG per (beam_y, beam_yp) condition."""
    os.makedirs(save_dir, exist_ok=True)

    lgap_mm    = lgap_array * 1e3
    conditions = get_all_conditions(data_dir)
    print(f"Individual plots: {len(conditions)} conditions → {save_dir}")

    worker_args = [
        (by, byp, data_dir, save_dir, screen_y_mins, screen_y_pts, lgap_mm)
        for by, byp in conditions
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_plot_worker, a): a[:2] for a in worker_args}
        for future in as_completed(futures):
            by, byp = futures[future]
            try:
                _, _, ok = future.result()
                status = "done" if ok else "SKIP (no data)"
                print(f"  [{status}]  BeamY={by:.1f} mm, BeamYP={byp:.1f} mrad")
            except Exception as exc:
                print(f"  [ERROR] BeamY={by:.1f} BeamYP={byp:.1f}: {exc}")

    print("Individual plots done.")

# =====================================================================
# Grid plot (all conditions in one image)
# =====================================================================

def run_grid(data_dir, save_dir, lgap_array, screen_y_mins, screen_y_pts, n_workers):
    """
    Load all conditions in parallel, then plot a single grid image.

    Layout
    ------
    rows    : BeamY values  (ascending top → bottom)
    columns : BeamYP values (ascending left → right)
    cells   : intensity(Lgap, Screen Y) colormap  (per-condition color scale)

    Output: save_dir/Intensity2D_grid.png
    """
    os.makedirs(save_dir, exist_ok=True)

    lgap_mm    = lgap_array * 1e3
    conditions = get_all_conditions(data_dir)

    beam_y_vals  = sorted({c[0] for c in conditions})
    beam_yp_vals = sorted({c[1] for c in conditions})
    n_rows = len(beam_y_vals)
    n_cols = len(beam_yp_vals)

    by_idx  = {v: i for i, v in enumerate(beam_y_vals)}
    byp_idx = {v: i for i, v in enumerate(beam_yp_vals)}

    print(f"Grid: {n_rows} BeamY × {n_cols} BeamYP = {n_rows * n_cols} panels")

    # --- Parallel data loading ---
    load_args = [
        (by, byp, data_dir, screen_y_mins, screen_y_pts)
        for by, byp in conditions
    ]
    data_store: dict = {}   # (by, byp) → (intensity, screen_y)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_worker, a): a[:2] for a in load_args}
        for future in as_completed(futures):
            by, byp = futures[future]
            try:
                _, _, intensity, screen_y = future.result()
                data_store[(by, byp)] = (intensity, screen_y)
                print(f"  [loaded] BeamY={by:.1f} BeamYP={byp:.1f}")
            except Exception as exc:
                print(f"  [ERROR]  BeamY={by:.1f} BeamYP={byp:.1f}: {exc}")

    # --- Build figure ---
    cell_w, cell_h = 3.0, 2.5
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols + 1.5, cell_h * n_rows + 1.2),
        sharex=True, sharey=True,
        squeeze=False,
    )
    fig.suptitle(
        "Intensity(Lgap, Screen Y)\n"
        "rows: BeamY [mm]   /   cols: BeamYP [mrad]",
        fontsize=11, y=0.998,
    )

    # --- Fill each cell ---
    for (by, byp), (intensity, screen_y) in data_store.items():
        i = by_idx.get(by)
        j = byp_idx.get(byp)
        if i is None or j is None:
            continue
        ax = axes[i, j]

        if intensity is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="gray")
            continue

        y_mm = screen_y * 1e3
        ax.imshow(
            intensity.T,        # (n_lgap, n_y).T → (n_y, n_lgap) for imshow
            aspect="auto",
            origin="lower",
            extent=[lgap_mm[0], lgap_mm[-1], y_mm[0], y_mm[-1]],
            cmap="viridis",
        )
        ax.tick_params(labelsize=6)

    # Mark any missing conditions
    for by in beam_y_vals:
        for byp in beam_yp_vals:
            if (by, byp) not in data_store:
                axes[by_idx[by], byp_idx[byp]].set_facecolor("#222222")

    # --- Column headers: BeamYP values (top row) ---
    for j, byp in enumerate(beam_yp_vals):
        axes[0, j].set_title(f"YP = {byp:.1f} mrad", fontsize=8, pad=3)

    # --- Row headers: BeamY values (leftmost column, combined with Screen Y label) ---
    for i, by in enumerate(beam_y_vals):
        axes[i, 0].set_ylabel(f"Y = {by:.1f} mm\nScreen Y [mm]", fontsize=7)

    # --- x-axis label on bottom row only ---
    for j in range(n_cols):
        axes[-1, j].set_xlabel("Lgap [mm]", fontsize=7)

    plt.subplots_adjust(
        left=0.10, right=0.98,
        bottom=0.06, top=0.92,
        wspace=0.04, hspace=0.10,
    )

    fpath = os.path.join(save_dir, "Intensity2D_grid.png")
    fig.savefig(fpath, dpi=120)
    plt.close(fig)
    print(f"Grid saved → {fpath}")

# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    # Individual plots (one per condition)
    run(
        data_dir      = DATA_DIR,
        save_dir      = SAVE_DIR,
        lgap_array    = LGAP_ARRAY,
        screen_y_mins = SCREEN_Y_MINS,
        screen_y_pts  = SCREEN_Y_PTS,
        n_workers     = N_WORKERS,
    )

    # Combined grid image
    run_grid(
        data_dir      = DATA_DIR,
        save_dir      = SAVE_DIR,
        lgap_array    = LGAP_ARRAY,
        screen_y_mins = SCREEN_Y_MINS,
        screen_y_pts  = SCREEN_Y_PTS,
        n_workers     = N_WORKERS,
    )
