"""
SRWLIB_URI_integrated_scan_yy_yp_tracking.py
=============================================

Electron beam trajectory tracking for all (beam_y, beam_yp) conditions
used in the yy_yp makedata scan.

Beam/field parameters are imported from srw_uri_yy_yp_common.py and are
therefore identical to those used by makedata.

For each condition, trajectories are computed at two representative Lgap
values (LGAP_TRACK[0] and LGAP_TRACK[-1]).  Results are saved as:
  - individual/Tracking_BeamY*_BeamYP*.png : X and Y planes per condition
  - Tracking_grid_Y.png : grid of Y(ct), unified axis range, all conditions
  - Tracking_grid_X.png : grid of X(ct), unified axis range, all conditions

Output directory: SAVE_DIR
"""

import sys
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import os

try:
    import sys
    sys.path.append('../')
    from srwlib import *
except Exception:
    from srwpy.srwlib import *

from srw_uri_yy_yp_common import (
    MAG_DIR,
    LGAP_ARRAY, BEAM_Y_LOOP, BEAM_YP_LOOP,
    GAMMA,
    MagFieldManager,
)


# =====================================================================
# Tracking-specific configuration
# =====================================================================

SAVE_DIR = "/home/nishi/SRW/env/python/srwpy/examples/Oscillation/yy_yp_tracking/"

# Representative Lgap values: min and max of the full scan range
LGAP_TRACK  = [float(LGAP_ARRAY[0]),  float(LGAP_ARRAY[-1])]   # m
LGAP_LABELS = [
    f"Lgap_min ({LGAP_TRACK[0]*1e3:.0f} mm)",
    f"Lgap_max ({LGAP_TRACK[1]*1e3:.0f} mm)",
]
LGAP_COLORS = ["tab:blue", "tab:red"]

NP_TRAJ   = 10001   # trajectory points (coarser than makedata's 50001)
N_WORKERS = 20

# Subset of conditions shown in the grid plot (must be a subset of BEAM_Y/YP_LOOP).
# 5 values each, including 0, as evenly spaced as possible.
GRID_BEAM_Y_MM   = [  0.0,   5.0,  10.0,  15.0,  20.0]   # mm
GRID_BEAM_YP_MRAD = [ -2.0,  -1.0,   0.0,   1.0,   1.9]   # mrad


# =====================================================================
# TrajectoryEngine  — data generation
# =====================================================================

class TrajectoryEngine:
    """
    Compute electron beam trajectories through the double-undulator.

    Mirrors the class structure of SRWSimulationEngine in makedata.py:
    one engine instance per worker process; call run_tracking_driving()
    to obtain trajectories for all LGAP_TRACK values at once.
    """

    def __init__(self, mag_manager: MagFieldManager):
        self.mag_manager = mag_manager
        self.np_traj     = NP_TRAJ
        # ct_end = distance from beam start to the downstream end of the 2nd
        # undulator (+ 0.1 m margin), using the largest Lgap so all trajectories
        # share the same time axis.  Mirrors makedata's z_end_field exactly.
        self.ct_end_ref  = (
            mag_manager.z_end(max(LGAP_TRACK)) - mag_manager.z_start
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_tracking_driving(
        self, beam_y: float, beam_yp: float
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute trajectories at every Lgap in LGAP_TRACK.

        Returns
        -------
        list of (ct_mesh [m], arX [mm], arY [mm]) — one entry per Lgap
        """
        return [self.calc_trajectory(beam_y, beam_yp, Lgap) for Lgap in LGAP_TRACK]

    # ------------------------------------------------------------------
    # Internal: single Lgap trajectory
    # ------------------------------------------------------------------

    def calc_trajectory(
        self, beam_y: float, beam_yp: float, Lgap: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fld_cnt = self.mag_manager.get_fld_cnt(Lgap)
        part    = self._make_particle(beam_y, beam_yp)

        partTraj = SRWLPrtTrj()
        partTraj.partInitCond = part
        partTraj.allocate(self.np_traj, True)
        partTraj.ctStart = 0.0
        partTraj.ctEnd   = self.ct_end_ref

        srwl.CalcPartTraj(partTraj, fld_cnt, [1])

        ct  = np.linspace(partTraj.ctStart, partTraj.ctEnd, partTraj.np)
        arX = np.array(partTraj.arX) * 1e3   # m → mm
        arY = np.array(partTraj.arY) * 1e3
        arBy = np.array(partTraj.arBy)        # T
        arBz = np.array(partTraj.arBz)        # T
        return ct, arX, arY, arBy, arBz

    # ------------------------------------------------------------------
    # Internal: particle initialisation
    # ------------------------------------------------------------------

    def _make_particle(self, beam_y: float, beam_yp: float) -> "SRWLParticle":
        part       = SRWLParticle()
        part.x     = 0.0
        part.y     = beam_y
        part.z     = self.mag_manager.z_start   # upstream of first undulator
        part.xp    = 0.0
        part.yp    = beam_yp
        part.gamma = GAMMA
        return part


# =====================================================================
# Parallel worker
# =====================================================================

_worker_engine: Optional[TrajectoryEngine] = None


def _init_worker() -> None:
    """Initialise one TrajectoryEngine per subprocess (called by ProcessPoolExecutor).

    Redirects the worker's stdout (both Python-level and C-level fd 1) to
    /dev/null so that SRW diagnostic messages do not scroll the progress grid
    in the parent process.
    """
    global _worker_engine
    # C-level redirect (catches SRW's printf / cout output)
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 1)
    os.close(_devnull_fd)
    # Python-level redirect
    sys.stdout = open(os.devnull, 'w')

    mag_mgr = MagFieldManager(MAG_DIR)
    _worker_engine = TrajectoryEngine(mag_mgr)


def _track_worker(
    args: Tuple[float, float]
) -> Tuple[float, float, Optional[list]]:
    """
    Worker function: compute trajectories for one (beam_y, beam_yp) condition.

    Returns
    -------
    (beam_y_mm, beam_yp_mrad, traj_list | None)
    """
    beam_y_m, beam_yp_m = args
    beam_y_mm    = beam_y_m  * 1e3
    beam_yp_mrad = beam_yp_m * 1e3
    try:
        traj_list = _worker_engine.run_tracking_driving(beam_y_m, beam_yp_m)
        return beam_y_mm, beam_yp_mrad, traj_list
    except Exception as exc:
        print(f"  [ERROR] BeamY={beam_y_mm:.1f} mm BeamYP={beam_yp_mrad:.1f} mrad: {exc}")
        return beam_y_mm, beam_yp_mrad, None


# =====================================================================
# Filename utility
# =====================================================================

def safe_num(v: float, decimals: int = 1) -> str:
    """Float → filename-safe string  (e.g. -1.5 → 'm1p5')."""
    return f"{v:.{decimals}f}".replace("-", "m").replace(".", "p")


# =====================================================================
# Plotting  — visualisation
# =====================================================================

def plot_individual(
    beam_y_mm:    float,
    beam_yp_mrad: float,
    traj_list:    list,
    save_dir:     str,
) -> None:
    """Save a 2-panel plot (X and Y planes) with both Lgap trajectories."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle(
        f"BeamY={beam_y_mm:.1f} mm, BeamYP={beam_yp_mrad:.1f} mrad",
        fontsize=11,
    )

    for (ct, arX, arY, *_), label, color in zip(traj_list, LGAP_LABELS, LGAP_COLORS):
        axes[0].plot(ct, arX, color=color, label=label, linewidth=0.3, linestyle="-", marker='')
        axes[1].plot(ct, arY, color=color, label=label, linewidth=0.3, linestyle="-", marker='')

    axes[0].set_ylabel("X [mm]")
    axes[0].set_title("Horizontal trajectory")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("Y [mm]")
    axes[1].set_xlabel("ct [m]")
    axes[1].set_title("Vertical trajectory")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    fname = f"Tracking_BeamY{safe_num(beam_y_mm)}_BeamYP{safe_num(beam_yp_mrad)}.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=120)
    plt.close(fig)


def _get_component(
    traj_entry: tuple, component: str
) -> np.ndarray:
    """Extract the requested component array from a trajectory tuple."""
    ct, arX, arY, arBy, arBz = traj_entry
    return {'X': arX, 'Y': arY, 'By': arBy, 'Bz': arBz}[component]


_COMPONENT_UNIT = {'X': 'mm', 'Y': 'mm', 'By': 'T', 'Bz': 'T'}


def _compute_global_limits(
    data_store: dict,
    component:  str,   # 'X', 'Y', 'By', or 'Bz'
) -> Tuple[float, float, float, float]:
    """Return (ct_min, ct_max, val_min, val_max) across all stored trajectories."""
    ct_all, val_all = [], []
    for traj_list in data_store.values():
        if traj_list is None:
            continue
        for traj_entry in traj_list:
            ct = traj_entry[0]
            vals = _get_component(traj_entry, component)
            ct_all.extend([float(ct[0]), float(ct[-1])])
            val_all.extend([float(vals.min()), float(vals.max())])

    if not val_all:
        return 0.0, 1.0, -1.0, 1.0

    v_min, v_max = min(val_all), max(val_all)
    v_pad = (v_max - v_min) * 0.05 or 0.1
    return min(ct_all), max(ct_all), v_min - v_pad, v_max + v_pad


def _build_grid_figure(
    data_store:   dict,
    beam_y_vals:  list,
    beam_yp_vals: list,
    by_idx:       dict,
    byp_idx:      dict,
    component:    str,   # 'X', 'Y', 'By', or 'Bz'
    ct_min:       float,
    ct_max:       float,
    val_min:      float,
    val_max:      float,
    save_dir:     str,
) -> None:
    """Build and save one grid figure with unified axis ranges.

    Layout
    ------
    rows    : BeamY values (ascending top → bottom)
    columns : BeamYP values (ascending left → right)
    cells   : trajectory plot, both Lgap values overlaid in different colours
    """
    n_rows = len(beam_y_vals)
    n_cols = len(beam_yp_vals)
    cell_w, cell_h = 0.5, 0.4

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols + 1.5, cell_h * n_rows + 1.2),
        squeeze=False,
    )
    unit = _COMPONENT_UNIT.get(component, '')
    fig.suptitle(
        f"Trajectory {component}(ct)  [{unit}]   rows: BeamY [mm]  /  cols: BeamYP [mrad]\n"
        + "  ".join(f"[{c}] {lbl}" for c, lbl in zip(LGAP_COLORS, LGAP_LABELS)),
        fontsize=4, y=0.999,
    )

    for (by, byp), traj_list in data_store.items():
        i = by_idx.get(by)
        j = byp_idx.get(byp)
        if i is None or j is None:
            continue
        ax = axes[i, j]
        ax.set_xlim(ct_min, ct_max)
        ax.set_ylim(val_min, val_max)
        ax.tick_params(labelsize=2, width=0.3, length=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

        if traj_list is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=5, color="gray")
            continue

        for traj_entry, color in zip(traj_list, LGAP_COLORS):
            ct   = traj_entry[0]
            vals = _get_component(traj_entry, component)
            step = max(1, len(ct) // 500)
            ax.plot(ct[::step], vals[::step],
                    color=color, linewidth=0.3, linestyle="-", marker='',
                    rasterized=True)

    # Mark missing conditions with dark background
    for by in beam_y_vals:
        for byp in beam_yp_vals:
            if (by, byp) not in data_store:
                axes[by_idx[by], byp_idx[byp]].set_facecolor("#222222")

    # Axis labels: column headers (top) and row labels (left)
    for j, byp in enumerate(beam_yp_vals):
        axes[0, j].set_title(f"YP={byp:.1f}", fontsize=2, pad=1)
    for i, by in enumerate(beam_y_vals):
        axes[i, 0].set_ylabel(f"Y={by:.0f}mm\n[{unit}]", fontsize=2)
    for j in range(n_cols):
        axes[-1, j].set_xlabel("ct [m]", fontsize=2)

    plt.subplots_adjust(
        left=0.07, right=0.99, bottom=0.04, top=0.93,
        wspace=0.07, hspace=0.13,
    )
    fpath = os.path.join(save_dir, f"Tracking_grid_{component}.png")
    fig.savefig(fpath, dpi=600)
    plt.close(fig)
    print(f"Grid ({component}) saved → {fpath}")


def plot_grid(data_store: dict, save_dir: str) -> None:
    """Generate Tracking_grid_Y.png and Tracking_grid_X.png.

    Grid dimensions are fixed to the full BEAM_Y_LOOP × BEAM_YP_LOOP scan
    range so that missing or failed conditions appear as dark cells rather
    than causing the grid to shrink or shift.
    """
    # Use the reduced grid lists (5×5 subset, 0 always included).
    # Values are rounded to 6 decimal places to match data_store key format.
    beam_y_vals  = [round(float(v), 6) for v in sorted(GRID_BEAM_Y_MM)]
    beam_yp_vals = [round(float(v), 6) for v in sorted(GRID_BEAM_YP_MRAD)]
    by_idx  = {v: i for i, v in enumerate(beam_y_vals)}
    byp_idx = {v: i for i, v in enumerate(beam_yp_vals)}

    for component in ('Y', 'X', 'By', 'Bz'):
        ct_min, ct_max, val_min, val_max = _compute_global_limits(data_store, component)
        # grid_X: fix y-axis to 0 ~ 120 µm (= 0 ~ 0.12 mm) with small margin
        if component == 'X':
            val_min = -0.005   # mm
            val_max =  0.140   # mm
        _build_grid_figure(
            data_store, beam_y_vals, beam_yp_vals,
            by_idx, byp_idx,
            component,
            ct_min, ct_max, val_min, val_max,
            save_dir,
        )


# =====================================================================
# Endpoint difference plot
# =====================================================================

def plot_endpoint_diff(data_store: dict, save_dir: str) -> None:
    """
    Plot the difference of X and Y endpoint positions between Lgap_max and
    Lgap_min trajectories.

    For each (beam_y, beam_yp) condition with valid data:
        diff_X = arX_lgap_max[-1] - arX_lgap_min[-1]   [mm]
        diff_Y = arY_lgap_max[-1] - arY_lgap_min[-1]   [mm]

    Two-panel figure:
        top : diff_X vs BeamYP, one series per BeamY
        bot : diff_Y vs BeamYP, one series per BeamY

    Only GRID_BEAM_Y_MM × GRID_BEAM_YP_MRAD conditions are shown.
    """
    beam_y_vals  = sorted(round(float(v), 6) for v in GRID_BEAM_Y_MM)
    beam_yp_vals = sorted(round(float(v), 6) for v in GRID_BEAM_YP_MRAD)

    # Colour map and marker cycle for BeamY series
    cmap    = plt.get_cmap("tab10")
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '+']

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    fig.suptitle(
        "Endpoint position difference  (Lgap_max − Lgap_min)\n"
        f"at ct = {LGAP_LABELS[1]} vs {LGAP_LABELS[0]}",
        fontsize=11,
    )

    for k, by in enumerate(beam_y_vals):
        diff_X_list, diff_Y_list, byp_list = [], [], []

        for byp in beam_yp_vals:
            traj_list = data_store.get((by, byp))
            if traj_list is None or len(traj_list) < 2:
                continue
            # traj_list[0] = Lgap_min,  traj_list[1] = Lgap_max
            arX_min, arY_min = traj_list[0][1], traj_list[0][2]
            arX_max, arY_max = traj_list[1][1], traj_list[1][2]
            diff_X_list.append(float(arX_max[-1] - arX_min[-1]))
            diff_Y_list.append(float(arY_max[-1] - arY_min[-1]))
            byp_list.append(byp)

        if not byp_list:
            continue

        kw = dict(
            color=cmap(k % 10),
            marker=markers[k % len(markers)],
            markersize=5,
            linewidth=0.8,
            linestyle="-",
            label=f"BeamY={by:.0f} mm",
        )
        axes[0].plot(byp_list, diff_X_list, **kw)
        axes[1].plot(byp_list, diff_Y_list, **kw)

    axes[0].axhline(0, color="gray", linewidth=0.4, linestyle="--")
    axes[0].set_ylabel("ΔX at endpoint [mm]")
    axes[0].set_title("Horizontal endpoint difference")
    axes[0].legend(fontsize=8, loc="best")

    axes[1].axhline(0, color="gray", linewidth=0.4, linestyle="--")
    axes[1].set_ylabel("ΔY at endpoint [mm]")
    axes[1].set_xlabel("BeamYP [mrad]")
    axes[1].set_title("Vertical endpoint difference")
    axes[1].legend(fontsize=8, loc="best")

    plt.tight_layout()
    fpath = os.path.join(save_dir, "Tracking_endpoint_diff.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"Endpoint diff plot saved → {fpath}")


# =====================================================================
# Main
# =====================================================================

def run(n_workers: int = N_WORKERS) -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    conditions = [
        (float(by), float(byp))
        for by  in BEAM_Y_LOOP
        for byp in BEAM_YP_LOOP
    ]
    print(
        f"Tracking {len(conditions)} conditions "
        f"({len(BEAM_Y_LOOP)} BeamY × {len(BEAM_YP_LOOP)} BeamYP), "
        f"Lgap = {LGAP_TRACK} m, {n_workers} workers."
    )

    data_store: Dict[Tuple[float, float], Optional[list]] = {}

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
        futures = {
            pool.submit(_track_worker, (by, byp)): (round(by * 1e3, 6), round(byp * 1e3, 6))
            for by, byp in conditions
        }
        for future in tqdm(as_completed(futures), total=len(conditions), desc="Tracking"):
            by_mm, byp_mrad = futures[future]
            try:
                _, _, traj_list = future.result()
            except Exception as exc:
                traj_list = None
                sys.stderr.write(f"  [ERROR] BeamY={by_mm:.1f} BeamYP={byp_mrad:.1f}: {exc}\n")
            data_store[(by_mm, byp_mrad)] = traj_list

    # --- Individual plots ---
    ind_dir = os.path.join(SAVE_DIR, "individual")
    os.makedirs(ind_dir, exist_ok=True)
    n_ok = 0
    for (by_mm, byp_mrad), traj_list in data_store.items():
        if traj_list is None:
            continue
        plot_individual(by_mm, byp_mrad, traj_list, ind_dir)
        n_ok += 1
    print(f"Individual plots saved ({n_ok}) → {ind_dir}")

    # --- Grid plots (unified axis ranges) ---
    plot_grid(data_store, SAVE_DIR)

    # --- Endpoint difference plot ---
    plot_endpoint_diff(data_store, SAVE_DIR)
    print("Done.")


if __name__ == "__main__":
    run()
