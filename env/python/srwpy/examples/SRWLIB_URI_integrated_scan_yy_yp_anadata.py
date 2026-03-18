import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import NamedTuple, List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import re
import warnings

# =====================================================================
# Data structures
# =====================================================================

@dataclass(eq=False)
class Config:
    """All run-time configuration in one place."""
    data_dir:      str
    save_dir:      str
    lgap_array:    np.ndarray     # metres
    screen_y_mins: np.ndarray     # metres
    screen_y_pts:  int = 51
    #n_workers:     int = field(default_factory=lambda: os.cpu_count() or 1)
    n_workers :    int = 20

class LgapSlice(NamedTuple):
    """A named sub-range of the Lgap scan."""
    name:      str            # safe for filenames / dict keys
    label:     str            # human-readable for plot titles
    idx_start: int            # start index into the full lgap array
    idx_end:   int            # end index (exclusive)
    lgap_mm:   np.ndarray     # actual lgap values [mm] for this slice


# prm_arr / err_arr column indices
I_AMP, I_PERIOD, I_PHASE = 0, 1, 2

# summary array column indices
I_BY, I_BYP, I_Y_MAXPER, I_Y_MINPHI, I_PERIOD_MAX, I_PHASE_MIN = range(6)

# curve_fit bounds: A > 0, T > 0, phi free (unbounded)
# Phase continuity is handled by unwrapping after each fit, not by hard bounds.
_FIT_BOUNDS = (
    [0.0,    0.0,    -np.inf],
    [np.inf, np.inf,  np.inf],
)


def make_lgap_slices(lgap_array_m: np.ndarray) -> List[LgapSlice]:
    """Return the 6 named Lgap sub-ranges (full + 5 partial)."""
    d = lgap_array_m * 1e3      # → mm
    n = len(d)
    q1, q2, q3 = n // 4, n // 2, 3 * n // 4

    specs = [
        ("full",     "Full (0–1)",   0,  n ),
        ("0_half",   "0 – 1/2",      0,  q2),
        ("0_3q",     "0 – 3/4",      0,  q3),
        ("1q_3q",    "1/4 – 3/4",   q1, q3),
        ("1q_end",   "1/4 – end",   q1,  n ),
        ("half_end", "1/2 – end",   q2,  n ),
    ]
    return [LgapSlice(name, label, i0, i1, d[i0:i1])
            for name, label, i0, i1 in specs]


# =====================================================================
# Sine model:  A · (1 + sin(2π·d/T + φ))
# =====================================================================

def sin_model(d: np.ndarray, A: float, T: float, phi: float) -> np.ndarray:
    return A * (1.0 + np.sin(d * 2.0 * np.pi / T + phi))


# =====================================================================
# Filename utility
# =====================================================================

def safe_num(v: float, decimals: int = 1) -> str:
    """Float → filename-safe string  (e.g. -1.5 → 'm1p5')."""
    return f"{v:.{decimals}f}".replace("-", "m").replace(".", "p")


# =====================================================================
# File discovery
# =====================================================================

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
# Sine fitting
# =====================================================================

def fit_sine_vs_lgap(
    intensity: np.ndarray,
    lgap_mm:   np.ndarray,
    init_val:  Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit A·(1+sin(2π·d/T+φ)) vs Lgap for each screen-y column.

    Iterates in screen-y order and propagates the previous successful fit as
    the initial guess → ensures smooth (continuous) period/phase curves.
    Phase is constrained to [0, 2π) via curve_fit bounds.

    Parameters
    ----------
    intensity : (n_lgap, n_y)
    lgap_mm   : (n_lgap,) in mm

    Returns
    -------
    prm_arr, err_arr : (n_y, 3)   columns: [amplitude, period_mm, phase_rad]
    """
    if init_val is None:
        init_val = [float(np.nanmean(intensity)), 115.0, np.pi]

    n_y     = intensity.shape[1]
    prm_arr = np.full((n_y, 3), np.nan)
    err_arr = np.full((n_y, 3), np.nan)
    current_p0 = list(init_val)

    for iy in range(n_y):
        p0 = [
            max(current_p0[I_AMP],    1e-30),
            max(current_p0[I_PERIOD], 1e-30),
            current_p0[I_PHASE],          # phase is free; no clipping needed
        ]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    sin_model, lgap_mm, intensity[:, iy],
                    p0=p0, bounds=_FIT_BOUNDS, maxfev=10000,
                )
            # Phase unwrapping: bring the result close to the previous value
            # by removing the nearest 2π multiple.  This prevents sudden ±2π
            # jumps while allowing the phase to drift freely beyond [0, 2π].
            dphi = popt[I_PHASE] - current_p0[I_PHASE]
            popt[I_PHASE] -= np.round(dphi / (2.0 * np.pi)) * 2.0 * np.pi

            prm_arr[iy] = popt
            if np.ndim(pcov) == 2:
                err_arr[iy] = np.sqrt(np.abs(np.diag(pcov)))
            current_p0 = list(popt)   # soft guide for next screen-y
        except Exception as exc:
            # Keep current_p0 so the next point still has a reasonable guess.
            print(f"\n    [warn] fit failed iy={iy}: {exc}")

    return prm_arr, err_arr


# =====================================================================
# Quadratic extremum extraction
# =====================================================================

def quadratic_extremum(
    x: np.ndarray, y: np.ndarray, kind: str
) -> Tuple[float, bool, Optional[np.ndarray]]:
    """
    Fit a quadratic to (x, y) and return the vertex x-coordinate.

    Parameters
    ----------
    kind : 'max' or 'min'

    Returns
    -------
    x_ext  : float
    fit_ok : bool   True when parabola opens correctly and vertex is in range
    coeffs : (a, b, c) or None
    """
    valid = np.isfinite(y)
    if valid.sum() < 3:
        idx = np.nanargmax(y) if kind == "max" else np.nanargmin(y)
        return float(x[idx]), False, None

    coeffs = np.polyfit(x[valid], y[valid], 2)
    a, b, _ = coeffs
    x_vertex = -b / (2.0 * a)

    correct_dir     = (a < 0 and kind == "max") or (a > 0 and kind == "min")
    vertex_in_range = float(x[valid][0]) <= x_vertex <= float(x[valid][-1])

    if correct_dir and vertex_in_range:
        return x_vertex, True, coeffs

    idx = np.nanargmax(y) if kind == "max" else np.nanargmin(y)
    return float(x[idx]), False, coeffs


# =====================================================================
# Per-condition plot
# =====================================================================

def plot_period_phase(
    screen_y_m:   np.ndarray,
    prm_arr:      np.ndarray,
    err_arr:      np.ndarray,
    beam_y_mm:    float,
    beam_yp_mrad: float,
    save_dir:     str,
    slice_name:   str,
    slice_label:  str,
) -> Tuple[float, float]:
    """
    Plot period and phase vs screen-y with quadratic fits and extremum markers.

    Returns
    -------
    (y_max_period_mm, y_min_phase_mm)
    """
    y_mm         = screen_y_m * 1e3
    period       = prm_arr[:, I_PERIOD]
    phase_raw    = prm_arr[:, I_PHASE]          # unwrapped; used for quadratic fit
    phase_disp   = phase_raw % (2.0 * np.pi)   # wrapped to [0, 2π] for display
    e_per        = err_arr[:, I_PERIOD]
    e_phi        = err_arr[:, I_PHASE]

    y_maxper, per_ok, per_coeffs = quadratic_extremum(y_mm, period,    "max")
    y_minphi, phi_ok, phi_coeffs = quadratic_extremum(y_mm, phase_raw, "min")

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    fig.suptitle(
        f"BeamY={beam_y_mm:.1f} mm, BeamYP={beam_yp_mrad:.1f} mrad"
        f"  [{slice_label}]",
        fontsize=11,
    )

    # --- Period panel ---
    axes[0].errorbar(y_mm, period, yerr=e_per,
                     fmt="-o", markersize=2, linewidth=0.8, label="fitted period")
    if per_ok and per_coeffs is not None:
        yq = np.linspace(y_mm[0], y_mm[-1], 300)
        axes[0].plot(yq, np.polyval(per_coeffs, yq), "--", linewidth=0.8,
                     label="quadratic fit")
    axes[0].axvline(y_maxper, color="red", linestyle=":", linewidth=1.0,
                    label=f"max at {y_maxper:.2f} mm")
    axes[0].set_ylabel("Fitted Period [mm]")
    axes[0].set_title("Oscillation Period vs Screen Y")
    axes[0].legend(fontsize=8)

    # --- Phase panel ---
    # Display wrapped phase [0, 2π] for readability.
    # Quadratic fit and extremum marker use the unwrapped phase internally.
    axes[1].errorbar(y_mm, phase_disp, yerr=e_phi,
                     fmt="-o", markersize=2, linewidth=0.8,
                     color="tab:orange", label="fitted phase (wrapped)")
    if phi_ok and phi_coeffs is not None:
        yq = np.linspace(y_mm[0], y_mm[-1], 300)
        quad_vals = np.polyval(phi_coeffs, yq) % (2.0 * np.pi)
        axes[1].plot(yq, quad_vals, "--", linewidth=0.8,
                     color="tab:red", label="quadratic fit (wrapped)")
    axes[1].axvline(y_minphi, color="red", linestyle=":", linewidth=1.0,
                    label=f"min at {y_minphi:.2f} mm")
    axes[1].set_ylabel("Fitted Phase [rad]")
    axes[1].set_xlabel("Screen Y [mm]")
    axes[1].set_title("Oscillation Phase vs Screen Y")
    axes[1].set_ylim(0.0, 2.0 * np.pi)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fname = (
        f"PeriodPhase"
        f"_BeamY{safe_num(beam_y_mm)}"
        f"_BeamYP{safe_num(beam_yp_mrad)}"
        f"_{slice_name}.png"
    )
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)

    return y_maxper, y_minphi


# =====================================================================
# Parallel worker  (top-level function required for pickling)
# =====================================================================

def _process_condition(args: tuple) -> Tuple[float, float, Optional[Dict]]:
    """
    Process one (beam_y, beam_yp) condition for all Lgap slices.

    Returns
    -------
    (beam_y_mm, beam_yp_mrad, results)
    results : dict  slice_name → [by, byp, y_maxper, y_minphi, period_max, phase_min]
              None if no data files were found.
    """
    (beam_y_mm, beam_yp_mrad,
     data_dir, save_dir,
     screen_y_mins, screen_y_pts,
     slices) = args

    intensity, screen_y = load_condition(
        data_dir, beam_y_mm, beam_yp_mrad, screen_y_mins, screen_y_pts
    )
    if intensity is None:
        return beam_y_mm, beam_yp_mrad, None

    y_mm    = screen_y * 1e3
    results: Dict[str, list] = {}

    for sl in slices:
        intensity_sl = intensity[sl.idx_start:sl.idx_end, :]

        prm_arr, err_arr = fit_sine_vs_lgap(intensity_sl, sl.lgap_mm)

        y_max_period, y_min_phase = plot_period_phase(
            screen_y, prm_arr, err_arr,
            beam_y_mm, beam_yp_mrad,
            save_dir, sl.name, sl.label,
        )

        period  = prm_arr[:, I_PERIOD]
        phase   = prm_arr[:, I_PHASE]
        idx_max = np.nanargmin(np.abs(y_mm - y_max_period))
        idx_min = np.nanargmin(np.abs(y_mm - y_min_phase))

        results[sl.name] = [
            beam_y_mm, beam_yp_mrad,
            y_max_period, y_min_phase,
            float(period[idx_max]), float(phase[idx_min]),
        ]

    return beam_y_mm, beam_yp_mrad, results


# =====================================================================
# Main analysis loop
# =====================================================================

def run_analysis(cfg: Config) -> Tuple[Dict[str, np.ndarray], List[LgapSlice]]:
    """
    Run the full analysis for all conditions and Lgap slices in parallel.

    Returns
    -------
    all_summaries : dict  slice_name → ndarray (n_conditions, 6)
    slices        : list of LgapSlice
    """
    os.makedirs(cfg.save_dir, exist_ok=True)

    slices     = make_lgap_slices(cfg.lgap_array)
    conditions = get_all_conditions(cfg.data_dir)

    print(
        f"Found {len(conditions)} conditions, {len(slices)} Lgap slices, "
        f"{cfg.n_workers} workers."
    )

    per_slice_rows: Dict[str, List[list]] = {sl.name: [] for sl in slices}

    worker_args = [
        (by, byp,
         cfg.data_dir, cfg.save_dir,
         cfg.screen_y_mins, cfg.screen_y_pts,
         slices)
        for by, byp in conditions
    ]

    with ProcessPoolExecutor(max_workers=cfg.n_workers) as pool:
        futures = {pool.submit(_process_condition, a): a[:2] for a in worker_args}
        for future in as_completed(futures):
            by, byp = futures[future]
            try:
                _, _, results = future.result()
                if results is None:
                    print(f"  [SKIP]  BeamY={by:.1f} BeamYP={byp:.1f}: no data")
                    continue
                for sname, row in results.items():
                    per_slice_rows[sname].append(row)
                print(f"  [done]  BeamY={by:.1f} mm, BeamYP={byp:.1f} mrad")
            except Exception as exc:
                print(f"  [ERROR] BeamY={by:.1f} BeamYP={byp:.1f}: {exc}")

    # Convert to arrays and save per-slice summaries
    all_summaries: Dict[str, np.ndarray] = {}
    for sl in slices:
        rows = per_slice_rows[sl.name]
        if not rows:
            continue
        arr = np.array(rows)
        np.savetxt(
            os.path.join(cfg.save_dir, f"summary_{sl.name}.txt"),
            arr,
            header=(
                f"Lgap slice: {sl.label}  (idx {sl.idx_start}:{sl.idx_end})\n"
                "beam_y_mm  beam_yp_mrad  y_maxPeriod_mm  y_minPhase_mm"
                "  period_max_mm  phase_min_rad"
            ),
            fmt="%.4f",
        )
        all_summaries[sl.name] = arr
        print(f"  Summary [{sl.name}] saved ({len(rows)} conditions).")

    return all_summaries, slices


# =====================================================================
# Summary plots
# =====================================================================

def _plot_1d_summary(
    summary:     np.ndarray,
    slice_label: str,
    save_dir:    str,
    slice_name:  str,
) -> None:
    """1D line plots: y_maxperiod and y_minphase vs beam_yp (and vs beam_y)."""
    beam_y_vals  = np.unique(summary[:, I_BY])
    beam_yp_vals = np.unique(summary[:, I_BYP])

    for by in beam_y_vals:
        mask = np.isclose(summary[:, I_BY], by)
        sub  = summary[mask][np.argsort(summary[mask][:, I_BYP])]

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle(f"BeamY = {by:.1f} mm  [{slice_label}]", fontsize=12)
        axes[0].plot(sub[:, I_BYP], sub[:, I_Y_MAXPER], "-o", markersize=4)
        axes[0].set_ylabel("y at max Period [mm]")
        axes[0].set_title("Screen Y of max Oscillation Period")
        axes[1].plot(sub[:, I_BYP], sub[:, I_Y_MINPHI], "-o", markersize=4,
                     color="tab:orange")
        axes[1].set_ylabel("y at min Phase [mm]")
        axes[1].set_xlabel("BeamYP [mrad]")
        axes[1].set_title("Screen Y of min Oscillation Phase")
        plt.tight_layout()
        fig.savefig(
            os.path.join(
                save_dir,
                f"Summary_vs_BeamYP_BeamY{safe_num(by)}_{slice_name}.png",
            ),
            dpi=150,
        )
        plt.close(fig)

    for byp in beam_yp_vals:
        mask = np.isclose(summary[:, I_BYP], byp)
        sub  = summary[mask][np.argsort(summary[mask][:, I_BY])]

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle(f"BeamYP = {byp:.1f} mrad  [{slice_label}]", fontsize=12)
        axes[0].plot(sub[:, I_BY], sub[:, I_Y_MAXPER], "-o", markersize=4)
        axes[0].set_ylabel("y at max Period [mm]")
        axes[0].set_title("Screen Y of max Oscillation Period")
        axes[1].plot(sub[:, I_BY], sub[:, I_Y_MINPHI], "-o", markersize=4,
                     color="tab:orange")
        axes[1].set_ylabel("y at min Phase [mm]")
        axes[1].set_xlabel("BeamY [mm]")
        axes[1].set_title("Screen Y of min Oscillation Phase")
        plt.tight_layout()
        fig.savefig(
            os.path.join(
                save_dir,
                f"Summary_vs_BeamY_BeamYP{safe_num(byp)}_{slice_name}.png",
            ),
            dpi=150,
        )
        plt.close(fig)


def _plot_2d_summary(
    summary:     np.ndarray,
    slice_label: str,
    save_dir:    str,
    slice_name:  str,
) -> None:
    """2D colormap on (beam_y, beam_yp) grid."""
    beam_y_vals  = np.unique(summary[:, I_BY])
    beam_yp_vals = np.unique(summary[:, I_BYP])

    grid_ymp = np.full((len(beam_y_vals), len(beam_yp_vals)), np.nan)
    grid_phi = np.full((len(beam_y_vals), len(beam_yp_vals)), np.nan)
    for row in summary:
        iy  = np.argmin(np.abs(beam_y_vals  - row[I_BY]))
        iyp = np.argmin(np.abs(beam_yp_vals - row[I_BYP]))
        grid_ymp[iy, iyp] = row[I_Y_MAXPER]
        grid_phi[iy, iyp] = row[I_Y_MINPHI]

    extent = [beam_yp_vals[0], beam_yp_vals[-1], beam_y_vals[0], beam_y_vals[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"[{slice_label}]", fontsize=12)

    im0 = axes[0].imshow(grid_ymp, aspect="auto", origin="lower", extent=extent)
    axes[0].set_xlabel("BeamYP [mrad]")
    axes[0].set_ylabel("BeamY [mm]")
    axes[0].set_title("Screen Y at max Period [mm]")
    plt.colorbar(im0, ax=axes[0], label="y [mm]")

    im1 = axes[1].imshow(grid_phi, aspect="auto", origin="lower", extent=extent)
    axes[1].set_xlabel("BeamYP [mrad]")
    axes[1].set_ylabel("BeamY [mm]")
    axes[1].set_title("Screen Y at min Phase [mm]")
    plt.colorbar(im1, ax=axes[1], label="y [mm]")

    plt.tight_layout()
    fig.savefig(
        os.path.join(save_dir, f"Summary_2D_{slice_name}.png"), dpi=150
    )
    plt.close(fig)


def plot_all_summaries(
    all_summaries: Dict[str, np.ndarray],
    slices:        List[LgapSlice],
    save_dir:      str,
) -> None:
    """Generate all 1D and 2D summary plots for every Lgap slice."""
    label_map = {sl.name: sl.label for sl in slices}
    for sname, summary in all_summaries.items():
        if len(summary) < 2:
            continue
        label = label_map.get(sname, sname)
        _plot_1d_summary(summary, label, save_dir, sname)
        _plot_2d_summary(summary, label, save_dir, sname)
    print("Summary plots saved.")


# =====================================================================
# Default configuration and entry point
# =====================================================================

DEFAULT_CONFIG = Config(
    data_dir      = "/home/nishi/SRW/env/python/srwpy/examples/data_URI/",
    save_dir      = "/home/nishi/SRW/env/python/srwpy/examples/Oscillation/yy_yp_ana/",
    lgap_array    = np.linspace(0.860, 1.684, 825),     # metres
    screen_y_mins = np.arange(-20.0, 19.0, 2.0) * 1e-3,
    screen_y_pts  = 51,
)

if __name__ == "__main__":
    all_summaries, slices = run_analysis(DEFAULT_CONFIG)
    if all_summaries:
        plot_all_summaries(all_summaries, slices, DEFAULT_CONFIG.save_dir)
