import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import matplotlib
from scipy.optimize import leastsq
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import scipy.signal as signal
import os

try:
    import sys
    sys.path.append('../')
    from srwlib import *
    from uti_plot import *
except Exception:
    from srwpy.srwlib import *
    from srwpy.uti_plot import *

from srw_uri_yy_yp_common import (
    MAG_DIR, DATA_DIR,
    LGAP_ARRAY, BEAM_Y_LOOP, BEAM_YP_LOOP,
    BEAM_ENERGY, LAMBDA_OBS, ZCID,
    MagFieldManager,
)


# =====================================================================
# Makedata-specific configuration
# =====================================================================

SCREEN_Y_CONFIG_LIST = [[-20.e-3 + i*2.e-3, -18.e-3 + i*2.e-3, 51] for i in range(20)]
SCREEN_Y_ARRAY_LIST  = [np.linspace(cfg[0], cfg[1], cfg[2]) for cfg in SCREEN_Y_CONFIG_LIST]


# =====================================================================
# SRWSimulationEngine  — wavefront intensity calculation
# =====================================================================

class SRWSimulationEngine:
    """
    Compute SR intensity for one (beam_y, beam_yp, screen_y_window) condition
    across all Lgap values and save the result to DATA_DIR.

    Mirrors the class structure of TrajectoryEngine in tracking.py:
    one engine instance per worker process; call run_wfr_simulation_driving()
    to scan the full LGAP_ARRAY.
    """

    def __init__(self, mag_manager: MagFieldManager):
        self.mag_manager = mag_manager
        self.screen_z = 14.0    # m, screen longitudinal distance from ZCID
        self.np_traj  = 50001   # trajectory integration points

        self.base_beam = self._initialize_beam()
        self.base_wfr  = self._initialize_wavefront()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_wfr_simulation_driving(
        self, beam_y: float, beam_yp: float, screen_y: list
    ) -> None:
        """
        Compute intensity vs Lgap for one (beam_y, beam_yp, screen_y_window).
        Saves result to DATA_DIR; skips if the output file already exists.
        """
        filename = self._generate_filename(beam_y, beam_yp, screen_y[0])
        if os.path.exists(filename):
            print(f"File {filename} already exists. Skipping simulation.")
            return

        try:
            arI_lst = [
                self._run_single_shot(float(beam_y), float(beam_yp), float(Lgap), screen_y)
                for Lgap in LGAP_ARRAY
            ]
            np.savetxt(filename, np.array(arI_lst))
        except Exception as exc:
            print(f"Error in simulation: {exc}")
            raise

    # ------------------------------------------------------------------
    # Internal: single Lgap shot
    # ------------------------------------------------------------------

    def _run_single_shot(
        self, beam_y: float, beam_yp: float, Lgap: float, screen_y: list
    ):
        _mag_fld_cnt = self.mag_manager.get_fld_cnt(Lgap)

        _beam = deepcopy(self.base_beam)
        _beam.partStatMom1.y  = beam_y
        _beam.partStatMom1.yp = beam_yp

        _wfr = deepcopy(self.base_wfr)
        _wfr.partBeam    = _beam
        _wfr.mesh.yStart = screen_y[0]
        _wfr.mesh.yFin   = screen_y[1]
        _wfr.mesh.ny     = screen_y[2]

        return self._calculate_intensity(_mag_fld_cnt, _beam, _wfr)

    # ------------------------------------------------------------------
    # Internal: beam / wavefront initialisation
    # ------------------------------------------------------------------

    def _initialize_beam(self) -> "SRWLPartBeam":
        _elecBeam = SRWLPartBeam()
        _elecBeam.Iavg = 1e-9
        _elecBeam.partStatMom1.x     = 0.0
        _elecBeam.partStatMom1.y     = 0.0
        _elecBeam.partStatMom1.z     = self.mag_manager.z_start  # upstream of first ID
        _elecBeam.partStatMom1.xp    = 0.0
        _elecBeam.partStatMom1.yp    = 0.0
        _elecBeam.partStatMom1.gamma = BEAM_ENERGY / 0.51099890221e-03
        return _elecBeam

    def _initialize_wavefront(self) -> "SRWLWfr":
        _wfr = SRWLWfr()
        _wfr.allocate(_ne=1, _nx=1, _ny=51)
        _wfr.mesh.zStart = ZCID + self.screen_z
        _wfr.mesh.eStart = 1239.841984 / LAMBDA_OBS
        _wfr.mesh.eFin   = 1239.841984 / LAMBDA_OBS
        _wfr.mesh.xStart = 0.0
        _wfr.mesh.xFin   = 0.0
        _wfr.mesh.nx     = 1
        _wfr.mesh.yStart = 0.0
        _wfr.mesh.yFin   = 1.0e-3
        _wfr.mesh.ny     = 51
        _wfr.partBeam    = self.base_beam
        return _wfr

    # ------------------------------------------------------------------
    # Internal: intensity calculation
    # ------------------------------------------------------------------

    def _calculate_intensity(self, _mag_fld_cnt, _beam, _wfr):
        z_start_beam = _beam.partStatMom1.z
        z_end_field  = _mag_fld_cnt.arZc[1] + 0.5 * _mag_fld_cnt.arMagFld[1].rz + 0.1
        _arPrecPar = [1, 0.01, z_start_beam, z_end_field, self.np_traj, 1, 0]
        srwl.CalcElecFieldSR(_wfr, 0, _mag_fld_cnt, _arPrecPar)
        arI = array('f', [0] * _wfr.mesh.nx * _wfr.mesh.ny * _wfr.mesh.ne)
        srwl.CalcIntFromElecField(arI, _wfr, 6, 0, 3, _wfr.mesh.eStart, 0.0, 0.0)
        return arI

    # ------------------------------------------------------------------
    # Internal: filename
    # ------------------------------------------------------------------

    def _generate_filename(self, beam_y: float, beam_yp: float, screen_y_min: float) -> str:
        return (
            DATA_DIR
            + f"BeamY_{beam_y*1e3:.3f}mm"
            + f"_BeamYP_{beam_yp*1e3:.3f}mrad"
            + f"_ScreenYmin_{screen_y_min*1e3:.3f}mm.txt"
        )


# =====================================================================
# Parallel worker
# =====================================================================

def init_worker() -> None:
    """Initialise one SRWSimulationEngine per subprocess."""
    global worker_engine
    mag_mgr = MagFieldManager(MAG_DIR)
    worker_engine = SRWSimulationEngine(mag_mgr)


def get_all_tasks() -> list:
    return [
        (beam_y, beam_yp, screen_y_config)
        for beam_y          in BEAM_Y_LOOP
        for beam_yp         in BEAM_YP_LOOP
        for screen_y_config in SCREEN_Y_CONFIG_LIST
    ]


def worker(task: tuple) -> None:
    beam_y, beam_yp, screen_y_config = task
    worker_engine.run_wfr_simulation_driving(float(beam_y), float(beam_yp), screen_y_config)


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    tasks = get_all_tasks()
    with ProcessPoolExecutor(max_workers=20, initializer=init_worker) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks)):
            future.result()
