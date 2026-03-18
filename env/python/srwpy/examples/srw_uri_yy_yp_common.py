"""
srw_uri_yy_yp_common.py
=======================

Shared parameters, physics constants, and MagFieldManager for the yy_yp
scan suite.  Imported by:

  - SRWLIB_URI_integrated_scan_yy_yp_makedata.py
  - SRWLIB_URI_integrated_scan_yy_yp_tracking.py

Keeping all scan/physics parameters here ensures makedata and tracking
always use identical conditions.
"""

import numpy as np

try:
    import sys
    sys.path.append('../')
    from srwlib import *
except Exception:
    from srwpy.srwlib import *


# =====================================================================
# Shared paths
# =====================================================================

MAG_DIR  = "/home/nishi/Mainz/mainz/BeamEnergyCalib/2024Calib/MagField/"
DATA_DIR = "/home/nishi/SRW/env/python/srwpy/examples/data_URI/"


# =====================================================================
# Scan parameters  (single source of truth for makedata and tracking)
# =====================================================================

LGAP_ARRAY   = np.arange(0.860, 1.685, 0.001)      # m,   825 points
BEAM_Y_LOOP  = np.arange( 0.0, 21.0, 1.0) * 1e-3  # m,   0 .. 20 mm  (21 pts)
BEAM_YP_LOOP = np.arange(-2.0,  2.0, 0.1) * 1e-3  # rad, -2 .. 1.9 mrad (40 pts)


# =====================================================================
# Shared physics constants
# =====================================================================

BEAM_ENERGY = 0.195                          # GeV
LAMBDA_OBS  = 404.0                          # nm  (observation wavelength)
L_UND       = 0.520                          # m   (single undulator length)
XCID        = 0.0                            # m   (undulator centre x)
YCID        = 0.0                            # m   (undulator centre y)
ZCID        = 0.0                            # m   (first undulator centre z)
GAMMA       = BEAM_ENERGY / 0.51099890221e-03  # relativistic gamma


# =====================================================================
# MagFieldManager
# =====================================================================

class MagFieldManager:
    """
    Load the 3D magnetic field from disk once and build Lgap-dependent
    SRWLMagFldC (double-undulator) containers on demand (cached).

    Usage
    -----
    mag_mgr = MagFieldManager()
    fld_cnt = mag_mgr.get_fld_cnt(0.860)   # SRWLMagFldC for Lgap = 0.860 m
    z0      = mag_mgr.z_start              # initial z for the beam particle
    """

    def __init__(self, mag_dir: str = MAG_DIR):
        self.mag_dir = mag_dir
        self._fld3d: "SRWLMagFld3D" = self._load_fld3d()
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_fld_cnt(self, Lgap: float) -> "SRWLMagFldC":
        """Return (cached) SRWLMagFldC for the double-undulator at *Lgap* [m]."""
        key = round(float(Lgap), 6)
        if key not in self._cache:
            self._cache[key] = self._build_fld_cnt(Lgap)
        return self._cache[key]

    @property
    def z_start(self) -> float:
        """Initial longitudinal coordinate of the beam (upstream of first ID) [m]."""
        return ZCID - 0.5 * self._fld3d.rz - 0.1

    def z_end(self, Lgap: float) -> float:
        """Downstream end of the 2nd undulator + 0.1 m margin [m].

        Matches the integration end used by makedata's _calculate_intensity.
        """
        return ZCID + L_UND + float(Lgap) + 0.5 * self._fld3d.rz + 0.1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_fld3d(self) -> "SRWLMagFld3D":
        """Read field profile files and construct a SRWLMagFld3D object."""
        B_z = np.loadtxt(self.mag_dir + "Bz_profile.txt", dtype="float")
        B_y = np.loadtxt(self.mag_dir + "By_profile.txt", dtype="float")
        z   = np.loadtxt(self.mag_dir + "z_positions.txt", dtype="float")
        y   = np.loadtxt(self.mag_dir + "y_positions.txt", dtype="float")

        ny, nz = B_y.shape
        nx = 1
        Bx_arr = array('d', [0.0] * (nx * ny * nz))
        By_arr = array('d', [0.0] * (nx * ny * nz))
        Bz_arr = array('d', [0.0] * (nx * ny * nz))

        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    idx = ix + iy * nx + iz * nx * ny
                    By_arr[idx] = float(B_y[iy, iz]) * 1e-3   # mT → T
                    Bz_arr[idx] = float(B_z[iy, iz]) * 1e-3   # mT → T

        return SRWLMagFld3D(
            Bx_arr, By_arr, Bz_arr,
            nx, ny, nz,
            1e-3,                        # rx  (transverse x range, 1 point)
            float(y[-1] - y[0]),         # ry  (transverse y range)
            float(z[-1] - z[0]),         # rz  (longitudinal range)
            1,                           # nRep
        )

    def _build_fld_cnt(self, Lgap: float) -> "SRWLMagFldC":
        """Assemble a double-undulator SRWLMagFldC for the given *Lgap* [m]."""
        fld = self._fld3d
        return SRWLMagFldC(
            _arMagFld=[fld, fld],
            _arXc=array('d', [XCID, XCID]),
            _arYc=array('d', [YCID, YCID]),
            _arZc=array('d', [ZCID, ZCID + L_UND + Lgap]),
        )
