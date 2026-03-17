import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import matplotlib
from scipy.optimize import leastsq
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import scipy.signal as signal

try: #OC15112022
    import sys
    sys.path.append('../')
    from srwlib import *
    from uti_plot import *
except:
    from srwpy.srwlib import *
    from srwpy.uti_plot import *

# ----- Constants and Configurations -----
DATA_DIR = "/home/nishi/SRW/env/python/srwpy/examples/data_URI/"
LGAP_ARRAY = np.arange(0.860, 1.685, 0.001) # m
SCREEN_Y_CONFIG = [0e-3, +2.0e-3, 51] # m
SCREEN_Y_ARRAY = np.linspace(SCREEN_Y_CONFIG[0], SCREEN_Y_CONFIG[1], SCREEN_Y_CONFIG[2])

SCREEN_Y_CONFIG_LIST  = [ [-20.e-3+i*2.e-3, -18.e-3+i*2.e-3, 51] for i in range(20)]
SCREEN_Y_ARRAY_LIST   = [ np.linspace(config[0], config[1], config[2]) for config in SCREEN_Y_CONFIG_LIST ]

BEAM_Y_LOOP  = np.arange(0.e-3, 21.e-3, 1.e-3)
BEAM_YP_LOOP = np.arange(-2.e-3, 2.e-3, 0.1e-3)

#test benchmark
#SCREEN_Y_CONFIG_LIST  = [ [-20.e-3+i*2.e-3, -18.e-3+i*2.e-3, 51] for i in range(5)]
#BEAM_Y_LOOP  = np.arange(0.e-3, 5.e-3, 1.e-3)
#BEAM_YP_LOOP = np.arange(0.e-3, 0.5e-3, 0.1e-3)
# ------------------------*-------------------------

class MagFieldManager:
    def __init__(self, mag_field_dir):
        self.mag_field_dir = mag_field_dir
        self.mag_fld_cache = {}
        try:
            self.get_mag_field(LGAP_ARRAY[0]) # preload to cache
        except:
            self.get_mag_field(0.860) # preload to cache

    def _get_cache_key(self, Lgap):
        if isinstance(Lgap, (float,int)):
            return f"Lgap_{Lgap:.3f}"
        else:
            return Lgap

    def get_mag_field(self, Lgap):
        key = self._get_cache_key(Lgap)
        if key not in self.mag_fld_cache:
            self.mag_fld_cache[key] = self._configure_double_undulator_fld(Lgap)
        return self.mag_fld_cache[key]
    
    def _configure_double_undulator_fld(self, Lgap=0.860):
        B_z = np.loadtxt(self.mag_field_dir +"Bz_profile.txt", dtype='float')
        B_y = np.loadtxt(self.mag_field_dir+"By_profile.txt", dtype='float')
        z   = np.loadtxt(self.mag_field_dir+"z_positions.txt", dtype='float')
        y   = np.loadtxt(self.mag_field_dir+"y_positions.txt", dtype='float')

        n_y, n_z = B_y.shape
        n_x = 1
        Bx_arr = array('d', [0.]*(n_x*n_y*n_z))
        By_arr = array('d', [0.]*(n_x*n_y*n_z))
        Bz_arr = array('d', [0.]*(n_x*n_y*n_z))
        for iz in range(n_z):
            for iy in range(n_y):
                for ix in range(n_x):
                    idx = ix + iy*n_x + iz*n_x*n_y
                    By_arr[idx] = B_y[iy][iz]*1e-3  #Convert from mT to T
                    Bz_arr[idx] = B_z[iy][iz]*1e-3  #Convert from mT to T 
        magFld = SRWLMagFld3D( Bx_arr, By_arr, Bz_arr, n_x, n_y, n_z, 1e-3, y[-1] - y[0], z[-1] - z[0], 1 )

        self.xcID = 0. #Transverse Coordinates of ID Center [m]
        self.ycID = 0.
        self.zcID = 0. #Longitudinal Coordinate of ID Center [m]
        fieldInterpMeth = 4 #2 #Magnetic Field Interpolation Method, to be entered into 3D field structures below (to be used e.g. for trajectory calculation):
        #1- bi-linear (3D), 2- bi-quadratic (3D), 3- bi-cubic (3D), 4- 1D cubic spline (longitudinal) + 2D bi-cu
        L_und = 0.520
        magFldCnt = SRWLMagFldC(
            _arMagFld = [magFld, magFld],
            _arXc = array('d', [self.xcID, self.xcID]),
            _arYc = array('d', [self.ycID, self.ycID]),
            _arZc = array('d', [self.zcID, self.zcID + L_und + Lgap]),
        )
        return magFldCnt

class SRWSimulationEngine:
    def __init__(self, mag_manager):
        self.mag_manager = mag_manager
        self.zcID = mag_manager.zcID
        self.screen_z = 14. #m
        self.npTraj = 50001 #Number of Points for Trajectory calculation

        self.beam_energy = 0.195 # GeV
        self.lambda_L_obs = 404. # nm

        self.base_beam = self._initialize_beam()
        self.base_wfr = self._initialize_wavefront()

    def run_wfr_simulation_driving(self, beam_y, beam_yp, screen_y):
        filename = self.generate_filename(beam_y, beam_yp, screen_y[0])
        if os.path.exists(filename):
            print(f"File {filename} already exists. Skipping simulation.")
            return np.loadtxt(filename)
        try:
            arI_lst = []
            for Lgap in LGAP_ARRAY:
                arI = self.run_wfr_simulation_single_shot(float(beam_y), float(beam_yp), float(Lgap), screen_y)
                arI_lst.append(arI)
            arI_lst = np.array(arI_lst)
            self.save_raw_intensity(arI_lst, filename)
        except Exception as e:
            print(f"Error occurred while running simulation: {e}")
            raise
        

    def run_wfr_simulation_single_shot(self, beam_y, beam_yp, Lgap, screen_y):
        _mag_fld_cnt = self.mag_manager.get_mag_field(Lgap)

        _beam = deepcopy(self.base_beam)
        _beam.partStatMom1.y = beam_y
        _beam.partStatMom1.yp = beam_yp

        _wfr = deepcopy(self.base_wfr)
        _wfr.partBeam = _beam
        _wfr.mesh.yStart = screen_y[0] 
        _wfr.mesh.yFin   = screen_y[1]
        _wfr.mesh.ny     = screen_y[2]

        arI = self.calculate_intensity(_mag_fld_cnt, _beam, _wfr)
        return arI # extract center x point
    
    def _initialize_beam(self):
        temp_mag = self.mag_manager.get_mag_field(list(self.mag_manager.mag_fld_cache.keys())[0])
        _elecBeam = SRWLPartBeam()
        _elecBeam.Iavg = 1e-9 #Average Current [A]
        _elecBeam.partStatMom1.x = 0.
        _elecBeam.partStatMom1.y = 0.
        _elecBeam.partStatMom1.z = self.zcID - 0.5*temp_mag.arMagFld[0].rz - 0.1   #Initial Longitudinal Coordinate (set before the ID)
        _elecBeam.partStatMom1.xp = 0.
        _elecBeam.partStatMom1.yp = 0.
        _elecBeam.partStatMom1.gamma = self.beam_energy/0.51099890221e-03 #Relative
        return _elecBeam
    
    def _initialize_wavefront(self):
        _wfr = SRWLWfr()
        _wfr.allocate(_ne=1, _nx=1, _ny=51) #Numbers of points vs Photon Energy
        _wfr.mesh.zStart = self.zcID + self.screen_z #Longitudinal Position [m] where Electric Field will be calculated
        _wfr.mesh.eStart = 1239.841984/self.lambda_L_obs #Initial Photon Energy [eV]
        _wfr.mesh.eFin   = 1239.841984/self.lambda_L_obs #Final Photon Energy [eV]
        _wfr.mesh.xStart = 0. #Initial Horizontal Position [m]
        _wfr.mesh.xFin   = 0. #Final Horizontal Position [m]
        _wfr.mesh.nx     = 1. #Number of points vs Horizontal Position
        _wfr.mesh.yStart = 0. #Initial Vertical Position [m]
        _wfr.mesh.yFin   = 1.e-3 #Final Vertical Position [m]
        _wfr.mesh.ny     = 51 #Number of points vs Vertical Position
        _wfr.partBeam = self.base_beam
        return _wfr
    
    def calculate_intensity(self, _mag_fld_cnt, _beam, _wfr):
        _arPrecPar = [1, 0.01, _beam.partStatMom1.z, _mag_fld_cnt.arZc[1] + 0.5*_mag_fld_cnt.arMagFld[1].rz + 0.1, self.npTraj, 1, 0]
        srwl.CalcElecFieldSR(_wfr, 0, _mag_fld_cnt, _arPrecPar)
        arI = array('f', [0]*_wfr.mesh.nx * _wfr.mesh.ny * _wfr.mesh.ne)
        srwl.CalcIntFromElecField(arI, _wfr, 6, 0, 3, _wfr.mesh.eStart, 0., 0.)
        return arI
    
    def generate_filename(self, beam_y, beam_yp, screen_y_min):
        file_dir = "/home/nishi/SRW/env/python/srwpy/examples/data_URI/"
        return file_dir + f"BeamY_{beam_y*1e3:.3f}mm_BeamYP_{beam_yp*1e3:.3f}mrad_ScreenYmin_{screen_y_min*1e3:.3f}mm.txt"
    
    def save_raw_intensity(self, intensity_array, filename):
        np.savetxt(filename, intensity_array)
        return 0
    

def init_worker():
    global worker_engine
    mag_mgr = MagFieldManager(MAG_DIR)
    worker_engine = SRWSimulationEngine(mag_mgr)

def get_all_tasks():
    return list((beam_y, beam_yp, screen_y_config) for beam_y in BEAM_Y_LOOP for beam_yp in BEAM_YP_LOOP for screen_y_config in SCREEN_Y_CONFIG_LIST)

def worker(task):
    beam_y, beam_yp, screen_y_config = task
    arI = worker_engine.run_wfr_simulation_driving(float(beam_y), float(beam_yp), screen_y_config)
    return arI

if __name__ == "__main__":
    tasks = get_all_tasks()
    with ProcessPoolExecutor(max_workers=20, initializer=init_worker) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks)):
            results = future.result()

exit()

mag_mgr = MagFieldManager(MAG_DIR)
engine = SRWSimulationEngine(mag_mgr)

plt.figure(figsize=(10, 5))
arI_lst = []

for beam_y in BEAM_Y_LOOP:
    for beam_yp in BEAM_YP_LOOP:
        for screen_y_config in SCREEN_Y_CONFIG_LIST:
            #print(beam_y, beam_yp, screen_y_config)
            arI = engine.run_wfr_simulation_driving(beam_y, beam_yp, screen_y_config)

exit()