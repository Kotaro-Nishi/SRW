import numpy as np
import matplotlib.pyplot as plt

import matplotlib


try: #OC15112022
    import sys
    sys.path.append('../')
    from srwlib import *
    from uti_plot import *
except:
    from srwpy.srwlib import *
    from srwpy.uti_plot import *


#**********************Magnetic Field from Text Files:
DirPath = "/home/nishi/Mainz/mainz/BeamEnergyCalib/2024Calib/MagField/"
B_z = np.loadtxt(DirPath+"Bz_profile.txt", dtype='float')
B_y = np.loadtxt(DirPath+"By_profile.txt", dtype='float')
z = np.loadtxt(DirPath+"z_positions.txt", dtype='float')
y = np.loadtxt(DirPath+"y_positions.txt", dtype='float')

ny, nz = B_y.shape
nx = 1
Bx_arr = array('d', [0.]*(nx*ny*nz))
By_arr = array('d', [0.]*(nx*ny*nz))
Bz_arr = array('d', [0.]*(nx*ny*nz))
for iz in range(nz):
    for iy in range(ny):
        for ix in range(nx):
            idx = ix + iy*nx + iz*nx*ny
            By_arr[idx] = B_y[iy][iz]*1e-3  #Convert from mT to T
            Bz_arr[idx] = B_z[iy][iz]*1e-3  #Convert from mT to T

magFld = SRWLMagFld3D( Bx_arr, By_arr, Bz_arr, nx, ny, nz, 1e-3, y[-1] - y[0], z[-1] - z[0], 1 )

#**********************Configuration Parameters:
xcID = 0 #Transverse Coordinates of ID Center [m]
ycID = 0
zcID = 0 #Longitudinal Coordinate of ID Center [m]

npTraj = 10001 #Number of Points for Trajectory calculation
fieldInterpMeth = 4 #2 #Magnetic Field Interpolation Method, to be entered into 3D field structures below (to be used e.g. for trajectory calculation):
#1- bi-linear (3D), 2- bi-quadratic (3D), 3- bi-cubic (3D), 4- 1D cubic spline (longitudinal) + 2D bi-cubic

#**********************Defining Magnetic Field Structure:
L_und = z[-1] - z[0]
Lgap = 0.7325 # gap = 860 - 1685 mm in experiment. fringe range = 127.5, Lgap = 732.5 ~ 1557.5
#Lgap = 1.5575 # downstream limit
mag1, mag2 = magFld, magFld
magFldCnt = SRWLMagFldC(
    _arMagFld = [magFld, magFld],
    _arXc = array('d', [xcID, xcID]),
    _arYc = array('d', [ycID, ycID]),
    _arZc = array('d', [zcID, zcID + L_und + Lgap]),
)





#**********************
#**********************Defining Study Parameters:
file_id = "0011"
Lgap_config = [0.7325, 1.5575, 0.001] #m
Screen_x_config = [-3.e-3, 3.e-3, 3] #m
Screen_y_config = [-2.7e-3, -1.7e-3, 31] #m
beam_x, beam_y, beam_xp, beam_yp = 0., 1.5e-3, 0., -0.2e-3
lambda_L_obs = 404. #nm
screen_z = 14. #m




## process
Lgap_arange    = np.arange(Lgap_config[0], Lgap_config[1], Lgap_config[2])
screen_x_range = np.linspace(Screen_x_config[0], Screen_x_config[1], Screen_x_config[2])
screen_y_range = np.linspace(Screen_y_config[0], Screen_y_config[1], Screen_y_config[2])

#**********************Calculating Radiation Electric Field Spectrum
elecBeam = SRWLPartBeam()
elecBeam.Iavg = 1e-9 #Average Current [A]
elecBeam.partStatMom1.x = beam_x
elecBeam.partStatMom1.y = beam_y
elecBeam.partStatMom1.z = zcID - 0.5*magFldCnt.arMagFld[0].rz  #Initial Longitudinal Coordinate (set before the ID)
elecBeam.partStatMom1.xp = beam_xp
elecBeam.partStatMom1.yp = beam_yp
elecBeam.partStatMom1.gamma = .195/0.51099890221e-03 #Relative E

Intensity_list = []
for Lgap in Lgap_arange:
    magFldCnt.arZc[1] = zcID + L_und + Lgap

    wfr = SRWLWfr()
    wfr.allocate(_ne=1, _nx=11, _ny=11) #Numbers of points vs Photon Energy
    wfr.mesh.zStart = magFldCnt.arMagFld[0].rz + magFldCnt.arMagFld[1].rz + Lgap + screen_z #Longitudinal Position [m] where Electric Field will be calculated

    wfr.mesh.eStart = 1239.841984/lambda_L_obs #Initial Photon Energy [eV]
    wfr.mesh.eFin   = 1239.841984/lambda_L_obs #Final Photon Energy [eV]

    wfr.mesh.xStart = Screen_x_config[0] #Initial Horizontal Position [m]
    wfr.mesh.xFin   = Screen_x_config[1] #Final Horizontal Position [m]
    wfr.mesh.nx     = Screen_x_config[2] #Number of points vs Horizontal Position

    wfr.mesh.yStart = Screen_y_config[0] #Initial Vertical Position [m]
    wfr.mesh.yFin   = Screen_y_config[1] #Final Vertical Position [m]
    wfr.mesh.ny     = Screen_y_config[2] #Number of points vs Vertical Position

    wfr.partBeam = elecBeam
    arPrecPar = [0, 0.01, 0, 0, npTraj, 1, 0]

    srwl.CalcElecFieldSR(wfr, 0, magFldCnt, arPrecPar)
    arI = array('f', [0]*wfr.mesh.nx * wfr.mesh.ny * wfr.mesh.ne)
    srwl.CalcIntFromElecField(arI, wfr, 6, 0, 3, wfr.mesh.eStart, 0., 0.)

    arI = np.array(arI).reshape(wfr.mesh.ny, wfr.mesh.nx)
    arI = arI.T  # to make y first index
    Intensity_list.append(arI)

ar_Intensity = np.array(Intensity_list)
print(ar_Intensity.shape)


#***********************Plotting Results:
#norm = matplotlib.colors.Normalize(vmin=Screen_y_config[0], vmax=Screen_y_config[1])
#cmap = matplotlib.cm.viridis
#fig, ax = plt.subplots(figsize=(8,6))
#
#driving = (Lgap_arange - Lgap_arange[0])*1e3
#for y_idx, y_val in enumerate(screen_y_range):
#    line_color = cmap(norm(y_val))
#    ax.plot(driving,ar_Intensity[:, len(screen_x_range)//2, y_idx], color=line_color, linestyle="-", marker="")
#
#sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
#sm.set_array([])
#cbar = plt.colorbar(sm, ax=ax, label='Screen y [mm]')
#
#plt.xlabel('driving range [mm]')
#plt.ylabel('Intensity [a.u.]')
#plt.show()

#***********************Saving Results:
DirPath = "/home/nishi/SRW/env/python/srwpy/examples/Oscillation/"
ar_Intensity_File = ar_Intensity[:, len(screen_x_range)//2, :]
print(ar_Intensity_File.shape)
np.savetxt(DirPath+"Oscillation"+file_id+".txt", ar_Intensity_File)