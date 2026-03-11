import numpy as np
import matplotlib.pyplot as plt

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

#**********************Input Electron Beam Parameters:
numPer = 40 #Number of ID Periods (without counting for terminations)
xcID = 0 #Transverse Coordinates of ID Center [m]
ycID = 0
zcID = 0 #Longitudinal Coordinate of ID Center [m]


npTraj = 10001 #Number of Points for Trajectory calculation
fieldInterpMeth = 4 #2 #Magnetic Field Interpolation Method, to be entered into 3D field structures below (to be used e.g. for trajectory calculation):
#1- bi-linear (3D), 2- bi-quadratic (3D), 3- bi-cubic (3D), 4- 1D cubic spline (longitudinal) + 2D bi-cubic
arPrecPar = [1]

#**********************Defining Magnetic Field Structure:
L_und = z[-1] - z[0]
print("L_und=", L_und)
#Lgap = 0.7325 # gap = 860 - 1685 mm in experiment. fringe range = 127.5 ~ 1557.5
Lgap = 1.5575 # downstream limit
mag1, mag2 = magFld, magFld
magFldCnt = SRWLMagFldC(
    _arMagFld = [magFld, magFld],
    _arXc = array('d', [xcID, xcID]),
    _arYc = array('d', [ycID, ycID]),
    _arZc = array('d', [zcID, zcID + L_und + Lgap]),
)

partTrajLstY = []
y_shift_list = np.linspace(1e-3, -1e-3, 11)
for y_pos in y_shift_list:
    part = SRWLParticle()
    part.x = 0. #Initial Transverse Coordinates (initial Longitudinal Coordinate will be defined later on) [m]
    part.y = 0.
    part.z = zcID - 0.5*magFldCnt.arMagFld[0].rz  
    part.xp = 0. #Initial Transverse Velocities
    part.yp = 0.
    part.gamma = 0.195/0.51099890221e-03 #Relative Energy

    part.y = y_pos

    #  *----------------------*
    partTraj = SRWLPrtTrj()
    partTraj.partInitCond = part
    partTraj.allocate(npTraj, True)
    partTraj.ctStart = 0 #Start Time for the calculation
    #partTraj.ctEnd = magFldCnt.arMagFld[0].rz  #End Time
    partTraj.ctEnd = magFldCnt.arMagFld[0].rz + magFldCnt.arMagFld[1].rz + Lgap #End Time

    srwl.CalcPartTraj(partTraj, magFldCnt, arPrecPar)
    partTrajLstY.append(partTraj)
print("   Trajectory calculation completed.")

#**********************Plotting results
print('   Plotting the results (blocks script execution; close any graph windows to proceed) ... ', end='')
ctMesh = np.linspace(partTraj.ctStart, partTraj.ctEnd, partTraj.np)

for partTraj in partTrajLstY:
    for i in range(partTraj.np):
        partTraj.arX[i] *= 1000
        partTraj.arY[i] *= 1000

arX  = [partTrajLstY[i].arX for i in range(len(partTrajLstY))]
arY  = [partTrajLstY[i].arY for i in range(len(partTrajLstY))]
arBy = [partTrajLstY[i].arBy for i in range(len(partTrajLstY))]
arBz = [partTrajLstY[i].arBz for i in range(len(partTrajLstY))]

import matplotlib
norm = matplotlib.colors.Normalize(vmin=y_shift_list.min()*1000, vmax=y_shift_list.max()*1000)
cmap = matplotlib.cm.viridis

line_color = cmap(norm(y_shift_list[0]*1e3))

fig = plt.figure(figsize=(8,6))
ax = fig.subplots(2,1)
partTrajRef = partTrajLstY[len(partTrajLstY)//2]
for i, partTraj in enumerate(partTrajLstY):
    line_color = cmap(norm(y_shift_list[i]*1e3))
    ax[0].plot(ctMesh, partTraj.arX, color = line_color, linestyle="-", marker="None")
    arDy = np.array(partTraj.arX) - np.array(partTrajRef.arX)
    ax[1].plot(ctMesh, arDy, color = line_color, linestyle="-", marker="None")
ax[1].set_xlabel('ct [m]')
ax[0].set_ylabel('Horizontal Position [mm]')
ax[1].set_ylabel('Difference [mm]')
ax[0].set_title('Electron Trajectories in Horizontal Plane')
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Initial offset y [mm]')
plt.show()


fig= plt.figure(figsize=(8,6))
ax = fig.subplots(2,1)

for i, partTraj in enumerate(partTrajLstY):
    line_color = cmap(norm(y_shift_list[i]*1e3))
    ax[0].plot(ctMesh, partTraj.arY, color = line_color, linestyle="-", marker="None")
    y_track = np.array(partTraj.arY) - np.array(partTraj.arY)[0]
    ax[1].plot(ctMesh, y_track, color = line_color, linestyle="-", marker="None")
ax[1].set_xlabel('ct [m]')
ax[0].set_ylabel('Vertical Position  [mm]')
ax[1].set_ylabel('y shift [mm]')
ax[0].set_title('Electron Trajectories in Vertical Plane')
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Initial offset y [mm]')
plt.show()


#uti_plot1d(partTraj.arX, ctMesh, ['ct [m]', 'Horizontal Position [mm]'])
#uti_plot1d(partTraj.arY, ctMesh, ['ct [m]', 'Vertical Position [mm]'])
#uti_plot1d(partTraj.arBy, ctMesh, ['ct [m]', 'Vertical Magnetic Field [T]'])
#uti_plot1d(partTraj.arBz, ctMesh, ['ct [m]', 'Longitudinal Magnetic Field [T]'])
#uti_plot_show() #show all graphs (and block execution)
print('done')

