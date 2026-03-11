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
Lgap = 0.7325 # gap = 860 - 1685 mm in experiment. fringe range = 127.5, Lgap = 732.5 ~ 1557.5
#Lgap = 1.5575 # downstream limit
mag1, mag2 = magFld, magFld
magFldCntUp = SRWLMagFldC(
    _arMagFld = [magFld, magFld],
    _arXc = array('d', [xcID, xcID]),
    _arYc = array('d', [ycID, ycID]),
    _arZc = array('d', [zcID, zcID + L_und + Lgap]),
)

Lgap = 1.5575 # downstream limit
magFldCntDown = SRWLMagFldC(
    _arMagFld = [magFld, magFld],
    _arXc = array('d', [xcID, xcID]),
    _arYc = array('d', [ycID, ycID]),
    _arZc = array('d', [zcID, zcID + L_und + Lgap]),
)

partTrajLstDown = []
partTrajLstUp = []
yp_shift_list = np.linspace(-5.e-4, 0.e-5, 6)
yp_shift_list = np.array([-.2e-3])
for yp in yp_shift_list:
    ## UPSTREAM
    part = SRWLParticle()
    part.x = 0. #Initial Transverse Coordinates (initial Longitudinal Coordinate will be defined later on) [m]
    part.y = 1.5e-3
    part.z = zcID - 0.5*magFldCntUp.arMagFld[0].rz  
    part.xp = 0. #Initial Transverse Velocities
    part.yp = 0.
    part.gamma = 0.195/0.51099890221e-03 #Relative Energy

    part.yp = yp

    #  *----------------------*
    partTraj = SRWLPrtTrj()
    partTraj.partInitCond = part
    partTraj.allocate(npTraj, True)
    partTraj.ctStart = 0 #Start Time for the calculation
    #partTraj.ctEnd = magFldCnt.arMagFld[0].rz  #End Time
    partTraj.ctEnd = magFldCntUp.arMagFld[0].rz + magFldCntUp.arMagFld[1].rz + Lgap #End Time

    srwl.CalcPartTraj(partTraj, magFldCntUp, arPrecPar)
    partTrajLstUp.append(partTraj)

    ## DOWNSTREAM
    part = SRWLParticle()
    part.x = 0. #Initial Transverse Coordinates (initial Longitudinal Coordinate will be defined later on) [m]
    part.y = 1.5e-3
    part.z = zcID - 0.5*magFldCntDown.arMagFld[0].rz  
    part.xp = 0. #Initial Transverse Velocities
    part.yp = 0.
    part.gamma = 0.195/0.51099890221e-03 #Relative Energy

    part.yp = yp

    #  *----------------------*
    partTraj = SRWLPrtTrj()
    partTraj.partInitCond = part
    partTraj.allocate(npTraj, True)
    partTraj.ctStart = 0 #Start Time for the calculation
    #partTraj.ctEnd = magFldCnt.arMagFld[0].rz  #End Time
    partTraj.ctEnd = magFldCntDown.arMagFld[0].rz + magFldCntDown.arMagFld[1].rz + Lgap #End Time

    srwl.CalcPartTraj(partTraj, magFldCntDown, arPrecPar)
    partTrajLstDown.append(partTraj)

print("   Trajectory calculation completed.")

#**********************Plotting results
print('   Plotting the results (blocks script execution; close any graph windows to proceed) ... ', end='')
ctMeshUP = np.linspace(partTrajLstUp[0].ctStart, partTrajLstUp[0].ctEnd, partTrajLstUp[0].np)
ctMeshDOWN = np.linspace(partTrajLstDown[0].ctStart, partTrajLstDown[0].ctEnd, partTrajLstDown[0].np)

for partTraj in partTrajLstUp:
    for i in range(partTraj.np):
        partTraj.arX[i] *= 1000
        partTraj.arY[i] *= 1000
for partTraj in partTrajLstDown:
    for i in range(partTraj.np):
        partTraj.arX[i] *= 1000
        partTraj.arY[i] *= 1000


arXUp  = [partTrajLstUp[i].arX for i in range(len(partTrajLstUp))]
arYUp = [partTrajLstUp[i].arY for i in range(len(partTrajLstUp))]
arByUp = [partTrajLstUp[i].arBy for i in range(len(partTrajLstUp))]
arBzUp = [partTrajLstUp[i].arBz for i in range(len(partTrajLstUp))]

arXDown  = [partTrajLstDown[i].arX for i in range(len(partTrajLstDown))]
arYDown = [partTrajLstDown[i].arY for i in range(len(partTrajLstDown))]
arByDown = [partTrajLstDown[i].arBy for i in range(len(partTrajLstDown))]
arBzDown = [partTrajLstDown[i].arBz for i in range(len(partTrajLstDown))]

import matplotlib
norm = matplotlib.colors.Normalize(vmin=yp_shift_list.min(), vmax=yp_shift_list.max())
cmap = matplotlib.cm.viridis

line_color = cmap(norm(yp_shift_list[0]))

fig= plt.figure(figsize=(8,6))
ax = fig.subplots(2,1)

for i, partTraj in enumerate(partTrajLstUp):
    line_color = cmap(norm(yp_shift_list[i]))
    ax[0].plot(ctMeshUP, partTraj.arY, color = line_color, linestyle="-", marker="None")
for i, partTraj in enumerate(partTrajLstDown):
    line_color = cmap(norm(yp_shift_list[i]))
    ax[0].plot(ctMeshDOWN, partTraj.arY, color = line_color, linestyle="--", marker="None")
ax[1].set_xlabel('ct [m]')
ax[0].set_ylabel('Vertical Position  [mm]')
ax[0].set_title('Electron Trajectories in Vertical Plane')
ax[0].set_xlim(ctMeshDOWN[0], ctMeshDOWN[-1])
ax[1].plot([0.06375, 0.06375 + 0.520], [0, 0], color='red', linestyle='-', label='UND1')
ax[1].plot([0.06375, 0.06375 + 0.520], [1, 1], color='red', linestyle='--', label='UND1')
ax[1].plot([0.06375 +0.520 +0.7325,        0.06375 + 0.520 + 0.7325 + 0.520], [0, 0], color='blue', linestyle='-', label='UND2(up lim)')
ax[1].plot([0.06375 +0.520 +0.7325 +0.825, 0.06375 + 0.520 + 0.7325 + 0.825 + 0.520], [1, 1], color='blue', linestyle='--', label='UND2(down lim)')
ax[1].set_xlim(ctMeshDOWN[0], ctMeshDOWN[-1])
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Initial offset y ')
#ax[1].legend()
plt.show()

#uti_plot1d(partTraj.arX, ctMesh, ['ct [m]', 'Horizontal Position [mm]'])
#uti_plot1d(partTraj.arY, ctMesh, ['ct [m]', 'Vertical Position [mm]'])
#uti_plot1d(partTraj.arBy, ctMesh, ['ct [m]', 'Vertical Magnetic Field [T]'])
#uti_plot1d(partTraj.arBz, ctMesh, ['ct [m]', 'Longitudinal Magnetic Field [T]'])
#uti_plot_show() #show all graphs (and block execution)
print('done')

