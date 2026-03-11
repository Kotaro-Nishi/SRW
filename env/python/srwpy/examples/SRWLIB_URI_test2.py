#############################################################################
# URI test: Calculating electron trajectory in magnetic field a segmented planar undulator
# v 0.08
#############################################################################
from __future__ import print_function #Python 2.7 compatibility

import sys
sys.path.append('../')
from srwlib import *
from uti_plot import *

import os
import numpy as np
import matplotlib.pyplot as plt 
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.marker'] = 'None'

#**********************Input Parameters
#**********************Folder and File names
strExDataFolderName = 'data_example_02' #example data sub-folder name
strFldOutFileName = 'ex02_res_fld.dat' #file name for output (tabulated) magnetic field data
strTrajOutFileName = 'ex02_res_traj.dat' #file name for output trajectory data

#**********************Initial Conditions for Particle Trajectory Calculation
part = SRWLParticle()
part.x = 0.0001 #Initial Transverse Coordinates (initial Longitudinal Coordinate will be defined later on) [m]
part.y = 0.0001
part.xp = 0 #Initial Transverse Velocities
part.yp = 0
part.gamma = 0.195/0.51099890221e-03 #Relative Energy
part.relE0 = 1 #Electron Rest Mass
part.nq = -1 #Electron Charge

npTraj = 20001 #Number of Points for Trajectory calculation

arPrecPar = [1] #General Precision parameters for Trajectory calculation:
#[0]: integration method No:
    #1- fourth-order Runge-Kutta (precision is driven by number of points)
    #2- fifth-order Runge-Kutta
#[1],[2],[3],[4],[5]: absolute precision values for X[m],X'[rad],Y[m],Y'[rad],Z[m] (yet to be tested!!) - to be taken into account only for R-K fifth order or higher
#[6]: tolerance (default = 1) for R-K fifth order or higher
#[7]: max. number of auto-steps for R-K fifth order or higher (default = 5000)

#**********************Magnetic Field
numSegm = 1 #Number of ID Segments
numPer = 6 #Number of Periods in one Segment (without counting for terminations)
undPer = 0.08 #Period Length [m]
xcID = 0 #Transverse Coordinates of ID Center [m]
ycID = 0
zcID = 0 #Longitudinal Coordinate of ID Center [m]

und = SRWLMagFldU([SRWLMagFldH(1, 'v', 0.135, 0, 1)], undPer, numPer) #Undulator Segment

arZero = array('d', [0]*9)
magFldCnt = SRWLMagFldC([und]) #Container of all Field Elements

#**********************Trajectory structure, where the results will be stored
partTraj = SRWLPrtTrj()
partTraj.partInitCond = part
#partTraj.allocate(npTraj)
partTraj.allocate(npTraj, True) #True ensures that field along trajectory will also be extracted 
partTraj.ctStart = 0 #"Start Time" (c*t) for the calculation (0 corresponds to the time moment for which the initial conditions are defined)
partTraj.ctEnd = partTraj.ctStart + 0.60 #End Time

#**********************Trajectory Calculation (SRWLIB function call)
print('   Performing calculation ... ', end='')
partTraj = srwl.CalcPartTraj(partTraj, magFldCnt, arPrecPar)
print('done')

#**********************Plotting results
print('   Plotting the results (close all graph windows to proceed with the script execution) ... ', end='')
ctMesh = [partTraj.ctStart, partTraj.ctEnd, partTraj.np]
for i in range(partTraj.np): #converting from [m] to [mm] and from [rad] to [mrad]
    partTraj.arXp[i] *= 1000
    partTraj.arX[i] *= 1000
    partTraj.arYp[i] *= 1000
    partTraj.arY[i] *= 1000
uti_plot1d(partTraj.arBy, ctMesh, ['ct [m]', 'Vertical Magnetic Field [T]'])
#uti_plot1d(partTraj.arXp, ctMesh, ['ct [m]', 'Horizontal Angle [mrad]'])
#uti_plot1d(partTraj.arX, ctMesh, ['ct [m]', 'Horizontal Position [mm]'])
#uti_plot1d(partTraj.arBx, ctMesh, ['ct [m]', 'Horizontal Magnetic Field [T]'])
#uti_plot1d(partTraj.arYp, ctMesh, ['ct [m]', 'Vertical Angle [mrad]'])
#uti_plot1d(partTraj.arY, ctMesh, ['ct [m]', 'Vertical Position [mm]'])
uti_plot_show()
print('done')
