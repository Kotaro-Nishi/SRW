#############################################################################
# URI test: Calculating electron trajectory in magnetic field a segmented planar undulator
# v 0.08
#############################################################################
from __future__ import print_function #Python 2.7 compatibility

import sys
sys.path.append('../')
from srwpy.srwlib import *
from srwpy.uti_plot import *
import os
import numpy as np
import matplotlib.pyplot as plt
from array import array
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

und = SRWLMagFldU([SRWLMagFldH(1, 'v', 0.135, 0.01, 1)], undPer, numPer) #Undulator Segment

magFldCnt = SRWLMagFldC([und]) #Container of all Field Elements
magFldCnt.arZc[0] = zcID #Positioning the first Segment


Nx  = 21
Ny  = 21
Nz  = 101

xRange = 0.01   # [m]
yRange = 0.01
zRange = 0.80

mag3d = SRWLMagFld3D( _nx=Nx, _ny=Ny, _nz=Nz, _rx=xRange, _ry=yRange, _rz=zRange)

mag3d.arBx = array('d', [0.0]*mag3d.nx*mag3d.ny*mag3d.nz)
mag3d.arBy = array('d', [0.0]*mag3d.nx*mag3d.ny*mag3d.nz)
mag3d.arBz = array('d', [0.0]*mag3d.nx*mag3d.ny*mag3d.nz)

magFldCnt_out = SRWLMagFldC(_arMagFld=[mag3d], _arXc=[0.0], _arYc=[0.0], _arZc=[0.0])

#srwl.CalcMagnField(mag3d, magFldCnt, [0])
srwl.CalcMagnField(magFldCnt_out, magFldCnt, [0])


Bx_np = np.array(mag3d.arBx).reshape(Nz,Ny,Nx).T
By_np = np.array(mag3d.arBy).reshape(Nz,Ny,Nx).T
Bz_np = np.array(mag3d.arBz).reshape(Nz,Ny,Nx).T

x_arr = np.linspace(-xRange/2, xRange/2, Nx)
y_arr = np.linspace(-yRange/2, yRange/2, Ny)
z_arr = np.linspace(-zRange/2, zRange/2, Nz)

plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['image.interpolation'] = 'None'
plt.rcParams['font.size'] = 10
plt.figure(figsize=(8,6))
plt.subplot(3,1,1)
plt.imshow(Bx_np[:,:,Nz//2].T, extent= [x_arr[0]*1e3, x_arr[-1]*1e3, y_arr[0]*1e3, y_arr[-1]*1e3], origin='lower')
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.colorbar()
plt.title('Bx at z=0.0 m')
plt.subplot(3,1,2)
plt.imshow(By_np[:,:,Nz//2].T, extent= [x_arr[0]*1e3, x_arr[-1]*1e3, y_arr[0]*1e3, y_arr[-1]*1e3], origin='lower')
plt.colorbar()
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.title('By at z=0.0 m')
plt.subplot(3,1,3)
plt.imshow(Bz_np[:,:,Nz//2].T,  extent= [x_arr[0]*1e3, x_arr[-1]*1e3, y_arr[0]*1e3, y_arr[-1]*1e3], origin='lower')
plt.colorbar()
plt.title('Bz at z=0.0 m')
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.subplot(3,1,1)
plt.imshow(Bx_np[:,Ny//2,:], extent= [z_arr[0]*1e3, z_arr[-1]*1e3, x_arr[0]*1e3, x_arr[-1]*1e3], origin='lower', aspect='auto')
plt.colorbar()
plt.xlabel('z [mm]')
plt.ylabel('x [mm]')
plt.title('Bx at y=0.0 m')
plt.subplot(3,1,2)
plt.imshow(By_np[:,Ny//2,:], extent= [z_arr[0]*1e3, z_arr[-1]*1e3, x_arr[0]*1e3, x_arr[-1]*1e3], origin='lower', aspect='auto')
plt.colorbar()
plt.title('By at y=0.0 m')
plt.xlabel('z [mm]')
plt.ylabel('x [mm]')
plt.subplot(3,1,3)
plt.imshow(Bz_np[:,Ny//2,:], extent= [z_arr[0]*1e3, z_arr[-1]*1e3, x_arr[0]*1e3, x_arr[-1]*1e3], origin='lower', aspect='auto')
plt.colorbar()
plt.title('Bz at y=0.0 m')
plt.xlabel('z [mm]')
plt.ylabel('x [mm]')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
plt.subplot(3,1,1)
plt.imshow(Bx_np[Nx//2,:,:], extent= [z_arr[0]*1e3, z_arr[-1]*1e3, y_arr[0]*1e3, y_arr[-1]*1e3], origin='lower', aspect='auto')
plt.colorbar()
plt.xlabel('z [mm]')
plt.ylabel('y [mm]')
plt.title('Bx at x=0.0 m')
plt.subplot(3,1,2)
plt.imshow(By_np[Nx//2,:,:], extent= [z_arr[0]*1e2, z_arr[-1]*1e2, y_arr[0]*1e3, y_arr[-1]*1e3], origin='lower', aspect='auto')
plt.colorbar()
plt.xlabel('z [mm]')
plt.ylabel('y [mm]')
plt.title('By at x=0.0 m')
plt.subplot(3,1,3)
plt.imshow(Bz_np[Nx//2,:,:], extent= [z_arr[0]*1e3, z_arr[-1]*1e3, y_arr[0]*1e3, y_arr[-1]*1e3], origin='lower', aspect='auto')
plt.colorbar()
plt.xlabel('z [mm]')
plt.ylabel('y [mm]')
plt.title('Bz at x=0.0 m')
plt.tight_layout()
plt.show()
exit()

#**********************Trajectory structure, where the results will be stored
partTraj = SRWLPrtTrj()
partTraj.partInitCond = part
#partTraj.allocate(npTraj)
partTraj.allocate(npTraj, True) #True ensures that field along trajectory will also be extracted 
partTraj.ctStart = 0 #"Start Time" (c*t) for the calculation (0 corresponds to the time moment for which the initial conditions are defined)
partTraj.ctEnd = partTraj.ctStart + 0.80 #End Time

#***********Electron Beam
eBeam = SRWLPartBeam()
eBeam.Iavg = 0.005 #average current [A]
eBeam.partStatMom1.x = 0. #initial transverse positions [m]
eBeam.partStatMom1.y = 0.
eBeam.partStatMom1.z = 0. #initial longitudinal positions (set in the middle of undulator)
eBeam.partStatMom1.xp = 0 #initial relative transverse velocities
eBeam.partStatMom1.yp = 0
eBeam.partStatMom1.gamma = 0.195/0.51099890221e-03 #relative energy
sigEperE = 0.00089 #relative RMS energy spread
sigX = 50.e-06 #horizontal RMS size of e-beam [m]
sigXp = 10.e-06 #horizontal RMS angular divergence [rad]
sigY = 50.e-06 #vertical RMS size of e-beam [m]
sigYp = 10.e-06 #vertical RMS angular divergence [rad]
#2nd order stat. moments:
eBeam.arStatMom2[0] = sigX*sigX #<(x-<x>)^2>
eBeam.arStatMom2[1] = 0 #<(x-<x>)(x'-<x'>)>
eBeam.arStatMom2[2] = sigXp*sigXp #<(x'-<x'>)^2>
eBeam.arStatMom2[3] = sigY*sigY #<(y-<y>)^2>
eBeam.arStatMom2[4] = 0 #<(y-<y>)(y'-<y'>)>
eBeam.arStatMom2[5] = sigYp*sigYp #<(y'-<y'>)^2>
eBeam.arStatMom2[10] = sigEperE*sigEperE #<(E-<E>)^2>/<E>^2

#**********************Wavefront
wfr = SRWLWfr()
wfr.allocate(1, 100, 100) #Numbers of points vs Photon Energy, Horizontal and Vertical Positions
wfr.mesh.zStart = 10. #Longitudinal Position [m] at which Wavefront has to be calculated
wfr.mesh.eStart = 3.1 #Initial Photon Energy [eV]
wfr.mesh.eFin = 5. #Final Photon Energy [eV]

wfr.mesh.xStart = -0.01 #Initial Horizontal Position [m]
wfr.mesh.xFin = 0.01 #Final Horizontal Position [m]
wfr.mesh.yStart = -0.01 #Initial Vertical Position [m]
wfr.mesh.yFin = 0.01 #Final Vertical Position [m]

wfr.partBeam = eBeam
#**********************Trajectory Calculation (SRWLIB function call)
print('   Performing calculation ... ', end='')
partTraj = srwl.CalcPartTraj(partTraj, magFldCnt, arPrecPar)
print('done')

plt.plot(partTraj.arBy)
plt.show()
exit()

#**********************Wavefront Calculation (SRWLIB function call)
print('   Calculating wavefront ... ', end='')
srwl.CalcElecFieldSR(wfr, 1, magFldCnt, [1, 0.01, 0, 0, npTraj])
Ex = np.array(wfr.arEx).reshape(-1, 2)
Ex_c = Ex[:,0] + 1j*Ex[:,1]
Ex_c = Ex_c.reshape(wfr.mesh.ny, wfr.mesh.nx)
Ey = np.array(wfr.arEy).reshape(-1, 2)
Ey_c = Ey[:,0] + 1j*Ey[:,1]
Ey_c = Ey_c.reshape(wfr.mesh.ny, wfr.mesh.nx)
Intensity = np.abs(Ex_c)**2 + np.abs(Ey_c)**2 
print('done')
plt.imshow(Intensity.reshape(wfr.mesh.nx, wfr.mesh.ny).T, extent=(wfr.mesh.xStart*1e3, wfr.mesh.xFin*1e3, wfr.mesh.yStart*1e3, wfr.mesh.yFin*1e3), origin='lower')
plt.show()


#**********************Plotting results
print('   Plotting the results (close all graph windows to proceed with the script execution) ... ', end='')
ctMesh = [partTraj.ctStart, partTraj.ctEnd, partTraj.np]
for i in range(partTraj.np): #converting from [m] to [mm] and from [rad] to [mrad]
    partTraj.arXp[i] *= 1000
    partTraj.arX[i] *= 1000
    partTraj.arYp[i] *= 1000
    partTraj.arY[i] *= 1000
uti_plot1d(partTraj.arBy, ctMesh, ['ct [m]', 'Vertical Magnetic Field [T]'])
uti_plot1d(partTraj.arXp, ctMesh, ['ct [m]', 'Horizontal Angle [mrad]'])
uti_plot1d(partTraj.arX, ctMesh, ['ct [m]', 'Horizontal Position [mm]'])
uti_plot1d(partTraj.arBx, ctMesh, ['ct [m]', 'Horizontal Magnetic Field [T]'])
uti_plot1d(partTraj.arYp, ctMesh, ['ct [m]', 'Vertical Angle [mrad]'])
uti_plot1d(partTraj.arY, ctMesh, ['ct [m]', 'Vertical Position [mm]'])
uti_plot_show()
print('done')
