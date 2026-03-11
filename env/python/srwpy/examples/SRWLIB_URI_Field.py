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

part = SRWLParticle()
part.x = 0. #Initial Transverse Coordinates (initial Longitudinal Coordinate will be defined later on) [m]
part.y = 1e-3
part.xp = 0 #Initial Transverse Velocities
part.yp = 0
part.gamma = 0.195/0.51099890221e-03 #Relative Energy
#part.gamma = 3/0.51099890221e-03 #Relative Energy
part.relE0 = 1 #Electron Rest Mass
part.nq = -1 #Electron Charge

npTraj = 10001 #Number of Points for Trajectory calculation
fieldInterpMeth = 4 #2 #Magnetic Field Interpolation Method, to be entered into 3D field structures below (to be used e.g. for trajectory calculation):
#1- bi-linear (3D), 2- bi-quadratic (3D), 3- bi-cubic (3D), 4- 1D cubic spline (longitudinal) + 2D bi-cubic
arPrecPar = [1]

#**********************Defining Magnetic Field Structure:
L_und = z[-1] - z[0]
Lgap = 0.5
mag1, mag2 = magFld, magFld
magFldCntDouble = SRWLMagFldC(
    _arMagFld = [magFld, magFld],
    _arXc = array('d', [xcID, xcID]),
    _arYc = array('d', [ycID, ycID]),
    _arZc = array('d', [zcID, zcID + L_und + Lgap]),
)

magFldCnt = SRWLMagFldC(_arMagFld = [magFld], _arXc = array('d', [xcID]), _arYc = array('d', [ycID]), _arZc = array('d', [zcID]) )

part.z = zcID - 0.5*magFldCnt.arMagFld[0].rz  #Initial Longitudinal Coordinate (set before the ID)

#**********************Trajectory structure, where the results will be stored
partTraj = SRWLPrtTrj()
partTraj.partInitCond = part
#partTraj.allocate(npTraj)
partTraj.allocate(npTraj, True)
partTraj.ctStart = 0 #Start Time for the calculation
#partTraj.ctEnd = (numPer + 2)*per + magFldCnt.arMagFld[0].rz + magFldCnt.arMagFld[2].rz #End Time
#partTraj.ctEnd = magFldCnt.arMagFld[0].rz + magFldCnt.arMagFld[1].rz + Lgap #End Time
partTraj.ctEnd = magFldCnt.arMagFld[0].rz  #End Time

partTraj = srwl.CalcPartTraj(partTraj, magFldCnt, arPrecPar)
print("   Trajectory calculation completed.")

#**********************Plotting results
print('   Plotting the results (blocks script execution; close any graph windows to proceed) ... ', end='')
ctMesh = [partTraj.ctStart, partTraj.ctEnd, partTraj.np]
for i in range(partTraj.np):
    partTraj.arX[i] *= 1000
    partTraj.arY[i] *= 1000
    
#uti_plot1d(partTraj.arX, ctMesh, ['ct [m]', 'Horizontal Position [mm]'])
#uti_plot1d(partTraj.arY, ctMesh, ['ct [m]', 'Vertical Position [mm]'])
#uti_plot1d(partTraj.arBy, ctMesh, ['ct [m]', 'Vertical Magnetic Field [T]'])
#uti_plot1d(partTraj.arBz, ctMesh, ['ct [m]', 'Longitudinal Magnetic Field [T]'])
#uti_plot_show() #show all graphs (and block execution)
print('done')

#**********************Calculating Radiation Electric Field Spectrum
elecBeam = SRWLPartBeam()
elecBeam.Iavg = 1e-3 #Average Current [A]
elecBeam.partStatMom1.x = 0. #Initial Transverse Coordinates (initial Longitudinal Coordinate will be defined later on) [m]
elecBeam.partStatMom1.y = 0.
elecBeam.partStatMom1.z = zcID - 0.5*magFldCnt.arMagFld[0].rz  #Initial Longitudinal Coordinate (set before the ID)
elecBeam.partStatMom1.xp = 0 #Initial Relative Transverse Velocities
elecBeam.partStatMom1.yp = 0
elecBeam.partStatMom1.gamma = .195/0.51099890221e-03 #Relative E

wfr = SRWLWfr()
wfr.allocate(_ne=1001, _nx=1,_ny=1) #Numbers of points vs Photon Energy
wfr.mesh.zStart = partTraj.ctEnd + 10. #Longitudinal Position [m] where Electric Field will be calculated
wfr.mesh.eStart = 1239.841984/410. #Initial Photon Energy [eV]
wfr.mesh.eFin = 1239.841984/398. #Final Photon Energy [eV]

wfr.mesh.xStart = 0. #Initial Horizontal Position [m]
wfr.mesh.xFin = 0. #Final Horizontal Position [m]
wfr.mesh.nx = 1 #Number of points vs Horizontal Position
wfr.mesh.yStart = 0. #Initial Vertical Position [m]
wfr.mesh.yFin = 0. #Final Vertical Position [m]
wfr.mesh.ny = 1 #Number of points vs Vertical Position

wfr.partBeam = elecBeam

wfr2 = wfr
wfr3 = wfr
wfr4 = wfr
wfr5 = wfr

arPrecPar = [0, 0.01, 0, 0, npTraj, 1, 0]

srwl.CalcElecFieldSR(wfr, 0, magFldCnt, arPrecPar)
arI = array('f', [0]*wfr.mesh.ne)
srwl.CalcIntFromElecField(arI, wfr, 6, 0, 0, wfr.mesh.eStart, wfr.mesh.xStart, wfr.mesh.yStart)

srwl.CalcElecFieldSR(wfr2, 0, magFldCntDouble, arPrecPar)
arI2 = array('f', [0]*wfr2.mesh.ne)
srwl.CalcIntFromElecField(arI2, wfr2, 6, 0, 0, wfr2.mesh.eStart, wfr2.mesh.xStart, wfr2.mesh.yStart)

## dz
magFldCntDouble.arZc = array('d', [zcID, zcID + L_und + Lgap + 0.01])
srwl.CalcElecFieldSR(wfr3, 0, magFldCntDouble, arPrecPar)
arI3 = array('f', [0]*wfr3.mesh.ne)
srwl.CalcIntFromElecField(arI3, wfr3, 6, 0, 0, wfr3.mesh.eStart, wfr3.mesh.xStart, wfr3.mesh.yStart)

magFldCntDouble.arZc = array('d', [zcID, zcID + L_und + Lgap + 0.02])
srwl.CalcElecFieldSR(wfr4, 0, magFldCntDouble, arPrecPar)
arI4 = array('f', [0]*wfr4.mesh.ne)
srwl.CalcIntFromElecField(arI4, wfr4, 6, 0, 0, wfr4.mesh.eStart, wfr4.mesh.xStart, wfr4.mesh.yStart)

magFldCntDouble.arZc = array('d', [zcID, zcID + L_und + Lgap + 0.03])
srwl.CalcElecFieldSR(wfr5, 0, magFldCntDouble, arPrecPar)
arI5 = array('f', [0]*wfr5.mesh.ne)
srwl.CalcIntFromElecField(arI5, wfr5, 6, 0, 0, wfr5.mesh.eStart, wfr5.mesh.xStart, wfr5.mesh.yStart)

#lambda_nm = 1239.841984 / energy_eV
energy_eV = np.linspace(wfr.mesh.eStart, wfr.mesh.eFin, wfr.mesh.ne)
lambda_nm = 1239.841984 / energy_eV
plt.plot(lambda_nm, arI, linestyle="-", marker="", label="Single Undulator")
plt.plot(lambda_nm, arI2, linestyle="-", marker="", label="Double Undulator")
plt.plot(lambda_nm, arI3, linestyle="-", marker="", label="Double Undulator dz=10cm")
plt.xlabel('Wavelength [nm]')
plt.ylabel('Intensity [a.u.]')
plt.legend()
plt.show()

plt.plot(lambda_nm, arI2, linestyle="-", marker="", label="Double Undulator")
plt.plot(lambda_nm, arI3, linestyle="-", marker="", label="dz=1cm")
plt.plot(lambda_nm, arI4, linestyle="-", marker="", label="dz=2cm")
plt.plot(lambda_nm, arI5, linestyle="-", marker="", label="dz=3cm")
plt.xlabel('Wavelength [nm]')
plt.ylabel('Intensity [a.u.]')
plt.legend()
plt.show()


wfr_xy = SRWLWfr()
wfr_xy.allocate(_ne=1, _nx=201,_ny=201) #Numbers of points vs Photon Energy
wfr_xy.mesh.zStart = partTraj.ctEnd + 10. #Longitudinal Position [m] where Electric Field will be calculated
wfr_xy.mesh.eStart = 1239.841984/404. #Initial Photon Energy [eV]
wfr_xy.mesh.eFin = 1239.841984/404 #Final Photon Energy [eV]
wfr_xy.mesh.xStart = -0.003 #Initial Horizontal Position [m]
wfr_xy.mesh.xFin = 0.003 #Final Horizontal Position [m]
wfr_xy.mesh.nx = 201 #Number of points vs Horizontal Position
wfr_xy.mesh.yStart = -0.003 #Initial Vertical Position [m]
wfr_xy.mesh.yFin = 0.003 #Final Vertical Position [m]

wfr_xy.partBeam = elecBeam

srwl.CalcElecFieldSR(wfr_xy, 0, magFldCnt, arPrecPar)
ar_xy = array('f', [0]*wfr_xy.mesh.nx*wfr_xy.mesh.ny) #"flat" array to take 2D intensity data

nx, ny = wfr_xy.mesh.nx, wfr_xy.mesh.ny
Ey = np.zeros((ny, nx), dtype=complex)

for iy in range(ny):
    for ix in range(nx):
        idx = 2 * ( ix + nx * iy )
        Ey[iy, ix] = (
            wfr_xy.arEy[idx]
            + 1j * wfr_xy.arEy[idx + 1]
        )

phase = np.unwrap(np.angle(Ey))
phase = phase/2/np.pi
#print(phase.shape)
#plt.imshow(phase)
#plt.show()
#srwl.CalcIntFromElecField(ar_xy, wfr_xy, 6, 0, 3, wfr2.mesh.eStart, 0, 0)

#uti_plot2d(ar_xy, [1000*wfr_xy.mesh.xStart, 1000*wfr_xy.mesh.xFin, wfr_xy.mesh.nx], [1000*wfr_xy.mesh.yStart, 1000*wfr_xy.mesh.yFin, wfr_xy.mesh.ny], ['Horizontal Position [mm]', 'Vertical Position [mm]', 'Intensity at ' + str(wfr_xy.mesh.eStart) + ' eV'])
#uti_plot2d(phase, [1000*wfr_xy.mesh.xStart, 1000*wfr_xy.mesh.xFin, wfr_xy.mesh.nx], ['Horizontal Position [mm]', 'Vertical Position [mm]', 'Phase at ' + str(wfr_xy.mesh.eStart) + ' eV'])

#uti_plot_show()

uti_plot1d(arI, [wfr.mesh.eStart, wfr.mesh.eFin, wfr.mesh.ne], ['Photon Energy [eV]', 'Intensity [ph/s/.1%bw/mm^2]', 'On-Axis Spectrum'])
uti_plot_show()
