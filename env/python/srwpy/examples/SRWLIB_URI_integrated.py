import numpy as np
import matplotlib.pyplot as plt

import matplotlib
from scipy.optimize import leastsq

try: #OC15112022
    import sys
    sys.path.append('../')
    from srwlib import *
    from uti_plot import *
except:
    from srwpy.srwlib import *
    from srwpy.uti_plot import *

#**********************Defining Study Parameters:
file_id = "0011"
Lgap_config = [0.7325, 1.5575, 0.001] #m
Lgap_config = [0.860, 1.685, 0.001] #m
Screen_x_config = [0, 1.e-3, 1] #m
Screen_y_config = [-2.2e-3, -1.3e-3, 51] #m
beam_x, beam_y, beam_xp, beam_yp = 0., 0.26e-3, 0., -0.125e-3
lambda_L_obs = 404. #nm
screen_z = 14. #m
print("========================================================")
print("Study Parameters Defined")
print("Lgap_config:", Lgap_config)
print("x,y config:", Screen_x_config, Screen_y_config)
print("beam parameters (x,y,x',y'):", beam_x*1e3, beam_y*1e3, beam_xp*1e3, beam_yp*1e3)
print("Observation Wavelength [nm]:", lambda_L_obs)
print("Screen z [m]:", screen_z)
print("========================================================")


RunNum = input("Enter Run Number (for record purpose): ")
SaveDirPath = "/home/nishi/SRW/env/python/srwpy/examples/Oscillation/{:}/".format("run"+RunNum) 
os.makedirs(SaveDirPath, exist_ok=True)



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

npTraj = 50001 #Number of Points for Trajectory calculation
fieldInterpMeth = 4 #2 #Magnetic Field Interpolation Method, to be entered into 3D field structures below (to be used e.g. for trajectory calculation):
#1- bi-linear (3D), 2- bi-quadratic (3D), 3- bi-cubic (3D), 4- 1D cubic spline (longitudinal) + 2D bi-cubic

#**********************Defining Magnetic Field Structure:
#L_und = z[-1] - z[0]
L_und = 0.520
#Lgap = 0.7325 # gap = 860 - 1685 mm in experiment. fringe range = 127.5, Lgap = 732.5 ~ 1557.5
#Lgap = 1.5575 # downstream limit
Lgap = 0.860
#Lgap = 1.685
mag1, mag2 = magFld, magFld
magFldCnt = SRWLMagFldC(
    _arMagFld = [magFld, magFld],
    _arXc = array('d', [xcID, xcID]),
    _arYc = array('d', [ycID, ycID]),
    _arZc = array('d', [zcID, zcID + L_und + Lgap]),
)

#**********************Processing Configuration Ranges:
Lgap_arange    = np.arange(Lgap_config[0], Lgap_config[1], Lgap_config[2])

screen_x_range = np.linspace(Screen_x_config[0], Screen_x_config[1], Screen_x_config[2])
screen_y_range = np.linspace(Screen_y_config[0], Screen_y_config[1], Screen_y_config[2])

#**********************Calculating Radiation Electric Field Spectrum
elecBeam = SRWLPartBeam()
elecBeam.Iavg = 1e-9 #Average Current [A]
elecBeam.partStatMom1.x = beam_x
elecBeam.partStatMom1.y = beam_y
elecBeam.partStatMom1.z = zcID - 0.5*magFldCnt.arMagFld[0].rz - 0.1   #Initial Longitudinal Coordinate (set before the ID)
elecBeam.partStatMom1.xp = beam_xp
elecBeam.partStatMom1.yp = beam_yp
elecBeam.partStatMom1.gamma = .195/0.51099890221e-03 #Relative E

Intensity_list = []
for Lgap in Lgap_arange:
    magFldCnt.arZc[1] = zcID + L_und + Lgap

    wfr = SRWLWfr()
    wfr.allocate(_ne=1, _nx=1, _ny=31) #Numbers of points vs Photon Energy
    wfr.mesh.zStart = magFldCnt.arMagFld[0].rz + magFldCnt.arMagFld[1].rz + Lgap + screen_z #Longitudinal Position [m] where Electric Field will be calculated
    wfr.mesh.zStart = magFldCnt.arZc[0] + screen_z #Longitudinal Position [m] where Electric Field will be calculated
    print(wfr.mesh.zStart)
    wfr.mesh.eStart = 1239.841984/lambda_L_obs #Initial Photon Energy [eV]
    wfr.mesh.eFin   = 1239.841984/lambda_L_obs #Final Photon Energy [eV]

    #wfr.mesh.xStart = Screen_x_config[0] #Initial Horizontal Position [m]
    #wfr.mesh.xFin   = Screen_x_config[1] #Final Horizontal Position [m]
    #wfr.mesh.nx     = Screen_x_config[2] #Number of points vs Horizontal Position
    wfr.mesh.xStart = 0.
    wfr.mesh.xFin   = 0.
    wfr.mesh.nx     = 1

    wfr.mesh.yStart = Screen_y_config[0] #Initial Vertical Position [m]
    wfr.mesh.yFin   = Screen_y_config[1] #Final Vertical Position [m]
    wfr.mesh.ny     = Screen_y_config[2] #Number of points vs Vertical Position

    wfr.partBeam = elecBeam
    #arPrecPar = [1, 0.01, elecBeam.partStatMom1.z, magFldCnt.arMagFld[0].rz + magFldCnt.arMagFld[1].rz + Lgap, npTraj, 0, 1]
    #arPrecPar = [1, 0.01, elecBeam.partStatMom1.z, magFldCnt.arMagFld[0].rz + magFldCnt.arMagFld[1].rz + Lgap, npTraj, 1, 1] #NG
    arPrecPar = [1, 0.01, elecBeam.partStatMom1.z, magFldCnt.arMagFld[0].rz + magFldCnt.arMagFld[1].rz + Lgap, npTraj, 1, 0]
    #arPrecPar = [1, 0.01, 0, 0, npTraj, 0, 0] #OK

    srwl.CalcElecFieldSR(wfr, 0, magFldCnt, arPrecPar)
    arI = array('f', [0]*wfr.mesh.nx * wfr.mesh.ny * wfr.mesh.ne)
    srwl.CalcIntFromElecField(arI, wfr, 6, 0, 3, wfr.mesh.eStart, 0., 0.)

    arI = np.array(arI)#.reshape(wfr.mesh.ny, wfr.mesh.nx)
    #arI = arI.T  # to make y first index
    Intensity_list.append(arI)

ar_Intensity = np.array(Intensity_list) #825, 3, 31
print(ar_Intensity.shape) #OK:825, 3, 31, NG:8, 8, 8
#ar_Intensity = ar_Intensity[:,len(screen_x_range)//2, :]
#print(ar_Intensity.shape) #OK:825, 31, NG:8, 8
plt.plot(ar_Intensity[:,15])
plt.show()
exit()

#***********************Plotting Results:
position = Lgap_arange * 1e3  # in mm

prm_lst = []
cut_config = [
    [0, len(position)],
    [0, len(position)//2],
    [0, 3*len(position)//4],
    [len(position)//4, 3*len(position)//4],
    [len(position)//4, len(position)],
    [len(position)//2, len(position)],
]

### Fitting ###
def sin_fnc(prm, d):
    return prm[0] * (1 + np.sin(d *2*np.pi / prm[1] + prm[2]) ) 

def fit_fnc(prm, d, osc):
    return ( sin_fnc(prm, d) - osc )

for cut_l, cut_h in cut_config:
    Data_sampled = ar_Intensity[cut_l:cut_h, :]
    Position_sampled = position[cut_l:cut_h]
    for i in range(Data_sampled.shape[1]):
        temp_lst = [i]
        Data_split = Data_sampled[:, i]
        init_val = [1000, 115,  2.5]
        #init_val = [15, 117, 2.5]
        try:
            prm,cov,info,msg,ier= leastsq(fit_fnc, init_val, args=(Position_sampled, Data_split),full_output=True)
            for j in range(3):
                temp_lst.append(prm[j])
                temp_lst.append(np.sqrt(cov[j,j]))
        except:
            temp_lst = [i]
            for j in range(6):
                temp_lst.append(0)
        temp_lst.append(cut_l)
        temp_lst.append(cut_h)
        prm_lst.append(temp_lst)

print("Fitting completed.")
prm_lst = np.array(prm_lst).reshape(len(cut_config), Data_sampled.shape[1], 9)


#0    index
#1,2  ampl, err_ampl
#3,4  period, err_period
#5,6  phase, err_phase
#7,8  cut_l, cut_h

#prm_lst shape: (cut_csonfig, num data points along y dir, 9 prms)

fig = plt.figure(figsize=(8,6))
ax = fig.subplots(1,2)

fig_ext_val = plt.figure(figsize=(8,6))
ax_ext_val  = fig_ext_val.subplots(2,1)

cut_label = [f"{cut_config[i][0]:03d}:{cut_config[i][1]:03d}" for i in range(len(cut_config))]
y = screen_y_range # convert to mm
#y = range(Data_sampled.shape[1]) 
result_lst =[]
for cut_cond_i in range(len(cut_config)):
    ax[0].errorbar(y, prm_lst[cut_cond_i,:,3], yerr=prm_lst[cut_cond_i,:,4], label=cut_label[cut_cond_i])
    lambda_osc_y_idx = np.argmax(prm_lst[cut_cond_i,:,3])
    lambda_osc_y = y[lambda_osc_y_idx]
    ax[1].errorbar(y, prm_lst[cut_cond_i,:,5], yerr=prm_lst[cut_cond_i,:,6], label=cut_label[cut_cond_i])
    print(cut_label[cut_cond_i], 
        np.argmax(prm_lst[cut_cond_i,:,3]), 
        y[np.argmax(prm_lst[cut_cond_i,:,3])]*1.e3, 
        np.argmin(prm_lst[cut_cond_i,:,5]),
        y[np.argmin(prm_lst[cut_cond_i,:,5])]*1.e3
    )
    result_lst.append([
        cut_config[cut_cond_i][0], cut_config[cut_cond_i][1],
        np.argmax(prm_lst[cut_cond_i,:,3]), y[np.argmax(prm_lst[cut_cond_i,:,3])]*1.e3,
        np.argmin(prm_lst[cut_cond_i,:,5]), y[np.argmin(prm_lst[cut_cond_i,:,5])]*1.e3
    ])

    if cut_cond_i == 0 : continue
    cut_cond_mid = (cut_config[cut_cond_i][0] + cut_config[cut_cond_i][1]) // 2
    cut_cond_dev = (cut_config[cut_cond_i][1] - cut_config[cut_cond_i][0]) // 2
    ext_val_lam = y[np.argmax(prm_lst[cut_cond_i,:,3])]*1.e3
    ext_val_phi = y[np.argmin(prm_lst[cut_cond_i,:,5])]*1.e3
    yerr = screen_y_range[1] - screen_y_range[0]
    ax_ext_val[0].errorbar(cut_cond_mid, ext_val_lam , xerr = cut_cond_dev, yerr= yerr, color = 'tab:blue')
    ax_ext_val[1].errorbar(cut_cond_mid, ext_val_phi , xerr = cut_cond_dev, yerr= yerr, color = 'tab:orange')
    ax_ext_val[0].set_ylabel("Fitted Oscillation Period [mm]")
    ax_ext_val[1].set_xlabel("Cut Range along y index")
    ax_ext_val[1].set_ylabel("Fitted Oscillation Phase [mm]")

plt.legend()
fig.savefig(SaveDirPath+"Oscillation_Fitting_Lgap.png", dpi=300)
fig_ext_val.savefig(SaveDirPath+"Oscillation_Fitting_Lgap_ExtVal.png", dpi=300)
#plt.show()
plt.close()

np.savetxt(SaveDirPath+RunNum+"_Fitting_Results.txt", np.array(result_lst))


#***********************Tracking:
part = SRWLParticle()
part.x = elecBeam.partStatMom1.x
part.y = elecBeam.partStatMom1.y
part.z = elecBeam.partStatMom1.z 
part.xp = elecBeam.partStatMom1.xp
part.yp = elecBeam.partStatMom1.yp
part.gamma = 0.195/0.51099890221e-03 #Relative Energy

#  *----------------------*

partTrajLst = []
ctMeshLst = []
for Lgap in (Lgap_arange[0], Lgap_arange[-1]):
    magFldCnt.arZc[1] = zcID + L_und + Lgap
    partTraj = SRWLPrtTrj()
    partTraj.partInitCond = part
    partTraj.allocate(npTraj, True)
    partTraj.ctStart = 0 #Start Time for the calculation
    #partTraj.ctEnd = magFldCnt.arMagFld[0].rz  #End Time
    partTraj.ctEnd = magFldCnt.arMagFld[0].rz + magFldCnt.arMagFld[1].rz + Lgap_arange[-1] #End Time

    srwl.CalcPartTraj(partTraj, magFldCnt, [1])
    partTrajLst.append(partTraj)

    ctMesh = np.linspace(partTraj.ctStart, partTraj.ctEnd, partTraj.np)
    ctMeshLst.append(ctMesh)


print("   Trajectory calculation completed.")
for partTraj, ctMesh in zip(partTrajLst, ctMeshLst):
    for i in range(partTraj.np):
        partTraj.arX[i] *= 1000
        partTraj.arY[i] *= 1000

arX  = [partTrajLst[i].arX for i in range(len(partTrajLst))]
arY  = [partTrajLst[i].arY for i in range(len(partTrajLst))]
arBy = [partTrajLst[i].arBy for i in range(len(partTrajLst))]
arBz = [partTrajLst[i].arBz for i in range(len(partTrajLst))]

fig = plt.figure(figsize=(8,6))
ax =fig.subplots(3,1)


linear = beam_yp*1.e3 * ctMeshLst[0] + beam_y*1.e3
for i, partTraj in enumerate(partTrajLst):
    ax[0].plot(ctMeshLst[i], arX[i],  label="Lgap={:.4f} m".format(Lgap_arange[i]),linestyle="-", marker="None")
    ax[1].plot(ctMeshLst[i], arY[i], label="Lgap={:.4f} m".format(Lgap_arange[i]), linestyle="-", marker="None")

#resid = np.array(arY[i]) - np.array(linear)
resid = np.array(arY[1]) - np.array(arY[0])
ax[2].plot(ctMeshLst[0], resid, linestyle="-", marker="None")
ax[1].set_xlabel("ct [m]")
ax[0].set_ylabel("X [mm]")
ax[1].set_ylabel("Y [mm]")
ax[0].set_title("Electron Trajectory \n ")
plt.savefig(SaveDirPath+"Electron_Trajectory_Lgap.png", dpi=300)
#plt.show()
plt.close()

#***********************Saving Results:
np.savetxt(SaveDirPath+RunNum+".txt", ar_Intensity)