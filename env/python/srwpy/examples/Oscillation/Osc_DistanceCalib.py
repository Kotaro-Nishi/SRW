import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.optimize import leastsq


DirPath = "/home/nishi/SRW/env/python/srwpy/examples/Oscillation/"
file_id = "0011"

ar_Intensity = np.loadtxt(DirPath+"Oscillation"+file_id+".txt", dtype='float')
ar_Intensity = np.array(ar_Intensity)

data_len = ar_Intensity.shape[0]
if data_len == 825:
    position = np.arange(0, 825, 1)
elif data_len == 165:
    position = np.arange(0, 825, 5)

prm_lst = []
cut_config = [
    [0, len(position)],
    [0, len(position)//2],
    [len(position)//2, len(position)]
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

prm_lst = np.array(prm_lst).reshape(len(cut_config), Data_sampled.shape[1], 9)
#0    index
#1,2  ampl, err_ampl
#3,4  period, err_period
#5,6  phase, err_phase
#7,8  cut_l, cut_h

#prm_lst shape: (cut_config, num data points along y dir, 9 prms)

fig = plt.figure(figsize=(8,6))
ax = fig.subplots(1,2)
cut_label = ['Full', 'First Half', 'Second Half']
y = np.linspace(-2.7, -1.7, Data_sampled.shape[1]) # in mm
#y = range(Data_sampled.shape[1]) 
for cut_cond_i in range(len(cut_config)):
    ax[0].errorbar(y, prm_lst[cut_cond_i,:,3], yerr=prm_lst[cut_cond_i,:,4], label=cut_label[cut_cond_i])
    lambda_osc_y_idx = np.argmax(prm_lst[cut_cond_i,:,3])
    lambda_osc_y = y[lambda_osc_y_idx]
    print(cut_label[cut_cond_i], np.argmax(prm_lst[cut_cond_i,:,3]))
    print(y[np.argmax(prm_lst[cut_cond_i,:,3])])
    ax[1].errorbar(y, prm_lst[cut_cond_i,:,5], yerr=prm_lst[cut_cond_i,:,6], label=cut_label[cut_cond_i])
    print(cut_label[cut_cond_i], np.argmin(prm_lst[cut_cond_i,:,5]))
    print(y[np.argmin(prm_lst[cut_cond_i,:,5])])

plt.legend()
plt.show()

## Fitting ##
#fit_result,residual,ndf,chi2,chi2_lst,result = MODfit.guessed_to_optimized(MODconf.x,Data,EData,DataOn,Position,prm_add_lst)
#fit_result,residual,ndf,chi2,chi2_lst,result,m = MODfit.minuit_optimize(MODconf.x,Data,EData,DataOn,Position,prm_add_lst)
