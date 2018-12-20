#%matplotlib inline
#import sdf
import matplotlib
import matplotlib as mpl
#mpl.style.use('https://raw.githubusercontent.com/Michael-Gong/DLA_project/master/style')
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal
from optparse import OptionParser
import os
from mpl_toolkits.mplot3d import Axes3D
import random
from mpl_toolkits import mplot3d
from matplotlib import rc
import matplotlib.transforms as mtransforms
import sys
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
font = {'family' : 'monospace',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 25,
       }
font_size = 25
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')


def pxpy_to_energy(gamma, weight):
    binsize = 100
    en_grid = np.linspace(100,19900,100)
    en_bin  = np.linspace(0,20000.0,101)
    en_value = np.zeros_like(en_grid) 
    for i in range(binsize):
#        if i == binsize-1:
#            en_value[i] = sum(weight[en_bin[i]<=gamma])
#        else:
            en_value[i] = sum(weight[ (en_bin[i]<=gamma) & (gamma<en_bin[i+1]) ])
    return (en_grid, en_value)



if __name__ == "__main__":
  part_number = 50000
  from_path = './Data_no_T500_p50000/'
  nsteps      = int(sum(1 for line in open(from_path+'px_tot_s.txt'))/part_number)


  from_path = './Data_no_T500_p50000/'
#  to_path   = './Data_no/'
  t0  = np.loadtxt(from_path+'t_tot_s.txt')/2/np.pi
  rad0= np.loadtxt(from_path+'radt_tot_s.txt')
  px0 = np.loadtxt(from_path+'px_tot_s.txt')
  py0 = np.loadtxt(from_path+'py_tot_s.txt')

  t0  = np.reshape(t0,(part_number,nsteps))
  rad0= np.reshape(rad0,(part_number,nsteps))
  px0 = np.reshape(px0,(part_number,nsteps))
  py0 = np.reshape(py0,(part_number,nsteps))
  gg0 = (px0**2+py0**2+1)**0.5
  ww0 = np.zeros_like(gg0)+1



  from_path = './Data_qe_T500_p50000/'
#  to_path   = './jpg_log/'
  t2  = np.loadtxt(from_path+'t_tot_s.txt')/2/np.pi
  rad2= np.loadtxt(from_path+'radt_tot_s.txt')
  px2 = np.loadtxt(from_path+'px_tot_s.txt')
  py2 = np.loadtxt(from_path+'py_tot_s.txt')

  t2  = np.reshape(t2,(part_number,nsteps))
  rad2= np.reshape(rad2,(part_number,nsteps))
  px2 = np.reshape(px2,(part_number,nsteps))
  py2 = np.reshape(py2,(part_number,nsteps))
  gg2 = (px2**2+py2**2+1)**0.5
  ww2 = np.zeros_like(gg2)+1



  from_path = './Data_rr_T500_p50000/'
#  to_path   = './jpg_log/'
  t1  = np.loadtxt(from_path+'t_tot_s.txt')/2/np.pi
  rad1= np.loadtxt(from_path+'radt_tot_s.txt')
  px1 = np.loadtxt(from_path+'px_tot_s.txt')
  py1 = np.loadtxt(from_path+'py_tot_s.txt')

  t1  = np.reshape(t1,(part_number,nsteps))
  rad1= np.reshape(rad1,(part_number,nsteps))
  px1 = np.reshape(px1,(part_number,nsteps))
  py1 = np.reshape(py1,(part_number,nsteps))
  gg1 = (px1**2+py1**2+1)**0.5
  ww1 = np.zeros_like(gg1)+1


  
  time_series = np.zeros(nsteps)
  electron_0  = np.zeros(nsteps)
  electron_1  = np.zeros(nsteps)
  electron_2  = np.zeros(nsteps)
  photon_1    = np.zeros(nsteps)
  photon_2    = np.zeros(nsteps)

  for i in range(nsteps):
          electron_0[i] = (sum(gg0[:,i]) - sum(gg0[:,0]))*0.51*1.6e-19
          electron_1[i] = (sum(gg1[:,i]) - sum(gg1[:,0]))*0.51*1.6e-19
          electron_2[i] = (sum(gg2[:,i]) - sum(gg2[:,0]))*0.51*1.6e-19
          photon_1[i] = sum(rad1[:,i])*0.51*1.6e-19
          photon_2[i] = sum(rad2[:,i])*0.51*1.6e-19
          time_series[i]= t2[0,i]
  #### manifesting colorbar, changing label and axis properties ####
  plt.plot(time_series*10./3., electron_0,'-r',linewidth=4,label='Electron w/o RR')
  plt.plot(time_series*10./3., electron_1,'-g',linewidth=4,label='Electron w LL RR')
  plt.plot(time_series*10./3., electron_2,'-b',linewidth=4,label='Electron w QED RR')
  plt.plot(time_series*10./3., photon_1,'--g',linewidth=4,label='Electron w LL loss')
  plt.plot(time_series*10./3., photon_2,'--b',linewidth=4,label='Electron w QED loss')
  plt.legend(loc='best',fontsize=18)
  plt.xlabel('time [fs]',fontdict=font)
  plt.ylabel('Energy [J]',fontdict=font)
  plt.xticks([0,500,1000,1500],fontsize=font_size); plt.yticks(fontsize=font_size);
  #plt.yscale('log') 
  plt.xlim(0,1650)
  plt.legend(loc='best',fontsize=font_size,framealpha=1)
  #plt.text(250,6e9,'t='+str(round(time/1.0e-15,0))+' fs',fontdict=font)
  #plt.title('t = '+str(round(t0[0,i]*10./3.,0))+' fs',fontdict=font)
  plt.subplots_adjust(top=0.98, bottom=0.14, left=0.14, right=0.99, hspace=0.10, wspace=0.15)
  fig = plt.gcf()
  fig.set_size_inches(10, 6.5)
  fig.savefig('./wrap_energy_evolution.png',format='png',dpi=160)
  plt.close("all")
#  print('plotting '+str(i).zfill(4))
          
  
          
          
