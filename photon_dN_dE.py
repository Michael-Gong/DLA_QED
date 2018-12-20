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
  #x0  = np.loadtxt(from_path+'x_tot.txt')/2/np.pi
  #y0  = np.loadtxt(from_path+'y_tot.txt')/2/np.pi
  px0 = np.loadtxt(from_path+'px_tot_s.txt')
  py0 = np.loadtxt(from_path+'py_tot_s.txt')
  t0  = np.reshape(t0,(part_number,nsteps))
  #x0  = np.reshape(x0,(part_number,nsteps))
  #y0  = np.reshape(y0,(part_number,nsteps))
  px0 = np.reshape(px0,(part_number,nsteps))
  py0 = np.reshape(py0,(part_number,nsteps))
  gg0 = (px0**2+py0**2+1)**0.5
  ww0 = np.zeros_like(gg0)+1



  from_path = './Data_qe_T500_p50000/'
#  to_path   = './jpg_log/'
  t2  = np.loadtxt(from_path+'t_tot_s.txt')/2/np.pi
  #x2  = np.loadtxt(from_path+'x_tot.txt')/2/np.pi
  #y2  = np.loadtxt(from_path+'y_tot.txt')/2/np.pi
  px2 = np.loadtxt(from_path+'px_tot_s.txt')
  py2 = np.loadtxt(from_path+'py_tot_s.txt')
  t2  = np.reshape(t2,(part_number,nsteps))
  #x2  = np.reshape(x1,(part_number,nsteps))
  #y2  = np.reshape(y1,(part_number,nsteps))
  px2 = np.reshape(px2,(part_number,nsteps))
  py2 = np.reshape(py2,(part_number,nsteps))
  gg2 = (px2**2+py2**2+1)**0.5
  ww2 = np.zeros_like(gg2)+1


  
  from_path = './Data_rr_T500_p50000/'
#  to_path   = './jpg_log/'
  t1  = np.loadtxt(from_path+'t_tot_s.txt')/2/np.pi
  #x1  = np.loadtxt(from_path+'x_tot.txt')/2/np.pi
  #y1  = np.loadtxt(from_path+'y_tot.txt')/2/np.pi
  px1 = np.loadtxt(from_path+'px_tot_s.txt')
  py1 = np.loadtxt(from_path+'py_tot_s.txt')
  t1  = np.reshape(t1,(part_number,nsteps))
  #x1  = np.reshape(x1,(part_number,nsteps))
  #y1  = np.reshape(y1,(part_number,nsteps))
  px1 = np.reshape(px1,(part_number,nsteps))
  py1 = np.reshape(py1,(part_number,nsteps))
  gg1 = (px1**2+py1**2+1)**0.5
  ww1 = np.zeros_like(gg1)+1



  for i in range(224,227,1):
          dist_x0, den0 = pxpy_to_energy(gg0[:,i],ww0[:,i])
          dist_x1, den1 = pxpy_to_energy(gg1[:,i],ww1[:,i])
          dist_x2, den2 = pxpy_to_energy(gg2[:,i],ww2[:,i])
          #### manifesting colorbar, changing label and axis properties ####
          plt.plot(dist_x0*0.51*1e-3,den0,'-r',linewidth=4,label='Electron w/o RR')
          plt.plot(dist_x1*0.51*1e-3,den1,'-g',linewidth=4,label='Electron w LL RR')
          plt.plot(dist_x2*0.51*1e-3,den2,'-b',linewidth=4,label='Electron w QED  RR')
          plt.legend(loc='upper right',fontsize=20)
          plt.xlabel(r'$\epsilon_e$'+' [GeV]',fontdict=font)
          plt.ylabel('dN/dE [A.U.]',fontdict=font)
          plt.xticks(fontsize=font_size); plt.yticks(fontsize=font_size);
          plt.yscale('log')
          plt.xlim(0,10)
          plt.ylim(5,5e4)
          plt.legend(loc='best',fontsize=font_size,framealpha=1)
          #plt.text(250,6e9,'t='+str(round(time/1.0e-15,0))+' fs',fontdict=font)
          plt.title('t = '+str(round(t0[0,i]*10./3.,0))+' fs',fontdict=font)
          plt.subplots_adjust(top=0.98, bottom=0.14, left=0.12, right=0.97, hspace=0.10, wspace=0.15)
          fig = plt.gcf()
          fig.set_size_inches(10, 6.5)
          fig.savefig('./dN_dE_comb_s_'+str(i).zfill(4)+'.png',format='png',dpi=160)
          plt.close("all")
          print('plotting '+str(i).zfill(4))
          
  
          
          
