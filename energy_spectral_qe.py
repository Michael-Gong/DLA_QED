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

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')


def pxpy_to_energy(gamma, weight):
    binsize = 100
    en_grid = np.linspace(25,4975,5000)
    en_bin  = np.linspace(0,5000.0,5001)
    en_value = np.zeros_like(en_grid) 
    for i in range(binsize):
#        if i == binsize-1:
#            en_value[i] = sum(weight[en_bin[i]<=gamma])
#        else:
            en_value[i] = sum(weight[ (en_bin[i]<=gamma) & (gamma<en_bin[i+1]) ])
    return (en_grid, en_value)



if __name__ == "__main__":
  from_path = './Data_qe/'
  to_path   = './Data_qe/'
  
  
  part_number = 1000
  core_number = 50
  part_per_txt= int(part_number/core_number)
  nsteps      = sum(1 for line in open(from_path+'x_0000.txt'))/part_per_txt
  
  part_tt = np.zeros([part_number,nsteps])
  part_xx = np.zeros([part_number,nsteps])
  part_yy = np.zeros([part_number,nsteps])
  part_px = np.zeros([part_number,nsteps])
  part_py = np.zeros([part_number,nsteps])
  
  for n_th in range(core_number):
          insert_n='_'+str(n_th).zfill(4)
          t1  = np.loadtxt(from_path+'t'+insert_n+'.txt')/2/np.pi
          y1  = np.loadtxt(from_path+'y'+insert_n+'.txt')/2/np.pi
          x1  = np.loadtxt(from_path+'x'+insert_n+'.txt')/2/np.pi
          px1 = np.loadtxt(from_path+'px'+insert_n+'.txt')
          py1 = np.loadtxt(from_path+'py'+insert_n+'.txt')
          
          part_tt[n_th*part_per_txt:(n_th+1)*part_per_txt,:] = np.reshape(t1,(part_per_txt,nsteps))
          part_xx[n_th*part_per_txt:(n_th+1)*part_per_txt,:] = np.reshape(x1,(part_per_txt,nsteps))
          part_yy[n_th*part_per_txt:(n_th+1)*part_per_txt,:] = np.reshape(y1,(part_per_txt,nsteps))
          part_px[n_th*part_per_txt:(n_th+1)*part_per_txt,:] = np.reshape(px1,(part_per_txt,nsteps))
          part_py[n_th*part_per_txt:(n_th+1)*part_per_txt,:] = np.reshape(py1,(part_per_txt,nsteps))
   
          print('finishing '+insert_n+' !')
  
  part_gg = (part_px**2+part_py**2+1)**0.5
  part_ww = np.zeros_like(part_xx)+1
  for i in list(range(0,1000,10)):
#          dist_x, den = pxpy_to_energy(part_gg[:,i],part_ww[:,i])
#          plt.plot(dist_x,den,'-k',linewidth=4,label='Total')
          width = 100
#          kwargs = dict(histtype='stepfilled', alpha=0.3, normed=None, color='b', bins=300, range=(0,600), weights=ww_1/2)
          kwargs = dict(histtype='stepfilled', alpha=0.9, normed=None, color='b', bins=100, range=(0,10000), weights=part_ww[:,i])
          plt.hist(part_gg[:,i], **kwargs)
          #### manifesting colorbar, changing label and axis properties ####
          plt.xlabel('$\gamma$',fontdict=font)
          plt.ylabel('dN/dE [A.U.]',fontdict=font)
          plt.xticks(fontsize=20); plt.yticks(fontsize=20);
          #plt.yscale('log')
          plt.xlim(0,5000)
          plt.legend(loc='best',fontsize=20,framealpha=0.5)
          #plt.text(250,6e9,'t='+str(round(time/1.0e-15,0))+' fs',fontdict=font)
          plt.title('t='+str(round(part_tt[0,i],0))+' $T_0$',fontdict=font)
          fig = plt.gcf()
          fig.set_size_inches(10, 7.)
          fig.savefig(to_path+'dN_dE_qe'+str(i).zfill(4)+'.png',format='png',dpi=80)
          plt.close("all")
          print('plotting '+str(i).zfill(4))
          
  
          
          
