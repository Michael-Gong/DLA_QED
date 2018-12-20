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
        'size'   : 28,
       }

font2 = {'family' : 'monospace',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 15,
       }

font_size = 28
font_size_2 = 15
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
  part_number = 12000
  nsteps      = 101 #sum(1 for line in open(from_path+'x_0000.txt'))/part_number


  from_path = './Data_no_T050_p50000/'
  to_path   = from_path
  #x0  = np.loadtxt(from_path+'x_tot.txt')/2/np.pi
  #y0  = np.loadtxt(from_path+'y_tot.txt')/2/np.pi
  px0 = np.loadtxt(from_path+'photon_px_tot_s.txt')
  py0 = np.loadtxt(from_path+'photon_py_tot_s.txt')
  gg0 = (px0**2+py0**2)**0.5*0.51
  ww0 = np.zeros_like(gg0)+1
  #gg0 = np.log10(gg0)

  from_path = './Data_qe_T050_p50000/'
  to_path   = from_path
  px2 = np.loadtxt(from_path+'photon_px_tot_s.txt')
  py2 = np.loadtxt(from_path+'photon_py_tot_s.txt')
  gg2 = (px2**2+py2**2)**0.5*0.51
  ww2 = np.zeros_like(gg2)+1
  #gg2 = np.log10(gg2)

  ax=plt.subplot(1,1,1)

  width = 100
  kwargs = dict(histtype='stepfilled', alpha=0.4, normed=None, color='r', bins=100, range=(0,200), weights=ww0,label='w/o  RR')
  plt.hist(gg0, **kwargs)
  kwargs = dict(histtype='stepfilled', alpha=0.4, normed=None, color='b', bins=100, range=(0,200), weights=ww2,label='with RR')
  plt.hist(gg2, **kwargs)
  #### manifesting colorbar, changing label and axis properties ####
  plt.legend(loc='upper right',fontsize=font_size)
  plt.xlabel(r'$\varepsilon_\gamma$'+' [MeV]',fontdict=font)
  plt.ylabel('dN/d'+r'$\varepsilon_\gamma$'+' [A.U.]',fontdict=font)
  plt.xticks(fontsize=font_size); 
  plt.yticks([1e1,1e3,1e5,1e7],fontsize=font_size);
  plt.yscale('log')
#  plt.xscale('log')
  plt.xlim(0,200)
  plt.ylim(10,1e7)
  plt.legend(loc='best',fontsize=20,framealpha=0.5)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

  #plt.text(250,6e9,'t='+str(round(time/1.0e-15,0))+' fs',fontdict=font)
#  plt.title('t='+str(round(t1[0,i],0))+' $T_0$',fontdict=font)
  plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95, hspace=0.10, wspace=0.05)
  fig = plt.gcf()
  fig.set_size_inches(10, 7.)
  fig.savefig('ph_dN_dE_050.png',format='png',dpi=160)
  plt.close("all")
  #print('plotting '+str(i).zfill(4))
          
  
          
          
