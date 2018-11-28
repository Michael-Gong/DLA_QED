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

upper = matplotlib.cm.jet(np.arange(256))
lower = np.ones((int(256/4),4))
for i in range(3):
  lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
cmap = np.vstack(( lower, upper ))
mycolor_jet = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

def pxpy_to_energy(gamma, weight):
    binsize = 100
    en_grid = np.linspace(20,3980,100)
    en_bin  = np.linspace(0,4000.0,101)
    en_value = np.zeros_like(en_grid) 
    for i in range(binsize):
#        if i == binsize-1:
#            en_value[i] = sum(weight[en_bin[i]<=gamma])
#        else:
            en_value[i] = sum(weight[ (en_bin[i]<=gamma) & (gamma<en_bin[i+1]) ])
    return (en_grid, en_value)



if __name__ == "__main__":
  from_path = './Data_no/'
  to_path   = './Data_no/'
  
  
  part_number = 20000
  nsteps      = 101 #sum(1 for line in open(from_path+'x_0000.txt'))/part_number
 

  t1  = np.loadtxt(from_path+'t_tot.txt')/2/np.pi
  #x1  = np.loadtxt(from_path+'x_tot.txt')/2/np.pi
  #y1  = np.loadtxt(from_path+'y_tot.txt')/2/np.pi
  px1 = np.loadtxt(from_path+'px_tot.txt')/2/np.pi
  py1 = np.loadtxt(from_path+'py_tot.txt')/2/np.pi
  
  t1  = np.reshape(t1,(part_number,nsteps))
  #x1  = np.reshape(x1,(part_number,nsteps))
  #y1  = np.reshape(y1,(part_number,nsteps))
  px1 = np.reshape(px1,(part_number,nsteps))
  py1 = np.reshape(py1,(part_number,nsteps))
   
  gg1 = (px1**2+py1**2+1)**0.5
  ww1 = np.zeros_like(gg1)+1


  from_path = './Data_qe/'
  to_path   = './Data_qe/'

  t2  = np.loadtxt(from_path+'t_tot.txt')/2/np.pi
  #x2  = np.loadtxt(from_path+'x_tot.txt')/2/np.pi
  #y2  = np.loadtxt(from_path+'y_tot.txt')/2/np.pi
  px2 = np.loadtxt(from_path+'px_tot.txt')/2/np.pi
  py2 = np.loadtxt(from_path+'py_tot.txt')/2/np.pi
  
  t2  = np.reshape(t2,(part_number,nsteps))
  #x2  = np.reshape(x1,(part_number,nsteps))
  #y2  = np.reshape(y1,(part_number,nsteps))
  px2 = np.reshape(px2,(part_number,nsteps))
  py2 = np.reshape(py2,(part_number,nsteps))
   
  gg2 = (px2**2+py2**2+1)**0.5
  ww2 = np.zeros_like(gg2)+1

  axis_time = np.linspace(0,1000,101)  
  axis_gg   = np.linspace(20,3980,100)  
  data_no   = np.zeros([100,101])
  data_qe   = np.zeros([100,101])

  for i in range(nsteps):
      axis_temp, data_no[:,i] = pxpy_to_energy(gg1[:,i], ww1[:,i]) 
      axis_temp, data_qe[:,i] = pxpy_to_energy(gg2[:,i], ww2[:,i]) 

  plt.subplot(1,2,1)        
  x,y=np.meshgrid(axis_gg,axis_time)
  levels = np.logspace(0, np.log10(4e3), 41)
  plt.pcolormesh(x, y, data_no.T, norm=colors.LogNorm(vmin=1, vmax=4e3), cmap=mycolor_jet)
#plt.pcolormesh(x, y, ex.T, norm=mpl.colors.Normalize(vmin=0,vmax=100,clip=True), cmap=cm.cubehelix_r)
  plt.axis([x.min(), x.max(), y.min(), y.max()])
#### manifesting colorbar, changing label and axis properties ####
  cbar=plt.colorbar(pad=0.1)#ticks=[np.min(ex), -eee/2, 0, eee/2, np.min()])
  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
  cbar.set_label('dN/dE [A.U.]',fontdict=font)
#  a0=200.0
#  alpha=np.linspace(-3.5,0.5,501)
  plt.xlabel('$\gamma$',fontdict=font)
  plt.ylabel('time [$T_0$]',fontdict=font)
  plt.xticks([0,1000,2000,3000,4000],fontsize=25); plt.yticks(fontsize=25);
  plt.title('dN/dE for no RR', fontsize=25)
  plt.xlim(0,4000)
  plt.ylim(0,1000)
#plt.title('electron at y='+str(round(y[n,0]/2/np.pi,4)),fontdict=font)

  plt.subplot(1,2,2)        
  x,y=np.meshgrid(axis_gg,axis_time)
  levels = np.logspace(0, 4, 41)
  plt.pcolormesh(x, y, data_qe.T, norm=colors.LogNorm(vmin=1, vmax=1e4), cmap=mycolor_jet)
#plt.pcolormesh(x, y, ex.T, norm=mpl.colors.Normalize(vmin=0,vmax=100,clip=True), cmap=cm.cubehelix_r)
  plt.axis([x.min(), x.max(), y.min(), y.max()])
#### manifesting colorbar, changing label and axis properties ####
  cbar=plt.colorbar(pad=0.1)#ticks=[np.min(ex), -eee/2, 0, eee/2, np.min()])
  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
  cbar.set_label('dN/dE [A.U.]',fontdict=font)
#  a0=200.0
#  alpha=np.linspace(-3.5,0.5,501)
  plt.xlabel('$\gamma$',fontdict=font)
  plt.ylabel('time [$T_0$]',fontdict=font)
  plt.xticks([0,1000,2000,3000,4000],fontsize=25); plt.yticks(fontsize=25);
  plt.title('dN/dE for QED RR', fontsize=25)
  plt.xlim(0,4000)
  plt.ylim(0,1000)
#plt.title('electron at y='+str(round(y[n,0]/2/np.pi,4)),fontdict=font)

  plt.subplots_adjust(top=0.92, bottom=0.12, left=0.1, right=0.95, hspace=0.10, wspace=0.12)

  fig = plt.gcf()
  fig.set_size_inches(18, 7.5)
#fig.set_size_inches(5, 4.5)
  fig.savefig('./jpg/figure.png',format='png',dpi=160)
  plt.close("all")

  
          
          
