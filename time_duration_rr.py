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

upper = matplotlib.cm.rainbow(np.arange(256))
lower = np.ones((int(256/4),4))
for i in range(3):
  lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
cmap = np.vstack(( lower, upper ))
mycolor_rainbow = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

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
  if 0>1:
      part_number = 10000
      
      nsteps      = int(sum(1 for line in open('./old_qe_b15000/'+'t_tot_s.txt'))/part_number)
    
      axis_time   = np.linspace(0,(nsteps-1.)*5.,nsteps)
      axis_alpha  = np.linspace(15000,25000,21)/1e4-4
      data_enhance= np.zeros([np.size(axis_alpha),np.size(axis_time)])
      for i in range(21):
          from_path = './old_qe_b'+str(int(i*500+15000))+'/'
          t0  = np.loadtxt(from_path+'t_tot_s.txt')/2/np.pi
          px0 = np.loadtxt(from_path+'px_tot_s.txt')
          py0 = np.loadtxt(from_path+'py_tot_s.txt')
          t0  = np.reshape(t0,(part_number,nsteps))
          px0 = np.reshape(px0,(part_number,nsteps))
          py0 = np.reshape(py0,(part_number,nsteps))
          gg0 = (px0**2+py0**2+1)**0.5
          ww0 = np.zeros_like(gg0)+1
          gg0_ave = np.sum(gg0,axis=0)/np.size(gg0[:,0])
          data_enhance[i,:] = gg0_ave/(200.)
          print('read data from '+str(int(i*500+15000))+' finished !') 
    
      np.save('./rr_axis_time',axis_time)
      np.save('./rr_axis_alpha',axis_alpha)
      np.save('./rr_data_enhance',data_enhance)

  axis_time=np.load('./rr_axis_time.npy')
  axis_alpha=np.load('./rr_axis_alpha.npy')
  data_enhance=np.load('./rr_data_enhance.npy')

  plt.subplot(1,1,1)        
  x,y=np.meshgrid(axis_alpha,axis_time[::5])
  levels = np.logspace(0, np.log10(4e3), 41)
  plt.pcolormesh(x, y*3.33, data_enhance[:,::5].T, norm=colors.LogNorm(vmin=1, vmax=100), cmap=mycolor_jet)
#plt.pcolormesh(x, y, ex.T, norm=mpl.colors.Normalize(vmin=0,vmax=100,clip=True), cmap=cm.cubehelix_r)
#  plt.axis([x.min(), x.max(), y.min(), y.max()])
#### manifesting colorbar, changing label and axis properties ####
  cbar=plt.colorbar(pad=0.003)#ticks=[np.min(ex), -eee/2, 0, eee/2, np.min()])
  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
#  cbar.set_label('dN/dE [A.U.]',fontdict=font)
#  a0=200.0
#  alpha=np.linspace(-3.5,0.5,501)
  plt.xlim(-2.5,-1.5)
  plt.ylim(100,1600)
  plt.xlabel('lg'+r'$\alpha$',fontdict=font)
  plt.ylabel('time [fs]',fontdict=font)
  plt.xticks([-2.5,-2.0,-1.5],fontsize=25); 
  plt.yticks([500,1000,1500],fontsize=25);
  plt.title('$\overline{\gamma}_e/\gamma_0$ for QED RR', fontsize=25)
#plt.title('electron at y='+str(round(y[n,0]/2/np.pi,4)),fontdict=font)

  plt.subplots_adjust(top=0.92, bottom=0.12, left=0.15, right=0.97, hspace=0.10, wspace=0.15)

  fig = plt.gcf()
  fig.set_size_inches(10, 8.5)
#fig.set_size_inches(5, 4.5)
  fig.savefig('./rr_data_enhance.png',format='png',dpi=160)
  plt.close("all")

  
          
          
