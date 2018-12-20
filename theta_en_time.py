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

upper = matplotlib.cm.jet(np.arange(256))
lower = np.ones((int(256/4),4))
for i in range(3):
  lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
cmap = np.vstack(( lower, upper ))
mycolor_jet = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

upper = matplotlib.cm.viridis(np.arange(256))
lower = np.ones((int(256/4),4))
for i in range(3):
  lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
cmap = np.vstack(( lower, upper ))
mycolor_viridis = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

upper = matplotlib.cm.rainbow(np.arange(256))
lower = np.ones((int(256/4),4))
for i in range(3):
  lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
cmap = np.vstack(( lower, upper ))
mycolor_rainbow = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

def pxpy_to_energy(gamma, weight):
    binsize = 200
    en_grid = np.linspace(50,19950,200)
    en_bin  = np.linspace(0,20000.0,201)
    en_value = np.zeros_like(en_grid) 
    for i in range(binsize):
#        if i == binsize-1:
#            en_value[i] = sum(weight[en_bin[i]<=gamma])
#        else:
            en_value[i] = sum(weight[ (en_bin[i]<=gamma) & (gamma<en_bin[i+1]) ])
    return (en_grid, en_value)


def theta_to_grid(theta, weight):
    binsize = 240
    theta_grid = np.linspace(-119.5,119.5,240)
    theta_bin  = np.linspace(-120,120,241)
    theta_value = np.zeros_like(theta_grid) 
    for i in range(binsize):
#        if i == binsize-1:
#            en_value[i] = sum(weight[en_bin[i]<=gamma])
#        else:
            theta_value[i] = sum(weight[ (theta_bin[i]<=theta) & (theta<theta_bin[i+1]) ])
    return (theta_grid, theta_value)






if __name__ == "__main__":
  part_number = 50000
  from_path = './Data_no_T500_p50000/'
 # nsteps      = int(sum(1 for line in open(from_path+'t_tot_s.txt'))/part_number)

  ntheta = 180
  ngg    = 160

  from_path = './Data_no_T500_p50000/'
  to_path   = from_path
  #x0  = np.loadtxt(from_path+'x_tot_s.txt')/2/np.pi
  #y0  = np.loadtxt(from_path+'y_tot_s.txt')/2/np.pi
  px0 = np.loadtxt(from_path+'photon_px_tot_s.txt')
  py0 = np.loadtxt(from_path+'photon_py_tot_s.txt')
  #x0  = np.reshape(x0,(part_number,nsteps))
  #y0  = np.reshape(y0,(part_number,nsteps))
  gg0 = (px0**2+py0**2)**0.5*0.51
  ww0 = np.zeros_like(gg0)+gg0
  theta0 = np.arctan2(py0,px0)/np.pi*180.

  theta_edges = np.linspace(-90,90,  ntheta +1)
  gg_edges    = np.logspace(-1,4,  ngg +1)
  
  H, _, _ = np.histogram2d(gg0, theta0, [gg_edges, theta_edges], weights=gg0)
  print('Max H:',np.max(H))
  Theta, R = np.meshgrid(theta_edges,gg_edges)

  fig = plt.figure()
  ax  = fig.add_subplot(111)
  levels = np.logspace(2,5, 101)
  img = ax.pcolormesh(Theta,  R,  H, norm=colors.LogNorm(vmin=1e2, vmax=1e5), cmap=mycolor_rainbow)
  cax = fig.add_axes([0.65,0.94,0.25,0.02])
  cbar=fig.colorbar(img,cax=cax, ticks=[1e2,1e5],orientation='horizontal')
  cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), fontsize=font_size)
  cbar.set_label(r'dI/d$\theta$dE [A.U.]',fontdict=font)

      #ax.set_xlim(10,50)
      #ax.set_ylim(0.,1.)
  ax.set_xlabel(r'$\theta\ [^o]$',fontdict=font)
  ax.set_ylabel(r'$\varepsilon_{\gamma}$ [MeV]',fontdict=font)
  ax.set_yscale('log')
  ax.set_ylim(1e0,5e3)
  ax.set_xlim(-90,90)
  #ax.set_xticklabels([0,90,180,270])
  #ax.set_yticklabels([0.1,1,10,100,1000])

  #ax.set_theta_zero_location('N')
    #  ax.set_ylabel(r'$\theta\ [^o]$',fontdict=font)
  ax.tick_params(axis='x',labelsize=font_size) 
  ax.tick_params(axis='y',labelsize=font_size)
  #ax.set_title('proton_angular_time='+str(time1), va='bottom', y=1., fontsize=20)
    #  plt.text(-100,650,' t = '++' fs',fontdict=font)

#plt.pcolormesh(x, y, ex.T, norm=mpl.colors.Normalize(vmin=0,vmax=100,clip=True), cmap=cm.cubehelix_r)
#  plt.axis([x.min(), x.max(), y.min(), y.max()])
#### manifesting colorbar, changing label and axis properties ####
#  cbar=plt.colorbar(pad=0.01)#ticks=[np.min(ex), -eee/2, 0, eee/2, np.min()])
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=font_size)
#  cbar.set_label('dN/dE [A.U.]',fontdict=font)
#  a0=200.0
#  alpha=np.linspace(-3.5,0.5,501)
#  plt.xlabel(r'$\theta$'+' [degree]',fontdict=font)
#  plt.ylabel('time [fs]',fontdict=font)
#  plt.xticks([-135,-90,-45,0,45,90,135],fontsize=font_size); plt.yticks([0,500,1000,1500],fontsize=font_size);
#  plt.title(r'$dN/d\theta$'+' for no RR', fontsize=font_size)
#  plt.xlim(-120,120)
#  plt.ylim(0,1650)
#plt.title('electron at y='+str(round(y[n,0]/2/np.pi,4)),fontdict=font)

  plt.subplots_adjust(top=0.99, bottom=0.13, left=0.16, right=0.98, hspace=0.10, wspace=0.05)

  fig = plt.gcf()
  fig.set_size_inches(7.5, 6.5)
#fig.set_size_inches(5, 4.5)
  fig.savefig(to_path+'theta_en_dist.png',format='png',dpi=160)
  plt.close("all")

  
          
          
