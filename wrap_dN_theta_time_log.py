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
  nsteps      = int(sum(1 for line in open(from_path+'t_tot_s.txt'))/part_number)


  from_path = './Data_no_T500_p50000/'
  #to_path   = './Data_no/'
  t0  = np.loadtxt(from_path+'t_tot_s.txt')/2/np.pi
  #x0  = np.loadtxt(from_path+'x_tot_s.txt')/2/np.pi
  #y0  = np.loadtxt(from_path+'y_tot_s.txt')/2/np.pi
  px0 = np.loadtxt(from_path+'px_tot_s.txt')
  py0 = np.loadtxt(from_path+'py_tot_s.txt')
  t0  = np.reshape(t0,(part_number,nsteps))
  #x0  = np.reshape(x0,(part_number,nsteps))
  #y0  = np.reshape(y0,(part_number,nsteps))
  px0 = np.reshape(px0,(part_number,nsteps))
  py0 = np.reshape(py0,(part_number,nsteps))
  gg0 = (px0**2+py0**2+1)**0.5
  ww0 = np.zeros_like(gg0)+1
  theta0 = np.arctan2(py0,px0)/np.pi*180.


  from_path = './Data_rr_T500_p50000/'
  to_path   = './Data_rr_T500_p50000/'
  t1  = np.loadtxt(from_path+'t_tot_s.txt')/2/np.pi
  #x1  = np.loadtxt(from_path+'x_tot_s.txt')/2/np.pi
  #y1  = np.loadtxt(from_path+'y_tot_s.txt')/2/np.pi
  px1 = np.loadtxt(from_path+'px_tot_s.txt')
  py1 = np.loadtxt(from_path+'py_tot_s.txt')
  t1  = np.reshape(t1,(part_number,nsteps))
  #x1  = np.reshape(x1,(part_number,nsteps))
  #y1  = np.reshape(y1,(part_number,nsteps))
  px1 = np.reshape(px1,(part_number,nsteps))
  py1 = np.reshape(py1,(part_number,nsteps))
  gg1 = (px1**2+py1**2+1)**0.5
  ww1 = np.zeros_like(gg1)+1
  theta1 = np.arctan2(py1,px1)/np.pi*180.


  from_path = './Data_qe_T500_p50000/'
  to_path   = './Data_qe_T500_p50000/'
  t2  = np.loadtxt(from_path+'t_tot_s.txt')/2/np.pi
  #x2  = np.loadtxt(from_path+'x_tot_s.txt')/2/np.pi
  #y2  = np.loadtxt(from_path+'y_tot_s.txt')/2/np.pi
  px2 = np.loadtxt(from_path+'px_tot_s.txt')
  py2 = np.loadtxt(from_path+'py_tot_s.txt')
  t2  = np.reshape(t2,(part_number,nsteps))
  #x2  = np.reshape(x1,(part_number,nsteps))
  #y2  = np.reshape(y1,(part_number,nsteps))
  px2 = np.reshape(px2,(part_number,nsteps))
  py2 = np.reshape(py2,(part_number,nsteps))
  gg2 = (px2**2+py2**2+1)**0.5
  ww2 = np.zeros_like(gg2)+1
  theta2 = np.arctan2(py2,px2)/np.pi*180.



  axis_time = np.linspace(0,500,251)  
  axis_theta= np.linspace(-119.5,119.5,240)  
  data_no   = np.zeros([240,251])
  data_rr   = np.zeros([240,251])
  data_qe   = np.zeros([240,251])

  for i in range(nsteps):
      axis_temp, data_no[:,i] = theta_to_grid(theta0[:,i], ww0[:,i]) 
      axis_temp, data_qe[:,i] = theta_to_grid(theta2[:,i], ww2[:,i]) 
      axis_temp, data_rr[:,i] = theta_to_grid(theta1[:,i], ww1[:,i]) 

  plt.subplot(1,3,1)        
  x,y=np.meshgrid(axis_theta,axis_time)
  levels = np.logspace(1, np.log10(1e4), 41)
  plt.pcolormesh(x, y*10./3., data_no.T, norm=colors.LogNorm(vmin=10, vmax=1e4), cmap=mycolor_viridis)
#plt.pcolormesh(x, y, ex.T, norm=mpl.colors.Normalize(vmin=0,vmax=100,clip=True), cmap=cm.cubehelix_r)
  plt.axis([x.min(), x.max(), y.min(), y.max()])
#### manifesting colorbar, changing label and axis properties ####
#  cbar=plt.colorbar(pad=0.01)#ticks=[np.min(ex), -eee/2, 0, eee/2, np.min()])
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=font_size)
#  cbar.set_label('dN/dE [A.U.]',fontdict=font)
#  a0=200.0
#  alpha=np.linspace(-3.5,0.5,501)
  plt.xlabel(r'$\theta$'+' [degree]',fontdict=font)
  plt.ylabel('time [fs]',fontdict=font)
  plt.xticks([-135,-90,-45,0,45,90,135],fontsize=font_size); plt.yticks([0,500,1000,1500],fontsize=font_size);
#  plt.title(r'$dN/d\theta$'+' for no RR', fontsize=font_size)
  plt.xlim(-120,120)
  plt.ylim(0,1650)
#plt.title('electron at y='+str(round(y[n,0]/2/np.pi,4)),fontdict=font)

  plt.subplot(1,3,2)        
  x,y=np.meshgrid(axis_theta,axis_time)
  levels = np.logspace(1, np.log10(1e4), 41)
  plt.pcolormesh(x, y*10./3., data_qe.T, norm=colors.LogNorm(vmin=10, vmax=1e4), cmap=mycolor_viridis)
#plt.pcolormesh(x, y, ex.T, norm=mpl.colors.Normalize(vmin=0,vmax=100,clip=True), cmap=cm.cubehelix_r)
  plt.axis([x.min(), x.max(), y.min(), y.max()])
#### manifesting colorbar, changing label and axis properties ####
#  cbar=plt.colorbar(pad=0.01)#ticks=[np.min(ex), -eee/2, 0, eee/2, np.min()])
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=font_size)
#  cbar.set_label(r'$dN/d\theta$'+' [A.U.]',fontdict=font)
#  a0=200.0
#  alpha=np.linspace(-3.5,0.5,501)
  plt.xlabel(r'$\theta$'+' [degree]',fontdict=font)
  #plt.ylabel('time [fs]',fontdict=font)
  plt.xticks([-135,-90,-45,0,45,90,135],fontsize=font_size); plt.yticks([0,500,1000,1500],fontsize=0.000001);
#  plt.title(r'$dN/d\theta$'+' for QED RR', fontsize=font_size)
  plt.xlim(-120,120)
  plt.ylim(0,1650)
#plt.title('electron at y='+str(round(y[n,0]/2/np.pi,4)),fontdict=font)

  plt.subplot(1,3,3)        
  x,y=np.meshgrid(axis_theta,axis_time)
  levels = np.logspace(1, np.log10(1e4), 41)
  plt.pcolormesh(x, y*10./3., data_rr.T, norm=colors.LogNorm(vmin=10, vmax=1e4), cmap=mycolor_viridis)
#plt.pcolormesh(x, y, ex.T, norm=mpl.colors.Normalize(vmin=0,vmax=100,clip=True), cmap=cm.cubehelix_r)
  plt.axis([x.min(), x.max(), y.min(), y.max()])
#### manifesting colorbar, changing label and axis properties ####
#  cbar=plt.colorbar(pad=0.01)#ticks=[np.min(ex), -eee/2, 0, eee/2, np.min()])
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=font_size)
#  cbar.set_label(r'$dN/d\theta$'+' [A.U.]',fontdict=font)
#  a0=200.0
#  alpha=np.linspace(-3.5,0.5,501)
  plt.xlabel(r'$\theta$'+' [degree]',fontdict=font)
  #plt.ylabel('time [fs]',fontdict=font)
  plt.xticks([-135,-90,-45,0,45,90,135],fontsize=font_size); plt.yticks([0,500,1000,1500],fontsize=0.000001);
#  plt.title(r'$dN/d\theta$'+' for QED RR', fontsize=font_size)
  plt.xlim(-120,120)
  plt.ylim(0,1650)
#plt.title('electron at y='+str(round(y[n,0]/2/np.pi,4)),fontdict=font)

  plt.subplots_adjust(top=0.92, bottom=0.13, left=0.1, right=0.95, hspace=0.10, wspace=0.05)

  fig = plt.gcf()
  fig.set_size_inches(22, 6.5)
#fig.set_size_inches(5, 4.5)
  fig.savefig('./wrap_dN_theta.png',format='png',dpi=160)
  plt.close("all")

  
          
          
