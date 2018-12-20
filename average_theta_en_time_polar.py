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
 # from_path = './Data_no_T250_p50000/'
 # nsteps      = int(sum(1 for line in open(from_path+'t_tot_s.txt'))/part_number)

  ntheta = 270
  ngg    = 120

  from_path_list = ['./Data_qe_T050_p50000/','./Data_qe_T250_p50000/','./Data_qe_T500_p50000/','./Data_no_T050_p50000/','./Data_no_T250_p50000/','./Data_no_T500_p50000/']
  #from_path_list = ['./Data_qe_T500_p50000_try/']

  for i in range(np.size(from_path_list)):
      from_path = from_path_list[i] #'./Data_qe_T050_p50000/'
      to_path   = from_path
      #x0  = np.loadtxt(from_path+'x_tot_s.txt')/2/np.pi
      #y0  = np.loadtxt(from_path+'y_tot_s.txt')/2/np.pi
      px0 = np.loadtxt(from_path+'photon_px_tot_s.txt')
      py0 = np.loadtxt(from_path+'photon_py_tot_s.txt')
      #x0  = np.reshape(x0,(part_number,nsteps))
      #y0  = np.reshape(y0,(part_number,nsteps))
      gg0 = (px0**2+py0**2)**0.5*0.51
      ww0 = np.zeros_like(gg0)+gg0
      theta0 = np.arctan2(py0,px0)
    
      theta_edges = np.linspace(-np.pi,np.pi,  ntheta +1)
      gg_edges    = np.logspace(-1,np.log10(3e3),  ngg +1)
    
      theta_edges_1 = np.linspace(-np.pi,np.pi,  ntheta)
      gg_edges_1    = np.linspace(0,np.log10(3e3),  ngg)
       
     
      H, _, _ = np.histogram2d(gg0, theta0, [gg_edges, theta_edges], weights=gg0)
      print('Max H:',np.max(H))
      Theta, R = np.meshgrid(theta_edges_1,gg_edges_1)
      H_temp = np.sum(H[:,:]*R,0)
      print('averaged |theta|=',np.sum(gg0*abs(theta0))/np.sum(gg0)/np.pi*180)
