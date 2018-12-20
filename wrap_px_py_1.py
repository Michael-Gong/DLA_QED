import sdf
import matplotlib
matplotlib.use('agg')
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
#from numpy import ma
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal
from optparse import OptionParser
import os
import matplotlib.colors as mcolors


######## Constant defined here ########
pi        =     3.1415926535897932384626
q0        =     1.602176565e-19 # C
m0        =     9.10938291e-31  # kg
v0        =     2.99792458e8    # m/s^2
kb        =     1.3806488e-23   # J/K
mu0       =     4.0e-7*pi       # N/A^2
epsilon0  =     8.8541878176203899e-12 # F/m
h_planck  =     6.62606957e-34  # J s
wavelength=     1.0e-6
frequency =     v0*2*pi/wavelength

exunit    =     m0*v0*frequency/q0
bxunit    =     m0*frequency/q0
denunit    =     frequency**2*epsilon0*m0/q0**2
print('electric field unit: '+str(exunit))
print('magnetic field unit: '+str(bxunit))
print('density unit nc: '+str(denunit))

font = {'family' : 'monospace',  
        'style'  : 'normal',
        'color'  : 'black',  
	    'weight' : 'normal',  
        'size'   : 20,  
       }  
######### Parameter you should set ###########

upper = matplotlib.cm.jet(np.arange(256))
lower = np.ones((int(256/4),4))
for i in range(3):
    lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
cmap = np.vstack(( lower, upper ))
mycolor_jet = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])


if __name__ == "__main__":
  part_number = 1200
  nsteps      = 801 #sum(1 for line in open(from_path+'x_0000.txt'))/part_number

  from_path = './Data_no_T400/'
  to_path   = './jpg_no_T400/'
  t0  = np.loadtxt(from_path+'t_tot_s.txt')/2/np.pi
  x0  = np.loadtxt(from_path+'x_tot_s.txt')/2/np.pi
  y0  = np.loadtxt(from_path+'y_tot_s.txt')/2/np.pi
  px0 = np.loadtxt(from_path+'px_tot_s.txt')
  py0 = np.loadtxt(from_path+'py_tot_s.txt')
  t0  = np.reshape(t0,(part_number,nsteps))
  x0  = np.reshape(x0,(part_number,nsteps))
  y0  = np.reshape(y0,(part_number,nsteps))
  px0 = np.reshape(px0,(part_number,nsteps))
  py0 = np.reshape(py0,(part_number,nsteps))
  gg0 = (px0**2+py0**2+1)**0.5
  ww0 = np.zeros_like(gg0)+1
  dy0 = y0[:,1:]*y0[:,:-1]
  px0 = px0[:,:-1]
  py0 = py0[:,:-1]
  t0  = t0[:,:-1]


  from_path = './Data_qe_T400/'
  to_path   = './jpg_qe_T400/'
  t1  = np.loadtxt(from_path+'t_tot_s.txt')/2/np.pi
  x1  = np.loadtxt(from_path+'x_tot_s.txt')/2/np.pi
  y1  = np.loadtxt(from_path+'y_tot_s.txt')/2/np.pi
  px1 = np.loadtxt(from_path+'px_tot_s.txt')
  py1 = np.loadtxt(from_path+'py_tot_s.txt')
  t1  = np.reshape(t1,(part_number,nsteps))
  x1  = np.reshape(x1,(part_number,nsteps))
  y1  = np.reshape(y1,(part_number,nsteps))
  px1 = np.reshape(px1,(part_number,nsteps))
  py1 = np.reshape(py1,(part_number,nsteps))
  gg1 = (px1**2+py1**2+1)**0.5
  ww1 = np.zeros_like(gg1)+1
  dy1 = y1[:,1:]*y1[:,:-1]
  px1 = px1[:,:-1]
  py1 = py1[:,:-1]
  t1  = t1[:,:-1]

  print(dy1[0,:])


  px = np.linspace(-400,10400,501)
  py = np.linspace(-800,800,401)

  px, py = np.meshgrid(px, py)

  R = (1+px**2+py**2)**0.5-px
  R[R<0.1]=0.1
  levels = np.logspace(-1,3,101)
  print(np.min(R),np.max(R))
  plt.contourf(px, py, R, levels=levels, norm=mcolors.LogNorm(vmin=levels.min(), vmax=levels.max()), cmap='RdBu')
  #plt.contourf(px, py, R, levels=levels, norm=mcolors.LogNorm(vmin=levels.min(), vmax=levels.max()), cmap=mycolor_jet)
  cbar=plt.colorbar(pad=0.005, ticks=np.logspace(-1,3,5))#,orientation="horizontal")
  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
  cbar.set_label('R', fontdict=font)

   
  condition= (dy0<0)
  plt.scatter(px0[condition], py0[condition], c=t0[condition], norm=colors.Normalize(vmin=0,vmax=400), s=20, cmap='autumn', marker=(5, 1), edgecolors='None', alpha=1)
  plt.scatter(px1[condition], py1[condition], c=t1[condition], norm=colors.Normalize(vmin=0,vmax=400), s=20, cmap='winter', edgecolors='None', alpha=1)
#  cbar=plt.colorbar( ticks=np.linspace(0, 2000, 5) ,pad=0.005)
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
#  cbar.set_label('$\gamma$',fontdict=font)


  plt.xlabel('$p_x$ [$m_ec$]',fontdict=font)
  plt.ylabel('$p_y$ [$m_ec$]',fontdict=font)
  plt.xticks([0,5000,10000],fontsize=20); plt.yticks([-500,0,500],fontsize=20);
  #plt.xscale('log')
  plt.xlim(-200,2200)
  plt.ylim(-300,300)
  plt.subplots_adjust(left=0.16, bottom=None, right=0.97, top=None,
                wspace=None, hspace=None)

  #plt.show()
  #lt.figure(figsize=(100,100))
  fig = plt.gcf()
  fig.set_size_inches(12, 6.5)
  fig.savefig('./p_scatter_.png',format='png',dpi=160)
  plt.close("all")
  print('plotting '+str(i).zfill(4))
