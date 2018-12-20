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
        'size'   : 25,  
       }  
font_size = 25
######### Parameter you should set ###########

upper = matplotlib.cm.jet(np.arange(256))
lower = np.ones((int(256/4),4))
for i in range(3):
    lower[:,i] = np.linspace(1, upper[0,i], lower.shape[0])
cmap = np.vstack(( lower, upper ))
mycolor_jet = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])


if __name__ == "__main__":
  part_number = 2
  nsteps      = 79580 #sum(1 for line in open(from_path+'x_0000.txt'))/part_number

  from_path = './Data_no_part01/'
  to_path   = './jpg_no_part01/'
  t0  = np.loadtxt(from_path+'t_0000.txt')/2/np.pi
  x0  = np.loadtxt(from_path+'x_0000.txt')/2/np.pi
  y0  = np.loadtxt(from_path+'y_0000.txt')/2/np.pi
  px0 = np.loadtxt(from_path+'px_0000.txt')
  py0 = np.loadtxt(from_path+'py_0000.txt')
  t0  = np.reshape(t0,(part_number,nsteps))
  x0  = np.reshape(x0,(part_number,nsteps))
  y0  = np.reshape(y0,(part_number,nsteps))
  px0 = np.reshape(px0,(part_number,nsteps))
  py0 = np.reshape(py0,(part_number,nsteps))
  gg0 = (px0**2+py0**2+1)**0.5
  ww0 = np.zeros_like(gg0)+1
  R0  = gg0-px0

  n=1
  print('no max gg:',np.max(gg0[n,:]))
  plt.scatter(px0[n,:], py0[n,:], c=gg0[n,:], norm=colors.Normalize(vmin=0,vmax=500), s=10, cmap='autumn', edgecolors='None', alpha=1,zorder=3)
#  cbar=plt.colorbar( ticks=np.linspace(0, 500, 3) ,pad=0.005)
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=font_size)
#  cbar.set_label('$\gamma$',fontdict=font)



  from_path = './Data_qe_part01/'
  to_path   = './jpg_qe_part01/'
  t1  = np.loadtxt(from_path+'t_0000.txt')/2/np.pi
  x1  = np.loadtxt(from_path+'x_0000.txt')/2/np.pi
  y1  = np.loadtxt(from_path+'y_0000.txt')/2/np.pi
  px1 = np.loadtxt(from_path+'px_0000.txt')
  py1 = np.loadtxt(from_path+'py_0000.txt')
  radt1 = np.loadtxt(from_path+'radt_0000.txt')
  radn1 = np.loadtxt(from_path+'radn_0000.txt')
  t1  = np.reshape(t1,(part_number,nsteps))
  x1  = np.reshape(x1,(part_number,nsteps))
  y1  = np.reshape(y1,(part_number,nsteps))
  px1 = np.reshape(px1,(part_number,nsteps))
  py1 = np.reshape(py1,(part_number,nsteps))
  radt1 = np.reshape(radt1,(part_number,nsteps))
  radn1 = np.reshape(radn1,(part_number,nsteps))
  gg1 = (px1**2+py1**2+1)**0.5
  ww1 = np.zeros_like(gg1)+1
  R1  = gg1-px1


  print('qe max gg:',np.max(gg1[n,:]))
  plt.scatter(px1[n,:], py1[n,:], c=gg1[n,:], norm=colors.Normalize(vmin=0,vmax=12000), s=10, cmap='winter', edgecolors='None', alpha=1,zorder=1)
#  cbar=plt.colorbar( ticks=np.linspace(0, 12000, 3) ,pad=0.005)
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=font_size)
#  cbar.set_label('$\gamma$',fontdict=font)
  condition = ((radn1[n,1:]-radn1[n,:-1]) > 0) & ((radt1[n,1:]-radt1[n,:-1]) > 50.0 )
  plt.scatter(px1[n,:-1][condition], py1[n,:-1][condition], c=(radt1[n,1:]-radt1[n,:-1])[condition]*0.51, norm=colors.Normalize(vmin=50,vmax=500), s=250, cmap='hot', marker='*', edgecolors='k', alpha=1,zorder=2)
#  cbar=plt.colorbar( ticks=np.linspace(0, 500, 6) ,pad=0.005)
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=font_size)
#  cbar.set_label('$E_{ph}$ [MeV]',fontdict=font)


  px = np.linspace(-400,14000,501)
  py = np.linspace(-800,800,401)

  px, py = np.meshgrid(px, py)

  R = (1+px**2+py**2)**0.5-px
  R[R<0.1]=0.1
  levels = np.logspace(1,3,5)
  print(np.min(R),np.max(R))
  plt.contour(px, py, R, levels=levels, norm=mcolors.LogNorm(vmin=levels.min(), vmax=levels.max()), linestyles='dashed', cmap='gist_gray',zorder=0,linewidths=2.5)
  #plt.contourf(px, py, R, levels=levels, norm=mcolors.LogNorm(vmin=levels.min(), vmax=levels.max()), cmap=mycolor_jet)
#  cbar=plt.colorbar(pad=0.1, ticks=np.logspace(0,3,7))#,orientation="horizontal")
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=font_size)
#  cbar.set_label('R', fontdict=font)

   
#  plt.scatter(px0[condition], py0[condition], c=t0[condition], norm=colors.Normalize(vmin=0,vmax=400), s=20, cmap='nipy_spectral', marker=(5, 1), edgecolors='None', alpha=1)
#  plt.scatter(px1[condition], py1[condition], c=t1[condition], norm=colors.Normalize(vmin=0,vmax=400), s=20, cmap='winter', edgecolors='None', alpha=1)
#  cbar=plt.colorbar( ticks=np.linspace(0, 2000, 5) ,pad=0.005)
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
#  cbar.set_label('$\gamma$',fontdict=font)


  plt.xlabel('$p_x$ [$m_ec$]',fontdict=font)
  plt.ylabel('$p_y$ [$m_ec$]',fontdict=font)
  plt.xticks([0,5000,10000,15000],fontsize=font_size); plt.yticks([-500,0,500],fontsize=font_size);
  #plt.xscale('log')
  plt.xlim(-200,14000)
  plt.ylim(-800,800)
  plt.subplots_adjust(left=0.16, bottom=0.16, right=0.99, top=0.98,
                wspace=None, hspace=None)

  #plt.show()
  #lt.figure(figsize=(100,100))
  fig = plt.gcf()
  fig.set_size_inches(10, 8.)
  fig.savefig('./wrap_px_py.png',format='png',dpi=160)
  plt.close("all")
