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
  part_number = 2
  nsteps      = 79580 #sum(1 for line in open(from_path+'x_0000.txt'))/part_number
  n=1
  f,(ax,ax2) = plt.subplots(1,2,sharey=True,facecolor='w')
  from_path = './Data_qe_part01/'
  to_path   = './jpg_qe_part01/'
  t1  = np.loadtxt(from_path+'t_0000.txt')/2/np.pi
  x1  = np.loadtxt(from_path+'x_0000.txt')/2/np.pi
  y1  = np.loadtxt(from_path+'y_0000.txt')/2/np.pi
  px1 = np.loadtxt(from_path+'px_0000.txt')
  py1 = np.loadtxt(from_path+'py_0000.txt')
  radt1 = np.loadtxt(from_path+'radt_0000.txt')
  radn1 = np.loadtxt(from_path+'radn_0000.txt')
  radpx1 = np.loadtxt(from_path+'rad_px_0000.txt')
  t1  = np.reshape(t1,(part_number,nsteps))
  x1  = np.reshape(x1,(part_number,nsteps))
  y1  = np.reshape(y1,(part_number,nsteps))
  px1 = np.reshape(px1,(part_number,nsteps))
  py1 = np.reshape(py1,(part_number,nsteps))
  radt1 = np.reshape(radt1,(part_number,nsteps))
  radn1 = np.reshape(radn1,(part_number,nsteps))
  radpx1 = np.reshape(radpx1,(part_number,nsteps))
  gg1 = (px1**2+py1**2+1)**0.5
  ww1 = np.zeros_like(gg1)+1
  R1  = gg1-px1
  C1  = (py1[1,0]**2+1)**0.5 - (radt1-radpx1)

  alpha=10**(-1.5)
  y_max = (C1/alpha)**0.5/2/np.pi


  ax.scatter(x1[n,:], y1[n,:], c=gg1[n,:], norm=colors.Normalize(vmin=1000,vmax=14000), s=10, cmap='rainbow', edgecolors='None', alpha=1,zorder=3)
  ax2.scatter(x1[n,:], y1[n,:], c=gg1[n,:], norm=colors.Normalize(vmin=1000,vmax=14000), s=10, cmap='rainbow', edgecolors='None', alpha=1,zorder=3)
  ax.plot(x1[n,:], y_max[n,:],'--k',linewidth=2.5,label=r'$Reduced\ w=1$',zorder=4)
  ax2.plot(x1[n,:], y_max[n,:],'--k',linewidth=2.5,label=r'$Reduced\ w=1$',zorder=4)
  ax.plot(x1[n,:], -y_max[n,:],'--k',linewidth=2.5,label=r'$Reduced\ w=1$',zorder=5)
  ax2.plot(x1[n,:], -y_max[n,:],'--k',linewidth=2.5,label=r'$Reduced\ w=1$',zorder=5)
  condition = ((radn1[n,1:]-radn1[n,:-1]) > 0) & ((radt1[n,1:]-radt1[n,:-1]) > 50.0 )
  ax.scatter(x1[n,:-1][condition], y1[n,:-1][condition], c=(radt1[n,1:]-radt1[n,:-1])[condition]*0.51, norm=colors.Normalize(vmin=50,vmax=500), s=150, cmap='hot', marker='*', edgecolors='k', alpha=1,zorder=6)
  ax2.scatter(x1[n,:-1][condition], y1[n,:-1][condition], c=(radt1[n,1:]-radt1[n,:-1])[condition]*0.51, norm=colors.Normalize(vmin=50,vmax=500), s=150, cmap='hot', marker='*', edgecolors='k', alpha=1,zorder=6)
#  cbar=plt.colorbar( ticks=np.linspace(0, 500, 6) ,pad=0.005)
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
#  cbar.set_label('$E_{ph}$ [MeV]',fontdict=font)


  x = x1[n,:] #np.linspace(-20,500,2000)
  y = np.linspace(-10,10,101)
  X, Y = np.meshgrid(x, y)
  
  xi = (t1-x1)[n,:]
  XI,Y1 = np.meshgrid(xi,y)  

  print(np.shape(x),np.shape(y_max[n,:]))
  Y_max, Y2 = np.meshgrid(y_max[n,:],y)
#  R = (1+px**2+py**2)**0.5-px
#  R[R<0.1]=0.1
  Ey = np.cos(XI*2*np.pi) 
  Ey[abs(Y)>Y_max] = 0
  levels = np.linspace(-1,1,101)
#  print(np.min(R),np.max(R))
#  plt.contour(px, py, R, levels=levels, norm=mcolors.LogNorm(vmin=levels.min(), vmax=levels.max()), linestyles='dashed', cmap='copper',zorder=0)
  ax.contourf(X, Y, Ey, levels=levels, norm=mcolors.Normalize(vmin=levels.min(), vmax=levels.max()), cmap='bwr',zorder=1,alpha=1)
  ax2.contourf(X, Y, Ey, levels=levels, norm=mcolors.Normalize(vmin=levels.min(), vmax=levels.max()), cmap='bwr',zorder=1,alpha=1)
  #cbar=ax2.colorbar(pad=0.005, ticks=np.linspace(-1,1,3))#,orientation="horizontal")
  #cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
  #cbar.set_label('$E_y$', fontdict=font)

#  px, py = np.meshgrid(px, py)

#  R = (1+px**2+py**2)**0.5-px
#  R[R<0.1]=0.1
#  levels = np.logspace(1,3,5)
#  print(np.min(R),np.max(R))
#  plt.contour(px, py, R, levels=levels, norm=mcolors.LogNorm(vmin=levels.min(), vmax=levels.max()), linestyles='dashed', cmap='copper',zorder=0)
  #plt.contourf(px, py, R, levels=levels, norm=mcolors.LogNorm(vmin=levels.min(), vmax=levels.max()), cmap=mycolor_jet)
#  cbar=plt.colorbar(pad=0.005, ticks=np.logspace(0,3,7))#,orientation="horizontal")
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
#  cbar.set_label('R', fontdict=font)

   
#  plt.scatter(px0[condition], py0[condition], c=t0[condition], norm=colors.Normalize(vmin=0,vmax=400), s=20, cmap='nipy_spectral', marker=(5, 1), edgecolors='None', alpha=1)
#  plt.scatter(px1[condition], py1[condition], c=t1[condition], norm=colors.Normalize(vmin=0,vmax=400), s=20, cmap='winter', edgecolors='None', alpha=1)
#  cbar=plt.colorbar( ticks=np.linspace(0, 2000, 5) ,pad=0.005)
#  cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
#  cbar.set_label('$\gamma$',fontdict=font)

  ax.set_xlim(0,50)
  ax2.set_xlim(250,500)
  ax.spines['right'].set_visible(False)
  ax2.spines['left'].set_visible(False)
  ax.yaxis.tick_left()
  ax.tick_params(labelright='off')
  ax2.yaxis.tick_right()

  #plt.xlabel('X [$\mu m$]',fontdict=font)
  #plt.ylabel('Y [$\mu m$]',fontdict=font)
  #plt.xticks([0,5000,15000],fontsize=20); plt.yticks([-500,0,500],fontsize=20);
  #plt.xscale('log')
  #plt.xlim(00,50)
  #plt.ylim(-800,800)
  plt.subplots_adjust(left=0.16, bottom=None, right=0.97, top=None,
                wspace=None, hspace=None)

  #plt.show()
  #lt.figure(figsize=(100,100))
  fig = plt.gcf()
  fig.set_size_inches(12, 7.5)
  fig.savefig('./wrap_x_y_qe.png',format='png',dpi=160)
  plt.close("all")
