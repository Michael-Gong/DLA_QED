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


if __name__ == "__main__":
  part_number = 2
  from_path = './Data_no_part01/'
  nsteps      = int(sum(1 for line in open(from_path+'x_0000.txt'))/part_number)


  from_path = './Data_no_part01/'
  to_path   = './Data_no_part01/'
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

  plt.subplot(1,1,1)
#  for n in range(25,75,10):
  n=1
  #plt.scatter((t0-x0)[n,:], (gg0-px0)[n,:], c=np.zeros_like(px0[n,:])+py0[n,0], norm=colors.Normalize(vmin=50,vmax=150), s=5, cmap='rainbow', edgecolors='None', alpha=1)
  plt.plot((t0-x0)[n,:], np.zeros_like(px0[n,:])+py0[n,0],':',color='k',linewidth=3, zorder=0)
  plt.scatter((t0-x0)[n,:], (gg0-px0)[n,:], c=gg0[n,:],cmap='autumn', norm=colors.Normalize(vmin=0,vmax=500), s=4, edgecolors='None', alpha=1,zorder=1)
 #   plt.legend(loc='upper right')

  from_path = './Data_qe_part01/'
  to_path   = './Data_qe_part01/'
  t0  = np.loadtxt(from_path+'t_0000.txt')/2/np.pi
  x0  = np.loadtxt(from_path+'x_0000.txt')/2/np.pi
  y0  = np.loadtxt(from_path+'y_0000.txt')/2/np.pi
  px0 = np.loadtxt(from_path+'px_0000.txt')
  py0 = np.loadtxt(from_path+'py_0000.txt')
  radt0 = np.loadtxt(from_path+'radt_0000.txt')
  radn0 = np.loadtxt(from_path+'radn_0000.txt')
  rad_px0 = np.loadtxt(from_path+'rad_px_0000.txt')
  eta0 = np.loadtxt(from_path+'eta_0000.txt')

  t0  = np.reshape(t0,(part_number,nsteps))
  x0  = np.reshape(x0,(part_number,nsteps))
  y0  = np.reshape(y0,(part_number,nsteps))
  px0 = np.reshape(px0,(part_number,nsteps))
  py0 = np.reshape(py0,(part_number,nsteps))
  radt0 = np.reshape(radt0,(part_number,nsteps))
  radn0 = np.reshape(radn0,(part_number,nsteps))
  rad_px0 = np.reshape(rad_px0,(part_number,nsteps))
  eta0 = np.reshape(eta0,(part_number,nsteps))
  gg0 = (px0**2+py0**2+1)**0.5
  ww0 = np.zeros_like(gg0)+1

  rad_C0 = radt0-rad_px0
#  for n in range(25,75,10):
  n=1
  plt.plot((t0-x0)[n,:], np.zeros_like(px0[n,:])+py0[n,0]-rad_C0[n,:],'--',color='k',linewidth=3,zorder=2)
  plt.scatter((t0-x0)[n,:], (gg0-px0)[n,:], c=gg0[n,:], cmap='winter', norm=colors.Normalize(vmin=0,vmax=12000), s=4, edgecolors='None', alpha=1,zorder=3)
  condition = ((radn0[n,1:]-radn0[n,:-1]) > 0) & ((rad_C0[n,1:]-rad_C0[n,:-1]) > 1.0 )
  #plt.scatter((t0-x0)[n,:-1][condition], (gg0-px0)[n,:-1][condition], c=(rad_C0[n,1:]-rad_C0[n,:-1])[condition], norm=colors.Normalize(vmin=1,vmax=2.4), s=120, cmap='viridis', marker='*', edgecolors='k', alpha=1,zorder=4)

  #cbar=plt.colorbar( ticks=np.linspace(1, 2.4, 6) ,pad=0.005)
  #cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=font_size)
  #cbar.set_label('$C_{rad}$',fontdict=font)
#plt.plot(np.linspace(-500,900,1001), 200-np.linspace(-500,900,1001),'-',color='grey',linewidth=3)
 #   plt.legend(loc='upper right')
  plt.xlim(-0.2,30.2)
  plt.ylim(0,105)
  plt.xlabel(r'$\xi$'+'$\ [2\pi]$',fontdict=font)
  plt.ylabel('$R=\gamma-p_x/m_ec$',fontdict=font)
  plt.xticks(fontsize=font_size); plt.yticks([0,25,50,75,100],fontsize=font_size);
#  plt.title('t='+str(round(t0[0,i],0))+' $T_0$',fontdict=font)
  #plt.text(-100,650,' t = 400 fs',fontdict=font)

  plt.subplots_adjust(left=0.15, bottom=0.16, right=0.98, top=0.98,
                wspace=None, hspace=None)

  plt.show()
  #lt.figure(figsize=(100,100))
  fig = plt.gcf()
  fig.set_size_inches(10, 6.5)
  fig.savefig('wrap_R_t.png',format='png',dpi=160)
  plt.close("all")
