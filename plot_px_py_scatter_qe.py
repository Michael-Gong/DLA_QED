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
        'size'   : 20,  
       }  
######### Parameter you should set ###########


if __name__ == "__main__":
  part_number = 1200
  nsteps      = 8001 #sum(1 for line in open(from_path+'x_0000.txt'))/part_number


  from_path = './Data_qe_T400/'
  to_path   = './jpg_qe_T400/'
  t0  = np.loadtxt(from_path+'t_tot_s.txt')/2/np.pi
  x0  = np.loadtxt(from_path+'x_tot_s.txt')/2/np.pi
  y0  = np.loadtxt(from_path+'y_tot_s.txt')/2/np.pi
  px0 = np.loadtxt(from_path+'px_tot_s.txt')/2/np.pi
  py0 = np.loadtxt(from_path+'py_tot_s.txt')/2/np.pi
  t0  = np.reshape(t0,(part_number,nsteps))
  x0  = np.reshape(x0,(part_number,nsteps))
  y0  = np.reshape(y0,(part_number,nsteps))
  px0 = np.reshape(px0,(part_number,nsteps))
  py0 = np.reshape(py0,(part_number,nsteps))
  gg0 = (px0**2+py0**2+1)**0.5
  ww0 = np.zeros_like(gg0)+1

  for i in range(0,nsteps,2):
#      plt.subplot()
      plt.scatter(px0[:,i], py0[:,i], c=gg0[:,i], norm=colors.Normalize(vmin=0,vmax=2e3), s=10, cmap='rainbow', edgecolors='None', alpha=0.66)
      cbar=plt.colorbar( ticks=np.linspace(0, 2000, 5) ,pad=0.005)
      cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=20)
      cbar.set_label('$\gamma$',fontdict=font)

#plt.plot(np.linspace(-500,900,1001), 200-np.linspace(-500,900,1001),'-',color='grey',linewidth=3)
 #   plt.legend(loc='upper right')
      plt.xlim(-100,3100)
      plt.ylim(-250,250)
      plt.xlabel('$P_x$ [$m_ec$]',fontdict=font)
      plt.ylabel('$P_y$ [$m_ec$]',fontdict=font)
      plt.xticks(fontsize=20); plt.yticks(fontsize=20);
      plt.title('t='+str(round(t0[0,i],0))+' $T_0$',fontdict=font)
      #plt.text(-100,650,' t = 400 fs',fontdict=font)
      plt.subplots_adjust(left=0.16, bottom=None, right=0.97, top=None,
                wspace=None, hspace=None)

      #plt.show()
      #lt.figure(figsize=(100,100))
      fig = plt.gcf()
      fig.set_size_inches(12, 6.5)
      fig.savefig(to_path+'p_scatter_'+str(i).zfill(4)+'.png',format='png',dpi=80)
      plt.close("all")

      print('plotting '+str(i).zfill(4))
