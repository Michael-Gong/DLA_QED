%matplotlib inline
#import sdf
import matplotlib
import matplotlib as mpl
mpl.style.use('https://raw.githubusercontent.com/Michael-Gong/DLA_project/master/style')
#matplotlib.use('agg')
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
font = {'family' : 'Carlito',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 25,
       }

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

def momentum_to_theta(arg_x, arg_y):
    if arg_x == 0.00:
        arg_x += (random.uniform(0, 1)-0.5)/1000.0
    if arg_x >= 0:
        return np.arctan(arg_y/arg_x)
    elif arg_y > 0:
        return np.arctan(arg_y/arg_x)+np.pi
    else:
        return np.arctan(arg_y/arg_x)-np.pi





part_number=100
nsteps=20001

insert1='./Data/'
insert_n='_0'
t1=np.loadtxt(insert1+'t'+insert_n+'.txt')
#z1=np.loadtxt(insert1+'z'+insert_n+'.txt')
y1=np.loadtxt(insert1+'y'+insert_n+'.txt')
x1=np.loadtxt(insert1+'x'+insert_n+'.txt')
px1=np.loadtxt(insert1+'px'+insert_n+'.txt')
py1=np.loadtxt(insert1+'py'+insert_n+'.txt')
#pz1=np.loadtxt(insert1+'pz'+insert_n+'.txt')
#ey=np.loadtxt(insert+'e_part'+'.txt')
#bz=np.loadtxt(insert+'b_part'+'.txt')
#ay=np.loadtxt(insert+'a_part'+'.txt')
radn1=np.loadtxt(insert1+'radn'+insert_n+'.txt')
radt1=np.loadtxt(insert1+'radt'+insert_n+'.txt')
radpx1=np.loadtxt(insert1+'rad_px'+insert_n+'.txt')
#opt1=np.loadtxt(insert1+'opt'+insert_n+'.txt')
#eta1=np.loadtxt(insert1+'eta'+insert_n+'.txt')

t=np.reshape(t1,(part_number,nsteps))
x=np.reshape(x1,(part_number,nsteps))
y=np.reshape(y1,(part_number,nsteps))
#z=np.reshape(z1,(part_number,nsteps))
px=np.reshape(px1,(part_number,nsteps))
py=np.reshape(py1,(part_number,nsteps))
#pz=np.reshape(pz1,(part_number,nsteps))
#ey=np.reshape(ey,(part_number,nsteps))
#ay=np.reshape(ay,(part_number,nsteps))
radn=np.reshape(radn1,(part_number,nsteps))
radt=np.reshape(radt1,(part_number,nsteps))
radpx=np.reshape(radpx1,(part_number,nsteps))
#opt=np.reshape(opt1,(part_number,nsteps))
#eta=np.reshape(eta1,(part_number,nsteps))

gamma=np.sqrt(px**2+py**2+1)

R_dep=gamma-px

py_0 = 1.00*int(150)
R_max = py_0-(radt-radpx)

y_max0=(py_0/0.02)**0.5
y_max=((py_0-(radt-radpx))/0.02)**0.5/2.0/np.pi

index=0

radn_x=(radn[index,1:]-radn[index,:-1])
radt_x=(radt[index,1:]-radt[index,:-1])

condition = np.where(radt_x>2)

arg_px = px[index,condition]
arg_py = py[index,condition]
radt_x=(radt[index,1:]-radt[index,:-1])
arg_gg = radt_x[condition]
theta_x = np.zeros_like(arg_px)
energy_x = np.zeros_like(arg_px)
#print(arg_py/arg_px)
for i in range(np.size(theta_x)):
    theta_x[0,i]=momentum_to_theta(arg_px[0,i],arg_py[0,i])
    energy_x[0,i]=arg_gg[i]

theta_grid = np.linspace(-180.0, 180.0, 61)
energy_grid= np.linspace(2,2002,51)

x_grid = np.linspace(-180.0, 180.0, 60)
y_grid = np.linspace(2,2002,50)
x_grid,y_grid = np.meshgrid(x_grid,y_grid)

theta_energy = np.zeros_like(x_grid)
theta_energy = theta_energy.T
theta_x/np.pi*180

for i in range(np.size(theta_x/np.pi*180)):
    index_x = np.min(np.where(theta_x[0,i]/np.pi*180 < theta_grid))-1
    index_y = np.min(np.where(energy_x[0,i] < energy_grid))-1
    theta_energy[index_x,index_y]+=1
#print(theta_energy)

print(x_grid.shape,y_grid.shape,theta_energy.shape)

ax=plt.subplot(2,1,1)
levels = np.linspace(np.min(theta_energy), np.max(theta_energy), 10)
plt.pcolormesh(x_grid, y_grid, theta_energy.T, cmap='nipy_spectral')
#plt.contourf(x_grid, y_grid, theta_energy.T, levels=levels, cmap='hot_r', alpha=.7)
cbar=plt.colorbar(ticks=np.linspace(np.min(levels), np.max(levels), 5))
cbar.set_label(r'$N$', fontdict=font)#plt.xlim(47,53)
#ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
plt1 = plt.twinx()
z_value = y_grid * theta_energy.T
plt.plot(x_grid[0,:],np.sum(z_value,axis=0),'--y',linewidth=1.5)
plt1.set_ylabel('Normalized')  
plt.xlabel(r'$\theta\ [degree]$',fontdict=font)
plt.ylabel(r'$\gamma$',fontdict=font)
#plt.xticks(fontsize=1.0); plt.yticks(fontsize=26);
#plt.ylim(-1.025,1.025)
plt.xlim(-180,180)
#plt.legend(loc='best')

ax=plt.subplot(2,1,2)
plt.scatter(theta_x/np.pi*180, arg_gg, c=np.linspace(1,np.size(theta_x),np.size(theta_x))[np.newaxis,:], s=20, cmap='nipy_spectral', edgecolors='None')
cbar=plt.colorbar(ticks=np.linspace(1, np.size(theta_x), 5), shrink=1)# orientation='horizontal', shrink=0.2)
cbar.set_label(r'$Nth$', fontdict=font)
plt.xlim(-45,45)
#print(theta_x)
plt.xlabel(r'$\theta\ [degree]$',fontdict=font)
plt.ylabel(r'$\gamma$',fontdict=font)
#plt.xticks(fontsize=30); plt.yticks(fontsize=30);
#plt.ylim(0,2000.0)


plt.subplots_adjust(top=0.95, bottom=0.10, left=0.15, right=0.95, hspace=0.25, wspace=0.30)

fig = plt.gcf()
#fig.set_size_inches(30, 15)
fig.set_size_inches(8, 13)
#fig.savefig('./png_a200_b002_p150/qed_photon_1000_com.png',format='png',dpi=80)
#plt.close("all")
