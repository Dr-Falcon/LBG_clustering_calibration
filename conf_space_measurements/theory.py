#import 
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.integrate import simpson
from scipy.special import jv
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RectBivariateSpline as rbs

#path
root = os.path.abspath(os.getcwd())
print(root)

#setting cosmology
from dark_emulator import darkemu
from dark_emulator import pyfftlog_interface
#
emu = darkemu.base_class()
cparam = np.array([0.02225,0.1198,0.6844,3.094,0.9645,-1.])
emu.set_cosmology(cparam)

import classy
from classy import Class
cosmo = Class()

params = {'output': 'tCl  lCl mPk', 
          'l_max_scalars': 4000, 
          #'reio_parametrization':'reio_none',
          'P_k_max_h/Mpc':4000,
          'N_ur':3.046,
          'tau_reio':0.0971,
          'YHe':0.25,
          'z_pk':1089, 
          'lensing': 'yes', 
          'non linear':'Halofit',
          'A_s': np.exp(cparam[3])*np.power(10.,-10.), 
          'n_s': cparam[4],
          'h': 0.6724008, 
          'omega_b': cparam[0], 
          'omega_cdm': cparam[1]}

cosmo.set(params)
cosmo.compute()

def get_comoving_distance(z):
    return emu.cosmo.get_comoving_distance(z)

def mPk(k,z):
    pk = np.zeros(k.size)
    for i in range(k.size):
        pk[i] = cosmo.pk(k[i],z)
        #print(k[i],pk[i],z)
    return pk

# theoretical template for dark matter projected correlation functions
def get_wp(theta,z):
    theta=(np.pi/180.)*theta #deg to rad
    c = 2.99792458e05 #speed of light [km/s]

    x=emu.cosmo.get_comoving_distance(z)
    E=emu.cosmo.get_Ez(z)
    h = 0.6724008
    H0 = 100.*h
    H=H0*emu.cosmo.get_Ez(z)
    
    # 各zに対するxi_proj をここで作っておく 
    xi_start=time.time()
    xi_proj_z = [0] * len(z)
    for i in range(len(z)):
        #xi_proj_z[i] = pyfftlog_interface.pk2xiproj_J0_pyfftlog(pk_func=emu.get_pklin_from_z, args=z[i])
        xi_proj_z[i] = pyfftlog_interface.pk2xiproj_J0_pyfftlog(pk_func=mPk,logkmax=3., args=z[i])
    xi_end=time.time()
    print("{0:2d} loop time {1:g} [sec]".format(i,xi_end-xi_start))
    #print("")
    #print("----------------------")
    th_loop_start = time.time()
    
    wp = np.empty((len(theta),len(z)))
    
    for i in range (len(theta)):
        z_loop_start = time.time()
        R=x*theta[i]
        for j in range(len(z)):
            z_start = time.time()           
            wp[i,j] = xi_proj_z[j](R[j]) # dimension-less
            z_end = time.time()
            #print("{0:2d} calc time {1:g} [sec]".format(i,end-start))
        z_loop_end = time.time()
        
    th_loop_end = time.time()
    elapsed_time = th_loop_end - th_loop_start
    #print('{}s'.format(elapsed_time))
    return wp

def get_wp_finite(theta,z,pimax):
    theta=(np.pi/180.)*theta #deg to rad     
    c = 2.99792458e05 #speed of light [km/s]

    x=emu.cosmo.get_comoving_distance(z)
    E=emu.cosmo.get_Ez(z)
    h = 0.6724008
    H0 = 100.*h
    H=H0*emu.cosmo.get_Ez(z)
    pi = np.logspace(-3.,np.log10(pimax*h),num=1000,base=10.)
    
    # 各zに対するxi_proj をここで作っておく 
    xi_start=time.time()
    xi_z = [0] * len(z)
    for i in range(len(z)):
        xi_z[i] = pyfftlog_interface.pk2xi_pyfftlog(pk_func=mPk,logkmax=3., args=z[i])
    xi_end=time.time()
    print("{0:2d} loop time {1:g} [sec]".format(i,xi_end-xi_start))
    #print("")
    #print("----------------------")
    th_loop_start = time.time()
    
    wp = np.empty((len(theta),len(z)))
    
    for i in range (len(theta)):
        z_loop_start = time.time()
        R=x*theta[i]
        for j in range(len(z)):
            z_start = time.time()
            r = np.sqrt(R[j]**2.+pi**2.)
            wp[i,j] = 2.*simpson(xi_z[j](r),r) # dimension-full #factor 2 needed
            z_end = time.time()
            #print("{0:2d} calc time {1:g} [sec]".format(i,end-start))
        z_loop_end = time.time()
        
    th_loop_end = time.time()
    elapsed_time = th_loop_end - th_loop_start
    #print('{}s'.format(elapsed_time))
    return wp

# theoretical template for dark matter correlation function
def get_wthetaz(theta,z):
    theta=(np.pi/180.)*theta #deg to rad
    c = 2.99792458e05 #speed of light [km/s]

    x=emu.cosmo.get_comoving_distance(z)
    E=emu.cosmo.get_Ez(z)
    h = 0.6724008
    H0 = 100.*h
    H=H0*emu.cosmo.get_Ez(z)
    
    # 各zに対するxi_proj をここで作っておく 
    xi_start=time.time()
    xi_proj_z = [0] * len(z)
    for i in range(len(z)):
        #xi_proj_z[i] = pyfftlog_interface.pk2xiproj_J0_pyfftlog(pk_func=emu.get_pklin_from_z, args=z[i])
        xi_proj_z[i] = pyfftlog_interface.pk2xiproj_J0_pyfftlog(pk_func=mPk,logkmax=3., args=z[i])
    xi_end=time.time()
    print("{0:2d} loop time {1:g} [sec]".format(i,xi_end-xi_start))
    #print("")
    #print("----------------------")
    th_loop_start = time.time()
    
    wz = np.empty((len(theta),len(z)))
    
    for i in range (len(theta)):
        z_loop_start = time.time()
        R=x*theta[i]
        for j in range(len(z)):
            z_start = time.time()
            xi_proj=xi_proj_z[j](R[j])            
            wz[i,j] = (H[j]/c)*xi_proj # dimension-less
            z_end = time.time()
            #print("{0:2d} calc time {1:g} [sec]".format(i,end-start))
        z_loop_end = time.time()
        
    th_loop_end = time.time()
    elapsed_time = th_loop_end - th_loop_start
    #print('{}s'.format(elapsed_time))
    return wz

# CMB lensing angular power spectrum
def get_clkkz(ell,z):
    c = 2.99792458e05 #speed of light [km/s]
    k = np.logspace(-3.,np.log10(4.)*3.,num=1000,base=10.) 
    pkz= mPk(k,z)       
    x=emu.cosmo.get_comoving_distance(z)
    xls=emu.cosmo.get_comoving_distance(1089.)
    E=emu.cosmo.get_Ez(z)
    h = 0.6724008
    H0 = 100.*h
    H=H0*emu.cosmo.get_Ez(z)
    Omega_0 = emu.cosmo.get_Omega0()
    
    C = np.empty(len(ell),dtype=float)
    Wk = (1.5*Omega_0*H0**2./c**2.)*x*(xls-x)*(1.+z)/xls
    C = (c/H)*Wk**2./x**2.*ius(k,pkz)(ell/x)
  
    return C

def get_clkk(ell):
    z = np.logspace(-3.,np.log10(1089.),num=1000,base=10.)
    C = np.empty(len(ell),dtype=float)
    Cz = np.empty((len(ell),len(z)),dtype=float)
    for i in range(len(z)):
        Cz[:,i] = get_clkkz(ell[:],z[i])
    for i in range(len(ell)):
        C[i] = simpson(Cz[i,:],z[:])
    return C

# CMB lensing - LSS 
def get_clkgz(ell,z):
    c = 2.99792458e05 #speed of light [km/s]
    k = np.logspace(-3.,np.log10(4.)*3.,num=1000,base=10.) 
    pkz= mPk(k,z)       
    x=emu.cosmo.get_comoving_distance(z)
    xls=emu.cosmo.get_comoving_distance(1089.)
    E=emu.cosmo.get_Ez(z)
    h = 0.6724008
    H0 = 100.*h
    H=H0*emu.cosmo.get_Ez(z)
    Omega_0 = emu.cosmo.get_Omega0()
    dpdz = dpdz_goldrush(z)
    
    C = np.empty(len(ell),dtype=float)
    Wk = (1.5*Omega_0*H0**2./c**2.)*x*(xls-x)*(1.+z)/xls
    Wg = (H/c)*dpdz
    C = (c/H)*Wk*Wg/x**2.*ius(k,pkz)(ell/x)
  
    return C

def get_clkg(ell):
    z = np.logspace(np.log10(3.2),np.log10(4.5),num=1000,base=10.)
    C = np.empty(len(ell),dtype=float)
    Cz = np.empty((len(ell),len(z)),dtype=float)
    for i in range(len(z)):
        Cz[:,i] = get_clkgz(ell[:],z[i])
    for i in range(len(ell)):
        C[i] = simpson(Cz[i,:],z[:])
    return C

# galaxy clustering
def get_clggz(ell,z):
    c = 2.99792458e05 #speed of light [km/s]
    k = np.logspace(-3.,np.log10(4.)*3.,num=1000,base=10.) 
    pkz= mPk(k,z)       
    x=emu.cosmo.get_comoving_distance(z)
    xls=emu.cosmo.get_comoving_distance(1089.)
    E=emu.cosmo.get_Ez(z)
    h = 0.6724008
    H0 = 100.*h
    H=H0*emu.cosmo.get_Ez(z)
    Omega_0 = emu.cosmo.get_Omega0()
    dpdz = dpdz_goldrush(z)
    
    C = np.empty(len(ell),dtype=float)
    Wg = (H/c)*dpdz
    C = (c/H)*Wg**2./x**2.*ius(k,pkz)(ell/x)
  
    return C

def get_clgg(ell):
    z = np.logspace(np.log10(3.2),np.log10(4.5),num=1000,base=10.)
    C = np.empty(len(ell),dtype=float)
    Cz = np.empty((len(ell),len(z)),dtype=float)
    for i in range(len(z)):
        Cz[:,i] = get_clggz(ell[:],z[i])
    for i in range(len(ell)):
        C[i] = simpson(Cz[i,:],z[:])
    return C


#theoretical template for angular cross correlation btw. reconsctructed radio kernel and CMB convergence map
def get_rec_cxl(ell,z,Wr):
    c = 2.99792458e05 #speed of light [km/s]
    k = np.logspace(-3.,3.,num=1000,base=10.) 
    pkz= mPk(k,z)
        
    x=emu.cosmo.get_comoving_distance(z)
    xls=emu.cosmo.get_comoving_distance(1100)
    E=emu.cosmo.get_Ez(z)
    h = 0.6724008
    H0 = 100.*h
    H=H0*emu.cosmo.get_Ez(z)
    Omega_0 = emu.cosmo.get_Omega0()
    
    C = np.empty(len(ell),dtype=float)
    Wk = (1.5*Omega_0*H0**2./c**2.)*x*(xls-x)*(1.+z)/xls
    C = Wr*Wk/x**2.*ius(k,pkz)(ell/x)
    Cnorm = ell*(ell+1.)*C/(2.*np.pi)
    
    return Cnorm

def get_wtheta(theta,z):
    theta=(np.pi/180.)*theta #deg to rad
    c = 2.99792458e05 #speed of light [km/s]
    b=1.6+0.85*z+0.33*pow(z,2.)#linear bias
    dpdz=dpdz_kono2019_8000mujy(z) #dpdz
    dpdz_up = dpdz_kono2019_8000mujy_err(z,1)
    dpdz_low = dpdz_kono2019_8000mujy_err(z,-1)
    
    x=emu.cosmo.get_comoving_distance(z)
    E=emu.cosmo.get_Ez(z)
    h = 0.6724008
    H0 = 100.*h
    H=H0*emu.cosmo.get_Ez(z)
    
    # 各zに対するxi_proj をここで作っておく 
    xi_start=time.time()
    xi_proj_z = [0] * len(z)
    for i in range(len(z)):
        xi_proj_z[i] = pyfftlog_interface.pk2xiproj_J0_pyfftlog(pk_func=mPk,logkmax=3., args=z[i])
    xi_end=time.time()
    print("{0:2d} loop time {1:g} [sec]".format(i,xi_end-xi_start))
    #print("")
    #print("----------------------")
    th_loop_start = time.time()
    
    w = np.empty(len(theta))
    w_up = np.empty(len(theta))
    w_low = np.empty(len(theta))
    
    wz = np.empty((len(theta),len(z)))
    wz_up = np.empty((len(theta),len(z)))
    wz_low = np.empty((len(theta),len(z)))
    
    K = np.empty(len(z))
    K_up = np.empty(len(z))
    K_low = np.empty(len(z))
    
    for i in range (len(theta)):
        z_loop_start = time.time()
        R=x*theta[i]
        for j in range(len(z)):
            z_start = time.time()
            xi_proj=xi_proj_z[j](R[j])
            K[j]=b[j]*dpdz[j]
            K_up[j]=b[j]*dpdz_up[j]
            K_low[j]=b[j]*dpdz_low[j]
            
            wz[i,j] = (H[j]/c)*K[j]**2.*xi_proj # dimension-less
            wz_up[i,j] = (H[j]/c)*K_up[j]**2.*xi_proj
            wz_low[i,j] = (H[j]/c)*K_low[j]**2.*xi_proj
            z_end = time.time()
            #print("{0:2d} calc time {1:g} [sec]".format(i,end-start))
        z_loop_end = time.time()
        
        w[i] = simpson(wz[i,:],z,0.01)
        w_up[i] = simpson(wz_up[i,:],z,0.01)
        w_low[i] = simpson(wz_low[i,:],z,0.01)

    th_loop_end = time.time()
    elapsed_time = th_loop_end - th_loop_start
    #print('{}s'.format(elapsed_time))
    return w, w_up, w_low

def get_rec_wtheta(theta,z,Wr):
    theta=(np.pi/180.)*theta #deg to rad
    c = 2.99792458e05 #speed of light [km/s]    
    x=emu.cosmo.get_comoving_distance(z)
    E=emu.cosmo.get_Ez(z)
    h = 0.6724008
    H0 = 100.*h
    H=H0*emu.cosmo.get_Ez(z)
    
    # 各zに対するxi_proj をここで作っておく 
    xi_start=time.time()
    xi_proj_z = [0] * len(z)
    for i in range(len(z)):
        xi_proj_z[i] = pyfftlog_interface.pk2xiproj_J0_pyfftlog(pk_func=mPk,logkmax=3., args=z[i])
    xi_end=time.time()
    print("{0:2d} loop time {1:g} [sec]".format(i,xi_end-xi_start))
    #print("")
    #print("----------------------")
    th_loop_start = time.time()
    
    w = np.empty(len(theta))
    wz = np.empty((len(theta),len(z)))
        
    for i in range (len(theta)):
        z_loop_start = time.time()
        R=x*theta[i]
        for j in range(len(z)):
            z_start = time.time()
            xi_proj=xi_proj_z[j](R[j])
            
            wz[i,j] = (H[j]/c)*Wr[j]**2.*xi_proj # dimension-less
            z_end = time.time()
            #print("{0:2d} calc time {1:g} [sec]".format(i,end-start))
        z_loop_end = time.time()
        
        w[i] = simpson(wz[i,:],z,0.01)

    th_loop_end = time.time()
    elapsed_time = th_loop_end - th_loop_start
    #print('{}s'.format(elapsed_time))
    return w

#distribution of GOLDRUSH
def dpdz_goldrush(z):
    f=open('../../data/goldrush/redshift_distribution_23.50_25.50.dat','r')
    z_data=[]
    dpdz_data=[]
    n=14
    for i in range(n):
        list=f.readline()
        column=list.split()
        z_data.append(float(column[0]))
        dpdz_data.append(float(column[1]))
    f.close()
    
    dpdz_interp = ius(z_data,dpdz_data)(z)
  
    return dpdz_interp
    

#distribution of radio galaxies computed by Kono 2019
def dpdz_kono2019_8000mujy(z):
    f=open('../../../data/lumi_func/NVSS_8000muJy.dat','r')
    z_data=[]
    dndz_data=[]
    n=29
    for i in range(n):
        list=f.readline()
        column=list.split()
        if i>0:
            z_data.append(float(column[0]))
            dndz_data.append(float(column[1]))
    f.close()

    f_interp = interpolate.interp1d(z_data,dndz_data,kind="quadratic")
    dndz_interp = f_interp(z)

    #normalisation scheme
    dz=0.01
    zint=np.arange(0.,5.8,dz)
    norm=0.
    
    for i in range(len(zint)):
        norm = norm + dz*f_interp(zint)[i]

    dpdz_interp=dndz_interp/norm
    
    return dpdz_interp

#distribution of radio galaxies computed by Kono 2019
def dpdz_kono2019_8000mujy_err(z,switch):
    f=open('../../../data/lumi_func/NVSS_8000muJy_err.dat','r')
    z_data=[]
    dndz_data=[]
    dndz_data_low=[]
    dndz_data_up=[]
    n=29
    for i in range(n):
        list=f.readline()
        column=list.split()
        if i>0:
            z_data.append(float(column[0]))
            dndz_data.append(float(column[1]))
            dndz_data_low.append(float(column[2]))
            dndz_data_up.append(float(column[3]))
            
    f.close()

    if switch == 1:
        f_interp = interpolate.interp1d(z_data,dndz_data_up,kind="quadratic")
        dndz_interp = f_interp(z)

        #normalisation scheme
        dz=0.01
        zint=np.arange(0.,5.8,dz)
        norm=0.
        for i in range(len(zint)):
            norm = norm + dz*f_interp(zint)[i]

        dpdz_interp=dndz_interp/norm
    
        return dpdz_interp

    if switch == -1:
        f_interp = interpolate.interp1d(z_data,dndz_data_low,kind="quadratic")
        dndz_interp = f_interp(z)

        #normalisation scheme
        dz=0.01
        zint=np.arange(0.,5.8,dz)
        norm=0.
        for i in range(len(zint)):
            norm = norm + dz*f_interp(zint)[i]

        dpdz_interp=dndz_interp/norm
    
        return dpdz_interp