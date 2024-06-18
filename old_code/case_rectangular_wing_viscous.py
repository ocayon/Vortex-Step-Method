# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 00:26:08 2022

@author: oriol
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:09:16 2022

@author: oriol2
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import src.functions_VSM_LLT as VSM
    
plt.close('all')
#%%  Input DATA
N = 30
#   Rectangular wing coordinates
span = 6
chord = 1
AR = span/chord
Atot = span*chord
dist = 'cos'
beta = np.zeros(N)
twist = np.zeros(N)
chord_arr = np.ones(N)*chord
coord = VSM.generate_coordinates_rect_wing(chord_arr, span, twist, beta, N,dist)

# Value of the initial circulation
Gamma0 = 0
Gamma_i = VSM.get_circulation_distribution_elliptical_wing(Gamma0, span, N) # Ellyptical wing Gamma distribution


# Wind speed mag and direction  
Umag = 20
aoa = 16
aoa = aoa*np.pi/180
Uinf = np.array([np.cos(aoa),0,np.sin(aoa)])*Umag

# Convergence criteria
conv_crit_damp = {
    'Niterations': 3000,
    'error' : 1e-6,
    'Relax_factor': 0.01,  
    'k2': 0.05,
    'k4': 0 
    }

conv_crit_nodamp = {
    'Niterations': 3000,
    'error' : 1e-6,
    'Relax_factor': 0.01,  
    'k2': 0,
    'k4': 0
    }

data_airf = np.loadtxt(r'./polars/clarky.csv',delimiter = ',')



# Generate geometry
controlpoints,rings,bladepanels,ringvec,coord_L = VSM.create_geometry_general(
        coord, Uinf, int(len(coord)/2),'5fil','VSM')

#%% SOLVER  
t1 = time.time()
Gamma_i = VSM.get_circulation_distribution_elliptical_wing(Gamma0, span, N)
F_relv,CLv,CDv,CSv,Gammav,aero_coeffsv = VSM.run_VSM(coord, Uinf, Gamma_i,data_airf,conv_crit_damp, Atot,rho = 1.225)
print(CLv,CDv)
t2 = time.time()
Gamma_i = VSM.get_circulation_distribution_elliptical_wing(Gamma0, span, N)
F_rel,CL,CD,CS,Gamma,aero_coeffs = VSM.run_VSM(coord, Uinf, Gamma_i,data_airf,conv_crit_nodamp, Atot,rho = 1.225)
print(CL,CD)
t3 =time.time()
print('Runtime with damping:', t2-t1)
print('Runtime with no damping:', t3-t2)
#%% PLOTS

plt_path = './plots/'
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})
plt.rcParams.update({'font.size': 14})


plt.figure(figsize = (6,4))
plt.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffs[:,1], label = 'No visc.') 
plt.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffsv[:,1], label =  'Artificial visc.')
plt.xlabel(r'$y/2b$')
plt.ylabel(r'$C_l$')
plt.legend()
# plt.ylim((0,1.9))
# plt.xlim((-1,1))
plt.grid()


plt.figure(figsize = (6,4))
plt.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffs[:,0]*180/np.pi, label =   'No visc.')  
plt.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffsv[:,0]*180/np.pi, label =  'Artificial visc.')
plt.xlabel(r'$y/2b$')
plt.ylabel(r'$\alpha$')
# plt.ylim((0,25))
# plt.xlim((-1,1))
plt.grid()


plt.figure(figsize = (6,4))
plt.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffs[:,2], label =   'No visc.') 
plt.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffsv[:,2], label = 'Artificial visc.')    
plt.xlabel(r'$y/s$')
plt.ylabel(r'Viscous $C_d$')
plt.legend()


plt.figure(figsize = (6,4))
plt.plot(coord_L[:,1]/max(coord_L[:,1]),Gamma, label =  'No visc.') 
plt.plot(coord_L[:,1]/max(coord_L[:,1]),Gammav, label = 'Artificial visc.')
plt.xlabel(r'$y/s$')
plt.ylabel(r'$\Gamma$')
plt.legend()
plt.grid()



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

ax1.plot(coord_L[:,1]/max(coord_L[:,1]),Gamma, label =  'No visc.') 
ax1.plot(coord_L[:,1]/max(coord_L[:,1]),Gammav, label = 'Artificial visc.')
ax1.set_ylabel(r'$\Gamma$')
ax1.legend()
ax1.grid()

ax2.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffs[:,0]*180/np.pi, label =  'No visc.') 
ax2.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffsv[:,0]*180/np.pi, label = 'Artificial visc.')
ax2.set_ylabel(r'$\alpha$')
ax2.legend()
ax2.grid()

ax2.set_xlabel(r'$y/s$')
# %% Solver 2
# if converged is False:
#     ring_geo = '5fil_stall'
#     controlpoints,rings,bladepanels,ringvec,coord_L = VSM.create_geometry_general(
#     coord, Uinf, N,ring_geo,model)
#     # Solve for Gamma
#     Fmag, Gamma,aero_coeffs,converged = VSM.solve_lifting_line_system_matrix_approach_semiinfinite(
#         ringvec, controlpoints, rings,Uinf,data_airf,False,Gamma0)
#     #%% PLOTS

#     fig = plt.figure(1, figsize = (6,5))
#     plt.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffs[:,1], label = model) 
#     plt.title('Curved wing AR =' + str(AR)+r'$, \alpha = $' + str(aoa*180/np.pi))    
#     plt.xlabel(r'$y/s$')
#     plt.ylabel(r'$C_l$')
#     plt.legend()
#     plt.show()
    
#     fig = plt.figure(2, figsize = (6,5))
#     plt.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffs[:,0]*180/np.pi, label = model) 
#     plt.title('Curved wing AR =' + str(AR)+r'$, \alpha = $' + str(aoa*180/np.pi))    
#     plt.xlabel(r'$y/s$')
#     plt.ylabel(r'$\alpha$')
#     plt.legend()
#     plt.show()
    
#     fig = plt.figure(2, figsize = (6,5))
#     plt.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffs[:,2], label = model) 
#     plt.title('Curved wing AR =' + str(AR)+r'$, \alpha = $' + str(aoa*180/np.pi))    
#     plt.xlabel(r'$y/s$')
#     plt.ylabel(r'Viscous $C_d$')
#     plt.legend()
#     plt.show()
    
#     fig = plt.figure(2, figsize = (6,5))
#     plt.plot(coord_L[:,1]/max(coord_L[:,1]),Gamma, label = model) 
#     plt.title('Curved wing AR =' + str(AR)+r'$, \alpha = $' + str(aoa*180/np.pi))    
#     plt.xlabel(r'$y/s$')
#     plt.ylabel(r'$\Gamma$')
#     plt.legend()
#     plt.show()
# # print(end_time-start_time)

