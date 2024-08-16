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
plt.rcParams.update({'font.size': 14})
#%%  Input DATA
N = 50
#   Rectangular wing coordinates
chord = np.ones(N)*1
span = 20
AR = span/max(chord)
Atot = span*max(chord)
beta = np.zeros(N)
twist = np.ones(N)*0
dist = 'lin'
coord = VSM.generate_coordinates_rect_wing(chord, span, twist, beta, N,dist)

# Convergence criteria
conv_crit = {
    'Niterations': 1500,
    'error' : 1e-5,
    'Conv_weight': 0.03
    }

# Wind speed mag and direction  
Umag = 20
aoa = 3
aoa = aoa*np.pi/180
Uinf = np.array([np.cos(aoa),0,np.sin(aoa)])*Umag

#   Model and program specifics
ring_geo = '5fil' # '3fil' or '5fil', meaning each horseshoe vortex has 5 filaments or 3. 5 filaments generally yields better results
model = 'VSM'       # 'VSM': Vortex Step Method or 'LLT': Lifting Line Theory

# Airfoil data flat plate
alpha_airf = np.arange(-10,30)
data_airf = np.zeros((len(alpha_airf),4))
data_airf[:,0] = alpha_airf                     # Angle of attack
data_airf[:,1] = alpha_airf/180*np.pi*2*np.pi   # Cl
data_airf[:,2] = alpha_airf*0                   # Cd
data_airf[:,3] = alpha_airf*0                   # Cm 
#%% SOLVER  
start_time = time.time()
# Define system of vorticity
controlpoints,rings,bladepanels,ringvec,coord_L = VSM.create_geometry_general(
    coord, Uinf, N,ring_geo,model)
# Solve for Gamma
Fmag, Gamma,aero_coeffs= VSM.solve_lifting_line_system_matrix_approach_semiinfinite(
    ringvec, controlpoints, rings,Uinf,data_airf,conv_crit,model)
end_time = time.time()
print(end_time-start_time)
# Calculate global coefficients and forces
F_rel,F_gl,Ltot,Dtot,CL,CD = VSM.output_results(Fmag,aero_coeffs,ringvec,Uinf,controlpoints,Atot)

print('CL = ', CL, 'CD = ', CD)

#%% PLOTS

fig = plt.figure(1, figsize = (6,5))
plt.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffs[:,1], label = model) 
plt.title('Rectangular wing AR =' + str(AR)+r'$, \alpha = $' + str(aoa*180/np.pi))    
plt.xlabel(r'$y/s$')
plt.ylabel(r'$C_l$')
plt.legend()
plt.show()

fig = plt.figure(2, figsize = (6,5))
plt.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffs[:,0]*180/np.pi, label = model) 
plt.title('Rectangular wing AR =' + str(AR)+r'$, \alpha = $' + str(aoa*180/np.pi))    
plt.xlabel(r'$y/s$')
plt.ylabel(r'$\alpha$')
plt.legend()
plt.show()

fig = plt.figure(3, figsize = (6,5))
plt.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffs[:,2], label = model) 
plt.title('Rectangular wing AR =' + str(AR)+r'$, \alpha = $' + str(aoa*180/np.pi))    
plt.xlabel(r'$y/s$')
plt.ylabel(r'Viscous $C_d$')
plt.legend()
plt.show()

VSM.plot_geometry(bladepanels,controlpoints,rings,F_gl,coord_L,'True')

