import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.insert(0, '../functions/')
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
Gamma0 = 2
Gamma_i = VSM.get_circulation_distribution_elliptical_wing(Gamma0, span, N) # Ellyptical wing Gamma distribution


# Wind speed mag and direction  
Umag = 20
aoas = np.linspace(-5, 32,20)*np.pi/180


# Convergence criteria
conv_crit_damp = {
    'Niterations': 3000,
    'error' : 1e-6,
    'Relax_factor': 0.01,  
    'k2': 0.0,
    'k4': 0.0   
    }


data_airf = np.loadtxt(r'./polars/clarky.csv',delimiter = ',')

#%% SOLVER  
F_rel = []; CL = []; CD = []; CS = []; Gamma = []; aero_coeffs = []

for i,aoa in enumerate(aoas): 
    Uinf = np.array([np.cos(aoa),0,np.sin(aoa)])*Umag
    Gamma_i = VSM.get_circulation_distribution_elliptical_wing(Gamma0, span, N) # Ellyptical wing Gamma distribution
    F_reli,CLi,CDi,CSi,Gammai,aero_coeffsi = VSM.run_VSM(coord, Uinf, Gamma_i,data_airf,conv_crit_damp, Atot,rho = 1.225)
    print(str(round((i+1)/len(aoas)*100,1))+' %')
    F_rel.append(F_reli)
    CL.append(CLi)
    CD.append(CDi)
    CS.append(CSi)
    Gamma.append(Gammai)
    aero_coeffs.append(aero_coeffsi)
    

#%% PLOTS

# Plot lift coefficient
plt.figure()
plt.plot(aoas*180/np.pi, CL)
plt.xlabel('Angle of Attack (degrees)')
plt.ylabel('CL')
plt.title('Lift Coefficient')
plt.grid()
plt.show()

# Plot drag coefficient
plt.figure()
plt.plot(aoas*180/np.pi, CD)
plt.xlabel('Angle of Attack (degrees)')
plt.ylabel('CD')
plt.title('Drag Coefficient')
plt.grid()
plt.show()

#%% Plot spanwise at an angle
aoas_plot = np.arange(16,33,2)
for i,aoa_plot in enumerate(aoas_plot):
    idx = np.argmin(np.abs(aoas*180/np.pi - aoa_plot))
    # Generate geometry
    controlpoints,rings,bladepanels,ringvec,coord_L = VSM.create_geometry_general(
            coord, Uinf, int(len(coord)/2),'5fil','VSM')
    
    aoa = aero_coeffs[idx][:, 0]
    cl = aero_coeffs[idx][:, 1]
    cd = aero_coeffs[idx][:, 2]
    cm = aero_coeffs[idx][:, 3]
    
    print('Angle of attack: ', round(aoas[idx]*180/np.pi,2), 'deg')
    # normalize the x-axis
    x = coord_L[:, 1] / np.max(coord_L[:, 1])
    
    # plot aoa
    figaoa = plt.figure(101)
    plt.plot(x, aoa*180/np.pi,label = str(round(aoa_plot,1)))
    plt.xlabel('Normalized X-Axis')
    plt.ylabel('AoA')
    plt.title('Angle of Attack')
    plt.ylim((0,35))
    
    plt.show()
    
    # plot cl
    figcl = plt.figure(102)
    plt.plot(x, cl,label = str(round(aoa_plot,1)))
    plt.xlabel('Normalized X-Axis')
    plt.ylabel('Cl')
    plt.title('Lift Coefficient')

    plt.show()
    
    # plot cd
    figcd = plt.figure(103)
    plt.plot(x, cd,label = str(round(aoa_plot,1)))
    plt.xlabel('Normalized X-Axis')
    plt.ylabel('Cd')
    plt.title('Drag Coefficient')

    plt.show()

figaoa.legend()
figcl.legend()
figcd.legend()
