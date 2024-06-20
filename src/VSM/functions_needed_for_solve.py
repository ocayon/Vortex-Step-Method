import numpy as np


def vec_norm(v):
    """
    Norm of a vector

    """
    return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def dot_product(r1, r2):
    """
    Dot product between r1 and r2

    """
    return r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]


def cross_product(r1, r2):
    """
    Cross product between r1 and r2

    """

    return np.array(
        [
            r1[1] * r2[2] - r1[2] * r2[1],
            r1[2] * r2[0] - r1[0] * r2[2],
            r1[0] * r2[1] - r1[1] * r2[0],
        ]
    )


def find_stall_angle(polar):
    # polar should be a two-dimensional numpy array with columns for angle of attack and coefficient of lift
    # get the angle of attack and coefficient of lift data from the polar
    aoa = polar[:, 0]
    cl = polar[:, 1]
    # use numpy's gradient function to find the rate of change of coefficient of lift with respect to angle of attack
    dcl_daoa = np.gradient(cl, aoa)
    # find the index where the gradient crosses zero
    zero_crossings = np.where(np.diff(np.sign(dcl_daoa)))[0]
    if zero_crossings.size == 0:
        # if no zero crossings are found, the airfoil may not have a distinct stall angle
        return 50
    else:
        stall_index = zero_crossings[0]
        # return the corresponding angle of attack
        aoa_stall = aoa[stall_index + 1]
        return aoa_stall


def velocity_induced_bound_2D(ringvec):

    r0 = ringvec["r0"]
    r3 = ringvec["r3"]

    cross = [
        r0[1] * r3[2] - r0[2] * r3[1],
        r0[2] * r3[0] - r0[0] * r3[2],
        r0[0] * r3[1] - r0[1] * r3[0],
    ]

    ind_vel = (
        cross
        / (cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
        / 2
        / np.pi
        * np.linalg.norm(r0)
    )

    return ind_vel


def find_stall_angle(polar):
    # polar should be a two-dimensional numpy array with columns for angle of attack and coefficient of lift
    # get the angle of attack and coefficient of lift data from the polar
    aoa = polar[:, 0]
    cl = polar[:, 1]
    # use numpy's gradient function to find the rate of change of coefficient of lift with respect to angle of attack
    dcl_daoa = np.gradient(cl, aoa)
    # find the index where the gradient crosses zero
    zero_crossings = np.where(np.diff(np.sign(dcl_daoa)))[0]
    if zero_crossings.size == 0:
        # if no zero crossings are found, the airfoil may not have a distinct stall angle
        return 50
    else:
        stall_index = zero_crossings[0]
        # return the corresponding angle of attack
        aoa_stall = aoa[stall_index + 1]
        return aoa_stall


def update_Gamma_single_ring(ring, GammaNew, WeightNew):
    """
    Update Gamma of all the filaments in a horshoe ring

    """
    # Runs through all filaments
    for filament in ring:
        filament["Gamma"] = (
            filament["Gamma"] * (1 - WeightNew) + WeightNew * GammaNew
        )  # Update each Gamma

    return ring


def velocity_3D_trailing_vortex_semiinfinite(XV1, Vf, XVP, GAMMA, Uinf):
    """
        Calculate the velocity induced by a semiinfinite trailing vortex filament in a point in space

        Vortex core correction from:
            Rick Damiani et al. “A vortex step method for nonlinear airfoil polar data as implemented in
    KiteAeroDyn”.
        ----------
        XV1 : Point A of the vortex filament (array)
        XV2 : Point B of the vortex filament (array)
        XVP : Controlpoint (array)
        gamma : Strength of the vortex (scalar)
        Uinf : Inflow velocity modulus (scalar)

        Returns
        -------
        vel_ind : induced velocity by the trailing fil. (array)

    """
    r1 = XVP - XV1  # Vector from XV1 to XVP
    r1XVf = cross_product(r1, Vf)

    alpha0 = 1.25643  # Oseen parameter
    nu = 1.48e-5  # Kinematic viscosity of air
    r_perp = (
        dot_product(r1, Vf) * Vf
    )  # Vector from XV1 to XVP perpendicular to the core radius
    epsilon = np.sqrt(4 * alpha0 * nu * vec_norm(r_perp) / Uinf)  # Cut-off radius

    if vec_norm(r1XVf) / vec_norm(Vf) > epsilon:
        # determine scalar
        K = (
            GAMMA
            / 4
            / np.pi
            / vec_norm(r1XVf) ** 2
            * (1 + dot_product(r1, Vf) / vec_norm(r1))
        )
        # determine the three velocity components
        vel_ind = K * r1XVf
    else:
        r1_proj = dot_product(r1, Vf) * Vf + epsilon * (
            r1 / vec_norm(r1) - Vf
        ) / vec_norm(r1 / vec_norm(r1) - Vf)
        r1XVf_proj = cross_product(r1_proj, Vf)
        K = (
            GAMMA
            / 4
            / np.pi
            / vec_norm(r1XVf_proj) ** 2
            * (1 + dot_product(r1_proj, Vf) / vec_norm(r1_proj))
        )
        # determine the three velocity components
        vel_ind = K * r1XVf_proj
    # output results, vector with the three velocity components
    return vel_ind


def velocity_3D_bound_vortex(XV1, XV2, XVP, gamma):
    """
        Calculate the velocity induced by a bound vortex filament in a point in space ()

        Vortex core correction from:
            Rick Damiani et al. “A vortex step method for nonlinear airfoil polar data as implemented in
    KiteAeroDyn”.

        ----------
        XV1 : Point A of Bound vortex (array)
        XV2 : Point B of Bound vortex (array)
        XVP : Control point (array)
        gamma : Strength of the vortex (scalar)

        Returns
        -------
        vel_ind : Induced velocity (array)

    """
    r0 = XV2 - XV1  # Vortex filament
    r1 = XVP - XV1  # Controlpoint to one end of the vortex filament
    r2 = XVP - XV2  # Controlpoint to one end of the vortex filament

    # Cross products used for later computations
    r1Xr0 = cross_product(r1, r0)
    r2Xr0 = cross_product(r2, r0)

    epsilon = 0.05 * vec_norm(r0)  # Cut-off radius

    if (
        vec_norm(r1Xr0) / vec_norm(r0) > epsilon
    ):  # Perpendicular distance from XVP to vortex filament (r0)
        r1Xr2 = cross_product(r1, r2)
        vel_ind = (
            gamma
            / (4 * np.pi)
            * r1Xr2
            / (vec_norm(r1Xr2) ** 2)
            * dot_product(r0, r1 / vec_norm(r1) - r2 / vec_norm(r2))
        )
    else:
        # The control point is placed on the edge of the radius core
        # proj stands for the vectors respect to the new controlpoint
        r1_proj = dot_product(r1, r0) * r0 / (
            vec_norm(r0) ** 2
        ) + epsilon * r1Xr0 / vec_norm(r1Xr0)
        r2_proj = dot_product(r2, r0) * r0 / (
            vec_norm(r0) ** 2
        ) + epsilon * r2Xr0 / vec_norm(r2Xr0)
        r1Xr2_proj = cross_product(r1_proj, r2_proj)
        vel_ind_proj = (
            gamma
            / (4 * np.pi)
            * r1Xr2_proj
            / (vec_norm(r1Xr2_proj) ** 2)
            * dot_product(r0, r1_proj / vec_norm(r1_proj) - r2_proj / vec_norm(r2_proj))
        )
        vel_ind = vec_norm(r1Xr0) / (vec_norm(r0) * epsilon) * vel_ind_proj
    return vel_ind


def velocity_3D_trailing_vortex(XV1, XV2, XVP, gamma, Uinf):
    """
        Calculate the velocity induced by a trailing vortex filament in a point in space

        Vortex core correction from:
            Rick Damiani et al. “A vortex step method for nonlinear airfoil polar data as implemented in
    KiteAeroDyn”.
        ----------
        XV1 : Point A of the vortex filament (array)
        XV2 : Point B of the vortex filament (array)
        XVP : Controlpoint (array)
        gamma : Strength of the vortex (scalar)
        Uinf : Inflow velocity modulus (scalar)

        Returns
        -------
        vel_ind : induced velocity by the trailing fil. (array)

    """
    r0 = XV2 - XV1  # Vortex filament
    r1 = XVP - XV1  # Controlpoint to one end of the vortex filament
    r2 = XVP - XV2  # Controlpoint to one end of the vortex filament

    alpha0 = 1.25643  # Oseen parameter
    nu = 1.48e-5  # Kinematic viscosity of air
    r_perp = (
        dot_product(r1, r0) * r0 / (vec_norm(r0) ** 2)
    )  # Vector from XV1 to XVP perpendicular to the core radius
    epsilon = np.sqrt(4 * alpha0 * nu * vec_norm(r_perp) / Uinf)  # Cut-off radius

    # Cross products used for later computations
    r1Xr0 = cross_product(r1, r0)
    r2Xr0 = cross_product(r2, r0)

    if (
        vec_norm(r1Xr0) / vec_norm(r0) > epsilon
    ):  # Perpendicular distance from XVP to vortex filament (r0)
        r1Xr2 = cross_product(r1, r2)
        vel_ind = (
            gamma
            / (4 * np.pi)
            * r1Xr2
            / (vec_norm(r1Xr2) ** 2)
            * dot_product(r0, r1 / vec_norm(r1) - r2 / vec_norm(r2))
        )
    else:
        # The control point is placed on the edge of the radius core
        # proj stands for the vectors respect to the new controlpoint
        r1_proj = dot_product(r1, r0) * r0 / (
            vec_norm(r0) ** 2
        ) + epsilon * r1Xr0 / vec_norm(r1Xr0)
        r2_proj = dot_product(r2, r0) * r0 / (
            vec_norm(r0) ** 2
        ) + epsilon * r2Xr0 / vec_norm(r2Xr0)
        r1Xr2_proj = cross_product(r1_proj, r2_proj)
        vel_ind_proj = (
            gamma
            / (4 * np.pi)
            * r1Xr2_proj
            / (vec_norm(r1Xr2_proj) ** 2)
            * dot_product(r0, r1_proj / vec_norm(r1_proj) - r2_proj / vec_norm(r2_proj))
        )
        vel_ind = vec_norm(r1Xr0) / (vec_norm(r0) * epsilon) * vel_ind_proj
    return vel_ind


def velocity_induced_single_ring_semiinfinite(ring, controlpoint, model, Uinf):
    """
    Calculates the velocity induced by a ring at a certain controlpoint

    Parameters
    ----------
    ring : List of dictionaries defining the filaments of a vortex ring
    controlpoint : Dictionary defining a controlpoint
    model : VSM: Vortex Step method/ LLT: Lifting Line Theory
    Uinf : Wind speed vector

    Returns
    -------
    velind : Induced velocity

    """

    velind = [0, 0, 0]
    for filament in ring:
        GAMMA = filament["Gamma"]
        XV1 = filament["x1"]

        XVP = controlpoint
        if filament["id"] == "trailing_inf1":
            Vf = filament["dir"]
            tempvel = velocity_3D_trailing_vortex_semiinfinite(
                XV1, Vf, XVP, GAMMA, Uinf
            )
        elif filament["id"] == "trailing_inf2":
            Vf = filament["dir"]
            tempvel = velocity_3D_trailing_vortex_semiinfinite(
                XV1, Vf, XVP, -GAMMA, Uinf
            )
        elif filament["id"] == "bound":
            if model == "VSM":
                XV2 = filament["x2"]
                tempvel = velocity_3D_bound_vortex(XV1, XV2, XVP, GAMMA)
            else:
                tempvel = [0, 0, 0]
        else:
            XV2 = filament["x2"]
            tempvel = velocity_3D_trailing_vortex(XV1, XV2, XVP, GAMMA, Uinf)

        velind[0] += tempvel[0]
        velind[1] += tempvel[1]
        velind[2] += tempvel[2]

    return velind


def velocity_3D_from_vortex_filament_semiinfinite(XV1, Vf, XVP, GAMMA, CORE):

    # function to calculate the velocity induced by a straight 3D vortex filament
    # with circulation GAMMA at a point VP1. The geometry of the vortex filament
    # is defined by its edges: the filaments start at XV1 and ends at XV2.
    # the input CORE defines a vortex core radius, inside which the velocity
    # is defined  as a solid body rotation.
    # The function is adapted from the algorithm presented in:
    #                Katz, Joseph, and Allen Plotkin. Low-speed aerodynamics.
    #                Vol. 13. Cambridge university press, 2001.

    # read coordinates that define the vortex filament
    X1 = XV1[0]
    Y1 = XV1[1]
    Z1 = XV1[2]  # start point of vortex filament
    Vfx = Vf[0]
    Vfy = Vf[1]
    Vfz = Vf[2]  # end point of vortex filament
    # read coordinates of target point where the velocity is calculated
    XP = XVP[0]
    YP = XVP[1]
    ZP = XVP[2]
    # calculate geometric relations for integral of the velocity induced by filament
    R1 = np.sqrt((XP - X1) ** 2 + (YP - Y1) ** 2 + (ZP - Z1) ** 2)

    R1XV_X = (YP - Y1) * Vfz - (ZP - Z1) * Vfy
    R1XV_Y = -(XP - X1) * Vfz + (ZP - Z1) * Vfx
    R1XV_Z = (XP - X1) * Vfy - (YP - Y1) * Vfx
    R1XR_SQR = R1XV_X**2 + R1XV_Y**2 + R1XV_Z**2
    VfR1 = Vfx * (XP - X1) + Vfy * (YP - Y1) + Vfz * (ZP - Z1)
    # check if target point is in the vortex filament core,
    # and modify to solid body rotation
    if R1XR_SQR < CORE**2:
        R1XR_SQR = CORE**2
    # GAMMA = 0;
    if R1 < CORE:
        R1 = CORE

    # GAMMA = 0
    # determine scalar
    K = GAMMA / 4 / np.pi / R1XR_SQR * (1 + VfR1 / R1)
    # determine the three velocity components
    U = K * R1XV_X
    V = K * R1XV_Y
    W = K * R1XV_Z
    # output results, vector with the three velocity components
    results = [U, V, W]
    return results


def velocity_3D_from_vortex_filament(XV1, XV2, XVP, GAMMA, CORE):

    # function to calculate the velocity induced by a straight 3D vortex filament
    # with circulation GAMMA at a point VP1. The geometry of the vortex filament
    # is defined by its edges: the filaments start at XV1 and ends at XV2.
    # the input CORE defines a vortex core radius, inside which the velocity
    # is defined  as a solid body rotation.
    # The function is adapted from the algorithm presented in:
    #                Katz, Joseph, and Allen Plotkin. Low-speed aerodynamics.
    #                Vol. 13. Cambridge university press, 2001.

    # read coordinates that define the vortex filament
    X1 = XV1[0]
    Y1 = XV1[1]
    Z1 = XV1[2]  # start point of vortex filament
    X2 = XV2[0]
    Y2 = XV2[1]
    Z2 = XV2[2]  # end point of vortex filament
    # read coordinates of target point where the velocity is calculated
    XP = XVP[0]
    YP = XVP[1]
    ZP = XVP[2]
    # calculate geometric relations for integral of the velocity induced by filament
    R1 = np.sqrt((XP - X1) ** 2 + (YP - Y1) ** 2 + (ZP - Z1) ** 2)
    R2 = np.sqrt((XP - X2) ** 2 + (YP - Y2) ** 2 + (ZP - Z2) ** 2)
    R1XR2_X = (YP - Y1) * (ZP - Z2) - (ZP - Z1) * (YP - Y2)
    R1XR2_Y = -(XP - X1) * (ZP - Z2) + (ZP - Z1) * (XP - X2)
    R1XR2_Z = (XP - X1) * (YP - Y2) - (YP - Y1) * (XP - X2)
    R1XR_SQR = R1XR2_X**2 + R1XR2_Y**2 + R1XR2_Z**2
    R0R1 = (X2 - X1) * (XP - X1) + (Y2 - Y1) * (YP - Y1) + (Z2 - Z1) * (ZP - Z1)
    R0R2 = (X2 - X1) * (XP - X2) + (Y2 - Y1) * (YP - Y2) + (Z2 - Z1) * (ZP - Z2)
    # check if target point is in the vortex filament core,
    # and modify to solid body rotation
    if R1XR_SQR < CORE**2:
        R1XR_SQR = CORE**2
    # GAMMA = 0;
    if R1 < CORE:
        R1 = CORE
    # GAMMA = 0;
    if R2 < CORE:
        R2 = CORE
    # GAMMA = 0
    # determine scalar
    K = GAMMA / 4 / np.pi / R1XR_SQR * (R0R1 / R1 - R0R2 / R2)
    # determine the three velocity components
    U = K * R1XR2_X
    V = K * R1XR2_Y
    W = K * R1XR2_Z
    # output results, vector with the three velocity components
    results = [U, V, W]
    return results


def velocity_induced_single_ring_semiinfinite_nocore(ring, controlpoint, model):
    CORE = 1e-20
    velind = [0, 0, 0]
    for filament in ring:
        GAMMA = filament["Gamma"]
        XV1 = filament["x1"]

        XVP = controlpoint
        if filament["id"] == "trailing_inf1":
            Vf = filament["dir"]
            tempvel = velocity_3D_from_vortex_filament_semiinfinite(
                XV1, Vf, XVP, GAMMA, CORE
            )
        elif filament["id"] == "trailing_inf2":
            Vf = filament["dir"]
            tempvel = velocity_3D_from_vortex_filament_semiinfinite(
                XV1, Vf, XVP, -GAMMA, CORE
            )
        elif filament["id"] == "bound":
            if model == "VSM":
                XV2 = filament["x2"]
                tempvel = velocity_3D_from_vortex_filament(XV1, XV2, XVP, GAMMA, CORE)
            else:
                tempvel = [0, 0, 0]
        else:
            XV2 = filament["x2"]
            tempvel = velocity_3D_from_vortex_filament(XV1, XV2, XVP, GAMMA, CORE)

        velind[0] += tempvel[0]
        velind[1] += tempvel[1]
        velind[2] += tempvel[2]

    return velind


def airfoil_coeffs(alpha, coeffs):

    cl = np.interp(alpha * 180 / np.pi, coeffs[:, 0], coeffs[:, 1])
    cd = np.interp(alpha * 180 / np.pi, coeffs[:, 0], coeffs[:, 2])
    cm = np.interp(alpha * 180 / np.pi, coeffs[:, 0], coeffs[:, 3])

    return cl, cd, cm


def solve_lifting_line_system_matrix_approach_art_visc(
    ringvec, controlpoints, rings, Uinf, Gamma0, data_airf, conv_crit, model, rho=1.225
):
    """
    Solve the VSM or LLM by finding the distribution of Gamma

    Parameters
    ----------
    ringvec : List of dictionaries containing the vectors that define each ring
    controlpoints :List of dictionaries with the variables needed to define each wing section
    rings : List of list with the definition of each vortex filament
    Uinf : Wind speed velocity vector
    data_airf : 2D airfoil data with alpha, Cl, Cd, Cm
    recalc_alpha : True if you want to recalculate the induced angle of attack at 1/4 of the chord (VSM)
    Gamma0 : Initial Guess of Gamma
    model : VSM: Vortex Step method/ LLT: Lifting Line Theory

    Returns
    -------
    F: Lift, Drag and Moment at each section
    Gamma: Gamma at each section
    aero_coeffs: alpha, cl, cd, cm at each wing section

    """

    nocore = False  # To shut down core corrections input True
    # Initialization of the parameters
    velocity_induced = []
    u = 0
    v = 0
    w = 0
    N = len(rings)
    Gammaini = Gamma0
    Gamma = np.zeros(N)
    GammaNew = Gammaini
    Lift = np.zeros(N)
    Drag = np.zeros(N)
    Ma = np.zeros(N)
    alpha = np.zeros(N)
    cl = np.zeros(N)
    cd = np.zeros(N)
    cm = np.zeros(N)
    MatrixU = np.empty((N, N))
    MatrixV = np.empty((N, N))
    MatrixW = np.empty((N, N))
    U_2D = np.zeros((N, 3))

    # Number of iterations and convergence criteria
    Niterations = conv_crit["Niterations"]
    errorlimit = conv_crit["error"]
    relax = conv_crit["Relax_factor"]
    k2 = conv_crit["k2"]
    k4 = conv_crit["k4"]
    converged = False
    rho = 1.225

    # Look-up airfoil 2D coefficients
    if data_airf.ndim == 3:
        aoa_stall = find_stall_angle(data_airf[:, :, int(N / 2)]) * np.pi / 180
    else:
        aoa_stall = find_stall_angle(data_airf) * np.pi / 180

    coord_cp = [controlpoints[icp]["coordinates"] for icp in range(N)]
    chord = [controlpoints[icp]["chord"] for icp in range(N)]
    airf_coord = [controlpoints[icp]["airf_coord"] for icp in range(N)]

    # TODO: take this as AIC matrix
    for icp in range(N):

        if model == "VSM":
            # Velocity induced by a infinte bound vortex with Gamma = 1
            U_2D[icp] = velocity_induced_bound_2D(ringvec[icp])

        for jring in range(N):
            rings[jring] = update_Gamma_single_ring(rings[jring], 1, 1)
            # Calculate velocity induced by a ring to a control point
            velocity_induced = velocity_induced_single_ring_semiinfinite(
                rings[jring], coord_cp[icp], model, vec_norm(Uinf)
            )
            # If CORE corrections are deactivated
            if nocore == True:
                # Calculate velocity induced by a ring to a control point
                velocity_induced = velocity_induced_single_ring_semiinfinite_nocore(
                    rings[jring], coord_cp[icp], model
                )

            # AIC Matrix
            MatrixU[icp, jring] = velocity_induced[0]
            MatrixV[icp, jring] = velocity_induced[1]
            MatrixW[icp, jring] = velocity_induced[2]

    # Start solving iteratively
    for kiter in range(Niterations):

        for ig in range(len(Gamma)):
            Gamma[ig] = GammaNew[ig]

        for icp in range(N):
            # Initialize induced velocity to 0
            u = 0
            v = 0
            w = 0
            # Compute induced velocities with previous Gamma distribution
            for jring in range(N):
                u = u + MatrixU[icp][jring] * Gamma[jring]
                # x-component of velocity
                v = v + MatrixV[icp][jring] * Gamma[jring]
                # y-component of velocity
                w = w + MatrixW[icp][jring] * Gamma[jring]
                # z-component of velocity

            u = u - U_2D[icp, 0] * Gamma[icp]
            v = v - U_2D[icp, 1] * Gamma[icp]
            w = w - U_2D[icp, 2] * Gamma[icp]

            # Calculate terms of induced corresponding to the airfoil directions
            norm_airf = airf_coord[icp][:, 0]
            tan_airf = airf_coord[icp][:, 1]
            z_airf = airf_coord[icp][:, 2]

            # Calculate relative velocity and angle of attack
            Urel = [Uinf[0] + u, Uinf[1] + v, Uinf[2] + w]
            vn = dot_product(norm_airf, Urel)
            vtan = dot_product(tan_airf, Urel)
            alpha[icp] = np.arctan(vn / vtan)

            Urelcrossz = np.cross(Urel, z_airf)
            Umag = np.linalg.norm(Urelcrossz)
            Uinfcrossz = np.cross(Uinf, z_airf)
            Umagw = np.linalg.norm(Uinfcrossz)

            # Look-up airfoil 2D coefficients
            if data_airf.ndim == 3:
                cl[icp], cd[icp], cm[icp] = airfoil_coeffs(
                    alpha[icp], data_airf[:, :, icp]
                )
            else:
                cl[icp], cd[icp], cm[icp] = airfoil_coeffs(alpha[icp], data_airf)

            # Retrieve forces and moments
            Lift[icp] = 0.5 * rho * Umag**2 * cl[icp] * chord[icp]
            Drag[icp] = 0.5 * rho * Umag**2 * cd[icp] * chord[icp]
            Ma[icp] = 0.5 * rho * Umag**2 * cm[icp] * chord[icp] ** 2

            # Find the new gamma using Kutta-Joukouski law
            GammaNew[icp] = 0.5 * Umag**2 / Umagw * cl[icp] * chord[icp]

        # stall = []

        # for i,aoa in enumerate(alpha):
        #     if aoa>aoa_stall:
        #         stall = np.arange(i-3, len(alpha)-i+3)
        #         break
        npan = len(Gamma)
        stall = []

        # threshold = 2/180*np.pi
        # Loop through the array, checking each pair of adjacent items to see if damping is needed
        # for i in range(len(alpha) - 1):
        #     if abs(alpha[i+1] - alpha[i]) > threshold and alpha[i]>aoa_stall:
        #        stall = np.arange(i-4, len(alpha)-i+4)
        #        break
        # Loop through the array, checking each pair of adjacent items to see if damping is needed
        stall = False
        for i in range(len(alpha) - 1):
            if alpha[i] > aoa_stall:
                stall = True
                break
        # If not converged, apply convergence weighting and continue the iteration
        for ig in range(len(Gamma)):
            if ig == 0:
                Gim2 = Gamma[0]
                Gim1 = Gamma[0]
                Gi = Gamma[0]
                Gip1 = Gamma[1]
                Gip2 = Gamma[2]
            elif ig == 1:
                Gim2 = Gamma[0]
                Gim1 = Gamma[0]
                Gi = Gamma[1]
                Gip1 = Gamma[2]
                Gip2 = Gamma[3]
            elif ig == npan - 2:
                Gim2 = Gamma[npan - 4]
                Gim1 = Gamma[npan - 3]
                Gi = Gamma[npan - 2]
                Gip1 = Gamma[npan - 1]
                Gip2 = Gamma[npan - 1]
            elif ig == npan - 1:
                Gim2 = Gamma[npan - 3]
                Gim1 = Gamma[npan - 2]
                Gi = Gamma[npan - 1]
                Gip1 = Gamma[npan - 1]
                Gip2 = Gamma[npan - 1]
            else:
                Gim2 = Gamma[ig - 2]
                Gim1 = Gamma[ig - 1]
                Gi = Gamma[ig]
                Gip1 = Gamma[ig + 1]
                Gip2 = Gamma[ig + 2]

            dif2 = (Gip1 - Gi) - (Gi - Gim1)
            dif4 = (Gip2 - 3.0 * Gip1 + 3.0 * Gi - Gim1) - (
                Gip1 - 3.0 * Gi + 3.0 * Gim1 - Gim2
            )

            if stall is False:
                damp = 0
            # elif ig <1 or ig >npan-2:
            #     damp =0
            else:
                damp = k2 * dif2 - k4 * dif4
            # damp = k2*dif2 - k4*dif4
            GammaNew[ig] = (1 - relax) * Gamma[ig] + relax * GammaNew[ig] + damp

        # check convergence of solution
        refererror = np.amax(np.abs(GammaNew))
        refererror = np.amax([refererror, 0.001])
        # define scale of bound circulation
        error = np.amax(np.abs(GammaNew - Gamma))
        # difference betweeen iterations
        error = error / refererror
        # relative error
        if error < errorlimit:
            # if error smaller than limit, stop iteration cycle
            converged = True

            break
    if converged == False:
        print("Not converged after " + str(Niterations) + " iterations")

    # In case VSM, calculate the effective angle of attack at a 1/4 chord
    if model == "VSM":

        for ig in range(len(Gamma)):
            Gamma[ig] = GammaNew[ig]
        for icp in range(N):
            # Compute induced velocities at 1/4 chord
            for jring in range(N):
                rings[jring] = update_Gamma_single_ring(rings[jring], 1, 1)
                velocity_induced = velocity_induced_single_ring_semiinfinite(
                    rings[jring],
                    controlpoints[icp]["coordinates_aoa"],
                    "LLT",
                    vec_norm(Uinf),
                )
                if nocore == True:
                    velocity_induced = velocity_induced_single_ring_semiinfinite_nocore(
                        rings[jring], controlpoints[icp]["coordinates_aoa"], model
                    )
                MatrixU[icp, jring] = velocity_induced[0]
                MatrixV[icp, jring] = velocity_induced[1]
                MatrixW[icp, jring] = velocity_induced[2]

        for icp in range(N):
            u = 0
            v = 0
            w = 0
            for jring in range(N):
                u = u + MatrixU[icp][jring] * Gamma[jring]
                # x-component of velocity
                v = v + MatrixV[icp][jring] * Gamma[jring]
                # y-component of velocity
                w = w + MatrixW[icp][jring] * Gamma[jring]
                # z-component of velocity

            # Calculate terms of induced corresponding to the airfoil directions
            norm_airf = airf_coord[icp][:, 0]
            tan_airf = airf_coord[icp][:, 1]
            z_airf = airf_coord[icp][:, 2]

            Urel = [Uinf[0] + u, Uinf[1] + v, Uinf[2] + w]
            vn = np.dot(norm_airf, Urel)
            vtan = np.dot(tan_airf, Urel)
            # New relative angle of attack
            alpha[icp] = np.arctan(vn / vtan)

    aero_coeffs = np.column_stack([alpha, cl, cd, cm])
    F = np.column_stack([Lift, Drag, Ma])
    return F, Gamma, aero_coeffs
