'''
Author: Vishnu Suresh Nair
Code to simulate 3 stage launch vehicle
lv_init.py - loads all initial conditions and prerequisites
mass_optimiser.py - stage optimizer for 3 stage LV
coe_from_sv.py - state vector to orbital elements conversion
'''
## Ascent Trajectory Design
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import numpy as np
from numpy import sin, cos, tan, pi, zeros as zr, matrix as mat, arctan as atan, \
    arcsin as asin, arccos as acos, cross, square, shape, ones, append, \
    multiply as mul, dot, arctan2 as atan2, matmul, exp, vstack, \
    cumsum, arange, array, transpose as T, mean
from functools import reduce
import math as math
from numpy.linalg import norm, solve, inv
from math import log, radians, sqrt, degrees as deg
import os
from bisect import bisect_left
from matplotlib import pyplot as plt
import xlrd
import xlsxwriter
from lv_init_fin import *
from coe_from_sv import *
from multiprocessing import Pool, cpu_count
import itertools as it
import time

start_time = time.time()

##----------------------------- Function Definitions ---------------------------

def rocket():
    global m_hs, m_init

    Th[x + 1], m[x + 1], mprop[x + 1], g[x + 1], Q[x + 1], D[x + 1], mach[x + 1], a[x + 1], am[x + 1], \
    v[x + 1], vr[x + 1], vm[x + 1], vrm[x + 1], r[x + 1], alt[x + 1], aoa[x + 1], fpa[x + 1], \
    lat_F[x + 1], long_F[x + 1], steerI[x + 1], rateI[x + 1], vAB[x + 1], AziA[x + 1], \
    LB[x + 1], IB[x + 1], IG[x + 1], GA[x + 1], AB[x + 1], GB[x + 1] \
        = lv_calc(mode, deltt, IL, gamma_air, R_air, j2, Re, Rs, mu, alt[x], m_init, \
                  mprop_init, flow_con0, mrate, r[x], v[x], vr[x], vAB[x], a[x], fpa[x], area, \
                  cd_data, mach_data, alt_data, P_data, T_data, rho_data, steerI[x], LB[x], IB[x], \
                  IG[x], GA[x], GB[x], Thrust, rateI[x], lat_F, long_F, Om_p)

    roll = steerI[x][0]

    # Roll Correction    
    if roll < 0:
        steerI[i][0] = roll + min([abs(roll) / 2, 5 * delt]) * delt
    elif roll > 0:
        steerI[i][0] = roll - min([roll / 2, 5 * delt]) * delt

    if 115 <= alt[x] / 1000 <= 120:
        m_init -= m_hs
        if m_hs != 0:
            '''
            print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')
            print('\tPAYLOAD FAIRING SEPERATED')
            print('\tAltitude:', round(alt[x] / 1000, 2), 'kms')
            print('\tTime:', round(tin, 3), 's')
            print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')
            '''
        m_hs = 0


# ----------------------------------Table Lookup---------------------------------
def lookup(x, xs, ys):
    if x <= xs[0]:  return ys[0]
    if x >= xs[-1]: return ys[-1]

    i = bisect_left(xs, x)
    k = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    y = k * (ys[i] - ys[i - 1]) + ys[i - 1]

    return y


# ---------------------------RungeKutta Integration---------------------------#
def RK4(dx1, dx2, sep, y):
    # This program numericaly solves equations using RK4 iterations
    # To be used only for linear functions of time alone

    dx3 = (dx1 + dx2) / 2  # dx value at delt/2
    k1 = sep * dx1
    k2 = sep * dx3
    k3 = k2  # k2=k3 as only time is variable
    k4 = sep * dx2

    y += k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
    return y


# ----------------------------------Gravity Model-------------------------------#
# As per NASA CR-132689
def gravity(dist, J2, RE, MU):
    [x, y, z] = dist
    re = sqrt(x ** 2 + y ** 2 + z ** 2)
    R = RE / re
    Z = z / re
    J = (3 / 2) * J2

    p = 1 + J * (R ** 2) * (1 - 5 * Z ** 2)

    gx = -1 * MU * x * p / (re ** 3)
    gy = -1 * MU * y * p / (re ** 3)
    gz = -1 * (MU / re ** 3) * (1 + J * R ** 2 * (3 - 5 * Z ** 2)) * z

    g = [gx, gy, gz]

    return g


# ------------------------------Co-ordinate Transformations----------------------
# Inertial to Launch Pad frame
def ILmatrix(lat_L, long_L, AzL):
    IL1 = [cos(lat_L) * cos(long_L), cos(lat_L) * sin(long_L), sin(lat_L)]
    IL2 = [sin(lat_L) * cos(long_L) * sin(AzL) - cos(AzL) * sin(long_L),
           cos(AzL) * cos(long_L) + sin(AzL) * sin(lat_L) * sin(long_L), -sin(AzL) * cos(lat_L)]
    IL3 = [-sin(AzL) * sin(long_L) - cos(AzL) * sin(lat_L) * cos(long_L),
           sin(AzL) * cos(long_L) - cos(AzL) * sin(lat_L) * sin(long_L), cos(AzL) * cos(lat_L)]
    ILmat = [IL1, IL2, IL3]
    return array(ILmat)


# -----------------Launch Vehicle to body frame transformation-------------------
def LBmatrix(phi, theta, psi):
    LB1 = [cos(psi) * cos(theta), cos(phi) * sin(psi) * cos(theta) + sin(phi) * sin(theta),
           sin(phi) * sin(psi) * cos(theta) - cos(phi) * sin(theta)]
    LB2 = [-sin(psi), cos(phi) * cos(psi), sin(phi) * cos(psi)]
    LB3 = [cos(psi) * sin(theta), cos(phi) * sin(psi) * sin(theta) - sin(phi) * cos(theta),
           sin(phi) * sin(psi) * sin(theta) + cos(phi) * cos(theta)]
    LBmat = [LB1, LB2, LB3]
    return array(LBmat)


# ------------Earth Centered Inertial to body frame transformation---------------
def IBmatrix(LB_in, IL_in):
    IBmat = LB_in @ IL_in
    return IBmat


# ----------------- Inertial to Geographic frame transformation-------------------
def IGmatrix(phi_c, theta_i):
    IGmat = [[-sin(phi_c) * cos(theta_i), -sin(phi_c) * sin(theta_i), cos(phi_c)],
             [-sin(theta_i), cos(theta_i), 0],
             [-cos(phi_c) * cos(theta_i), -cos(phi_c) * sin(theta_i), -sin(phi_c)]]

    return array(IGmat)


# --------------------Geographic to Body frame transformation--------------------
def GBmatrix(phi_r, theta_r, psi_r):
    GBmat = [[cos(theta_r) * cos(psi_r), cos(theta_r) * sin(psi_r), -sin(theta_r)],
             [sin(phi_r) * sin(theta_r) * cos(psi_r) - cos(phi_r) * sin(psi_r),
              sin(phi_r) * sin(theta_r) * sin(psi_r) + cos(phi_r) * cos(psi_r),
              sin(phi_r) * cos(theta_r)],
             [cos(phi_r) * sin(theta_r) * cos(psi_r) + sin(phi_r) * sin(psi_r),
              cos(phi_r) * sin(theta_r) * sin(psi_r) - sin(phi_r) * cos(psi_r),
              cos(phi_r) * cos(theta_r)]]

    return array(GBmat)


# ---Geographic to Atmospheric Relative Velocity System (ARVS) transformation----
def GAmatrix(gamma_a, lambda_a):
    GAmat = [[cos(gamma_a) * cos(lambda_a), cos(gamma_a) * sin(lambda_a), -sin(gamma_a)],
             [-sin(lambda_a), cos(lambda_a), 0],
             [sin(gamma_a) * cos(lambda_a), sin(gamma_a) * sin(lambda_a), cos(gamma_a)]]

    return array(GAmat)


# -------------------------ARVS to body transformation---------------------------
def ABmatrix(al, bet, sig):
    ABmat = [[cos(al) * cos(bet), -cos(al) * sin(bet) * cos(sig) + sin(al) * sin(sig),
              -cos(al) * sin(bet) * sin(sig) - sin(al) * cos(sig)],
             [sin(bet), cos(bet) * cos(sig), cos(bet) * sin(sig)],
             [sin(al) * cos(bet), -sin(al) * sin(bet) * cos(sig) - cos(al) * sin(sig),
              -sin(al) * sin(bet) * sin(sig) + cos(al) * cos(sig)]]

    return array(ABmat)


# -----------------------Inertial to Planet Relative transformation--------------
def IPmatrix(om, time):
    IPmat = [[cos(om * time), sin(om * time), 0],
             [-sin(om * time), cos(om * time), 0],
             [0, 0, 1]]

    return array(IPmat)


# --------------------------------Drag Calculation-------------------------------
def drag_calc(V, a, rho, area, cd_data, mach_data):
    mach = norm(V) / a;
    if mach < max(mach_data):
        cd = lookup(mach, mach_data, cd_data)  # import Cd as function of Mach
    else:
        cd = 0
    # cd = 0.3
    Q = 0.5 * rho * (norm(V)) ** 2
    drag = array([cd * area * Q, 0, 0])
    return drag, Q, mach


# --------------Calculate r,v,a,gamma,alpha,phi,theta,psi,mach,D,Q---------------
def lv_calc(fmode, dt, IL, gamma_amb, R_amb, j2, Re, Rs, mu, alt_i, m0, mprop0, mc, md, rI_i, \
            vI_i, vAI_i, vAB_i, aI_i, fpa_i, area, cd_data, mach_data, alt_data, Pr, Temp, rho, \
            steerI_i, LB_i, IB_i, IG_i, GA_i, GB_i, Th, rate_i, lat_i, long_i, Omp):
    # global tin,alt,r,v,vr,vm,vrm
    global tin

    # Ambient data
    T_amb, rho_amb = lookup(alt_i / 1000, alt_data, Temp), lookup(alt_i / 1000, alt_data, rho)
    a_amb = sqrt(gamma_amb * R_amb * T_amb)

    # Mass updation
    # deltat = t-t0
    mprop_out, m_out = mprop0 - mc - md * (tin - t0), m0 - mc - md * (tin - t0)

    g = array(gravity(rI_i, j2, Re, mu))
    Df, Q, mach = drag_calc(vAI_i, a_amb, rho_amb, area, cd_data, mach_data)
    Dfm = norm(Df)

    if fmode != 'GT':
        # Initiation of Velocities and Attitude angles (co-ordinate frame)
        phiI_i, thetaI_i, psiI_i = steerI_i[0], steerI_i[1], steerI_i[2]
        rr_i, pr_i, yr_i = rate_i[0], rate_i[1], rate_i[2]
        # (Inertial Euler Angles - final states)
        rr_f, pr_f, yr_f = rr_i, pr_i, yr_i
        phiI_f, thetaI_f, psiI_f = phiI_i + rr_f * dt, thetaI_i + pr_f * dt, psiI_i + yr_f * dt
        rate_out = array([rr_f, pr_f, yr_f])

        # Acceleration, Velocity and Position integration
        LB_f = LBmatrix(phiI_f, thetaI_f, psiI_f)
        IB_f = IBmatrix(LB_f, IL)
        Tf = array([Th, 0, 0])
        atB = (Tf - Df) / m_out
        aI_f = IB_f.T @ atB + g
        vI_f = RK4(aI_f, aI_i, dt, vI_i)
        rI_f = RK4(vI_f, vI_i, dt, rI_i)
        alt_f = norm(rI_f) - Rs

        # Holding criteria
        if alt_f < 0.0264:
            aI_f, vI_f, rI_f, alt_f = aI_i, vI_i, rI_i, norm(rI_f) - Rs

        vWI = array([0, 0, 0])
        vRI = vI_f - v[0]
        vAI_out = vRI + vWI

        # uRI = rI_f/norm(rI_f)
        # uVI = vI_f/norm(vI_f)
        # uVR = vRI/norm(vRI)
        # uVA = vAI_out/norm(vAI_out)
        # (Latitude and Longitude)
        # latc = asin(rI_f[2]/norm(rI_f))
        latc = atan2(rI_f[2], sqrt(rI_f[0] ** 2 + rI_f[1] ** 2))
        longI = atan2(rI_f[1], rI_f[0])
        # longR = longI-Omp[2]*(tin-t0)
        # Transformation to Geographic frame
        IG_f = IGmatrix(latc, longI)
        # vIG = IG_f@vI_f
        # vRG = IG_f@vRI
        vAG = IG_f @ vAI_out
        # (Flight Path Angles)
        # gammaI = asin(dot(uRI,uVI))
        # gammaR = asin(dot(uRI,uVR))
        # gammaA = asin(dot(uRI,uVA))
        gammaA = acos(vAG[0] / norm(vAG))
        # gammaA = atan2(vAG[2],vAG[0])
        # (Azimuth angles)
        # AzI = atan(vIG[1]/vIG[0])
        # AzR = atan(vRG[1]/vRG[0])
        AzA = atan(vAG[1] / vAG[0])
        # (Relative roll, pitch and yaw angles)
        GA_f = GAmatrix(gammaA, AzA)
        GB_f = IG_f.T @ IB_f  # For symmetrix matrix, inv(IG) = transpose(IG)
        # psiR = atan(GB_f[0,1]/GB_f[0,0])
        # thetaR = -asin(GB_f[0,2])
        # phiR = atan(GB_f[1,2]/GB_f[2,2])
        # Transformation to body frame
        vAB_f = IB_f @ vAI_out
        # (Aerodynamic Angles)
        alpha = atan(vAB_f[2] / vAB_f[0])
        beta = atan(vAB_f[1] / sqrt(vAB_f[2] ** 2 + vAB_f[0] ** 2))
        sigma = atan((GB_f[1, 2] + sin(beta) * sin(alpha)) \
                     / ((GB_f[1, 1] * cos(AzA) - GB_f[1, 0]) * sin(AzA) * cos(gammaA)))
        AB_f = ABmatrix(alpha, beta, sigma)

    else:

        vAG = IG_i @ vAI_i
        gammaA = acos(vAG[0] / norm(vAG))
        AzA = atan(vAG[1] / vAG[0])
        GA_f = GAmatrix(gammaA, AzA)
        # Initiation of Velocities and Attitude angles (co-ordinate frame)
        phiI_i = steerI_i[0]
        thetaI_i = steerI_i[1]
        psiI_i = steerI_i[2]
        # (Inertial Euler Angles during Gravity Turn)
        # phiI_f = atan(LB_f[1,2]/LB_f[1,1])
        phiI_f = 0
        # psiI_f = -asin(LB_f[1,0])
        psiI_f = 0
        thetaI_f = gammaA - radians(90)
        rr_f, pr_f, yr_f = (phiI_f - phiI_i) / dt, (thetaI_f - thetaI_i) / dt, (psiI_f - psiI_i) / dt
        rate_out = array([rr_f, pr_f, yr_f])
        # Acceleration, Velocity and Position integration
        LB_f = LBmatrix(phiI_f, thetaI_f, psiI_f)
        IB_f = IBmatrix(LB_f, IL)
        Tf = array([Th, 0, 0])
        atB = (Tf - Df) / m_out
        aI_f = IB_f.T @ atB + g
        vI_f = RK4(aI_f, aI_i, dt, vI_i)
        rI_f = RK4(vI_f, vI_i, dt, rI_i)
        alt_f = norm(rI_f) - Rs

        # Holding criteria
        if alt_f < 0.0264:
            aI_f, vI_f, rI_f, alt_f = aI_i, vI_i, rI_i, norm(rI_f) - Rs

        vWI = array([0, 0, 0])
        vRI = vI_f - v[0]
        vAI_out = vRI + vWI
        # (Latitude and Longitude)
        # latc = asin(rI_f[2]/norm(rI_f))
        latc = atan2(rI_f[2], sqrt(rI_f[0] ** 2 + rI_f[1] ** 2))
        longI = atan2(rI_f[1], rI_f[0])
        # longR = longI-Omp[2]*(tin-t0)
        # Transformation to Geographic frame
        IG_f = IGmatrix(latc, longI)
        GB_f = IG_f.T @ IB_f  # For symmetrix matrix, inv(IG) = transpose(IG)
        # Transformation to body frame
        vAB_f = IB_f @ vAI_out
        # (Aerodynamic Angles)
        alpha = 0
        beta = 0
        sigma = 0
        AB_f = ABmatrix(alpha, beta, sigma)

    # IP = IPmatrix(Omp[2],t)
    Tfm, aIm_f, vIm_f, vAIm_out = norm(Tf), norm(aI_f), norm(vI_f), norm(vAI_out)
    steerI_f = [phiI_f, thetaI_f, psiI_f]

    return Tfm, m_out, mprop_out, g, Q, Dfm, mach, aI_f, aIm_f, vI_f, vAI_out, vIm_f, \
           vAIm_out, rI_f, alt_f, alpha, gammaA, latc, longI, steerI_f, rate_out, vAB_f, \
           AzA, LB_f, IB_f, IG_f, GA_f, AB_f, GB_f

# ----------------------------Rocket Main-------------------------------------
def rocket_main(*args):
    global mode, deltt, x, m_init, mprop_init, flow_con0, mrate, Thrust, tin, t0, \
        tseries, idx, op_log, tags, values

    if len(args) == 0:
        pass

    # krate = prate_in[0]

    ind = 0
    idn = 0
    idx = zr(len(MS) - 1)
    Tmax = MS[-1][1]
    Tmin = MS[0][1]
    tseries = list(arange(Tmin + delt, Tmax + delt, delt))

    y = 0
    string0, string = MS[y], MS[y + 1]

    st_n, flag = int(string0[0]), string0[4]

    t0, Thrust0, flow_con0 = float(string0[1]), float(string0[2]), float(string0[3])
    t1, Thrust1, flow_con1 = float(string[1]), float(string[2]), float(string[3])
    # t,Thrust,flow_con = t1,mean([Thrust0,Thrust1]),mean([flow_con0,flow_con1])
    t, Thrust, flow_con = t1, Thrust1, flow_con1
    mrate = (flow_con1 - flow_con0) / (t1 - t0)

    m_init = mi[st_n - 1]
    mprop_init = mp[st_n - 1]

    for x, tin in enumerate(tseries):
        # print("Step:    ",x)

        if tin < t and not np.isclose(tin, t):
            deltt = delt
            # print(not tin<=t<=tin+delt)
            if kstart <= tin <= kend:
                rateI[x] = array([0, krate, 0])
                mode = 'BR'
            elif gtstart < tin <= gtend:
                mode = 'GT'
            else:
                rateI[x] = array([0, 0, 0])
                mode = 'BR'

            rocket()

        elif tin >= t or np.isclose(tin, t):
            tseries[x] = t
            tin = t
            deltt = tseries[x] - tseries[x - 1]
            if kstart <= tin <= kend:
                rateI[x] = array([0, krate, 0])
                mode = 'BR'
            elif gtstart < tin <= gtend:
                mode = 'GT'
            else:
                mode = 'BR'

            rocket()

            y += 1

            if mode == 'BR':
                event = 'Variable Bodyrate Maneuver'
            elif mode == 'GT':
                event = 'Gravity Turn'
            elif mode == 'CO':
                event = 'Separation'
#            on_screen_disp(event, st_n, tin, alt[x] / 1000, vm[x], mprop[x], deg(fpa[x]), deg(aoa[x]), deg(steerI[x][1]))

            try:
                string0, string = MS[y], MS[y + 1]
                st_n, flag = int(string[0]), string[4]
                t0, Thrust0, flow_con0 = float(string0[1]), float(string0[2]), float(string0[3])
                t1, Thrust1, flow_con1 = float(string[1]), float(string[2]), float(string[3])
                # t,Thrust,flow_con = t1,mean([Thrust0,Thrust1]),mean([flow_con0,flow_con1])
                t, Thrust, flow_con = t1, Thrust1, flow_con1
                mrate = (flow_con1 - flow_con0) / (t1 - t0)

            except:
                print('\n\n1st Stage Burn Completed!')

            # Stage Initial Conditions

            idx[idn] = x;

            #    plot_results(idx[idn-1],idx[idn],[eval(x) for x in pl])
            idn += 1
            ind += 1
            # t_data += time

    print('\nStage ' + str(st_n) + 'Seperation\n')
    # second and third stage
    for _ in range(2):
        st_n += 1
        thrusts, Isp, sf, p_rate, flow_con0, f_lo, n_seg = Thrust_in[st_n - 1], Sp_imp[st_n - 1], \
                                                           strf[st_n - 1], prate_in[st_n - 1], 0, m_lo[st_n - 1], segn[
                                                               st_n - 1]
        if m_hs == 0:
            m_init = mi[st_n - 1] - m_PLF
            print('\nPayload Fairing mass ' + str(m_PLF) + ' kgs removed \n')
            print('\nInitial mass ' + str(round(m_init / 1000, 3)) + ' T\n')
        else:
            m_init = mi[st_n - 1]
            print('\nInitial mass ' + str(round(m_init / 1000, 3)) + ' T\n')

        mprop_init = mp[st_n - 1]
        Ms = mprop_init - f_lo
        Is = Ms * Isp * 9.8055
        tslices = zeros(n_seg)
        m_rate = zeros(n_seg)
        tsl = 0
        for l, thrust in enumerate(list(thrusts)):
            tsl = tsl + Is / (n_seg * thrust)
            tslices[l] = tseries[x] + tsl
            m_rate[l] = thrust / (9.8055 * Isp)
        tslices = [tseries[x]] + tslices.tolist()
        f = 0
        for _ in range(500000):
            t0 = tseries[x]
            x += 1
            tseries += [tseries[-1] + delt]
            tin = tseries[x]
            for f in range(len(tslices) - 1):
                tsi = tslices[f]
                tsf = tslices[f + 1]
                if tsi <= tin < tsf:
                    prate = p_rate[f]
                    Thrust = thrusts[f]
                    mrate = m_rate[f]
            rateI[x] = [0, prate, 0]
            mode = 'BR'
            rocket()
            flow_con0 += mrate * delt

            if mode == 'BR':
                event = 'Variable Bodyrate Maneuver'
            elif mode == 'GT':
                event = 'Gravity Turn'
            elif mode == 'CO':
                event = 'Separation'

            #if round(tseries[x], 2) in np.around(tslices, 2):
                #on_screen_disp(event, st_n, tin, alt[x] / 1000, vm[x], mprop[x], deg(fpa[x]), deg(aoa[x]), deg(steerI[x][1]))
            if mprop_init - flow_con0 <= f_lo:
                print('\nStage ' + str(st_n) + ' Seperation\n')
                break

    # --------------------------------Orbital Elements---------------------------
    O_E = coe_from_sv(r[x], v[x], mu)
    Eta = O_E[0]
    hm = O_E[1]
    ecc = O_E[2]
    Om = O_E[3]
    inc = O_E[4]
    w = O_E[5]
    nu = O_E[6]
    semimaj = O_E[7]

    apg = semimaj * (1 + ecc) - Re
    prg = semimaj * (1 - ecc) - Re

    print('\n=================================================\n')
    print('\tOrbit Properties')
    print('\n\t--------------------\n')
    print('Eccentricity:', round(ecc, 3))
    print('Inclination:', round(deg(inc), 3), 'degrees')
    print('Apogee:', round(apg / 1000, 3), 'kms')
    print('Perigee:', round(prg / 1000, 3), 'kms')
    print('Right Ascension of Ascending Node:', round(deg(Om), 3), 'degrees')
    print('Argument of Periapsis:', round(deg(w), 3), 'degrees')
    print('True Anomaly:', round(deg(nu), 3), 'degrees')
    print('Orbital Energy:', round(Eta, 3), 'J/kg')
    print('Angular Momentum:', round(hm, 3), 'mps')
    print('\n=================================================\n')

    return vm[x],alt[x],deg(fpa[x]),
    

################################ Function Definitions End ######################

# -------------------------Import Stage Data-------------------------------------
with open(savepath + 'stage_data.txt', 'r') as f:
    staging_data = json.load(f)
    tags = staging_data[0]
    values = np.array(staging_data[1:4]).transpose().tolist()
for tag, value in zip(tags, values):
    locals()[tag] = value
# ------------------------------ Preallocation ----------------------------------
t, m, vm, vrm, Q, mprop, mach, alt, a, v, vr, r, g, steerI, rateI, fpa, lat_F, long_F, aoa, am, Th, \
D, AziA, vAB, LB, IB, GA, IG, AB, GB, t_data = zr(500000), zr(500000), zr(500000), \
                                               zr(500000), zr(500000), zr(500000), zr(500000), zr(500000), zr(
    [500000, 3]), \
                                               zr([500000, 3]), zr([500000, 3]), zr([500000, 3]), zr([500000, 3]), zr(
    [500000, 3]), \
                                               zr([500000, 3]), zr(500000), zr(500000), zr(500000), zr(500000), zr(
    500000), zr(500000) \
    , zr(500000), zr(500000), zr([500000, 3]), zr([5000000, 3, 3]), zr([5000000, 3, 3]), \
                                               zr([5000000, 3, 3]), zr([5000000, 3, 3]), zr([5000000, 3, 3]), zr(
    [5000000, 3, 3]), []

# ------------------------------Initial Conditions-------------------------------
m[0] = mi[0]
mprop[0] = mp[0]
v[0] = vi
vr[0] = [0, 0, 0]
r[0] = ri
g[0] = gravity(ri, j2, Re, mu)
a[0] = g[0]
alt[0] = norm(r[0]) - Rs
t[0] = 0
fpa[0] = pi / 2
lat_F[0] = lat_L
long_F[0] = long_L
m_hs = m_PLF
steerI[0] = [0, 0, 0]
IL = ILmatrix(lat_L, long_L, AzL)
# delt = 0.01

# ------------------------------Run Main Program---------------------------------

#print("--- %s seconds ---" % (time.time() - start_time))
