#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:13:37 2023

Author: Alistair Bell

Contact: alistair.bell@unibe.ch
"""

import numpy as np
from scipy.stats import chi2

def wave_signif(Y, dt, scale1, sigtest=-1, lag1=-1, siglvl=-1, dof=-1, mother=-1, param=-1):
    if mother == -1:
        mother = "MORLET"
    mother = mother.upper()

    n1 = len(Y)
    J1 = len(scale1) - 1
    scale = scale1.copy()

    s0 = min(scale)
    dj = np.log(scale[1] / scale[0]) / np.log(2)

    variance = Y if n1 == 1 else np.var(Y)

    if lag1 == -1:
        lag1 = 0.0
    if siglvl == -1:
        siglvl = 0.95

    # Define parameters based on mother wavelet
    if mother == 'MORLET':
        param = 6. if param == -1 else param
        k0 = param
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))
        empir = [2., -1, -1, -1]
        if k0 == 6:
            empir[1:] = [0.776, 2.32, 0.60]
    # Add similar blocks for 'PAUL' and 'DOG'

    period = scale * fourier_factor
    dofmin = empir[0]
    Cdelta = empir[1]
    gamma_fac = empir[2]
    dj0 = empir[3]

    freq = dt / period
    fft_theor = (1 - lag1 ** 2) / (1 - 2 * lag1 * np.cos(freq * 2 * np.pi) + lag1 ** 2)
    fft_theor *= variance
    signif = fft_theor.copy()

    if dof == -1:
        dof = dofmin

    # Perform significance tests
    if sigtest == 0:
        dof = dofmin
        chisquare = chi2.ppf(siglvl, dof) / dof
        signif = fft_theor * chisquare
    elif sigtest == 1:
        # Add the time-averaged significance test implementation
        pass
    elif sigtest == 2:
        # Add the scale-averaged significance test implementation
        pass
    else:
        raise ValueError('sigtest must be either 0, 1, or 2')

    return signif, fft_theor


