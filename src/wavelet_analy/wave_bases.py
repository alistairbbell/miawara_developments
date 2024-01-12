#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:26:05 2024

Author: Alistair Bell

Contact: alistair.bell@unibe.ch
"""

import numpy as np
from scipy.special import gamma

def wave_bases(mother, k, scale, param):
    mother = mother.upper()
    n = len(k)

    if mother == 'MORLET':  # Morlet wavelet
        if param == -1:
            param = 6.
        k0 = param
        expnt = -((scale * k - k0) ** 2) / 2.0 * (k > 0)
        norm = np.sqrt(scale * k[1]) * (np.pi ** -0.25) * np.sqrt(n)
        daughter = norm * np.exp(expnt)
        daughter *= (k > 0)  # Heaviside step function
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))
        coi = fourier_factor / np.sqrt(2)
        dofmin = 2

    elif mother == 'PAUL':  # Paul wavelet
        if param == -1:
            param = 4.
        m = param
        expnt = -(scale * k) * (k > 0)
        norm = np.sqrt(scale * k[1]) * (2 ** m / np.sqrt(m * np.prod(np.arange(2, 2 * m)))) * np.sqrt(n)
        daughter = norm * ((scale * k) ** m) * np.exp(expnt)
        daughter *= (k > 0)  # Heaviside step function
        fourier_factor = 4 * np.pi / (2 * m + 1)
        coi = fourier_factor * np.sqrt(2)
        dofmin = 2

    elif mother == 'DOG':  # DOG wavelet
        if param == -1:
            param = 2.
        m = param
        expnt = -((scale * k) ** 2) / 2.0
        norm = np.sqrt(scale * k[1] / gamma(m + 0.5)) * np.sqrt(n)
        daughter = -norm * (1j ** m) * ((scale * k) ** m) * np.exp(expnt)
        fourier_factor = 2 * np.pi * np.sqrt(2.0 / (2 * m + 1))
        coi = fourier_factor / np.sqrt(2)
        dofmin = 1

    else:
        raise ValueError("Mother must be one of 'MORLET', 'PAUL', 'DOG'")

    return daughter, fourier_factor, coi, dofmin
