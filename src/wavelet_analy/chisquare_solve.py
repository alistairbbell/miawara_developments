#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:24:40 2024

Author: Alistair Bell

Contact: alistair.bell@unibe.ch
"""

import scipy.special as sp

def chisquare_solve(XGUESS, P, V):
    """
    Internal function used by chisquare_inv.

    Given XGUESS, a percentile P, and degrees-of-freedom V,
    return the difference between calculated percentile and P.

    Parameters:
    XGUESS: The guessed value of X
    P: The target probability
    V: Degrees of freedom

    Returns:
    PDIFF: The absolute difference between the guessed probability
           and the target probability P.
    """

    # Calculate the guessed probability using the incomplete gamma function
    PGUESS = sp.gammainc(V / 2, V * XGUESS / 2)

    # Calculate the error in the guessed probability
    PDIFF = abs(PGUESS - P)

    # Tolerance for near-1 probability
    TOL = 1E-4
    if PGUESS >= 1 - TOL:  # if P is very close to 1 (i.e., a bad guess)
        PDIFF = XGUESS  # then just assign some large number like XGUESS

    return PDIFF
