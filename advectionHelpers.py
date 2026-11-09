###############################################################################
# Authors: Hilary Weller <h.weller@reading.ac.uk>
# Maintained by: Daniel Ward <dtw26@ic.ac.uk>
# Description: Initial conditions and error measures for advection tests
###############################################################################

import numpy as np

def cosBell(x, a, b):
    '''
    Returns a cosine bell shape between parameters a and b
    Author: Hilary Weller <h.weller@reading.ac.uk>
    Parameters:
        x (1darray): The x-locations for calculating the cosine bell.
        a (float): The start of the non-zero bell
        b (float): The end of the non-zero bell
    Returns:
        phi : 1darray. A cosine bell shape between x=a and x=b
    '''
    # Apply the smooth bell only inside the open interval (a, b).
    return np.where((x > a) & (x < b), 
                    0.5*(1 - np.cos(2*np.pi*(x - a)/(b-a))), 0.)

def square(x, a, b):
    '''
    Returns a square pulse between parameters a and b.
    Author: Hilary Weller <h.weller@reading.ac.uk>
    Parameters:
        x (1darray): The x-locations for calculating the pulse.
        a (float): The start of the non-zero pulse.
        b (float): The end of the non-zero pulse.
    Returns:
        phi (1darray): One inside (a, b) and zero elsewhere.
    '''
    return np.where((x > a) & (x < b), 
                    1., 0.)

def cosBell_0_05(x):
    "cosBell initial conditions with a=0, x=0.5"
    return cosBell(x, 0, 0.5)

def square_0_05(x):
    "Square-pulse initial conditions with a=0 and b=0.5."
    return square(x, 0, 0.5)
