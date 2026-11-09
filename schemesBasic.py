###############################################################################
# Authors: Hilary Weller <h.weller@reading.ac.uk>
# Adapted by: Daniel Ward <dtw26@ic.ac.uk>
# Description: Standard finite-difference schemes for the advection equation
###############################################################################

import numpy as np

def FTBS(phiLevels, c):
    '''
    Author: Hilary Weller <h.weller@reading.ac.uk>
    Solves the linear advection equation for one time step using FTBS.
    phiLevels is updated. Nothing is returned
    Parameters:
        phiLevels ([1darray]): A list length one containing a 1darray which is
                         the latest value of phi, to be updated.
        c (float): The Courant number
    Returns:
        None
    '''
    # Keep the old level and write the next solution into a separate array.
    phiLevels.append(phiLevels[-1].copy())
    # Labels for phi at time levels n and n+1
    phiOld = phiLevels[0]
    phiNew = phiLevels[1]
    
    # Use a backward difference in space, wrapping indices periodically.
    nx = len(phiOld)
    for i in range(nx):
        phiNew[i] = phiOld[i] - c*(phiOld[i] - phiOld[(i-1)%nx])
    
    # Retain only the newly computed time level.
    phiLevels.pop(0)
    return None

def FTCS(phiLevels, c):
    '''
    Author: Hilary Weller <h.weller@reading.ac.uk>
    Solves the linear advection equation for one time step using FTCS.
    phiLevels is updated. Nothing is returned
    Parameters:
        phiLevels ([1darray]): A list length one containing a 1darray which is
                         the latest value of phi, to be updated.
        c (float): The Courant number
    Returns:
        None
    '''
    # Keep the old level and write the next solution into a separate array.
    phiLevels.append(phiLevels[-1].copy())
    # Labels for phi at time levels n and n+1
    phiOld = phiLevels[0]
    phiNew = phiLevels[1]
    
    # Use a centred difference in space, wrapping indices periodically.
    nx = len(phiOld)
    for i in range(nx):
        phiNew[i] = phiOld[i] - 0.5*c*(phiOld[(i+1)%nx] - phiOld[(i-1)%nx])
    
    # Retain only the newly computed time level.
    phiLevels.pop(0)
    return None


def CTCS(phiLevels, c):
    '''
    Author: Hilary Weller <h.weller@reading.ac.uk>
    Solves the linear advection equation for one time step using CTCS.
    phiLevels is updated. Nothing is returned.
    If phiLevels contains just one array, then FTCS is used instead.
    Parameters:
        phiLevels ([1darray,1darray]): A list of two consecutive values of phi.
                   If only one is present, the next is created using FTCS.
        c (float): The Courant number
    Returns:
        None
    '''
    # Bootstrap the two-level method with FTCS when no previous level exists.
    if len(phiLevels) == 1:
        phiOld = phiLevels[0].copy()
        FTCS(phiLevels, c)
        phiLevels.insert(0, phiOld)
        return None
    
    # Prepare a new level while preserving the previous and current values.
    phiLevels.append(phiLevels[0].copy())

    # Labels for phi at time levels n-1, n and n+1
    phiOld = phiLevels[0]
    phiMid = phiLevels[1]
    phiNew = phiLevels[2]
    
    # Use the centred spatial difference at the current time level.
    nx = len(phiOld)
    for i in range(nx):
        phiNew[i] = phiOld[i] - c*(phiMid[(i+1)%nx] - phiMid[(i-1)%nx])
    
    # Discard the oldest level so the list keeps only the two latest levels.
    phiLevels.pop(0)
    return None
