###############################################################################
# Author: Daniel Ward <dtw26@ic.ac.uk>
# Description: Custom three-point scheme and periodic inner products
###############################################################################
import numpy as np
from scipy.optimize import fsolve

def trans(oldVec):
    '''
    Returns the vector shifted by one point with periodic boundaries.
    Author: Daniel Ward <dtw26@ic.ac.uk>
    Parameters:
        oldVec (1darray): The vector to shift.
    Returns:
        newVec (1darray): The shifted vector.
    '''
    newVec = oldVec.copy()
    nx = len(oldVec)
    for i in range(nx):
        # Wrap around so each point takes the value of its left neighbour.
        newVec[i] = oldVec[(i-1)%nx]
    return newVec

def innerProd(v1, v2):
    '''
    Returns the dot product of two vectors.
    Author: Daniel Ward <dtw26@ic.ac.uk>
    Parameters:
        v1 (1darray): The first vector.
        v2 (1darray): The second vector.
    Returns:
        float: The dot product of the vectors.
    '''
    return np.dot(v1, v2)

def innerProd2(v1, v2):
    '''
    Returns the dot product of the first differences of two periodic vectors.
    Author: Daniel Ward <dtw26@ic.ac.uk>
    Parameters:
        v1 (1darray): The first vector.
        v2 (1darray): The second vector.
    Returns:
        float: The dot product of the first differences.
    '''
    return np.dot(v1 - trans(v1), v2 - trans(v2))

def myScheme(phiLevels, c):
    '''
    Advances the advection solution by one time step using a three-point scheme.
    Author: Daniel Ward <dtw26@ic.ac.uk>
    Parameters:
        phiLevels (list of 1darrays): The current solution time levels. The
                                      latest level is updated in place.
        c (float): The Courant number.
    Returns:
        None
    '''
    
    # Keep the old solution while preparing a separate array for the update.
    phiLevels.append(phiLevels[-1].copy())
    # The scheme reads from the old level and writes to the new one.
    phiOld = phiLevels[0]
    phiNew = phiLevels[1]

    # Inner products used to impose conservation of the quadratic invariant.
    Sn = innerProd(phiOld,phiOld)
    S1n = innerProd(trans(phiOld),phiOld)
    S2n = innerProd(trans(trans(phiOld)),phiOld)

    def func(x):
        # Solve for consistency, the advection speed, and invariant conservation.
        alpha = x[0]
        beta = x[1]
        gamma = x[2]
        return [alpha + beta + gamma - 1, alpha - gamma - c, (alpha**2 + beta**2 + gamma**2 - 1)*Sn + 2*beta*(alpha + gamma)*S1n + 2*alpha*gamma*S2n]

    # Start the nonlinear solve from the centred-scheme coefficients.
    root = fsolve(func, [c/2, 1, -c/2])

    alpha = root[0]
    beta = root[1]
    gamma = root[2]

    
    # Apply the solved three-point stencil with periodic boundaries.
    nx = len(phiOld)
    for i in range(nx):
        phiNew[i] = alpha * phiOld[(i-1)%nx] + beta * phiOld[i] + gamma * phiOld[(i+1)%nx]

    
    # Discard the old level, leaving the updated solution in phiLevels.
    phiLevels.pop(0)
    return None
