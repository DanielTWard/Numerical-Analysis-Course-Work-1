###############################################################################
# Date Revised: 19 Sept 2025
# Author: Hilary Weller <h.weller@reading.ac.uk>
# Adapted by: Daniel Ward <dtw26@ic.ac.uk>
# Description: Create a graph of errors growing over time for advection schemes
###############################################################################
import matplotlib.pyplot as plt
import numpy as np
import os
from advectionHelpers import cosBell_0_05, l2error
from schemesBasic import FTBS, CTCS, FTCS

# First make sure that the plots folder exists
if not os.path.exists("plots"):
    os.makedirs("plots")

def growingErrors(init, schemes, outFile,
                  nx = 40, nt = 10, u = 1, endTime = 0.4, 
                  xLim = [0,1], yLim = None):
    '''
    Author: Hilary Weller <h.weller@reading.ac.uk>
    Creates a graph of error growth over time for various advection schemes
    Parameters:
        nx (int): The number of points in space
        init (callable): Function to set the initial conditions.
        schemes (list of callable): List of functions to evaluate the schemes.
        nt (int): The number of time steps.
        u (float): The wind speed
        endTime (float) : The final time
        outFile (string): The name of the output file.
        xLim (tuple): The maximum and minimun spacial dimensions
        yLim (tuple): The minimum (yLim[0]) and maximum (yLim[1]) y values of
                      the graph. If None is specified then default limits
                      are used.
    Returns:
        None
    '''
    # Derived parameters
    dx = (xLim[1] - xLim[0])/nx
    x = np.arange(xLim[0], xLim[1], dx)
    dt = endTime/nt
    c = nx*u*endTime/(nt*(xLim[1] - xLim[0])) # Courant number, dt*u/dx
    
    # Set the initial conditions for all schemes
    phi0 = init(x)
    
    # Store an array of all times
    times = np.linspace(0, endTime, nt+1)
    
    # Colours for each line
    cols = ['blue', 'red', 'green', 'cyan', 'magenta', 'brown', 'purple',
            'pink', 'orange', 'gray', 'orange']
    
    # Loop over all the schemes and plot
    for i in range(len(schemes)):
        # The l2 error for each time step
        l2errors = np.zeros(nt+1)
        # The solution for this scheme
        phiLevels = [phi0.copy()]
        # Loop through all time steps
        for n in range(nt):
            # Update phi
            schemes[i](phiLevels, c)
            # Translate the initial profile and wrap it at the periodic boundary.
            phiA = init((x - u*times[n+1])%(xLim[1] - xLim[0]))
            # Compare the numerical solution with the exact translated profile.
            l2errors[n+1] = l2error(phiLevels[-1], phiA)

        # Plot the error for this scheme
        plt.semilogy(times, l2errors, cols[i], label = schemes[i].__name__)
    
    plt.xlim([0, endTime])
    plt.ylim(yLim)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel(r'$\ell_2$ error')
    plt.savefig("plots/"+outFile)
    plt.show()
    plt.clf()

