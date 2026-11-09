###############################################################################
# Date Revised: 19 Sept 2025
# Author: Hilary Weller <h.weller@reading.ac.uk>
# Adapted by: Daniel Ward <dtw26@ic.ac.uk>
# Description: Create a graph of errors growing over time for advection schemes
###############################################################################
from mimetypes import init

import matplotlib.pyplot as plt
import numpy as np
import os
from advectionHelpers import cosBell_0_05, square_0_05
from schemesBasic import FTBS, CTCS, FTCS
from myScheme import myScheme, innerProd, innerProd2

# First make sure that the plots folder exists
if not os.path.exists("plots"):
    os.makedirs("plots")

def initialise(init, nx, nt, u, endTime, xLim):
    '''
    Sets up the spatial grid, initial solution, and time levels.
    Author: Daniel Ward <dtw26@ic.ac.uk>
    Parameters:
        init (callable): Function that returns the initial solution at x.
        nx (int): The number of spatial points.
        nt (int): The number of time steps.
        u (float): The wind speed.
        endTime (float): The final time.
        xLim (tuple): The minimum and maximum spatial coordinates.
    Returns:
        tuple: The grid spacing, grid, time step, Courant number, initial
               solution, plot colours, and time levels.
    '''
    # Derived parameters
    dx = (xLim[1] - xLim[0])/nx
    x = np.arange(xLim[0], xLim[1], dx)
    dt = endTime/nt
    c = nx*u*endTime/(nt*(xLim[1] - xLim[0])) # Courant number, dt*u/dx
    
    # Set the initial conditions for all schemes
    phi0 = init(x)

    # Colours for each line
    cols = ['blue', 'red', 'green', 'cyan', 'magenta', 'brown', 'purple',
            'pink', 'orange', 'gray', 'orange']

    # Store an array of all times
    times = np.linspace(0, endTime, nt+1)
    
    return dx, x, dt, c, phi0, cols, times

def finalise(xLabel, yLabel, outFile, xLim = None, yLim = None):
    '''
    Labels, saves, and clears the current plot.
    Author: Daniel Ward <dtw26@ic.ac.uk>
    Parameters:
        xLabel (string): The label for the x-axis.
        yLabel (string): The label for the y-axis.
        outFile (string): The name of the output file in the plots directory.
        xLim (tuple): The optional minimum and maximum x-axis values.
        yLim (tuple): The optional minimum and maximum y-axis values.
    Returns:
        None
    '''
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xlim(xLim)
    plt.ylim(yLim)
    plt.legend()
    plt.savefig("plots/"+outFile)
    plt.clf()

def l2error(phi, init, x, u, t, x0, x1):
    '''
    Returns the root-mean-square difference from the analytic solution.
    Author: Daniel Ward <dtw26@ic.ac.uk>
    Parameters:
        phi (1darray): The numerical solution.
        init (callable): Function that returns the initial solution at x.
        x (1darray): The spatial grid.
        u (float): The wind speed.
        t (float): The time of the numerical solution.
        x0 (float): The minimum spatial coordinate.
        x1 (float): The maximum spatial coordinate.
    Returns
        float: The root-mean-square pointwise error.
    '''
    # Calculate the analytic solution
    phiA = init((x - u*t)%(x1 - x0))
    # Calculate the l2 error
    phiError = phi - phiA
    l2 = np.sqrt(np.sum(phiError**2)/len(phiError))
    return l2

def Serror(phi, init, x, u = None, t = None, x0 = None, x1 = None):
    '''
    Returns the absolute change in the solution's squared norm.
    Author: Daniel Ward <dtw26@ic.ac.uk>
    Parameters:
        phi (1darray): The numerical solution.
        init (callable): Function that returns the initial solution at x.
        x (1darray): The spatial grid.
        u (float): Unused; accepted for compatibility with other error functions.
        t (float): Unused; accepted for compatibility with other error functions.
        x0 (float): Unused; accepted for compatibility with other error functions.
        x1 (float): Unused; accepted for compatibility with other error functions.
    Returns:
        float: The absolute difference between initial and current squared norms.
    '''
    return abs(innerProd(init(x), init(x)) - innerProd(phi, phi))

def dSerror(phi, init, x, u = None, t = None, x0 = None, x1 = None):
    '''
    Returns the absolute change in the squared norm of the solution's first differences.
    Author: Daniel Ward <dtw26@ic.ac.uk>
    Parameters:
        phi (1darray): The numerical solution.
        init (callable): Function that returns the initial solution at x.
        x (1darray): The spatial grid.
        u (float): Unused; accepted for compatibility with other error functions.
        t (float): Unused; accepted for compatibility with other error functions.
        x0 (float): Unused; accepted for compatibility with other error functions.
        x1 (float): Unused; accepted for compatibility with other error functions.
    Returns:
        float: The absolute difference between the initial and current values.
    '''
    return abs(innerProd2(init(x), init(x)) - innerProd2(phi, phi))


def error(init, schemes, outFile, version, yLabel,
                  nx = 40, nt = 20, u = 1, endTime = 0.2, 
                  xLim = [0,1], yLim = None):
    '''
    Plots an error measure over time for each advection scheme.
    Author: Daniel Ward <dtw26@ic.ac.uk>
    Parameters:
        init (callable): Function that returns the initial solution at x.
        schemes (list of callable): Advection schemes to compare.
        outFile (string): The name of the output file in the plots directory.
        version (callable): Error measure called after each time step.
        yLabel (string): The label for the y-axis.
        nx (int): The number of spatial points.
        nt (int): The number of time steps.
        u (float): The wind speed.
        endTime (float): The final time.
        xLim (tuple): The minimum and maximum spatial coordinates.
        yLim (tuple): The optional minimum and maximum y-axis values.
    Returns:
        None
    '''

    dx, x, dt, c, phi0, cols, times = initialise(init, nx, nt, u, endTime, xLim)

    
    # Loop over all the schemes and plot
    for i in range(len(schemes)):
        # The l2 error for each time step
        err = np.zeros(nt+1)
        # The solution for this scheme
        phiLevels = [phi0.copy()]
        # Loop through all time steps
        for n in range(nt):
            # Update phi
            schemes[i](phiLevels, c)
            err[n+1] = version(phiLevels[-1], init, x, u, times[n+1], xLim[0], xLim[1])

        # Plot the error for this scheme
        if version == l2error:
            plt.plot(times, err, cols[i], label = schemes[i].__name__)
        else:
            plt.semilogy(times, err, cols[i], label = schemes[i].__name__)

    finalise("Time", yLabel, outFile, [0, endTime], yLim)

def fourierTrans(init, schemes, outFile,
                  nx = 40, nt = 20, u = 1, endTime = 0.2, 
                  xLim = [0,1]):
    '''
    Plots the Fourier amplitude of each scheme's solution at the final time.
    Author: Daniel Ward <dtw26@ic.ac.uk>
    Parameters:
        init (callable): Function that returns the initial solution at x.
        schemes (list of callable): Advection schemes to compare.
        outFile (string): The name of the output file in the plots directory.
        nx (int): The number of spatial points.
        nt (int): The number of time steps.
        u (float): The wind speed.
        endTime (float): The final time.
        xLim (tuple): The minimum and maximum spatial coordinates.
    Returns:
        None
    '''
 
    dx, x, dt, c, phi0, cols, times = initialise(init, nx, nt, u, endTime, xLim)
    
    # Loop over all the schemes and plot
    for i in range(len(schemes)):
        # The solution for this scheme
        phiLevels = [phi0.copy()]
        # Loop through all time steps
        for n in range(nt):
            # Update phi
            schemes[i](phiLevels, c)

        # Plot the magnitude of each spatial Fourier mode at the final time.
        finalTrans = np.abs(np.fft.fft(phiLevels[-1]))
        # Pair frequencies with amplitudes and order them for plotting.
        freqs = np.fft.fftfreq(len(finalTrans), d=dt)

        freqs, finalTrans = zip(*sorted(zip(freqs, finalTrans)))
        
        plt.semilogy(freqs, finalTrans, cols[i], label = schemes[i].__name__)

    finalise("Frequency", "Amplitude", outFile)


def courantError(init, schemes, outFile,
                  dxs = [0.1], dts = [0.1], u = 1, endTime = 1, 
                  xLim = [0,1]):
    '''
    Plots final-time solution error against Courant number.
    Author: Daniel Ward <dtw26@ic.ac.uk>
    Parameters:
        init (callable): Function that returns the initial solution at x.
        schemes (list of callable): Advection schemes to compare.
        outFile (string): The name of the output file in the plots directory.
        dxs (list of float): Spatial grid spacings to test.
        dts (list of float): Time-step sizes to test.
        u (float): The wind speed.
        endTime (float): The final time.
        xLim (tuple): The minimum and maximum spatial coordinates.
    Returns:
        None
    '''
    
    # Loop over all the schemes and plot
    for i in range(len(schemes)):
        cs = []
        errs = []
        # Test every requested combination of spatial and temporal resolution.
        for dx in dxs:
            for dt in dts:
                nx = int((xLim[1] - xLim[0])/dx)
                nt = int(endTime/dt)
                # Record the Courant number for this grid and time step.
                c = u*dt/dx
                cs.append(c)
                x = np.arange(xLim[0], xLim[1], dx)
                phi0 = init(x)
                cols = ['blue', 'red', 'green', 'cyan', 'magenta', 'brown', 'purple',
                        'pink', 'orange', 'gray', 'orange']
                times = np.linspace(0, endTime, nt+1)

                # The solution for this scheme
                phiLevels = [phi0.copy()]
                # Loop through all time steps
                for n in range(nt):
                    # Update phi
                    schemes[i](phiLevels, c)

                # Measure the error after evolving to the requested final time.
                errs.append(l2error(phiLevels[-1], init, x, u, endTime, xLim[0], xLim[1]))
                
        plt.semilogy(cs, errs, cols[i], marker='o', label = schemes[i].__name__)

    finalise("Courant Number", r"$l_2$ Error at $t = "+str(endTime)+"$", outFile)