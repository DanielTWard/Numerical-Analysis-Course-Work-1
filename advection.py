###############################################################################
# Date Revised: 17 Sept 2025
# Author: Hilary Weller <h.weller@reading.ac.uk>
# Adapted by: Daniel Ward <dtw26@ic.ac.uk>
# Description: Basic tests of advection schemes
###############################################################################
import matplotlib.pyplot as plt
import numpy as np
import os
from advectionHelpers import cosBell_0_05, square_0_05
from schemesBasic import FTBS, CTCS, FTCS
from myScheme import myScheme
from myTests import error, fourierTrans, l2error, Serror, dSerror, courantError
if not os.path.exists("plots"):
    os.makedirs("plots")

def compareSchemeSnapshots(init, schemes, outFile,
                           nx = 40, nt = 10, u = 1, endTime = 0.4,
                           xLim = [0,1], yLim = [None,None]):
    '''
    Author: Hilary Weller <h.weller@reading.ac.uk>
    Compares snapshots of results of advection schemes using periodic boundaries
    Parameters:
        nx (int): The number of points in space
        init (callable): Function to set the initial conditions.
        schemes (list of callable): List of functions to evaluate the schemes.
        nt (int): The number of time steps.
        u (float): The wind speed
        endTime (float) : The final time
        outFile (string): The name of the output file.
        xLim (tuple): The minimum (xLim[0]) and maximim (xLim[1]) x-values of
                      the domain. xLim[1] is not part of the domain when periodic
        yLim (tuple): The minimum (yLim[0]) and maximum (yLim[1]) y values of
                      the graph. If [None,None] is specified then default limits
                      are used.
    Returns:
        None
    '''
    # Derived parameters
    dx = (xLim[1] - xLim[0])/nx
    x = np.arange(xLim[0], xLim[1], dx)
    dt = endTime/nt
    c = nx*u*endTime/(nt*(xLim[1] - xLim[0])) # Courant number, dt*u/dx
    
    # Set the initial conditions
    phi0 = init(x)
    
    # Colours for each line
    cols = ['blue', 'red', 'green', 'cyan', 'magenta', 'brown', 'purple',
            'pink', 'orange', 'gray', 'orange']
    
    # Find and plot the analytic solution
    phiA = init((x - u*endTime)%(xLim[1] - xLim[0]))
    plt.plot(x, phiA, 'k', label = "Analytic", linewidth=2)
    
    # Calculate the results of all of the schemes and plot
    for i in range(len(schemes)):
        # The solution, phi, is stored as a list of arrays at different time
        # steps, phiLevels. Initially just the initial conditions. The latest
        # time level array is always phiLevels[-1]
        phiLevels = [phi0.copy()]
        for n in range(nt):
            schemes[i](phiLevels, c)
        plt.plot(x, phiLevels[-1], cols[i], label = schemes[i].__name__)
    
    plt.xlim(xLim)
    plt.ylim(yLim)
    plt.xlabel('x')
    plt.ylabel(r'$\phi$')
    plt.legend()
    plt.savefig("plots/"+outFile)
    plt.clf()


initConds = [cosBell_0_05, square_0_05]
initCondsNames = ["CosBell","Square"]

for i in range(len(initConds)):
    # List of functions to do the advection
    schemes = [FTBS, CTCS, FTCS]

    compareSchemeSnapshots(initConds[i], schemes, f"basicSchemes{initCondsNames[i]}.pdf",
                        nx = 40, nt = 20, u = 1, endTime = 0.2, xLim = [0,1])


    schemes = [FTBS, CTCS, myScheme]

    compareSchemeSnapshots(initConds[i], schemes, f"allSchemes{initCondsNames[i]}.pdf",
                       nx = 40, nt = 20, u = 1, endTime = 0.2, xLim = [0,1])

    errorTypes = [l2error, Serror, dSerror]
    errorFiles = ["Errors", "S", "DiffS"]
    errorTypesNames = [r"$l_2$ error", r"$|S_n-S_0|$", r"$|dS_n-dS_0|$"]

    for j in range(len(errorTypes)):


        error(initConds[i], schemes, f"allSchemes{initCondsNames[i]}{errorFiles[j]}.pdf", errorTypes[j], errorTypesNames[j],
                        nx = 40, nt = 2000, u = 1, endTime = 20, xLim = [0,1])
    
    fourierTrans(initConds[i], schemes, f"allSchemes{initCondsNames[i]}FT.pdf",
                       nx = 40, nt = 20, u = 1, endTime = 0.2, xLim = [0,1])

    incs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    dxs = [[0.01], incs]
    dts = [incs, [0.01]]
    names = ["dx", "dt"]

    for j in range(len(dxs)):
        courantError(initConds[i], schemes, f"allSchemes{initCondsNames[i]}CourantErrorFixed{names[j]}.pdf", dxs = dxs[j], dts = dts[j])

    courantError(initConds[i], schemes, f"allSchemes{initCondsNames[i]}SmallC.pdf", dxs = [0.01], dts = np.arange(0.0005,0.01,0.0005))