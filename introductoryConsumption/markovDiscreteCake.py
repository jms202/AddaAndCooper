# Script to solve a simple discrete consumption-savings problem with 
# Markovian uncertainty

# Copyright (c) 2017, 2018 Jonathan Shaw

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize_scalar, minimize

def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


class Model:

    def __init__(self, gamma=0.5, beta=0.97, R=1.025, rho=0.95, p=[[0.6, 0.4], [0.4, 0.6]], eps=[0.9, 1.1], WMax=100.0, Wngp=101):

        self.gamma = gamma
        self.beta = beta
        self.R = R
        self.rho = rho
        self.p = np.array(p)
        self.eps = np.array(eps)
        self.epsngp = len(eps)
        self.Wngp = Wngp
        self.WMax = WMax
        self.WMin = self.WMax**(self.Wngp-1)
        self.tol = 0.0001
        self.Wgrid = np.empty(self.Wngp)
        W = self.WMax
        for ixW in range(self.Wngp):
            self.Wgrid[-(ixW+1)] = W
            W *= self.rho
        self.EV = np.empty((self.epsngp, self.Wngp))

    # CRRA utility
    def CRRA(self, c):
        return (c**(1.0 - self.gamma)) / (1.0 - self.gamma)


    # Calculate the value of waiting until next period
    # Complicated because W depreciates
    def calcEVW(self):
    
        EVinterp = interpolate.interp1d(self.Wgrid, self.EV, kind='linear', bounds_error=False, fill_value='extrapolate')
        EVBottom = EVinterp(self.rho*self.Wgrid[0])
        rhoEV = np.column_stack((EVBottom, self.EV[:,:-1]))
        return self.beta * (self.p @ rhoEV)


    # Solution
    def solve(self, iter=1000, verbose=False):

        # Value of eating cake today (static)
        VE = np.outer(self.eps, self.CRRA(self.Wgrid))

        # Initial guess: optimal to eat cake today
        np.copyto(self.EV, VE)

        for ixiter in range(iter):

            EVW = self.calcEVW()
            EVNext = np.maximum(VE, EVW)

            # Check convergence of value function (we could also check convergence of marginal utility)
            maxpos = np.unravel_index(np.argmax(abs(self.EV - EVNext)), self.EV.shape)
            dist = abs(self.EV[maxpos] - EVNext[maxpos])
            if verbose:
                print("%d dist %0.6f at position %s (%0.6f vs %0.6f)" % (ixiter, dist, maxpos, self.EV[maxpos], EVNext[maxpos]))
            if dist < self.tol:
                if verbose:
                    print("Breaking on ixiter %d" % ixiter)
                break

            # Can't just assign because they are pointers
            np.copyto(self.EV, EVNext)



    # Plot value function
    def plotValue(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.Wgrid, self.EV.T)
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()


    # Interpolate EVW
    # eps is a real scalar
    # ixW is an integer index
    # EVW is a 2D array (epsngp x Wngp)
    def interpolateEVW(self, eps, ixW, EVW):
        EVWinterp = interpolate.interp1d(self.eps, EVW[:, ixW], kind='linear', bounds_error=False, fill_value='extrapolate')
        return EVWinterp(eps)
        
    

    # Simulate the consumption choices of a number of individuals
    def simulate(self, n=1000, dontDraw=False):

        # Calculate u
        u = self.CRRA(self.Wgrid)
        # Calculate beta*EVW
        EVW = self.calcEVW()

        # Draw sequence of random values for epsilon
        # The transpose ensures that simulated decisions remain the same when the number of simulations are increased 
        if not(dontDraw):
            self.eps_sims = np.random.uniform(0.5, 1.5, size=(n, self.Wngp)).T

        # Create simulated decisions array
        d_all_sims = np.empty(shape=(self.Wngp, n), dtype=np.int32)

        # Loop across periods (which is effectively looping down the asset grid)
        for ixW in range(self.Wngp-1, -1, -1):
            # Work out whether to consume in each period
            d_all_sims[ixW, :] = np.where(self.eps_sims[ixW, :] * u[ixW] > self.interpolateEVW(self.eps_sims[ixW, :], ixW, EVW), 1, 0)

        self.d_sims = d_all_sims.argmax(axis=0)


    # Calculate moments on simulated data
    def calcMoments(self):
        self.moments = np.array([self.d_sims.mean(), self.d_sims.var()])


    # Calculate objective function for given value of
    # For simplicity we will assume rho is the only unknown parameter
    def objective(self, theta, n, momentsData):

        # Copy theta into model
        self.rho = theta[0]

        # Solve model (populates self.EV, dimension epsngp x Wngp)
        self.solve()

        # Simulate model and calculate moments
        self.simulate(n=n, dontDraw=True)
        self.calcMoments()
        print(theta, self.moments)
        
        # Calculate objective (ignores weighting matrix)
        return ((momentsData - self.moments)**2).sum()


    # Use simulated method of moments to estimate the parameters
    def estimate(self, guess, momentsData):

        n = 1000

        # Draw shocks that will be used throughout the estimation
        self.eps_sims = np.random.uniform(0.5, 1.5, size=(n, self.Wngp)).T

        # Find estimate of rho by minimising self.objective()
        #res = minimize_scalar(self.objective, bounds=(0.5, 0.99), method='bounded', args=(n, momentsData))
        res = minimize(self.objective, x0=np.array(guess), args=(n, momentsData), method='SLSQP', bounds=((0.5, 0.99),), options={'eps': 1e-2, 'maxiter': 500})

        if (not res.success):
            raise ArithmeticError(res.message)
        # print(res)

        return res.x

