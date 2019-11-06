# Script to solve a simple discrete consumption-savings problem with 
# IID uncertainty

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
import scipy as sp
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import uniform
from scipy.optimize import minimize_scalar, minimize
import numdifftools as ndt



def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


class Model:

    def __init__(self, gamma=0.5, beta=0.97, R=1.025, rho=0.95, p=[0.6, 0.4], eps=[0.5, 1.5], WMax=100.0, Wngp=101, seed=121):

        self.gamma = gamma
        self.beta = beta
        self.R = R
        self.p = np.array(p)
        self.eps = np.array(eps)
        self.epsngp = len(eps)
        self.Wngp = Wngp
        self.WMax = WMax
        self.WMin = self.WMax**(self.Wngp-1)
        self.tol = 0.0001
        self.Wgrid = np.empty(self.Wngp)
        self.rho = rho  # Moved here so WMax, Wngp and Wgrid are already set when property is called
        self.EV = np.empty((self.epsngp, self.Wngp))
        np.random.seed(seed)


    # Getter function for rho
    @property
    def rho(self):
        return self._rho


    # Setter function for rho
    @rho.setter
    def rho(self, value):
        W = self.WMax
        for ixW in range(self.Wngp):
            self.Wgrid[-(ixW+1)] = W
            W *= value
        self._rho = value


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

        # Iterate on value function
        for ixiter in range(iter):

            # Calculate next iteration
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


    # Simulate the consumption choices of a number of individuals
    def simulate(self, n=1000):

        # Calculate u at every grid value
        u = self.CRRA(self.Wgrid)
        # Calculate EVW = beta*EV
        EVW = self.calcEVW()

        # Draw sequence of random values for epsilon
        # The transpose ensures that simulated decisions remain the same when the number of simulations are increased 
        self.eps_sims = np.random.uniform(0.5, 1.5, size=(n, self.Wngp)).T

        # Create empty array to store simulated decisions
        d_all_sims = np.empty(shape=(self.Wngp, n), dtype=np.int32)

        # Loop across periods (which is effectively looping down the asset grid)
        for ixW in range(self.Wngp-1, -1, -1):
            # Work out whether to consume in each period
            # Second argument is simulation number
            d_all_sims[ixW, :] = np.where(self.eps_sims[ixW, :] * u[ixW] > EVW[ixW], 1, 0)

        self.d_sims = d_all_sims.argmax(axis=0)


    # Negative of log likelihood function
    # For simplicity we will assume rho is the only unknown parameter
    def negLogLikelihood(self, theta, d_data):

        # print('Theta is', theta)

        # Copy theta into model
        self.rho = theta[0]

        # Solve model (populates self.EV, dimension epsngp x Wngp)
        self.solve()

        # Find the critical value of eps for each ixW
        u = self.CRRA(self.Wgrid)
        EVW = self.calcEVW()
        epsStar = EVW / u

        # Use this in combination with the data to construct the log likelihood
            # suppose d_data[0] = 2
            # Then we need
                # l_i = log(1 - F(epsStar[0])) + log(1 - F(epsStar[1])) + log(F(epsStar[2]))
        logl = np.zeros(len(d_data))
        for ix, d_i in enumerate(d_data):
            for dd_i in range(d_i):
                logl[ix] += uniform.logcdf(epsStar[dd_i], loc=0.5, scale=1.0)
            logl[ix] += uniform.logsf(epsStar[d_i], loc=0.5, scale=1.0)
        return -logl.sum()


    # Use maximum likelihood to estimate the parameters
    def estimate(self, guess, d_data, calcSE=True):

        # Find estimate of rho by minimising negLogLikelihood()
        res = minimize(self.negLogLikelihood, x0=np.array(guess), args=(d_data), method='SLSQP', bounds=((0.5, 0.99),))

        if (not res.success):
            raise ArithmeticError(res.message)
        # print(res)

        if not(calcSE):
            return res.x
            
        # What's the standard error?
        # step_max is required to prevent function calculations outside bounds
        Hfun = ndt.Hessian(self.negLogLikelihood, step=ndt.MaxStepGenerator(step_max=0.01), method='central', full_output=True)
        (hessian_ndt, diagnost) = Hfun(res.x, d_data)
        se = np.sqrt(np.diag(np.linalg.inv(hessian_ndt)))

        return (res.x, se)


    # Simulate the standard error
    def bootstrapStdErr(self, truth, guess, reps=10):

        estimates = np.empty(reps)

        num = 1000
        d_data = np.empty(num, dtype=np.int32)

        for i in range(reps):

            # Copy theta into model
            self.rho = truth[0]

            # Simulate some data
            self.solve()
            self.simulate(n=num)
            np.copyto(d_data, self.d_sims)

            estimates[i] = self.estimate([0.9], d_data, calcSE=False)
            
            print(".", end="", flush=True)
        
        print("")

        return estimates.std()



