# Next step: minimise_scalar is complaining that the bounds are not scalars - look for a different minimisation routine or loop across values of W


# Script to solve a simple continuous non-stochastic consumption-savings problem

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



import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize_scalar
from scipy.optimize import brentq
import sys

def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

class Model:

    def __init__(self, gamma=1.5, beta=0.97, R=1.025, y=5.0, WMax=100.0, order=8, interpMethod='linear'):

        self.gamma = gamma
        self.beta = beta
        self.R = R
        self.y = y
        self.WMax = WMax
        self.interpMethod = interpMethod                # interpMethod should be one of 'linear' or 'cubic'
        self.tol = 0.0001
        self.cMin = 0.000001
        self.order = order


    # Getter function for WMin
    @property
    def WMin(self):
        # W = c - y + (c - y)/R + (c - y)/(R**2) + ...
        return max(0.0, self.R*(self.cMin - self.y)/(self.R - 1.0))


    # Getter function for interpFillHow
    @property
    def interpFillHow(self):
        if (self.interpMethod == 'linear'):
            return 'extrapolate'
        else:
            return (self.V[0], self.V[-1])


    # CRRA utility
    def u(self, c):
        return (c**(1.0 - self.gamma)) / (1.0 - self.gamma)


    # Objective function
    def negObjFunc(self, W1, W):

        c = W + self.y - (W1/self.R)
        return -(self.u(c) + (self.beta*self.V(W1)))


    # Find optimal W1 given W
    def findOptW1(self, W):

        W1Max = self.R*(W + self.y - self.cMin)
        W1Min = self.WMin

        W1 = np.empty(len(W))

        for ixW, WVal in enumerate(W):

            # Find optimal W1 and VNext(W1) by minimising negObjFunc()
            res = minimize_scalar(self.negObjFunc, bounds=(W1Min, W1Max[ixW]), method='bounded', args=(WVal,))
            if (not res.success):
                raise ArithmeticError(res.message)
            W1[ixW] = res.x

        return W1
    

    # Value function
    def negValue(self, W):
        return self.negObjFunc(self.findOptW1(W), W)



    # Initial guess for V
    def guessSln(self):
        self.V = np.polynomial.chebyshev.Chebyshev.interpolate(lambda x : self.u(x + self.y), deg=self.order, domain=[self.WMin, self.WMax])



    # Find next iteration
    def nextIter(self):
        VNext = np.polynomial.chebyshev.Chebyshev.interpolate(lambda W : -self.negValue(W), deg=self.order, domain=[self.WMin, self.WMax])
        return VNext

            
    # Distance between two functions
    def calcDistance(self, f1, f2):
        domain = f1.domain
        xvals = np.linspace(domain[0], domain[1], num=1000)
        dist = f1(xvals) - f2(xvals)
        maxpos = np.argmax(abs(dist))
        dist = abs(dist[maxpos])
        return dist, maxpos


    # Find policy function using solution
    def findPolicy(self):
        self.policy = np.polynomial.chebyshev.Chebyshev.interpolate(lambda W : W + self.y - (self.findOptW1(W)/self.R), deg=self.order, domain=[self.WMin, self.WMax])


    # Solution
    def solve(self, iter=1000):

        self.guessSln()

        for ixiter in range(iter):

            VNext = self.nextIter()

            # Check convergence of value function
            dist, maxpos = self.calcDistance(self.V, VNext)
            print("%d dist %0.6f at position %d" % (ixiter, dist, maxpos))
            if dist < self.tol:
                print("Breaking on ixiter %d" % ixiter)
                break

            # Can't just assign because they are pointers
            self.V = VNext.copy()

        print("ixiter: %d" % ixiter)

        self.findPolicy()




    # Plot value function
    def plotValue(self):

        WVals = np.linspace(self.WMin, self.WMax, num=1000)
        fig, ax = plt.subplots()
        ax.plot(WVals, self.V(WVals))
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()


    # Plot policy function
    def plotPolicy(self):

        WVals = np.linspace(self.WMin, self.WMax, num=1000)
        fig, ax = plt.subplots()
        ax.plot(WVals, self.policy(WVals))
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('Optimal c', fontsize=14)
        ax.set_title('Policy function')
        plt.show()



