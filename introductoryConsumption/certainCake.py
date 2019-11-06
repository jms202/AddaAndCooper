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

    def __init__(self, gamma=1.5, beta=0.97, R=1.025, y=5.0, WMax=100.0, Wngp=101, approach='euler', uType='smooth', interpTranform='normal', interpMethod='linear'):

        self.gamma = gamma
        self.beta = beta
        self.R = R
        self.y = y
        self.WMax = WMax
        self.approach = approach                        # approach should be one of 'value' or 'euler'
        self.uType = uType                              # uType should be one of 'smooth' or 'kink'
        self.interpTranform = interpTranform            # interpTranform should be one of 'normal' or 'inverse'
        self.interpMethod = interpMethod                # interpMethod should be one of 'linear' or 'cubic'
        self.cKink = 0.000001
        self.tol = 0.0001
        self.cMin = 0.000001
        self.Wngp = Wngp
        self.Wgrid = np.linspace(self.WMin, self.WMax, self.Wngp)
        self.V = np.empty(self.Wngp)
        self.margUtil = np.empty(self.Wngp)
        self.policy = np.empty(self.Wngp)


    # Getter function for WMin
    @property
    def WMin(self):
        # W = c - y + (c - y)/R + (c - y)/(R**2) + ...
        return max(0.0, self.R*(self.cMin - self.y)/(self.R - 1.0))


    # Getter function for uType
    @property
    def uType(self):
        return self._uType
        
    # Setter function for uType
    @uType.setter
    def uType(self, value):
        if (value == 'smooth'):
            self.u = self.CRRA
            self.uPrime = self.CRRAPrime
            self.uPrimeInv = self.CRRAPrimeInv
        elif (value == 'kink'):
            self.u = self.CRRAKink
            self.uPrime = self.CRRAKinkPrime
            self.uPrimeInv = None                       # Can't interpolate inverse if 'kink'
        else:
            raise ValueError("Unknown uType")
        self._uType = value


    # Getter function for interpFillHow
    @property
    def interpFillHow(self):
        if (self.interpMethod == 'linear'):
            return 'extrapolate'
        else:
            return (self.V[0], self.V[-1])


    # CRRA utility
    def CRRA(self, c):
        return (c**(1.0 - self.gamma)) / (1.0 - self.gamma)


    # CRRA marginal utility
    def CRRAPrime(self, c):
        return (c**(-self.gamma))


    # CRRA inverse marginal utility
    def CRRAPrimeInv(self, u):
        return (u**(-1.0/self.gamma))


    # CRRA utility with kink to ensure utility function is defined for c=0. Not sure how much this is going to affect the solution
    def CRRAKink(self, c):
        if (c < self.cKink):
            return (c * (self.cKink**(-self.gamma))) + (self.gamma * (self.cKink**(1.0 - self.gamma))) / (1.0 - self.gamma)
        else:
            return (c**(1.0 - self.gamma)) / (1.0 - self.gamma)


    # CRRA marginal utility with kink to ensure marginal utility is defined for c=0
    def CRRAKinkPrime(self, c):
        if (c < self.cKink):
            return (self.cKink**(-self.gamma))
        else:
            return c**(-self.gamma)


    # Objective function
    def negObjFunc(self, W1, W):

        Vinterp = interpolate.interp1d(self.Wgrid, self.V, kind=self.interpMethod, bounds_error=False, fill_value=self.interpFillHow)
        V1 = Vinterp(W1)
        
        c = W + self.y - (W1/self.R)
        return -(self.u(c) + (self.beta*V1))


    # Euler equation difference
    def eulerDiff(self, W1, W):

        if (self.interpTranform == 'inverse'):
            margUtilInv = np.empty(self.Wngp)
            for ixW in range(self.Wngp):
                margUtilInv[ixW] = self.uPrimeInv(self.margUtil[ixW])
            margUtilInterp = interpolate.interp1d(self.Wgrid, margUtilInv, kind=self.interpMethod, bounds_error=False, fill_value=self.interpFillHow)
            margUtil1Inv = margUtilInterp(W1)
            margUtil1 = self.uPrime(margUtil1Inv)
            
        else:
            margUtilInterp = interpolate.interp1d(self.Wgrid, self.margUtil, kind=self.interpMethod, bounds_error=False, fill_value=self.interpFillHow)
            margUtil1 = margUtilInterp(W1)

        c = W + self.y - (W1/self.R)
        return self.uPrime(c) - (self.beta*self.R*margUtil1)


    # Initial guess for V and margUtil
    def guessSln(self):
        for ixW, W in enumerate(self.Wgrid):
            self.V[ixW] = self.u(W + self.y)
            self.margUtil[ixW] = self.uPrime(W + self.y)


    # Find optimal W1 given W
    def findOptW1(self, W1min, W1max, W):

        if (self.approach == 'value'):
            # Find optimal W1 and VNext(W1) by minimising negObjFunc()
            res = minimize_scalar(self.negObjFunc, bounds=(W1min, W1max), method='bounded', args=(W,))
            if (not res.success):
                raise ArithmeticError(res.message)
            W1 = res.x

        elif (self.approach == 'euler'):
            # Find optimal W1 and VNext(W1) by solving Euler equation
            if (W1min < W1max):
                if (self.eulerDiff(W1min, W) > 0.0):
                    W1 = W1min
                elif (self.eulerDiff(W1max, W) < 0.0):
                    W1 = W1max
                else:
                    W1, res = brentq(self.eulerDiff, W1min, W1max, args=(W,), full_output=True)
                    if (not res.converged):
                        raise ArithmeticError("Root-finding did not converge")
            else:
                W1 = W1min

        else:
            raise ValueError("Unknown approach")

        return W1
    

    # Find next iteration
    def nextIter(self, VNext, margUtilNext):
    
        for ixW, W in enumerate(self.Wgrid):

            W1max = self.R*(W + self.y - self.cMin)
            W1min = self.WMin

            W1 = self.findOptW1(W1min, W1max, W)

            VNext[ixW] = -self.negObjFunc(W1, W)
            c = W + self.y - (W1/self.R)
            margUtilNext[ixW] = self.uPrime(c)
            
    

    # Generate sequence of value function iterations
    def genSeq(self, n, guess=True):
    
        if (guess):
            self.guessSln()
    
        VNext = np.empty(self.Wngp)
        margUtilNext = np.empty(self.Wngp)
        seq = self.V
        for i in range(n):
            self.nextIter(VNext, margUtilNext)
            seq = np.vstack([seq, VNext])
            np.copyto(self.V, VNext)
            np.copyto(self.margUtil, margUtilNext)
        return seq        
    

    # Solution
    def solve(self, iter=1000):

        self.guessSln()

        VNext = np.empty(self.Wngp)
        margUtilNext = np.empty(self.Wngp)

        for ixiter in range(iter):

            self.nextIter(VNext, margUtilNext)

            # Check convergence of value function (we could also check convergence of marginal utility)
            maxpos = np.argmax(abs(self.V - VNext))
            dist = abs(self.V[maxpos] - VNext[maxpos])
            print("%d dist %0.6f at position %d (%0.6f vs %0.6f)" % (ixiter, dist, maxpos, self.V[maxpos], VNext[maxpos]))
            if dist < self.tol:
                print("Breaking on ixiter %d" % ixiter)
                break

            # Can't just assign because they are pointers
            np.copyto(self.V, VNext)
            np.copyto(self.margUtil, margUtilNext)

        print("ixiter: %d" % ixiter)
        print(self.V[:4])
        print(VNext[:4])
        print(self.V[-4:])
        print(VNext[-4:])


    # Find policy function using solution
    def findPolicy(self):

        for ixW, W in enumerate(self.Wgrid):

            W1max = self.R*(W + self.y - self.cMin)
            W1min = self.WMin

            W1 = self.findOptW1(W1min, W1max, W)
            self.policy[ixW] = W + self.y - (W1/self.R)


    # Plot value function
    def plotValue(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.Wgrid, self.V)
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()

    # Plot sequence of value functions    
    def plotValueSeq(self, seq):
        fig, ax = plt.subplots()
        ax.plot(self.Wgrid, seq.T)
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function iteration')
        plt.show()

    # Plot policy function
    def plotPolicy(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.Wgrid, self.policy)
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('Optimal c', fontsize=14)
        ax.set_title('Policy function')
        plt.show()



