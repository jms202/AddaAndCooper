# Script to solve a simple continuous consumption-savings problem with 
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
from scipy.optimize import minimize_scalar
from scipy.optimize import brentq

def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


class Model:

    def __init__(self, gamma=1.5, beta=0.97, R=1.025, p=[[0.75, 0.25], [0.25, 0.75]], y=[2.5, 7.5], WMax=100.0, Wngp=101, approach='euler', uType='smooth', interpTranform='normal', interpMethod='linear'):

        self.gamma = gamma
        self.beta = beta
        self.R = R
        self.p = np.array(p)
        self.y = np.array(y)
        self.Yngp = len(y)
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
        self.EV = np.empty((self.Yngp, self.Wngp))
        self.EmargUtil = np.empty((self.Yngp, self.Wngp))
        self.policy = np.empty((self.Yngp, self.Wngp))

    # Getter function for WMin
    @property
    def WMin(self):
        # W = c - y + (c - y)/R + (c - y)/(R**2) + ...
        return max(0.0, self.R*(self.cMin - self.y[0])/(self.R - 1.0))

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

    # How to interpolate
    def interpFillHow(self, ixY):
        if (self.interpMethod == 'linear'):
            return 'extrapolate'
        else:
            return (self.EV[ixY,0], self.EV[ixY,-1])

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
    def negObjFunc(self, W1, ixY, W):

        EVinterp = interpolate.interp1d(self.Wgrid, self.EV[ixY,:], kind=self.interpMethod, bounds_error=False, fill_value=self.interpFillHow(ixY))
        EV1 = EVinterp(W1)
        
        c = W + self.y[ixY] - (W1/self.R)
        return -(self.u(c) + (self.beta*EV1))


    # Euler equation difference
    def eulerDiff(self, W1, ixY, W):

        if (self.interpTranform == 'inverse'):
            EmargUtilInv = np.empty(self.Wngp)
            for ixW in range(self.Wngp):
                EmargUtilInv[ixW] = self.uPrimeInv(self.EmargUtil[ixY,ixW])
            EmargUtilInterp = interpolate.interp1d(self.Wgrid, EmargUtilInv, kind=self.interpMethod, bounds_error=False, fill_value=self.interpFillHow(ixY))
            EmargUtil1Inv = EmargUtilInterp(W1)
            EmargUtil1 = self.uPrime(EmargUtil1Inv)
            
        else:
            EmargUtilInterp = interpolate.interp1d(self.Wgrid, self.EmargUtil[ixY,:], kind=self.interpMethod, bounds_error=False, fill_value=self.interpFillHow(ixY))
            EmargUtil1 = EmargUtilInterp(W1)

        c = W + self.y[ixY] - (W1/self.R)
        return self.uPrime(c) - (self.beta*self.R*EmargUtil1)


    # Initial guess for V and margUtil
    def guessSln(self):

        V = np.empty((self.Yngp, self.Wngp))
        margUtil = np.empty((self.Yngp, self.Wngp))

        # Calculate value function assuming shock is known
        for ixY, y in enumerate(self.y):
            for ixW, W in enumerate(self.Wgrid):
                V[ixY,ixW] = self.u(W + y)
                margUtil[ixY,ixW] = self.uPrime(W + y)

        # Now integrate over uncertain shock
        self.EV = self.p @ V
        self.EmargUtil = self.p @ margUtil



    # Find optimal W1 given W
    def findOptW1(self, W1min, W1max, ixY, W):

        if (self.approach == 'value'):
            # Find optimal W1 and EVNext(W1) by minimising negObjFunc()
            res = minimize_scalar(self.negObjFunc, bounds=(W1min, W1max), method='bounded', args=(ixY,W))
            if (not res.success):
                raise ArithmeticError(res.message)
            W1 = res.x

        elif (self.approach == 'euler'):
            # Find optimal W1 and EVNext(W1) by solving Euler equation
            if (W1min < W1max):
                if (self.eulerDiff(W1min, ixY, W) > 0.0):
                    W1 = W1min
                elif (self.eulerDiff(W1max, ixY, W) < 0.0):
                    W1 = W1max
                else:
                    W1, res = brentq(self.eulerDiff, W1min, W1max, args=(ixY,W), full_output=True)
                    if (not res.converged):
                        raise ArithmeticError("Root-finding did not converge")
            else:
                W1 = W1min

        else:
            raise ValueError("Unknown approach")

        return W1


    # Find next iteration
    def nextIter(self, EVNext, EmargUtilNext):
    
        VNext = np.empty((self.Yngp, self.Wngp))
        margUtilNext = np.empty((self.Yngp, self.Wngp))
    
        # Calculate value function assuming shock is known
        for ixY, y in enumerate(self.y):
            for ixW, W in enumerate(self.Wgrid):

                W1max = self.R*(W + y - self.cMin)
                W1min = self.WMin

                W1 = self.findOptW1(W1min, W1max, ixY, W)

                VNext[ixY,ixW] = -self.negObjFunc(W1, ixY, W)
                c = W + y - (W1/self.R)
                margUtilNext[ixY,ixW] = self.uPrime(c)

        # Now integrate over uncertain shock
        #EVNext[ixY_1,ixW] = p[ixY_1,ixY] * VNext[ixY,ixW]
        np.copyto(EVNext, self.p @ VNext)
        np.copyto(EmargUtilNext, self.p @ margUtilNext)

#        set_trace()


    # Generate sequence of value function iterations
    def genSeq(self, n, guess=True):
    
        if (guess):
            self.guessSln()
    
        EVNext = np.empty((self.Yngp,self.Wngp))
        EmargUtilNext = np.empty((self.Yngp,self.Wngp))
        seq = self.EV
        for i in range(n):
            self.nextIter(EVNext, EmargUtilNext)
            seq = np.vstack([seq, EVNext])
            np.copyto(self.EV, EVNext)
            np.copyto(self.EmargUtil, EmargUtilNext)
        return seq.reshape(n+1, seq.shape[0]//(n+1), seq.shape[1])


    # Solution
    def solve(self, iter=1000):

        self.guessSln()

        EVNext = np.empty((self.Yngp, self.Wngp))
        EmargUtilNext = np.empty((self.Yngp, self.Wngp))

        for ixiter in range(iter):

            self.nextIter(EVNext, EmargUtilNext)

            # Check convergence of value function (we could also check convergence of marginal utility)
            maxpos = np.unravel_index(np.argmax(abs(self.EV - EVNext)), self.EV.shape)
            dist = abs(self.EV[maxpos] - EVNext[maxpos])
            print("%d dist %0.6f at position %s (%0.6f vs %0.6f)" % (ixiter, dist, maxpos, self.EV[maxpos], EVNext[maxpos]))
            if dist < self.tol:
                print("Breaking on ixiter %d" % ixiter)
                break

            # Can't just assign because they are pointers
            np.copyto(self.EV, EVNext)
            np.copyto(self.EmargUtil, EmargUtilNext)



    # Find policy function using solution
    def findPolicy(self):

        for ixY, y in enumerate(self.y):
            for ixW, W in enumerate(self.Wgrid):

                W1max = self.R*(W + y - self.cMin)
                W1min = self.WMin

                W1 = self.findOptW1(W1min, W1max, ixY, W)
                self.policy[ixY, ixW] = W + y - (W1/self.R)



    # Plot value function
    def plotValue(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.Wgrid, self.EV.T)
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()


    # Plot policy function
    def plotPolicy(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.Wgrid, self.policy.T)
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('Optimal c', fontsize=14)
        ax.set_title('Policy function')
        plt.show()




