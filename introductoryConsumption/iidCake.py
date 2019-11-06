# Cake-eating with stochastic IID income (two-point distribution)

# Script to solve a simple continuous consumption-savings problem with 
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



import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize_scalar
from scipy.optimize import brentq

class Model:

    def __init__(self, gamma=1.5, beta=0.97, R=1.025, p=[0.5, 0.5], y=[2.5, 7.5], XMax=100.0, Xngp=101, approach='euler', uType='smooth', interpTranform='normal', interpMethod='linear'):

        self.gamma = gamma
        self.beta = beta
        self.R = R
        self.p = np.array(p)
        self.y = np.array(y)
        self.XMax = XMax
        self.approach = approach                        # approach should be one of 'value' or 'euler'
        self.uType = uType                              # uType should be one of 'smooth' or 'kink'
        self.interpTranform = interpTranform            # interpTranform should be one of 'normal' or 'inverse'
        self.interpMethod = interpMethod                # interpMethod should be one of 'linear' or 'cubic'
        self.cKink = 0.000001
        self.tol = 0.0001
        self.cMin = 0.000001
        self.Xngp = Xngp
        self.Xgrid = np.linspace(self.XMin, self.XMax, self.Xngp)
        self.V = np.empty(self.Xngp)
        self.margUtil = np.empty(self.Xngp)
        self.policy = np.empty(self.Xngp)


    # Getter function for XMin
    @property
    def XMin(self):
        # X = c + (c - y)/R + (c - y)/(R**2) + ...
        return max(self.y[0], (self.R*self.cMin - self.y[0])/(self.R - 1.0))

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
    def negObjFunc(self, c, X):

        # Calculate EV[X']
        Vinterp = interpolate.interp1d(self.Xgrid, self.V, kind=self.interpMethod, bounds_error=False, fill_value=self.interpFillHow)
        EV1 = 0.0
        for p, y in zip(self.p, self.y):
        
            X1 = self.R*(X - c) + y
            V1 = Vinterp(X1)
            EV1 = EV1 + p*V1
        
        return -(self.u(c) + (self.beta*EV1))


    # Euler equation difference
    def eulerDiff(self, c, X):

        if (self.interpTranform == 'inverse'):
            margUtilInv = np.empty(self.Xngp)
            for ixX in range(self.Xngp):
                margUtilInv[ixX] = self.uPrimeInv(self.margUtil[ixX])
            margUtilInterp = interpolate.interp1d(self.Xgrid, margUtilInv, kind=self.interpMethod, bounds_error=False, fill_value=self.interpFillHow)
            EmargUtil1 = 0.0
            for p, y in zip(self.p, self.y):
                X1 = self.R*(X - c) + y
                margUtil1Inv = margUtilInterp(X1)
                EMargUtil1 = EMargUtil1 + p*self.uPrime(margUtil1Inv)
            
        else:
            # Calculate Eu'[X1]
            margUtilInterp = interpolate.interp1d(self.Xgrid, self.margUtil, kind=self.interpMethod, bounds_error=False, fill_value=self.interpFillHow)
            EMargUtil1 = 0.0
            for p, y in zip(self.p, self.y):
                X1 = self.R*(X - c) + y
                margUtil1 = margUtilInterp(X1)
                EMargUtil1 = EMargUtil1 + p*margUtil1

        return self.uPrime(c) - (self.beta*self.R*EMargUtil1)


    # Initial guess for V and margUtil
    def guessSln(self):
        for ixX, X in enumerate(self.Xgrid):
            self.V[ixX] = self.u(X)
            self.margUtil[ixX] = self.uPrime(X)


    # Find optimal c given X
    def findOptC(self, cMin, cMax, X):

        if (self.approach == 'value'):
            # Find optimal c by minimising negObjFunc()
            res = minimize_scalar(self.negObjFunc, bounds=(cMin, cMax), method='bounded', args=(X,))
            if (not res.success):
                raise ArithmeticError(res.message)
            c = res.x

        elif (self.approach == 'euler'):
            # Find optimal X1 and VNext(X1) by solving Euler equation
            if (cMin < cMax):
                if (self.eulerDiff(cMax, X) > 0.0):
                    c = cMax
                elif (self.eulerDiff(cMin, X) < 0.0):
                    c = cMin
                else:
                    c, res = brentq(self.eulerDiff, cMin, cMax, args=(X,), full_output=True)
                    if (not res.converged):
                        raise ArithmeticError("Root-finding did not converge")
            else:
                c = cMin

        else:
            raise ValueError("Unknown approach")

        return c


    # Find next iteration
    def nextIter(self, VNext, margUtilNext):
    
        for ixX, X in enumerate(self.Xgrid):

            cMin = self.cMin
            cMax = X - ((self.XMin - self.y[0])/self.R)
            c = self.findOptC(cMin, cMax, X)

            VNext[ixX] = -self.negObjFunc(c, X)
            margUtilNext[ixX] = self.uPrime(c)


    # Generate sequence of value function iterations
    def genSeq(self, n, guess=True):
    
        if (guess):
            self.guessSln()
    
        VNext = np.empty(self.Xngp)
        margUtilNext = np.empty(self.Xngp)
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

        VNext = np.empty(self.Xngp)
        margUtilNext = np.empty(self.Xngp)

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

        for ixX, X in enumerate(self.Xgrid):

            cMin = self.cMin
            cMax = X - (self.XMin - self.y[0])/self.R

            self.policy[ixX] = self.findOptC(cMin, cMax, X)


    # Plot value function
    def plotValue(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.Xgrid, self.V)
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()

    # Plot sequence of value functions    
    def plotValueSeq(self, seq):
        fig, ax = plt.subplots()
        ax.plot(self.Xgrid, seq.T)
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function iteration')
        plt.show()

    # Plot policy function
    def plotPolicy(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.Xgrid, self.policy)
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Optimal c', fontsize=14)
        ax.set_title('Policy function')
        plt.show()



