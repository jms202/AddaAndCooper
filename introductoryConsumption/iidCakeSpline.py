# Script to solve a simple continuous consumption-savings problem with 
# IID uncertainty using spline approximation and numerical quadrature

# Copyright (c) 2020 Jonathan Shaw

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
from scipy.interpolate import BSpline, make_interp_spline
import scipy.integrate as integrate
from scipy.stats import lognorm
import copy


class Model:

    def __init__(self, gamma=1.5, beta=0.97, R=1.025, mu=2., sigma=0.4, XMax=100.0, Xngp=8, spline_order=3, quad_order=5, approach='value'):

        self.gamma = gamma
        self.beta = beta
        self.R = R
        self.approach = approach                        # approach should be one of 'value' or 'euler'
        self.XMax = XMax
        self.tol = 0.0001
        self.cMin = 0.000001
        self.mu = mu
        self.sigma = sigma
        self.Xngp = Xngp
        self.Xgrid = np.logspace(np.log(self.XMin+1.0), np.log(self.XMax+1.0), num=self.Xngp, base=np.exp(1.0)) - 1.0
        self.Xgridfine = np.logspace(np.log(self.XMin+1.0), np.log(self.XMax+1.0), num=self.Xngp*100, base=np.exp(1.0)) - 1.0
        self.spline_order = spline_order
        self.quad_order = quad_order


    # Getter function for XMin
    @property
    def XMin(self):
        # X = c + (c - y)/R + (c - y)/(R**2) + ...
        return max(self.yMin, (self.R*self.cMin - self.yMin)/(self.R - 1.0))

    @property
    def yMin(self):
        return np.exp(self.mu - (3. * self.sigma))

    @property
    def yMax(self):
        return np.exp(self.mu + (3. * self.sigma))

    # CRRA utility
    def u(self, c):
        return (c**(1.0 - self.gamma)) / (1.0 - self.gamma)


    # CRRA marginal utility
    def uPrime(self, c):
        return (c**(-self.gamma))



    # Calculate part of expected value function
    def EVpart(self, y1, X, c):
        return self.V(self.R*(X - c) + y1) * lognorm.pdf(y1, s=self.sigma, scale=np.exp(self.mu))

    # Calculate part of expected marginal utility
    def EuPrimePart(self, y1, X, c):
        return self.uPrime(self.policy(self.R*(X - c) + y1)) * lognorm.pdf(y1, s=self.sigma, scale=np.exp(self.mu))

    # Objective function
    def negObjFunc(self, c, X):

        if (np.isscalar(c)):

            EV = integrate.fixed_quad(self.EVpart, self.yMin, self.yMax, args = (X, c), n=self.quad_order)
            return -(self.u(c) + (self.beta*EV[0]))
        
        obj = np.empty(len(c))

        for ix, (cVal, XVal) in enumerate(zip(c, X)):

            EV = integrate.fixed_quad(self.EVpart, self.yMin, self.yMax, args = (XVal, cVal), n=self.quad_order)
            obj[ix] = -(self.u(cVal) + (self.beta*EV[0]))
        
        return obj

    # Euler equation difference
    def eulerDiff(self, c, X):

        if (np.isscalar(c)):
            EuPrime = integrate.fixed_quad(self.EuPrimePart, self.yMin, self.yMax, args = (X, c), n=self.quad_order)
            return self.uPrime(c) - (self.beta*self.R*EuPrime[0])

        diff = np.empty(len(c))

        for ix, (cVal, XVal) in enumerate(zip(c, X)):

            EuPrime = integrate.fixed_quad(self.EuPrimePart, self.yMin, self.yMax, args = (XVal, cVal), n=self.quad_order)
            diff[ix] = self.uPrime(cVal) - (self.beta*self.R*EuPrime[0])

        return diff


    # Initial guess for V and policy
    def guessSln(self):
        self.V = make_interp_spline(self.Xgrid, self.u(self.Xgrid), k=self.spline_order, bc_type="natural")
        self.policy = make_interp_spline(self.Xgrid, self.Xgrid, k=self.spline_order, bc_type="natural")


    # Find optimal c given X
    def findOptC(self, X):

        c = np.empty(len(X))

        if (self.approach == 'value'):

            for ixX, XVal in enumerate(X):

                # Find optimal c by minimising negObjFunc()
                res = minimize_scalar(self.negObjFunc, bounds=(self.cMin, XVal), method='bounded', args=(XVal,))
                if (not res.success):
                    raise ArithmeticError(res.message)
                c[ixX] = res.x

        elif (self.approach == 'euler'):
            # Find optimal X1 and VNext(X1) by solving Euler equation
            for ixX, XVal in enumerate(X):
                if (self.cMin < XVal):
                    if (self.eulerDiff(XVal, XVal) > 0.0):
                        c[ixX] = XVal
                    elif (self.eulerDiff(self.cMin, XVal) < 0.0):
                        c[ixX] = self.cMin
                    else:
                        c[ixX], res = brentq(self.eulerDiff, self.cMin, XVal, args=(XVal,), full_output=True)
                        if (not res.converged):
                            raise ArithmeticError("Root-finding did not converge")
                else:
                    c[ixX] = self.cMin

        else:
            raise ValueError("Unknown approach")

        return c


    # Value function
    def negValue(self, X):
        return self.negObjFunc(self.findOptC(X), X)


    # Distance between two functions
    def calcDistance(self, f1, f2):
        dist = f1(self.Xgrid) - f2(self.Xgrid)
        maxpos = np.argmax(abs(dist))
        dist = abs(dist[maxpos])
        return dist, maxpos

    # Find next iteration
    def nextIter(self):
        VNext = make_interp_spline(self.Xgrid, -self.negValue(self.Xgrid), k=self.spline_order, bc_type="natural")
        policyNext = make_interp_spline(self.Xgrid, self.findOptC(self.Xgrid), k=self.spline_order, bc_type="natural")
        return VNext, policyNext


    # Solution
    def solve(self, iter=1000):

        self.guessSln()

        for ixiter in range(iter):

            VNext, policyNext = self.nextIter()

            # Check convergence of value function
            dist, maxpos = self.calcDistance(self.V, VNext)
            print("%d dist %0.6f at position %d" % (ixiter, dist, maxpos))
            if dist < self.tol:
                print("Breaking on ixiter %d" % ixiter)
                break

            # Can't just assign because they are pointers
            self.V = copy.copy(VNext)
            self.policy = copy.copy(policyNext)

        print("ixiter: %d" % ixiter)



    # Plot value function
    def plotValue(self):

        fig, ax = plt.subplots()
        ax.plot(self.Xgridfine, self.V(self.Xgridfine))
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()


    # Plot policy function
    def plotPolicy(self):

        fig, ax = plt.subplots()
        ax.plot(self.Xgridfine, self.policy(self.Xgridfine))
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Optimal c', fontsize=14)
        ax.set_title('Policy function')
        plt.show()


    # Plot policy function
    def plotPolicy(self):

        fig, ax = plt.subplots()
        ax.plot(self.Xgrid, self.findOptC(self.Xgrid), "o", label = "Actual")
        ax.plot(self.Xgridfine, self.policy(self.Xgridfine), label="Fitted")
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Optimal c', fontsize=14)
        ax.set_title('Policy function')
        plt.show()

