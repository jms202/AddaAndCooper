# Script to solve a simple continuous consumption-savings problem with 
# Markovian uncertainty using spline approximation

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


import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize_scalar
from scipy.optimize import brentq
from scipy.interpolate import SmoothBivariateSpline
import scipy.integrate as integrate
from scipy.integrate import simps
from scipy.stats import lognorm
import copy
from tauchen import approx_markov

def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


class Model:

    def __init__(self, gamma=1.5, beta=0.97, R=1.025, rho=0.9, sigma=0.1, yngp=7, WMax=100.0, Wngp=10, spline_order=3, quad_order=5, approach='value'):

        self.gamma = gamma
        self.beta = beta
        self.R = R
        self.approach = approach                        # approach should be one of 'value' or 'euler'
        self.WMax = WMax
        self.tol = 0.0001
        self.cMin = 0.000001
        self.rho = rho
        self.sigma = sigma
        self.yngp = yngp
        # yProb[0,:] contains the transition probabilities for yGrid[0]
        self.yGrid, self.yProb = approx_markov(rho=self.rho, sigma_u=self.sigma, m=3, n=self.yngp)
        self.yGrid = np.exp(self.yGrid)
        self.yGridFine, self.yProbFine = approx_markov(rho=self.rho, sigma_u=self.sigma, m=3, n=self.yngp*4)
        self.yGridFine = np.exp(self.yGridFine)
        # E[Fy|y=y[0]] = yProb[0, :] * yGrid
        self.Wngp = Wngp
        self.WGrid = np.linspace(self.WMin, self.WMax, num=self.Wngp)
        self.WGridFine = np.linspace(self.WMin, self.WMax, num=self.Wngp*10)
        self.spline_order = spline_order
        self.quad_order = quad_order


    # Getter function for WMin
    @property
    def WMin(self):
        # W = c - y + (c - y)/R + (c - y)/(R**2) + ...
        return max(0.0, self.R*(self.cMin - self.yGrid[0])/(self.R - 1.0))



    # CRRA utility
    def u(self, c):
        return (c**(1.0 - self.gamma)) / (1.0 - self.gamma)


    # CRRA marginal utility
    def uPrime(self, c):
        return (c**(-self.gamma))


    # Objective function
    def negObjFunc(self, W1, y, W):

        c = W + y - (W1/self.R)

        if (np.isscalar(c) | c.size == 1):
            val = -(self.u(c) + (self.beta*self.EV(y, W1, grid=False)))

        else:
            val = np.empty_like(W)
            for row in np.arange(W.shape[0]):
                val[row] = -(self.u(c[row]) + (self.beta*self.EV(y[row], W1[row], grid=False)))

        return val


    # Value function
    def negValue(self, y, W):
        W1 =self.findOptW1(y, W)
        return self.negObjFunc(W1, y, W)


    # Euler equation difference
    def eulerDiff(self, W1, y, W):

        c = W + y - (W1/self.R)

        if (np.isscalar(c) | c.size == 1):
            EUPrime = self.EUPrime(y, W)
            return self.uPrime(c) - (self.beta*self.R*EUPrime)

        diff = np.empty_like(W)

        for row in np.arange(W.shape[0]):
            diff[row] = self.uPrime(c[row]) - (self.beta * self.R * self.EUPrime(y[row], W[row], grid=False))

        return diff


    # Initial guess for V and policy
    def guessSln(self):

        # Calculate value assuming W and y are known
        WMesh, yMesh = np.meshgrid(self.WGrid, self.yGrid)
        cMesh = WMesh + yMesh

        VMesh = self.u(cMesh)
        UPrimeMesh = self.uPrime(cMesh)

        # Integrate over y to get expected value
        EVMesh =  self.yProb @ VMesh
        EUPrimeMesh =  self.yProb @ UPrimeMesh

        # Now do 2D spline interpolation
        # Next line returns a warning message - ignore for now because I think it's working
        self.EV = SmoothBivariateSpline(yMesh.flatten(), WMesh.flatten(), EVMesh.flatten(),  kx=self.spline_order, ky=self.spline_order, s=0)

        # Policy is to consume everything
        # I think we don't want to take the expected value here because we don't need to know the policy from the perspective of the previous period but only for the current period
        # Next line returns a warning message - ignore for now because I think it's working
        self.policy = SmoothBivariateSpline(yMesh.flatten(), WMesh.flatten(), cMesh.flatten(),  kx=self.spline_order, ky=self.spline_order, s=0)

        # Marginal utility at optimum
        self.EUPrime = SmoothBivariateSpline(yMesh.flatten(), WMesh.flatten(), EUPrimeMesh.flatten(),  kx=self.spline_order, ky=self.spline_order, s=0)

  

    def findOptW1(self, y, W):

        if (not np.array_equal(y.shape, W.shape)):
            raise ValueError("y and W must be the same shape")

        W1 = np.empty_like(W)

        if (self.approach == 'value'):

            with np.nditer([W1, y, W], op_flags=['readwrite']) as it:
                for W1Val, yVal, WVal in it:

                    # Find optimal W1 by minimising negObjFunc()
                    W1Max = np.minimum(self.WMax, (self.R * (WVal + yVal - self.cMin)) - self.tol)
                    res = minimize_scalar(self.negObjFunc, bounds=(self.WMin, W1Max), method='bounded', args=(yVal, WVal))
                    if (not res.success):
                        raise ArithmeticError(res.message)
                    W1Val[...] = res.x


        elif (self.approach == 'euler'):

            # Find optimal W1 by solving Euler equation
            with np.nditer([W1, y, W], op_flags=['readwrite']) as it:
                for W1Val, yVal, WVal in it:
                    if (self.cMin < yVal + WVal):
                        if (self.eulerDiff(self.WMin, yVal, WVal) > 0.0):
                            W1Val[...] = self.WMin
                        elif (self.eulerDiff(self.R * (WVal + yVal - self.cMin), yVal, WVal) < 0.0):
                            W1Val[...] = self.R * (WVal + yVal - self.cMin)
                        else:
                            W1Val[...], res = brentq(self.eulerDiff, self.WMin, self.R * (WVal + yVal - self.cMin), args=(yVal, WVal), full_output=True)
                            if (not res.converged):
                                raise ArithmeticError("Root-finding did not converge")
                    else:
                        W1Val[...] = self.WMin


        return W1



    # Distance between two functions
    def calcDistance(self, f1, f2):

        WMesh, yMesh = np.meshgrid(self.WGrid, self.yGrid)

        dist = np.empty_like(WMesh)
        for row in np.arange(WMesh.shape[0]):
            dist[row] = f1(yMesh[row], WMesh[row], grid=False) - f2(yMesh[row], WMesh[row], grid=False)
        maxpos = np.argmax(abs(dist))
        dist = abs(dist[np.unravel_index(maxpos, dist.shape)])
        return dist, maxpos


    # Find next iteration
    def nextIter(self):

        # Calculate value assuming W and y are known
        WMesh, yMesh = np.meshgrid(self.WGrid, self.yGrid)
        cMesh = WMesh + yMesh - (self.findOptW1(yMesh, WMesh)/self.R)

        VMesh = -self.negValue(yMesh, WMesh)
        UPrimeMesh = self.uPrime(cMesh)

        # Integrate over y to get expected value
        EVMesh =  self.yProb @ VMesh
        EUPrimeMesh =  self.yProb @ UPrimeMesh

        # Now do 2D spline interpolation
        # Next line returns a warning message - ignore for now because I think it's working
        EV = SmoothBivariateSpline(yMesh.flatten(), WMesh.flatten(), EVMesh.flatten(),  kx=self.spline_order, ky=self.spline_order, s=0)

        # Policy is to consume everything
        # I think we don't want to take the expected value here because we don't need to know the policy from the perspective of the previous period but only for the current period
        # Next line returns a warning message - ignore for now because I think it's working
        policy = SmoothBivariateSpline(yMesh.flatten(), WMesh.flatten(), cMesh.flatten(),  kx=self.spline_order, ky=self.spline_order, s=0)

        # Marginal utility at optimum
        EUPrime = SmoothBivariateSpline(yMesh.flatten(), WMesh.flatten(), EUPrimeMesh.flatten(),  kx=self.spline_order, ky=self.spline_order, s=0)


        return EV, policy, EUPrime



    # Solution
    def solve(self, iter=1000):

        self.guessSln()

        for ixiter in range(iter):

            print("ixiter:", ixiter)

            EVNext, policyNext, EUPrimeNext = self.nextIter()

            # Check convergence of value function
            dist, maxpos = self.calcDistance(self.EV, EVNext)
            print("%d dist %0.6f at position %d" % (ixiter, dist, maxpos))
            if dist < self.tol:
                print("Breaking on ixiter %d" % ixiter)
                break

            # Can't just assign because they are pointers
            self.EV = copy.copy(EVNext)
            self.policy = copy.copy(policyNext)
            self.EUPrime = copy.copy(EUPrimeNext)
        print("ixiter: %d" % ixiter)



    # Plot expected value function
    def plotEV(self):

        WMesh, yMesh = np.meshgrid(self.WGrid, self.yGrid)
        VMesh = -self.negValue(yMesh, WMesh)

        EVMesh = np.empty_like(WMesh)
        for row in np.arange(WMesh.shape[0]):
            EVMesh[row] = self.EV(yMesh[row], WMesh[row], grid = False)

        fig, ax = plt.subplots()
        ax.plot(self.WGrid, EVMesh.T)
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('EV', fontsize=14)
        ax.set_title('Expected value function')
        plt.show()

    # Plot expected value function
    def plotEVFine(self):

        WMeshFine, yMeshFine = np.meshgrid(self.WGridFine, self.yGridFine)
        VMeshFine = -self.negValue(yMeshFine, WMeshFine)

        EVMeshFine = np.empty_like(WMeshFine)
        for row in np.arange(WMeshFine.shape[0]):
            EVMeshFine[row] = self.EV(yMeshFine[row], WMeshFine[row], grid = False)

        fig, ax = plt.subplots()
        ax.plot(self.WGridFine, EVMeshFine.T)
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('EV', fontsize=14)
        ax.set_title('Expected value function')
        plt.show()


    # Plot value function
    def plotValue(self):

        WMeshFine, yMeshFine = np.meshgrid(self.WGridFine, self.yGridFine)
        VMeshFine = -self.negValue(yMeshFine, WMeshFine)

        fig, ax = plt.subplots()
        ax.plot(self.WGridFine, VMeshFine.T)
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()

    # Plot value function transposed
    def plotValueTranspose(self):

        WMeshFine, yMeshFine = np.meshgrid(self.WGridFine, self.yGridFine)
        VMeshFine = -self.negValue(yMeshFine, WMeshFine)

        fig, ax = plt.subplots()
        ax.plot(self.yGridFine, VMeshFine)
        ax.set_xlabel('y', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function transposed')
        plt.show()

    # Plot policy function
    def plotPolicy(self):

        WMeshFine, yMeshFine = np.meshgrid(self.WGridFine, self.yGridFine)

        policyMeshFine = np.empty_like(WMeshFine)
        for row in np.arange(WMeshFine.shape[0]):
            policyMeshFine[row] = self.policy(yMeshFine[row], WMeshFine[row], grid = False)

        fig, ax = plt.subplots()
        ax.plot(self.WGridFine, policyMeshFine.T)
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('Optimal c', fontsize=14)
        ax.set_title('Policy function')
        plt.show()


    # Plot expected marginal utility function
    def plotEUPrime(self):

        WMeshFine, yMeshFine = np.meshgrid(self.WGridFine, self.yGridFine)

        EUPrimeMeshFine = np.empty_like(WMeshFine)
        for row in np.arange(WMeshFine.shape[0]):
            EUPrimeMeshFine[row] = self.EUPrime(yMeshFine[row], WMeshFine[row], grid = False)


        fig, ax = plt.subplots()
        ax.plot(self.WGridFine, EUPrimeMeshFine.T)
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('Expected marginal utility', fontsize=14)
        ax.set_title('Expected marginal utility')
        plt.show()

