# Script to solve a simple continuous non-stochastic consumption-savings problem using spline function interpolation

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
from scipy.interpolate import BSpline, make_interp_spline
from scipy.optimize import minimize_scalar
import sys
import copy

def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

class Model:

    def __init__(self, gamma=1.5, beta=0.97, R=1.025, y=5.0, WMax=100.0, Wngp=8, order=3):

        self.gamma = gamma
        self.beta = beta
        self.R = R
        self.y = y
        self.cMin = 0.000001
        self.WMax = WMax
        self.Wngp = Wngp
        #self.Wgrid = np.linspace(self.WMin, self.WMax, num=self.Wngp)
        self.Wgrid = np.logspace(np.log(self.WMin+1.0), np.log(self.WMax+1.0), num=self.Wngp, base=np.exp(1.0)) - 1.0
        self.Wgridfine = np.logspace(np.log(self.WMin+1.0), np.log(self.WMax+1.0), num=self.Wngp*100, base=np.exp(1.0)) - 1.0
        self.tol = 0.0001
        self.order = order


    # Getter function for WMin
    @property
    def WMin(self):
        # W = c - y + (c - y)/R + (c - y)/(R**2) + ...
        return max(0.0, self.R*(self.cMin - self.y)/(self.R - 1.0))


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
        self.V = make_interp_spline(self.Wgrid, self.u(self.Wgrid + self.y), k=self.order, bc_type="natural")



    # Find next iteration
    def nextIter(self):
        VNext = make_interp_spline(self.Wgrid, -self.negValue(self.Wgrid), k=self.order, bc_type="natural")
        return VNext

            
    # Distance between two functions
    def calcDistance(self, f1, f2):
        dist = f1(self.Wgrid) - f2(self.Wgrid)
        maxpos = np.argmax(abs(dist))
        dist = abs(dist[maxpos])
        return dist, maxpos


    # Find policy function using solution
    def findPolicy(self):
        #knots = np.logspace(np.log(self.WMin + 1.0), np.log(self.WMax + 1.0), num=self.Wngp+self.order+1, base=np.exp(1.0)) - 1.0
        self.policy = make_interp_spline(self.Wgrid, self.Wgrid + self.y - (self.findOptW1(self.Wgrid)/self.R), k=self.order, bc_type="natural")


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
            self.V = copy.copy(VNext)

        print("ixiter: %d" % ixiter)

        self.findPolicy()




    # Plot value function
    def plotValue(self):

        fig, ax = plt.subplots()
        ax.plot(self.Wgridfine, self.V(self.Wgridfine))
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()


    # Plot policy function
    def plotPolicy(self):

        actual = self.Wgrid + self.y - (self.findOptW1(self.Wgrid)/self.R)
        fig, ax = plt.subplots()
        ax.plot(self.Wgrid, actual, "o", label = "Actual")
        ax.plot(self.Wgridfine, self.policy(self.Wgridfine), label="Fitted")
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('Optimal c', fontsize=14)
        ax.set_title('Policy function')
        plt.show()



