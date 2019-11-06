# Script to solve a simple continuous non-stochastic consumption-savings 
# problem with no interpolation

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
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg


class Model:

    def __init__(self, gamma=1.5, beta=0.97, R=1.025, y=5.0, WMax=100.0, Wngp=101, method='VFI'):

        self.gamma = gamma
        self.beta = beta
        self.R = R
        self.y = y
        self.WMax = WMax
        self.tol = 0.0001
        self.cMin = 0.000001
        self.Wngp = Wngp
        self.method = method
        self.Wgrid = np.linspace(self.WMin, self.WMax, self.Wngp)
        self.V = np.zeros(self.Wngp)
        self.policy = np.empty(self.Wngp)


    # Getter function for WMin
    @property
    def WMin(self):
        # W = c - y + (c - y)/R + (c - y)/(R**2) + ...
        return max(0.0, self.R*(self.cMin - self.y)/(self.R - 1.0))


    # CRRA utility
    def CRRA(self, c):
        return (c**(1.0 - self.gamma)) / (1.0 - self.gamma)



    # Find next iteration (using value function iteration)
    def nextIterValue(self):

        TV = np.empty(self.Wngp)
        ixW1Opt = np.empty(self.Wngp, dtype=int)

        # Find the optimal policy and value function given previous value function
        for ixW, W in enumerate(self.Wgrid):
            c = W + self.y - (self.Wgrid/self.R)
            c = c[c>self.cMin]
            u = self.CRRA(c)
            objective = u + (self.beta*self.V[:len(c)])
            ixW1Opt[ixW] = np.argmax(objective)
            TV[ixW] = objective[ixW1Opt[ixW]]

        W1Opt = self.Wgrid[ixW1Opt]
        cOpt = self.Wgrid + self.y - (W1Opt/self.R)
        return (TV, cOpt)


           
    # Find next iteration (using policy function iteration)
    def nextIterPolicy(self):

        ixW1Opt = np.empty(self.Wngp, dtype=int)

        # Policy improvement step (finding the optimal policy given the current value function)
        for ixW, W in enumerate(self.Wgrid):
            c = W + self.y - (self.Wgrid/self.R)
            c = c[c>self.cMin]
            u = self.CRRA(c)
            ixW1Opt[ixW] = np.argmax(u + (self.beta*self.V[:len(c)]))


        # Policy evaluation step (the value of following policy forever)

        W1Opt = self.Wgrid[ixW1Opt]
        cOpt = self.Wgrid + self.y - (W1Opt/self.R)
        uOpt = self.CRRA(cOpt)

        I = sparse.identity(self.Wngp, format="csr")
        
        rows = np.arange(self.Wngp)
        data = np.ones(self.Wngp)
        Q = sparse.coo_matrix((data, (rows, ixW1Opt)), shape=(self.Wngp, self.Wngp))
        Q = Q.tocsr()

        TV = linalg.spsolve(I - (self.beta*Q), uOpt)

        return (TV, cOpt)

    
    # Solution
    def solve(self, iter=1000):

        for ixiter in range(iter):

            if (self.method == 'VFI'):
                VNext, policyNext = self.nextIterValue()
            else:
                VNext, policyNext = self.nextIterPolicy()

            # Check convergence of value function (we could also check convergence of marginal utility)
            maxpos = np.argmax(abs(self.V - VNext))
            dist = abs(self.V[maxpos] - VNext[maxpos])
            print("%d dist %0.6f at position %d (%0.6f vs %0.6f)" % (ixiter, dist, maxpos, self.V[maxpos], VNext[maxpos]))
            if dist < self.tol:
                print("Breaking on ixiter %d" % ixiter)
                break

            # Can't just assign because they are pointers
            np.copyto(self.V, VNext)
            np.copyto(self.policy, policyNext)


    # Plot policy function
    def plotPolicy(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.Wgrid, self.policy)
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('Optimal c', fontsize=14)
        ax.set_title('Policy function')
        plt.show()

