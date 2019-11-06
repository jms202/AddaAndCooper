

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

class Model:

    def __init__(self, alpha=0.75, beta=0.9, gamma=1.5, delta=0.3, p=[[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]], A=[0.9, 1.0, 1.1], KRange=0.2, Kngp=101):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.p = np.array(p)
        self.A = np.array(A)
        self.Angp = len(A)
        self.KStar = ((1 - beta*(1 - delta))/(alpha*beta))**(1/(alpha - 1)) # This assumes that the mean value of A is 1.0
        self.KMax = self.KStar*(1 + KRange)
        self.KMin = self.KStar*(1 - KRange)
        self.tol = 0.0001
        self.cMin = 0.000001
        self.Kngp = Kngp
        self.Kgrid = np.linspace(self.KMin, self.KMax, self.Kngp)
        self.EV = np.zeros((self.Angp, self.Kngp))
        self.policy = np.empty((self.Angp, self.Kngp))


    # CRRA utility
    def CRRA(self, c):
        return (c**(1.0 - self.gamma)) / (1.0 - self.gamma)


    # Production function
    def prodFunc(self, A, K):
        return A * (K**self.alpha)


    # Initial guess for V
    def guessSln(self):

        V = np.empty((self.Angp, self.Kngp))

        # Calculate value function assuming shock is known and everything is consumed today
        for ixA, A in enumerate(self.A):
            for ixK, K in enumerate(self.Kgrid):
                V[ixA,ixK] = self.CRRA(self.prodFunc(A, K) + (1.0 - self.delta)*K)

        # Now integrate over uncertain shock
        self.EV = self.p @ V


    # Find next iteration
    def nextIter(self, EVNext):
    
        VNext = np.empty((self.Angp, self.Kngp))
    
        # Calculate value function assuming shock is known
        for ixA, A in enumerate(self.A):
            for ixK, K in enumerate(self.Kgrid):

                # Find optimal K1
                c = self.prodFunc(A, K) + (1 - self.delta)*K - self.Kgrid   # This is a vector of c, one for each value of next-period assets
                c = c[c>self.cMin]
                u = self.CRRA(c)
                objective = u + (self.beta*self.EV[ixA, :len(c)])
                ixK1Opt = np.argmax(objective)
                VNext[ixA,ixK] = objective[ixK1Opt]

        # Now integrate over uncertain shock
        np.copyto(EVNext, self.p @ VNext)




    # Solution
    def solve(self, iter=1000):

        self.guessSln()

        EVNext = np.empty((self.Angp, self.Kngp))

        for ixiter in range(iter):

            self.nextIter(EVNext)

            # Check convergence of value function (we could also check convergence of marginal utility)
            maxpos = np.unravel_index(np.argmax(abs(self.EV - EVNext)), self.EV.shape)
            dist = abs(self.EV[maxpos] - EVNext[maxpos])
            print("%d dist %0.6f at position %s (%0.6f vs %0.6f)" % (ixiter, dist, maxpos, self.EV[maxpos], EVNext[maxpos]))
            if dist < self.tol:
                print("Breaking on ixiter %d" % ixiter)
                break

            # Can't just assign because they are pointers
            np.copyto(self.EV, EVNext)



    # Find policy function using solution
    def findPolicy(self):

        for ixA, A in enumerate(self.A):
            for ixK, K in enumerate(self.Kgrid):

                # Find optimal K1
                c = self.prodFunc(A, K) + (1 - self.delta)*K - self.Kgrid   # This is a vector of c, one for each value of next-period assets
                c = c[c>self.cMin]
                u = self.CRRA(c)
                objective = u + (self.beta*self.EV[ixA, :len(c)])
                ixK1Opt = np.argmax(objective)

                self.policy[ixA, ixK] = self.Kgrid[ixK1Opt]



    # Plot value function
    def plotValue(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.Kgrid, self.EV.T)
        ax.set_xlabel('K', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()


    # Plot policy function
    def plotPolicy(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.Kgrid, self.policy.T)
        ax.set_xlabel('K', fontsize=14)
        ax.set_ylabel('Optimal K1', fontsize=14)
        ax.set_title('Policy function')
        plt.show()




