

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

class Model:

    def __init__(self, alpha=0.75, beta=0.9, gamma=1.5, delta=0.3, KRange=0.1, Kngp=101):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.KStar = ((1 - beta*(1 - delta))/(alpha*beta))**(1/(alpha - 1))
        self.KMax = self.KStar*(1 + KRange)
        self.KMin = self.KStar*(1 - KRange)
        self.tol = 0.0001
        self.cMin = 0.000001
        self.Kngp = Kngp
        self.Kgrid = np.linspace(self.KMin, self.KMax, self.Kngp)
        self.V = np.zeros(self.Kngp)
        self.policy = np.empty(self.Kngp)


    # CRRA utility
    def CRRA(self, c):
        return (c**(1.0 - self.gamma)) / (1.0 - self.gamma)


    # Production function
    def prodFunc(self, K):
        return K**self.alpha



    # Find next iteration (using value function iteration)
    def nextIterValue(self):

        # Initialise vectors to hold new value function and policy function
        TV = np.empty(self.Kngp)
        ixK1Opt = np.empty(self.Kngp, dtype=int)

        # Find the new optimal policy and value function given current value function
        # The loop is over K today assuming current value function applies tomorrow
        for ixK, K in enumerate(self.Kgrid):
            # c is a vector of possible consumption values today given capital today
            c = self.prodFunc(K) + (1 - self.delta)*K - self.Kgrid
            c = c[c>self.cMin]
            u = self.CRRA(c)
            objective = u + (self.beta*self.V[:len(c)])
            # Store the index of optimal next-period capital
            ixK1Opt[ixK] = np.argmax(objective)
            # Store the value that corresponds to new optimal next-period capital
            TV[ixK] = objective[ixK1Opt[ixK]]

        K1Opt = self.Kgrid[ixK1Opt]
        return (TV, K1Opt)


    # Solution
    def solve(self, iter=1000):

        for ixiter in range(iter):

            VNext, policyNext = self.nextIterValue()

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
        ax.plot(self.Kgrid, self.policy)
        ax.set_xlabel('K', fontsize=14)
        ax.set_ylabel("Optimal K'", fontsize=14)
        ax.set_title('Policy function')
        plt.show()


