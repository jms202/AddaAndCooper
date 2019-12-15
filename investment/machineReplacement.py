

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from tauchen import approx_markov


class Model:

    def __init__(self, alpha=0.7, beta=0.95, delta=0.15, llambda=0.7, rho=0.9, sigma=0.1, Kngp=7, Angp=7, p = 1.2):

        self.tol = 0.0001

        self.AReady = {'rho':False, 'sigma':False, 'Angp':False}
        self.KReady = {'delta':False, 'Kngp':False}
        self.slnReady = {'Kngp':False, 'Angp':False}

        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.llambda = llambda
        self.rho = rho
        self.sigma = sigma
        self.Angp = Angp
        self.p = p
        self.Kngp = Kngp



    # Function to calculate grid on A
    def setAGrid(self):
        if all([val for val in self.AReady.values()]):
            self.A, self.prob = approx_markov(rho=self.rho, sigma_u=self.sigma, m=3, n=self.Angp)
            self.A = np.exp(self.A)

    # Function to calculate grid on K
    def setKGrid(self):
        if all([val for val in self.KReady.values()]):
            self.Kgrid = np.empty(self.Kngp)
            # Set top of grid to 1
            K = 1.0
            for ixK in range(self.Kngp):
                self.Kgrid[-(ixK+1)] = K
                K *= (1.0 - self.delta)
            # Set bottom of grid to zero
            self.Kgrid[0] = 0.0


    # Function to allocate solution matrices
    def allocateSln(self):
        if all([val for val in self.slnReady.values()]):
            self.EV = np.empty((self.Angp, self.Kngp))
            self.policy = np.empty((self.Angp, self.Kngp), dtype=np.bool_)


    # Getter function for delta
    @property
    def delta(self):
        return self._delta

    # Setter function for delta
    @delta.setter
    def delta(self, value):
        self._delta = value
        self.KReady['delta'] = True
        self.setKGrid()

    # Getter function for rho
    @property
    def rho(self):
        return self._rho

    # Setter function for rho
    @rho.setter
    def rho(self, value):
        self._rho = value
        self.AReady['rho'] = True
        self.setAGrid()

    # Getter function for sigma
    @property
    def sigma(self):
        return self._sigma

    # Setter function for sigma
    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        self.AReady['sigma'] = True
        self.setAGrid()

    # Getter function for Angp
    @property
    def Angp(self):
        return self._Angp

    # Setter function for Angp
    @Angp.setter
    def Angp(self, value):
        self._Angp = value
        self.AReady['Angp'] = True
        self.setAGrid()
        self.slnReady['Angp'] = True
        self.allocateSln()

    # Getter function for Kngp
    @property
    def Kngp(self):
        return self._Kngp

    # Setter function for Kngp
    @Kngp.setter
    def Kngp(self, value):
        self._Kngp = value
        self.KReady['Kngp'] = True
        self.setKGrid()
        self.slnReady['Kngp'] = True
        self.allocateSln()



    # One period profit function
    def onePeriodProfit(self, A, K, invest):
        if (invest == True):
            return ((A * self.llambda) - self.p)
        else:
            return(A * (K**self.alpha))


    # Initial guess for EV
    def guessSln(self):

        V = np.empty((self.Angp, self.Kngp))
        
        # Starting guess for value function is one-period profit function
        # First assume shock is known
        for ixA, A in enumerate(self.A):
            for ixK, K in enumerate(self.Kgrid):
                V[ixA,ixK] = max(self.onePeriodProfit(A, K, invest=True), self.onePeriodProfit(A, K, invest=False))

        # Now integrate over uncertain shock
        self.EV = self.prob @ V

        # Note the ixK index in EV relates to K in the next period (the period to which it relates)
        # So if ixK = 6 this period, you should use self.EV[ixA, 5] to get the expected value next period
        # This is true except for the bottom of the grid: if ixK = 0 then use self.EV[ixA, 0]


    # Find next iteration
    def nextIter(self, EVNext):
    
        VNext = np.empty((self.Angp, self.Kngp))
    
        # Calculate value function assuming shock is known
        for ixA, A in enumerate(self.A):
            for ixK, K in enumerate(self.Kgrid):

                VIfInvest = self.onePeriodProfit(A, K, invest=True) + (self.beta * self.EV[ixA, self.Kngp - 1])
                ixKnext = max(ixK - 1, 0)
                VIfNotInvest = self.onePeriodProfit(A, K, invest=False) + (self.beta * self.EV[ixA, ixKnext])

                VNext[ixA,ixK] = max(VIfInvest, VIfNotInvest)

        # Now integrate over uncertain shock
        np.copyto(EVNext, self.prob @ VNext)


    # Find policy function using solution
    def findPolicy(self):

        for ixA, A in enumerate(self.A):
            for ixK, K in enumerate(self.Kgrid):

                VIfInvest = self.onePeriodProfit(A, K, invest=True) + (self.beta * self.EV[ixA, self.Kngp - 1])
                ixKnext = max(ixK - 1, 0)
                VIfNotInvest = self.onePeriodProfit(A, K, invest=False) + (self.beta * self.EV[ixA, ixKnext])

                self.policy[ixA, ixK] = (VIfInvest > VIfNotInvest)


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

        self.findPolicy()


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
        ax.set_ylabel('Invest?', fontsize=14)
        ax.set_title('Policy function')
        plt.show()


    def drawShocks(self, n=1000, horiz=100):
        
        sims = {'ixA':np.empty((n, horiz), dtype=np.int32)}
        sims['ixA'][:, 0] = np.random.choice(self.Angp, size=n)

        for i in range(n):
            for h in range(1, horiz):

                sims['ixA'][i, h] = np.random.choice(self.Angp, p = self.prob[sims['ixA'][i, h-1], :])

        return sims


    # Simulate the investment choices of a number of firms
    def simulate(self, sims, n=1000, horiz=100):

        sims['ixK'] = np.empty((n, horiz), dtype=np.int32)
        sims['ixK1'] = np.empty((n, horiz), dtype=np.int32)

        for i in range(n):

            sims['ixK'][i, 0] = np.random.choice(range(self.Kngp))
            if (self.policy[sims['ixA'][i, 0], sims['ixK'][i, 0]]):
                sims['ixK1'][i, 0] = self.Kngp
            else:
                sims['ixK1'][i, 0] = max(sims['ixK'][i, 0] - 1, 0)
                
            for h in range(1, horiz):

                # Copy forward optimal k
                sims['ixK'][i, h] = sims['ixK1'][i, h-1]
                # Calculate optimal k'
                if (self.policy[sims['ixA'][i, h], sims['ixK'][i, h]]):
                    sims['ixK1'][i, h] = self.Kngp
                else:
                    sims['ixK1'][i, h] = max(sims['ixK'][i, h] - 1, 0)
        

        # Now calculate other variables
        sims['A'] = self.A[sims['ixA']]
        sims['K'] = self.Kgrid[sims['ixK']]
        sims['K1'] = self.Kgrid[sims['ixK1']]
        sims['EV'] = self.EV[sims['ixA'], sims['ixK1']]

        return sims

