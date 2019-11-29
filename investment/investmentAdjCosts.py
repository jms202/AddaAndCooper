

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from tauchen import approx_markov


def solveStationary(A):
    """ x = xA where x is the answer
    x - xA = 0
    x( I - A ) = 0 and sum(x) = 1
    """
    n = A.shape[0]
    a = np.eye(n) - A
    a = np.vstack((a.T, np.ones(n)))
    b = np.array([0] * n + [1]).T
    return np.squeeze(np.array(np.linalg.lstsq(a, b, rcond=None)[0]))


class Model:

    def __init__(self, alpha=0.7, beta=0.95, gamma=0.2, delta=0.15, rho=0.9, sigma=0.1, KScale=0.5, Kngp=7, Angp=7, p = 1.2):

        self.tol = 0.0001

        self.KStarReady = {'alpha':False, 'gamma':False, 'delta':False, 'AMean':False, 'p':False}
        self.KReady = {'KStar':False, 'KScale':False, 'Kngp':False}
        self.AReady = {'rho':False, 'sigma':False, 'Angp':False}
        self.slnReady = {'Kngp':False, 'Angp':False}

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.rho = rho
        self.sigma = sigma
        self.Angp = Angp
        self.p = p
        self.Kngp = Kngp
        self.KScale = KScale



    # Function to calculate grid on A
    def setAGrid(self):
        if all([val for val in self.AReady.values()]):
            self.A, self.prob = approx_markov(rho=self.rho, sigma_u=self.sigma, m=3, n=self.Angp)
            self.A = np.exp(self.A)
            self.AMean = np.dot(solveStationary(self.prob), self.A)
            self.KStarReady['AMean'] = True
            self.setKStar()

    # Function to work out KStar
    def setKStar(self):
        if all([val for val in self.KStarReady.values()]):
            self.KStar = ((self.gamma * self.delta + 2.0 * self.p) * self.delta / (2.0 * self.alpha * self.AMean))**(1.0 / (self.alpha - 1.0))
            self.KReady['KStar'] = True
            self.setKGrid()

    # Function to calculate grid on K
    def setKGrid(self):
        if all([val for val in self.KReady.values()]):
            self.Kgrid = np.linspace(self.KStar*self.KScale, self.KStar/self.KScale, self.Kngp)

    # Function to allocate solution matrices
    def allocateSln(self):
        if all([val for val in self.slnReady.values()]):
            self.EV = np.empty((self.Angp, self.Kngp))
            self.ixK1policy = np.empty((self.Angp, self.Kngp), dtype=np.int32)


    # Getter function for alpha
    @property
    def alpha(self):
        return self._alpha

    # Setter function for alpha
    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        self.KStarReady['alpha'] = True
        self.setKStar()

    # Getter function for gamma
    @property
    def gamma(self):
        return self._gamma

    # Setter function for gamma
    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self.KStarReady['gamma'] = True
        self.setKStar()

    # Getter function for delta
    @property
    def delta(self):
        return self._delta

    # Setter function for delta
    @delta.setter
    def delta(self, value):
        self._delta = value
        self.KStarReady['delta'] = True
        self.setKStar()

    # Getter function for p
    @property
    def p(self):
        return self._p

    # Setter function for p
    @p.setter
    def p(self, value):
        self._p = value
        self.KStarReady['p'] = True
        self.setKStar()

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

    # Getter function for KScale
    @property
    def KScale(self):
        return self._KScale

    # Setter function for KScale
    @KScale.setter
    def KScale(self, value):
        self._KScale = value
        self.KReady['KScale'] = True
        self.setKGrid()


    # Adjustment costs function
    def adjCost(self, K, K1):
        return 0.5 * self.gamma * K * (((K1 - (1.0 - self.delta)*K) / K)**2)

    # One period profit function
    def onePeriodProfit(self, A, K, K1):
        return A * (K**self.alpha) - self.adjCost(K, K1) - self.p * (K1 - (1.0 - self.delta) * K)

    # Initial guess for EV
    def guessSln(self):

        # We already calculated the current return from all combinations of A, k and k1
        # Just need to integrate over uncertain shock
        self.EV = self.prob @ self.currReturn[:,:,0]


    # Initial guess for V
    def guessSln(self):

        V = np.empty((self.Angp, self.Kngp))

        # Calculate value function assuming shock is known and capital is kept constant
        for ixA, A in enumerate(self.A):
            for ixK, K in enumerate(self.Kgrid):
                V[ixA,ixK] = self.onePeriodProfit(A, K, K)

        # Now integrate over uncertain shock
        self.EV = self.prob @ V


    # Find next iteration
    def nextIter(self, EVNext):
    
        VNext = np.empty((self.Angp, self.Kngp))
    
        # Calculate value function assuming shock is known
        for ixA, A in enumerate(self.A):
            for ixK, K in enumerate(self.Kgrid):

                # For each value of (A, K), work out optimal K1
                objective = self.onePeriodProfit(A, K, self.Kgrid) + (self.beta*self.EV[ixA, :])
                ixK1Opt = np.argmax(objective)
                VNext[ixA,ixK] = objective[ixK1Opt]

        # Now integrate over uncertain shock
        np.copyto(EVNext, self.prob @ VNext)


    # Find policy function using solution
    def findPolicy(self):

        for ixA, A in enumerate(self.A):
            for ixK, K in enumerate(self.Kgrid):

                # Find optimal K1
                objective = self.onePeriodProfit(A, K, self.Kgrid) + (self.beta*self.EV[ixA, :])
                ixK1Opt = np.argmax(objective)

                self.ixK1policy[ixA, ixK] = ixK1Opt


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
        ax.plot(self.Kgrid, self.Kgrid[self.ixK1policy.T])
        ax.set_xlabel('K', fontsize=14)
        ax.set_ylabel('Optimal K1', fontsize=14)
        ax.set_title('Policy function')
        plt.show()


    # Plot relationship between investment and average Q
    def plotInvestmentQbar(self):

        Qbar = self.EV / self.Kgrid[None, ...]
        investment = self.Kgrid[self.ixK1policy] - (1.0 - self.delta)*self.Kgrid[None, ...]

        fig, ax = plt.subplots()
        ax.plot(Qbar.T, investment.T)
        ax.set_xlabel('Average Q', fontsize=14)
        ax.set_ylabel('Investment', fontsize=14)
        ax.set_title('Relationship between average Q and investment')
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
            sims['ixK1'][i, 0] = self.ixK1policy[sims['ixA'][i, 0], sims['ixK'][i, 0]]

            for h in range(1, horiz):

                # Copy forward optimal k
                sims['ixK'][i, h] = sims['ixK1'][i, h-1]
                # Calculate optimal k'
                sims['ixK1'][i, h] = self.ixK1policy[sims['ixA'][i, h], sims['ixK'][i, h]]
        

        # Now calculate other variables
        sims['A'] = self.A[sims['ixA']]
        sims['K'] = self.Kgrid[sims['ixK']]
        sims['K1'] = self.Kgrid[sims['ixK1']]
        sims['i'] = sims['K1'] - (1.0 - self.delta) * sims['K']
        sims['EV'] = self.EV[sims['ixA'], sims['ixK1']]
        sims['Qbar'] = sims['EV'] / sims['K1']

        return sims

