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

    def __init__(self, alpha=0.7, beta=0.95, delta=0.15, llambda=0.7, F=0.1, rho=0.9, sigma=0.1, KScale=0.5, Kngp=7, Angp=7, p = 1.2):

        self.tol = 0.0001

        self.KStarReady = {'alpha':False, 'llambda':False, 'delta':False, 'AMean':False, 'p':False, 'F':False}
        self.KReady = {'KStar':False, 'Kngp':False}
        self.AReady = {'rho':False, 'sigma':False, 'Angp':False}
        self.slnReady = {'Kngp':False, 'Angp':False}

        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.llambda = llambda
        self.F = F
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
            self.AMean = np.dot(solveStationary(self.prob), self.A)
            self.KStarReady['AMean'] = True
            self.setKStar()


    # Function to work out KStar (one-period optimal K if investing)
    def setKStar(self):
        if all([val for val in self.KStarReady.values()]):
            self.KStar = ((self.F + self.delta * self.p) / (self.alpha * self.llambda * self.AMean))**(1.0 / (self.alpha - 1.0))
            self.KReady['KStar'] = True
            self.setKGrid()


    # Function to calculate grid on K
    def setKGrid(self):
        if all([val for val in self.KReady.values()]):
            self.Kgrid = np.empty(self.Kngp)
            # Set midpoint to KStar
            ixMidpoint = int((self.Kngp - (self.Kngp % 2)) / 2)
            self.Kgrid[ixMidpoint] = self.KStar
            # Set lower grid points
            K = self.KStar
            for ixK in range(ixMidpoint,0,-1):
                K *= (1.0 - self.delta)
                self.Kgrid[ixK-1] = K
            # Set higher grid points
            K = self.KStar
            for ixK in range(ixMidpoint+1, self.Kngp):
                K /= (1.0 - self.delta)
                self.Kgrid[ixK] = K
            # Set bottom of grid to zero
            self.Kgrid[0] = 0.0


    # Function to allocate solution matrices
    def allocateSln(self):
        if all([val for val in self.slnReady.values()]):
            self.EV = np.empty((self.Angp, self.Kngp))
            self.Ipolicy = np.empty((self.Angp, self.Kngp), dtype=np.bool_)
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


    # Getter function for llambda
    @property
    def llambda(self):
        return self._llambda

    # Setter function for llambda
    @llambda.setter
    def llambda(self, value):
        self._llambda = value
        self.KStarReady['llambda'] = True
        self.setKStar()


    # Getter function for F
    @property
    def F(self):
        return self._F

    # Setter function for F
    @F.setter
    def F(self, value):
        self._F = value
        self.KStarReady['F'] = True
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



    # One period profit function
    def onePeriodProfit(self, A, K, K1, invest):
        profit = A * (K**self.alpha)
        if (invest == True):
            profit = (self.llambda * profit) - (self.F * K) - (self.p * (K1 - (1.0 - self.delta)*K)) 
        return(profit)


    # Initial guess for EV
    def guessSln(self):

        V = np.empty((self.Angp, self.Kngp))
        
        # Starting guess for value function is one-period profit function
        # First assume shock is known
        for ixA, A in enumerate(self.A):
            for ixK, K in enumerate(self.Kgrid):
                profitIfInvest = np.max(self.onePeriodProfit(A, K, self.Kgrid, invest=True))
                V[ixA,ixK] = max(profitIfInvest, self.onePeriodProfit(A, K, K*(1.0 - self.delta), invest=False))

        # Now integrate over uncertain shock
        self.EV = self.prob @ V

        # Note the ixK index in EV relates to K in the next period (the period to which it relates)
        # So if ixK = 6 this period and there is no investment, you should use self.EV[ixA, 5] to get the expected value next period
        # This is true except for the bottom of the grid: if ixK = 0 then use self.EV[ixA, 0] if there is no investment


    # Find next iteration
    def nextIter(self, EVNext):
    
        VNext = np.empty((self.Angp, self.Kngp))
    
        # Calculate value function assuming shock is known
        for ixA, A in enumerate(self.A):
            for ixK, K in enumerate(self.Kgrid):

                VIfInvest = np.max(self.onePeriodProfit(A, K, self.Kgrid, invest=True) + (self.beta * self.EV[ixA, :]))
                ixKnext = max(ixK - 1, 0)
                VIfNotInvest = self.onePeriodProfit(A, K, (1.0 - self.delta)*K, invest=False) + (self.beta * self.EV[ixA, ixKnext])

                VNext[ixA,ixK] = max(VIfInvest, VIfNotInvest)

        # Now integrate over uncertain shock
        np.copyto(EVNext, self.prob @ VNext)


    # Find policy function using solution
    def findPolicy(self):

        for ixA, A in enumerate(self.A):
            for ixK, K in enumerate(self.Kgrid):

                # Find value if invest
                objectiveIfInvest = self.onePeriodProfit(A, K, self.Kgrid, invest=True) + (self.beta * self.EV[ixA, :])
                ixK1OptIfInvest = np.argmax(objectiveIfInvest)
                VIfInvest = objectiveIfInvest[ixK1OptIfInvest]

                # Find value if not invest
                ixKnext = max(ixK - 1, 0)
                VIfNotInvest = self.onePeriodProfit(A, K, (1.0 - self.delta)*K, invest=False) + (self.beta * self.EV[ixA, ixKnext])

                # Find optimum
                if (VIfInvest > VIfNotInvest):
                    self.Ipolicy[ixA, ixK] = True
                    self.ixK1policy[ixA, ixK] = ixK1OptIfInvest
                else:
                    self.Ipolicy[ixA, ixK] = False
                    self.ixK1policy[ixA, ixK] = ixKnext


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

        return sims

