
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

    def __init__(self, beta=0.95, p=1.0, R=0.9, gamma=2.0, rho=0.8, sigma=0.1, sngp=7, scale=50, Ingp=7):

        self.tol = 0.0001

        self.etaReady = {'p':False, 'gamma':False, 'sMean':False}
        self.IReady = {'Ingp':False, 'scale':False, 'sMean':False}
        self.sReady = {'rho':False, 'sigma':False, 'sngp':False, 'scale':False}
        self.slnReady = {'sngp':False, 'Ingp':False}

        self.beta = beta
        self.p = p
        self.R = R
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.sngp = sngp
        self.scale = scale
        self.IMin = 0.0
        self.Ingp = Ingp
        self.yMin = 0.01


    # Function to suggest eta
    def setEta(self):
        if all([val for val in self.etaReady.values()]):
            self.eta = self.suggestEta()

    # Function to calculate grid on s
    def setsGrid(self):
        if all([val for val in self.sReady.values()]):
            self.s, self.prob = approx_markov(rho=self.rho, sigma_u=self.sigma, m=3, n=self.sngp)
            self.s = np.exp(self.s) * self.scale
            self.sMean = np.dot(solveStationary(self.prob), self.s)
            self.IReady['sMean'] = True
            self.setIGrid()
            self.etaReady['sMean'] = True
            self.setEta()

    # Function to calculate grid on I
    def setIGrid(self):
        if all([val for val in self.IReady.values()]):
            self.Igrid = np.linspace(self.IMin, self.sMean * 2.0, self.Ingp)

    # Function to allocate solution matrices
    def allocateSln(self):
        if all([val for val in self.slnReady.values()]):
            self.EV = np.empty((self.sngp, self.Ingp))
            self.ixI1policy = np.empty((self.sngp, self.Ingp), dtype=np.int32)


    # Getter function for p
    @property
    def p(self):
        return self._p

    # Setter function for p
    @p.setter
    def p(self, value):
        self._p = value
        self.etaReady['p'] = True
        self.setEta()

    # Getter function for gamma
    @property
    def gamma(self):
        return self._gamma

    # Setter function for gamma
    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self.etaReady['gamma'] = True
        self.setEta()

    # Getter function for rho
    @property
    def rho(self):
        return self._rho

    # Setter function for rho
    @rho.setter
    def rho(self, value):
        self._rho = value
        self.sReady['rho'] = True
        self.setsGrid()

    # Getter function for sigma
    @property
    def sigma(self):
        return self._sigma

    # Setter function for sigma
    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        self.sReady['sigma'] = True
        self.setsGrid()

    # Getter function for sngp
    @property
    def sngp(self):
        return self._sngp

    # Setter function for sngp
    @sngp.setter
    def sngp(self, value):
        self._sngp = value
        self.sReady['sngp'] = True
        self.setsGrid()
        self.slnReady['sngp'] = True
        self.allocateSln()

    # Getter function for scale
    @property
    def scale(self):
        return self._scale

    # Setter function for scale
    @scale.setter
    def scale(self, value):
        self._scale = value
        self.sReady['scale'] = True
        self.setsGrid()
        self.IReady['scale'] = True
        self.setIGrid()

    # Getter function for Ingp
    @property
    def Ingp(self):
        return self._Ingp

    # Setter function for Ingp
    @Ingp.setter
    def Ingp(self, value):
        self._Ingp = value
        self.IReady['Ingp'] = True
        self.setIGrid()
        self.slnReady['Ingp'] = True
        self.allocateSln()



    # Revenue function
    def revenue(self, s):
        return self.p * s

    # Cost function
    def cost(self, y):
        return self.eta * (y**self.gamma)

    # Work out a sensible value for eta given the value of sMean
    def suggestEta(self):
        return (self.p / self.gamma) * (self.sMean ** (1.0 - self.gamma))

    # Output/sales level at which MR = MC
    def equilibrium(self):
        return (self.p / (self.eta * self.gamma)) ** (1.0 / (self.gamma - 1.0))

    # Combine for period return function
    def periodReturn(self, s, y):
        return np.where(y < self.yMin, -1.0e100, self.revenue(s) - self.cost(y))


    # Initial guess for V
    def guessSln(self):

        V = np.empty((self.sngp, self.Ingp))

        # Calculate value function assuming shock is known and production is chosen to maximise one-period profit
        for ixs, s in enumerate(self.s):
            for ixI, I in enumerate(self.Igrid):
                V[ixs,ixI] = np.max(self.periodReturn(s, max(s - I, self.yMin)))

        # Now integrate over uncertain shock
        self.EV = self.prob @ V


    # Find next iteration
    def nextIter(self, EVNext):
    
        VNext = np.empty((self.sngp, self.Ingp))
    
        # Calculate value function assuming shock is known
        for ixs, s in enumerate(self.s):
            for ixI, I in enumerate(self.Igrid):

                # For each value of (s, I), work out optimal y
                yvals = (self.Igrid/self.R) + s - I
                yvalspos = np.where(yvals > self.yMin)[0]
                objective = self.periodReturn(s, yvals[yvalspos]) + (self.beta * self.EV[ixs, yvalspos])
                VNext[ixs,ixI] = np.max(objective)

        # Now integrate over uncertain shock
        np.copyto(EVNext, self.prob @ VNext)


    # Find policy function using solution
    def findPolicy(self):

        for ixs, s in enumerate(self.s):
            for ixI, I in enumerate(self.Igrid):

                # Find optimal y
                yvals = (self.Igrid/self.R) + s - I
                yvalspos = np.where(yvals > self.yMin)[0]
                objective = self.periodReturn(s, yvals[yvalspos]) + (self.beta * self.EV[ixs, yvalspos])
                maxpos = np.argmax(objective)
                self.ixI1policy[ixs, ixI] = yvalspos[maxpos]


    # Solution
    def solve(self, iter=1000, verbose=False):

        self.guessSln()

        EVNext = np.empty((self.sngp, self.Ingp))

        for ixiter in range(iter):

            self.nextIter(EVNext)

            # Check convergence of value function (we could also check convergence of marginal utility)
            maxpos = np.unravel_index(np.argmax(abs(self.EV - EVNext)), self.EV.shape)
            dist = abs(self.EV[maxpos] - EVNext[maxpos])
            if (verbose):
                print("%d dist %0.6f at position %s (%0.6f vs %0.6f)" % (ixiter, dist, maxpos, self.EV[maxpos], EVNext[maxpos]))
            if dist < self.tol:
                if (verbose):
                    print("Breaking on ixiter %d" % ixiter)
                break

            # Can't just assign because they are pointers
            np.copyto(self.EV, EVNext)

        self.findPolicy()


    # Plot value function
    def plotValue(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.Igrid, self.EV.T)
        ax.set_xlabel('I', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()


    # Plot future inventory policy function
    def plotI1Policy(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.Igrid, self.Igrid[self.ixI1policy.T])
        ax.set_xlabel('I', fontsize=14)
        ax.set_ylabel('Optimal I1', fontsize=14)
        ax.set_title('Policy function for I1')
        plt.show()


    # Plot production policy function
    def plotyPolicy(self):
    
        fig, ax = plt.subplots()
        # In self.ixI1policy.T, the rows are I and the columns are s
        # Need to use y = (self.Igrid/self.R) + s - I)
        smesh = np.repeat(self.s[np.newaxis, :], self.Ingp, axis=0)
        Imesh = np.repeat(self.Igrid[:, np.newaxis], self.sngp, axis=1)
        ax.plot(self.Igrid, (self.Igrid[self.ixI1policy.T] / self.R) + smesh - Imesh)
        ax.set_xlabel('I', fontsize=14)
        ax.set_ylabel('Optimal y', fontsize=14)
        ax.set_title('Policy function for y')
        plt.show()



