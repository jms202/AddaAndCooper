
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

    def __init__(self, beta=0.95, p=1.0, R=0.9, gamma=2.0, rho_s=0.8, sigma_s=0.1, sngp=7, scale=50, Ingp=7, rho_A=0.8, sigma_A=0.1, Angp=7):

        self.tol = 0.0001

        self.etaReady = {'p':False, 'gamma':False, 'sMean':False, 'AMean':False}
        self.IReady = {'Ingp':False, 'scale':False, 'sMean':False}
        self.sReady = {'rho_s':False, 'sigma_s':False, 'sngp':False, 'scale':False}
        self.AReady = {'rho_A':False, 'sigma_A':False, 'Angp':False}
        self.slnReady = {'sngp':False, 'Angp':False, 'Ingp':False}

        self.beta = beta
        self.p = p
        self.R = R
        self.gamma = gamma
        self.rho_s = rho_s
        self.sigma_s = sigma_s
        self.sngp = sngp
        self.scale = scale
        self.rho_A = rho_A
        self.sigma_A = sigma_A
        self.Angp = Angp
        self.IMin = 0.0
        self.Ingp = Ingp
        self.yMin = 0.01


    # Function to set eta
    def setEta(self):
        if all([val for val in self.etaReady.values()]):
            self.eta = self.suggestEta()

    # Function to calculate grid on s
    def setsGrid(self):
        if all([val for val in self.sReady.values()]):
            self.s, self.prob = approx_markov(rho=self.rho_s, sigma_u=self.sigma_s, m=3, n=self.sngp)
            self.s = np.exp(self.s) * self.scale
            self.sMean = np.dot(solveStationary(self.prob), self.s)
            self.IReady['sMean'] = True
            self.setIGrid()
            self.etaReady['sMean'] = True
            self.setEta()

    # Function to calculate grid on A
    def setAGrid(self):
        if all([val for val in self.AReady.values()]):
            self.A, self.Aprob = approx_markov(rho=self.rho_A, sigma_u=self.sigma_A, m=3, n=self.Angp)
            self.A = np.exp(self.A)
            self.AMean = np.dot(solveStationary(self.Aprob), self.A)
            self.etaReady['AMean'] = True
            self.setEta()

    # Function to calculate grid on I
    def setIGrid(self):
        if all([val for val in self.IReady.values()]):
            self.Igrid = np.linspace(self.IMin, self.sMean * 2.0, self.Ingp)

    # Function to allocate solution matrices
    def allocateSln(self):
        if all([val for val in self.slnReady.values()]):
            self.EV = np.empty((self.Ingp, self.sngp, self.Angp))
            self.ixI1policy = np.empty((self.Ingp, self.sngp, self.Angp), dtype=np.int32)


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

    # Getter function for rho_s
    @property
    def rho_s(self):
        return self._rho_s

    # Setter function for rho_s
    @rho_s.setter
    def rho_s(self, value):
        self._rho_s = value
        self.sReady['rho_s'] = True
        self.setsGrid()

    # Getter function for sigma_s
    @property
    def sigma_s(self):
        return self._sigma_s

    # Setter function for sigma_s
    @sigma_s.setter
    def sigma_s(self, value):
        self._sigma_s = value
        self.sReady['sigma_s'] = True
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

    # Getter function for rho_s
    @property
    def rho_A(self):
        return self._rho_A

    # Setter function for rho_A
    @rho_A.setter
    def rho_A(self, value):
        self._rho_A = value
        self.AReady['rho_A'] = True
        self.setAGrid()

    # Getter function for sigma_A
    @property
    def sigma_A(self):
        return self._sigma_A

    # Setter function for sigma_A
    @sigma_A.setter
    def sigma_A(self, value):
        self._sigma_A = value
        self.AReady['sigma_A'] = True
        self.setAGrid()

    # Setter function for sigma_s
    @sigma_s.setter
    def sigma_s(self, value):
        self._sigma_s = value
        self.sReady['sigma_s'] = True
        self.setsGrid()

    # Getter function for Angp
    @property
    def Angp(self):
        return self._Angp

    # Setter function for Angp
    @sngp.setter
    def Angp(self, value):
        self._Angp = value
        self.AReady['Angp'] = True
        self.setAGrid()
        self.slnReady['Angp'] = True
        self.allocateSln()


    # Revenue function
    def revenue(self, s):
        return self.p * s

    # Cost function
    def cost(self, y, A):
        return self.eta * A * (y**self.gamma)

    # Work out a sensible value for eta given the value of sMean
    def suggestEta(self):
        return (self.p / (self.gamma * self.AMean)) * (self.sMean ** (1.0 - self.gamma))

    # Output/sales level at which MR = MC
    def equilibrium(self):
        return (self.p / (self.eta * self.AMean * self.gamma)) ** (1.0 / (self.gamma - 1.0))

    # Combine for period return function
    def periodReturn(self, s, A, y):
        return np.where(y < self.yMin, -1.0e100, self.revenue(s) - self.cost(y, A))



    # Initial guess for V
    def guessSln(self):

        V = np.empty((self.Ingp, self.sngp, self.Angp))

        # Calculate value function assuming shocks are known and production is chosen to maximise one-period profit
        for ixI, I in enumerate(self.Igrid):
            for ixs, s in enumerate(self.s):
                for ixA, A in enumerate(self.A):
                    V[ixI, ixs, ixA] = np.max(self.periodReturn(s, A, max(s - I, self.yMin)))

        # Now integrate over uncertain shock
        self.EV = self.prob @ V

        # Now integrate over uncertain shocks
        # See https://www.tutorialspoint.com/numpy/numpy_matmul.htm for how N-dimensional multiplication works
        # (self.Ingp, self.sngp, self.Angp)
        self.EV[:,:,:] = self.prob @ V @ self.Aprob.T


## ## ## HERE ## ## ##


    # Find next iteration
    def nextIter(self, EVNext):
    
        VNext = np.empty((self.Ingp, self.sngp, self.Angp))
    
        # Calculate value function assuming shocks are known
        for ixI, I in enumerate(self.Igrid):
            for ixs, s in enumerate(self.s):
                for ixA, A in enumerate(self.A):

                    # For each value of (s, I), work out optimal y
                    yvals = (self.Igrid/self.R) + s - I
                    yvalspos = np.where(yvals > self.yMin)[0]
                    objective = self.periodReturn(s, A, yvals[yvalspos]) + (self.beta * self.EV[yvalspos, ixs, ixA])
                    VNext[ixI, ixs, ixA] = np.max(objective)

        # Now integrate over uncertain shock
        np.copyto(EVNext, self.prob @ VNext @ self.Aprob.T)


    # Find policy function using solution
    def findPolicy(self):

        for ixI, I in enumerate(self.Igrid):
            for ixs, s in enumerate(self.s):
                for ixA, A in enumerate(self.A):

                    # Find optimal y
                    yvals = (self.Igrid/self.R) + s - I
                    yvalspos = np.where(yvals > self.yMin)[0]
                    objective = self.periodReturn(s, A, yvals[yvalspos]) + (self.beta * self.EV[yvalspos, ixs, ixA])
                    maxpos = np.argmax(objective)
                    self.ixI1policy[ixI, ixs, ixA] = yvalspos[maxpos]


    # Solution
    def solve(self, iter=1000, verbose=False):

        self.guessSln()

        EVNext = np.empty((self.Ingp, self.sngp, self.Angp))

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
    def plotValue(self, ixA):
    
        fig, ax = plt.subplots()
        ax.plot(self.Igrid, self.EV[:, :, ixA])
        ax.set_xlabel('I', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()


    # Plot future inventory policy function
    def plotI1Policy(self, ixA):
    
        fig, ax = plt.subplots()
        ax.plot(self.Igrid, self.Igrid[self.ixI1policy[:, :, ixA]])
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



