
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from tauchen import approx_markov


class Model:

    def __init__(self, beta=0.95, lnb=-0.2, sigma=1.0, wngp=7):

        self.tol = 0.0001

        self.wReady = {'sigma':False, 'wngp':False}
        self.slnReady = {'wngp':False}

        self.beta = beta
        self.b = np.exp(lnb)
        self.sigma = sigma
        self.wngp = wngp


    # Function to calculate grid on w
    def setwGrid(self):
        if all([val for val in self.wReady.values()]):
            self.w, self.prob = approx_markov(rho=0.0, sigma_u=self.sigma, m=3, n=self.wngp)
            self.w = np.exp(self.w)

    # Function to allocate solution matrices
    def allocateSln(self):
        if all([val for val in self.slnReady.values()]):
            self.EV = np.empty((self.wngp))
            self.ixpolicy = np.empty((self.wngp), dtype=np.bool)


    # Getter function for sigma
    @property
    def sigma(self):
        return self._sigma

    # Setter function for sigma
    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        self.wReady['sigma'] = True
        self.setwGrid()

    # Getter function for wngp
    @property
    def wngp(self):
        return self._wngp

    # Setter function for wngp
    @wngp.setter
    def wngp(self, value):
        self._wngp = value
        self.wReady['wngp'] = True
        self.setwGrid()
        self.slnReady['wngp'] = True
        self.allocateSln()



    # utility function
    def util(self, w):
        return np.log(w)

    # Utility of accepting today
    def utilAccept(self, w):
        return self.util(w) / (1.0 - self.beta)

    # Initial guess for V: just accept today
    def guessSln(self):

        V = np.empty((self.wngp))

        # Calculate value function assuming wage draw is known and job is accepted
        V[:] = self.utilAccept(self.w)

        # Now integrate over uncertain wage draw
        self.EV = self.prob @ V


    # Find next iteration
    def nextIter(self, EVNext):
    
        VNext = np.empty(self.wngp)
    
        # Calculate value of reject
        Vreject = self.util(self.b) + (self.beta * self.EV)

        # Calculate value of accept
        Vaccept = self.utilAccept(self.w)

        # Calculate value function
        VNext = np.maximum(Vaccept, Vreject)

        # Now integrate over uncertain wage draw
        np.copyto(EVNext, self.prob @ VNext)


    # Find policy function using solution
    def findPolicy(self):

        # Find accept/reject
        self.ixpolicy[:] = (self.utilAccept(self.w) > self.util(self.b) + (self.beta * self.EV))


    # Solution
    def solve(self, iter=1000, verbose=False):

        self.guessSln()

        EVNext = np.empty((self.wngp))

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
        ax.plot(self.w, np.maximum(self.utilAccept(self.w), self.util(self.b) + self.beta*self.EV))
        ax.set_xlabel('w', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()


    # Plot optimal decision function
    def plotPolicy(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.w, self.ixpolicy.astype(float))
        ax.set_xlabel('w', fontsize=14)
        ax.set_ylabel('Optimal decision', fontsize=14)
        ax.set_title('Policy function for accept/reject')
        plt.show()


