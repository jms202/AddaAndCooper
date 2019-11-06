# Is negative k1 possible?

import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import sys
#from quantecon import tauchen
from tauchen import approx_markov

def set_trace():
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)


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

    def __init__(self, alpha=0.75, beta=0.9, gamma=1.5, delta=0.3, rho=0.9, sigma_u=0.1, Angp=7, B=2, kRange=0.8, kngp=101, nngp=96):

        self.isSet = {'alpha':False, 'beta':False, 'delta':False, 'AMean':False}
        self.isSet2 = {'kMin':False, 'kMax':False, 'kngp':False}
        self.isSet3 = {'rho':False, 'sigma_u':False, 'Angp':False}
        self.isSet4 = {'Angp':False, 'kngp':False}
        self.nMin = 0.05
        self.nMax = 1.0
        self.nngp = nngp
        self.ngrid = np.linspace(self.nMin, self.nMax, self.nngp)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.B = B
        self.kRange = kRange
        self.kngp = kngp
        self.rho = rho
        self.sigma_u = sigma_u
        self.Angp = Angp
        self.tol = 0.0001
        self.cMin = 0.000001


    # Getter function for nngp
    @property
    def nngp(self):
        return self._nngp

    # Setter function for nngp
    @nngp.setter
    def nngp(self, value):
        self._nngp = value
        self.ngrid = np.linspace(self.nMin, self.nMax, self.nngp)


    # Getter function for alpha
    @property
    def alpha(self):
        return self._alpha

    # Setter function for alpha
    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        self.isSet['alpha'] = True
        if all([val for val in self.isSet.values()]):
            self.kStarMult = ((self.alpha*self.AMean / (1.0/self.beta - (1.0 - self.delta)))**(1.0 / (1.0 - self.alpha)))
            

    # Getter function for beta
    @property
    def beta(self):
        return self._beta

    # Setter function for beta
    @beta.setter
    def beta(self, value):
        self._beta = value
        self.isSet['beta'] = True
        if all([val for val in self.isSet.values()]):
            self.kStarMult = ((self.alpha*self.AMean / (1.0/self.beta - (1.0 - self.delta)))**(1.0 / (1.0 - self.alpha)))
            

    # Getter function for delta
    @property
    def delta(self):
        return self._delta

    # Setter function for delta
    @delta.setter
    def delta(self, value):
        self._delta = value
        self.isSet['delta'] = True
        if all([val for val in self.isSet.values()]):
            self.kStarMult = ((self.alpha*self.AMean / (1.0/self.beta - (1.0 - self.delta)))**(1.0 / (1.0 - self.alpha)))


    # Getter function for rho
    @property
    def rho(self):
        return self._rho

    # Setter function for rho
    @rho.setter
    def rho(self, value):
        self._rho = value
        self.isSet3['rho'] = True
        if all([val for val in self.isSet3.values()]):
            #mc = tauchen(rho=self.rho, sigma_u=self.sigma_u, b=0., m=3, n=self.Angp)
            #self.p = mc.P
            #self.A = np.exp(mc.state_values)
            self.A, self.p = approx_markov(rho=self.rho, sigma_u=self.sigma_u, m=3, n=self.Angp)
            self.A = np.exp(self.A)
            self.AMean = np.dot(solveStationary(self.p), self.A)
            

    # Getter function for sigma_u
    @property
    def sigma_u(self):
        return self._sigma_u

    # Setter function for sigma_u
    @sigma_u.setter
    def sigma_u(self, value):
        self._sigma_u = value
        self.isSet3['sigma_u'] = True
        if all([val for val in self.isSet3.values()]):
            #mc = tauchen(rho=self.rho, sigma_u=self.sigma_u, b=0., m=3, n=self.Angp)
            #self.p = mc.P
            #self.A = np.exp(mc.state_values)
            self.A, self.p = approx_markov(rho=self.rho, sigma_u=self.sigma_u, m=3, n=self.Angp)
            self.A = np.exp(self.A)
            self.AMean = np.dot(solveStationary(self.p), self.A)


    # Getter function for Angp
    @property
    def Angp(self):
        return self._Angp

    # Setter function for Angp
    @Angp.setter
    def Angp(self, value):
        self._Angp = value
        self.isSet3['Angp'] = True
        self.isSet4['Angp'] = True
        if all([val for val in self.isSet3.values()]):
            #mc = tauchen(rho=self.rho, sigma_u=self.sigma_u, b=0., m=3, n=self.Angp)
            #self.p = mc.P
            #self.A = np.exp(mc.state_values)
            self.A, self.p = approx_markov(rho=self.rho, sigma_u=self.sigma_u, m=3, n=self.Angp)
            self.A = np.exp(self.A)
            self.AMean = np.dot(solveStationary(self.p), self.A)
        if all([val for val in self.isSet4.values()]):
            self.currReturn = np.empty((self.Angp, self.kngp, self.kngp))
            self.ixnBest = np.empty((self.Angp, self.kngp, self.kngp))
            self.EV = np.zeros((self.Angp, self.kngp))
            self.ixk1policy = np.empty((self.Angp, self.kngp), dtype=np.int32)
            self.ixnpolicy = np.empty((self.Angp, self.kngp), dtype=np.int32)


    # Getter function for AMean
    @property
    def AMean(self):
        return self._AMean

    # Setter function for AMean
    @AMean.setter
    def AMean(self, value):
        self._AMean = value
        self.isSet['AMean'] = True
        if all([val for val in self.isSet.values()]):
            self.kStarMult = ((self.alpha*self.AMean / (1.0/self.beta - (1.0 - self.delta)))**(1.0 / (1.0 - self.alpha)))


    # Getter function for kngp
    @property
    def kngp(self):
        return self._kngp

    # Setter function for kngp
    @kngp.setter
    def kngp(self, value):
        self._kngp = value
        self.isSet2['kngp'] = True
        self.isSet4['kngp'] = True
        if all([val for val in self.isSet2.values()]):
            self.kgrid = np.linspace(self.kMin, self.kMax, self.kngp)
        if all([val for val in self.isSet4.values()]):
            self.currReturn = np.empty((self.Angp, self.kngp, self.kngp))
            self.ixnBest = np.empty((self.Angp, self.kngp, self.kngp))
            self.EV = np.zeros((self.Angp, self.kngp))
            self.ixk1policy = np.empty((self.Angp, self.kngp), dtype=np.int32)
            self.ixnpolicy = np.empty((self.Angp, self.kngp), dtype=np.int32)


    # Getter function for kStarMult
    @property
    def kStarMult(self):
        return self._kStarMult

    # Setter function for kStarMult
    @kStarMult.setter
    def kStarMult(self, value):
        self._kStarMult = value
        self.kStarMin = self.nMin * self.kStarMult
        self.kStarMax = self.nMax * self.kStarMult


    # Getter function for kStarMin
    @property
    def kStarMin(self):
        return self._kStarMin

    # Setter function for kStarMin
    @kStarMin.setter
    def kStarMin(self, value):
        self._kStarMin = value
        self.kMin = self.kStarMin*(1 - self.kRange)

        
    # Getter function for kStarMax
    @property
    def kStarMax(self):
        return self._kStarMax

    # Setter function for kStarMax
    @kStarMax.setter
    def kStarMax(self, value):
        self._kStarMax = value
        self.kMax = self.kStarMax*(1 + self.kRange)

    # Getter function for kMin
    @property
    def kMin(self):
        return self._kMin

    # Setter function for kMin
    @kMin.setter
    def kMin(self, value):
        self._kMin = value
        self.isSet2['kMin'] = True
        if all([val for val in self.isSet2.values()]):
            self.kgrid = np.linspace(self.kMin, self.kMax, self.kngp)

    # Getter function for kMax
    @property
    def kMax(self):
        return self._kMax

    # Setter function for kMax
    @kMax.setter
    def kMax(self, value):
        self._kMax = value
        self.isSet2['kMax'] = True
        if all([val for val in self.isSet2.values()]):
            self.kgrid = np.linspace(self.kMin, self.kMax, self.kngp)




    # Utility
    def util(self, c, n):
        return np.log(c) - self.B * (n**(1.0 + 1.0/self.gamma)) / (1.0 + 1.0/self.gamma)


    # Production function
    def prodFunc(self, A, k, n):
        return A * (k**self.alpha) * (n**(1.0 - self.alpha))


    # Find return function
    def findCurrReturn(self):

        self.currReturn[:,:,:] = -1.0e300

        for ixA, A in enumerate(self.A):
            for ixk, k in enumerate(self.kgrid):
                for ixk1, k1 in enumerate(self.kgrid):

                    # calculate consumption at each labour supply point
                    c = self.prodFunc(A, k, self.ngrid) + ((1.0 - self.delta) * k) - k1
                    try:
                        c = c[c>self.cMin]
                    except RuntimeWarning:
                        print(c)

                    # Break out if none of the consumption points are feasible
                    if len(c) == 0:
                        self.ixnBest[ixA, ixk, ixk1] = -1
                        continue
                    
                    # calculate current-period utility at each feasible labour supply point
                    u = self.util(c, self.ngrid[-len(c):])

                    # Store optimal labour supply index
                    ixnOpt = np.argmax(u)
                    self.ixnBest[ixA, ixk, ixk1] = ixnOpt + self.nngp - len(c)

                    # Store maximum
                    self.currReturn[ixA, ixk, ixk1] = u[ixnOpt]



    # Initial guess for EV
    def guessSln(self):

        # We already calculated the current return from all combinations of A, k and k1
        # Just need to integrate over uncertain shock
        self.EV = self.p @ self.currReturn[:,:,0]


    # Find next iteration
    def nextIter(self, EVNext):
    
        VNext = np.empty((self.Angp, self.kngp))
    
        # Calculate value function assuming shock is known
        for ixA in range(self.Angp):
            for ixk in range(self.kngp):

                # For each value of (A, k), work out optimal k1
                objective = self.currReturn[ixA, ixk, :] + (self.beta*self.EV[ixA, :])
                ixk1Opt = np.argmax(objective)
                VNext[ixA,ixk] = objective[ixk1Opt]


        # Now integrate over uncertain shock
        np.copyto(EVNext, self.p @ VNext)



    # Solution
    def solve(self, iter=1000, verbose=False):

        self.findCurrReturn()

        self.guessSln()

        EVNext = np.empty((self.Angp, self.kngp))

        for ixiter in range(iter):

            self.nextIter(EVNext)

            # Check convergence of value function (we could also check convergence of marginal utility)
            maxpos = np.unravel_index(np.argmax(abs(self.EV - EVNext)), self.EV.shape)
            dist = abs(self.EV[maxpos] - EVNext[maxpos])
            if (verbose == True):
                print("%d dist %0.6f at position %s (%0.6f vs %0.6f)" % (ixiter, dist, maxpos, self.EV[maxpos], EVNext[maxpos]))
                if dist < self.tol:
                    print("Breaking on ixiter %d" % ixiter)
                    break

            # Can't just assign because they are pointers
            np.copyto(self.EV, EVNext)

        self.findPolicy()



    # Find policy function using solution
    def findPolicy(self):

        for ixA in range(self.Angp):
            for ixk in range(self.kngp):

                # Find optimal K1
                objective = self.currReturn[ixA, ixk, :] + (self.beta*self.EV[ixA, :])
                ixk1Opt = np.argmax(objective)

                self.ixk1policy[ixA, ixk] = ixk1Opt
                self.ixnpolicy[ixA, ixk] = self.ixnBest[ixA, ixk, ixk1Opt]



    # Plot value function
    def plotValue(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.kgrid, self.EV.T)
        ax.set_xlabel('k', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()


    # Plot k1 policy function
    def plotk1Policy(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.kgrid, self.kgrid[self.ixk1policy.T])
        ax.set_xlabel('k', fontsize=14)
        ax.set_ylabel('Optimal k1', fontsize=14)
        ax.set_title('Policy function for k1')
        plt.show()


    # Plot n policy function
    def plotnPolicy(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.kgrid, self.ngrid[self.ixnpolicy.T])
        ax.set_xlabel('k', fontsize=14)
        ax.set_ylabel('Optimal n', fontsize=14)
        ax.set_title('Policy function for n')
        plt.show()



    def drawShocks(self, n=1000, horiz=100):
        
        sims = {'ixA':np.empty((n, horiz), dtype=np.int32)}
        sims['ixA'][:, 0] = np.random.choice(self.Angp, size=n)

        for i in range(n):
            for h in range(1, horiz):

                sims['ixA'][i, h] = np.random.choice(self.Angp, p = self.p[sims['ixA'][i, h-1], :])

        return sims




    # Simulate the consumption and labour choices of a number of individuals
    def simulate(self, sims, n=1000, horiz=100):

        sims['ixk'] = np.empty((n, horiz), dtype=np.int32)
        sims['ixn'] = np.empty((n, horiz), dtype=np.int32)
        sims['ixk1'] = np.empty((n, horiz), dtype=np.int32)

        for i in range(n):

            sims['ixk'][i, 0] = np.random.choice(range(self.kngp))
            sims['ixn'][i, 0] = np.random.choice(range(self.nngp))
            sims['ixk1'][i, 0] = self.ixk1policy[sims['ixA'][i, 0], sims['ixk'][i, 0]]

            for h in range(1, horiz):

                # Copy forward optimal k
                sims['ixk'][i, h] = sims['ixk1'][i, h-1]
                # Calculate optimal n
                sims['ixn'][i, h] = self.ixnpolicy[sims['ixA'][i, h], sims['ixk'][i, h]]
                # Calculate optimal k'
                sims['ixk1'][i, h] = self.ixk1policy[sims['ixA'][i, h], sims['ixk'][i, h]]
        

        # Now calculate other variables
        sims['A'] = self.A[sims['ixA']]
        sims['k'] = self.kgrid[sims['ixk']]
        sims['k1'] = self.kgrid[sims['ixk1']]
        sims['n'] = self.ngrid[sims['ixn']]

        sims['y'] = self.prodFunc(sims['A'], sims['k'], sims['n'])
        sims['i'] = sims['k1'] - (1.0 - self.delta) * sims['k']
        sims['c'] = sims['y'] - sims['i']
        sims['w'] = (1.0 - self.alpha) * sims['A'] * (( sims['k'] / sims['n'] )**self.alpha)

        return sims






# A function to calculate the average correlation between the corresponding rows of two matrices
def corravg(A, B):

    from scipy.stats.stats import pearsonr

    assert (A.shape == B.shape), "corravg error: A and B must have the same shape"

    n = A.shape[0]
    corr = np.empty(n)

    # Loop across simulations
    for i in range(n):
        # If no variation in either vector, set correlation to zero
        if ((np.all(A[i, :] == A[i, 0])) or (np.all(B[i, :] == B[i, 0]))):
            corr[i] = 0.0
        else:
            corr[i] = pearsonr(A[i, :], B[i, :])[0]
    
    return np.mean(corr)





def calcMoments(data, n=1000, horiz=100, burn=1):

    moments = np.empty(8)

    stddev_y = np.average(np.std(data['y'][:, burn:], axis = 1))

    # Std dev relative to output
    # consumption
    moments[0] = np.average(np.std(data['c'][:, burn:], axis = 1)) / stddev_y
    # investment
    moments[1] = np.average(np.std(data['i'][:, burn:], axis = 1)) / stddev_y
    # hours
    moments[2] = np.average(np.std(data['n'][:, burn:], axis = 1)) / stddev_y
    # wages
    moments[3] = np.average(np.std(data['w'][:, burn:], axis = 1)) / stddev_y

    # correlation with output
    # consumption
    moments[4] = corravg(data['y'][:, burn:], data['c'][:, burn:])
    # investment
    moments[5] = corravg(data['y'][:, burn:], data['i'][:, burn:])
    # hours
    moments[6] = corravg(data['y'][:, burn:], data['n'][:, burn:])
    # wages
    moments[7] = corravg(data['y'][:, burn:], data['w'][:, burn:])

    return moments





def objective(theta, targetMoments, model, sims):

    # Solve model, simulate data and calculate moments
    modelMoments = solveSimMoments(theta, model, sims)

    # Calculate objective
    return (np.sum(((modelMoments - targetMoments) / targetMoments) ** 2))




def solveSimMoments(theta, model, sims):

    # Copy parameters from theta into model
    model.alpha = theta[0]
    model.beta = theta[1]
    model.gamma = theta[2]
    model.delta = theta[3]
    model.rho = theta[4]
    model.sigma_u = theta[5]
    model.B = theta[6]

    # Solve
    model.solve()

    # Simulate
    sims = model.simulate(sims)

    # Calculate moments
    modelMoments = calcMoments(sims)

    return(modelMoments)




def callback(xk):

    print(xk)


