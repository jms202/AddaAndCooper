
# This doesn't iterate properly towards a solution: it gets stuck in a loop


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

    def __init__(self, alpha=0.7, beta=0.95, w=0.05, w0=1.5, w1=0.19, w2=0.03, gammaPlus=0.05, gammaMinus=-0.05, rho=0.8, sigma=0.1, engp=7, Angp=7, hngp=7):

        self.tol = 0.0001

        self.eReady = {'engp':False}
        self.hReady = {'hngp':False}
        self.AReady = {'rho':False, 'sigma':False, 'Angp':False}
        self.slnReady = {'engp':False, 'Angp':False}

        self.alpha = alpha
        self.beta = beta
        self.w = w
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.gammaPlus = gammaPlus
        self.gammaMinus = gammaMinus
        self.rho = rho
        self.sigma = sigma
        self.Angp = Angp
        self.eMin = 1.0
        self.eMax = 500.0
        self.engp = engp
        self.hMin = 1.0
        self.hMax = 40.0
        self.hngp = hngp


    # Function to calculate grid on A
    def setAGrid(self):
        if all([val for val in self.AReady.values()]):
            self.A, self.prob = approx_markov(rho=self.rho, sigma_u=self.sigma, m=3, n=self.Angp)
            self.A = np.exp(self.A)
            self.AMean = np.dot(solveStationary(self.prob), self.A)

    # Function to calculate grid on e
    def seteGrid(self):
        if all([val for val in self.eReady.values()]):
            self.egrid = np.linspace(self.eMin, self.eMax, self.engp)

    # Function to calculate grid on h
    def sethGrid(self):
        if all([val for val in self.hReady.values()]):
            self.hgrid = np.linspace(self.hMin, self.hMax, self.hngp)

    # Function to allocate solution matrices
    def allocateSln(self):
        if all([val for val in self.slnReady.values()]):
            self.EV = np.empty((self.Angp, self.engp))
            self.ixepolicy = np.empty((self.Angp, self.engp), dtype=np.int32)
            self.ixhpolicy = np.empty((self.Angp, self.engp), dtype=np.int32)



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

    # Getter function for engp
    @property
    def engp(self):
        return self._engp

    # Setter function for engp
    @engp.setter
    def engp(self, value):
        self._engp = value
        self.eReady['engp'] = True
        self.seteGrid()
        self.slnReady['engp'] = True
        self.allocateSln()

    # Getter function for hngp
    @property
    def hngp(self):
        return self._hngp

    # Setter function for hngp
    @hngp.setter
    def hngp(self, value):
        self._hngp = value
        self.hReady['hngp'] = True
        self.sethGrid()


    # Revenue function
    def revenue(self, A, e, h):
        return A * ((e * h)**self.alpha)

    # Compensation function
    def compensation(self, e, h):
        return self.w * e * (self.w0 + h + (self.w1 * (h - 40.0)) + (self.w2 * ((h - 40.0)**2)))

    # Adjustment costs function
    def adjCost(self, e, Le):
        De = e - Le
        cost = np.where(De >= 0.0, self.gammaPlus * De, self.gammaMinus * De)
        return cost

    # Combine for period return function
    def periodReturn(self, A, Le, e, h):
        return self.revenue(A, e, h) - self.compensation(e, h) - self.adjCost(e, Le)


    # Initial guess for V
    def guessSln(self):

        V = np.empty((self.Angp, self.engp))

        emesh, hmesh = np.meshgrid(self.egrid, self.hgrid)

        # Calculate value function assuming shock is known and capital is chosen to maximise one-period profit
        for ixA, A in enumerate(self.A):
            for ixLe, Le in enumerate(self.egrid):
                V[ixA,ixLe] = np.max(self.periodReturn(A, Le, emesh, hmesh))

        # Now integrate over uncertain shock
        self.EV = self.prob @ V



    # Find next iteration
    def nextIter(self, EVNext):
    
        VNext = np.empty((self.Angp, self.engp))
    
        emesh, hmesh = np.meshgrid(self.egrid, self.hgrid)

        # Calculate value function assuming shock is known
        for ixA, A in enumerate(self.A):
            for ixLe, Le in enumerate(self.egrid):

                EVmesh = np.repeat(self.EV[np.newaxis, ixA, :], self.hngp, axis=0)

                # For each value of (A, Le), work out optimal e
                objective = self.periodReturn(A, Le, emesh, hmesh) + (self.beta*EVmesh)
                VNext[ixA,ixLe] = np.max(objective)

        # Now integrate over uncertain shock
        np.copyto(EVNext, self.prob @ VNext)


    # Find policy function using solution
    def findPolicy(self):

        emesh, hmesh = np.meshgrid(self.egrid, self.hgrid)

        for ixA, A in enumerate(self.A):
            for ixLe, Le in enumerate(self.egrid):

                EVmesh = np.repeat(self.EV[np.newaxis, ixA, :], self.hngp, axis=0)

                # Find optimal e and optimal h
                objective = self.periodReturn(A, Le, emesh, hmesh) + (self.beta*EVmesh)
                maxpos = np.unravel_index(np.argmax(objective), emesh.shape)
                self.ixhpolicy[ixA, ixLe] = maxpos[0]
                self.ixepolicy[ixA, ixLe] = maxpos[1]


    # Solution
    def solve(self, iter=1000, verbose=False):

        self.guessSln()

        EVNext = np.empty((self.Angp, self.engp))

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
        ax.plot(self.egrid, self.EV.T)
        ax.set_xlabel('e', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()


    # Plot employment policy function
    def plotePolicy(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.egrid, self.egrid[self.ixepolicy.T])
        ax.set_xlabel('Le', fontsize=14)
        ax.set_ylabel('Optimal e', fontsize=14)
        ax.set_title('Policy function')
        plt.show()

    # Plot hours policy function
    def plothPolicy(self):
    
        fig, ax = plt.subplots()
        ax.plot(self.egrid, self.hgrid[self.ixhpolicy.T])
        ax.set_xlabel('Le', fontsize=14)
        ax.set_ylabel('Optimal h', fontsize=14)
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

        sims['ixLe'] = np.empty((n, horiz), dtype=np.int32)
        sims['ixe'] = np.empty((n, horiz), dtype=np.int32)
        sims['ixh'] = np.empty((n, horiz), dtype=np.int32)

        for i in range(n):

            sims['ixLe'][i, 0] = np.random.choice(range(self.engp))
            sims['ixe'][i, 0] = self.ixepolicy[sims['ixA'][i, 0], sims['ixLe'][i, 0]]
            sims['ixh'][i, 0] = self.ixhpolicy[sims['ixA'][i, 0], sims['ixLe'][i, 0]]

            for h in range(1, horiz):

                # Copy forward optimal Le
                sims['ixLe'][i, h] = sims['ixe'][i, h-1]
                # Calculate optimal e
                sims['ixe'][i, h] = self.ixepolicy[sims['ixA'][i, h], sims['ixLe'][i, h]]
                # Calculate optimal h
                sims['ixh'][i, h] = self.ixhpolicy[sims['ixA'][i, h], sims['ixLe'][i, h]]
        

        # Now calculate other variables
        sims['A'] = self.A[sims['ixA']]
        sims['Le'] = self.egrid[sims['ixLe']]
        sims['e'] = self.egrid[sims['ixe']]
        sims['h'] = self.hgrid[sims['ixh']]
        sims['EV'] = self.EV[sims['ixA'], sims['ixe']]

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

    moments = np.empty(3)

    # Std dev of hours relative to employment
    moments[0] = np.average(np.std(data['h'][:, burn:], axis = 1)) / np.average(np.std(data['e'][:, burn:], axis = 1))

    # Serial correlation of hours
    moments[1] = corravg(data['h'][:, burn+1:], data['h'][:, burn:-1])
    # Serial correlation of employment
    moments[2] = corravg(data['e'][:, burn+1:], data['e'][:, burn:-1])

    return moments




def objective(theta, targetMoments, model, sims):

    # Solve model, simulate data and calculate moments
    modelMoments = solveSimMoments(theta, model, sims)

    # Calculate objective
    return (np.sum(((modelMoments - targetMoments) / targetMoments) ** 2))


def solveSimMoments(theta, model, sims):

    # Copy parameters from theta into model
    model.w2 = theta[0]
    model.gammaPlus = theta[1]
    model.gammaMinus = -theta[1]

    # Solve
    model.solve()

    # Simulate
    sims = model.simulate(sims)

    # Calculate moments
    modelMoments = calcMoments(sims)

    return(modelMoments)




def callback(xk):

    print(xk)



