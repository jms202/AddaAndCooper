# Next steps:
# Check functions to draw shocks and simulate are right
# Find Euler equations
# Estimate Euler equations on simulated data
# Add non-separability in utility function and re-estimate Euler equation


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from tauchen import approx_markov


class Model:

    def __init__(self, beta=0.97, gamma=1.5, delta=0.1, rho=0.9, sigma_u=0.1, phi=0.9, sigma_v=0.1, R=1.025, WMax=100.0, Wngp=7, DMax=100.0, Dngp=7, yngp=7, pngp=3):

        self.isSet = {'rho':False, 'sigma_u':False, 'yngp':False}
        self.isSet2 = {'phi':False, 'sigma_v':False, 'pngp':False}
        self.isSet3 = {'Wngp':False, 'Dngp':False, 'yngp':False, 'pngp':False}
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.R = R
        self.tol = 0.0001
        self.rho = rho
        self.sigma_u = sigma_u
        self.yngp = yngp
        self.phi = phi
        self.sigma_v = sigma_v
        self.pngp = pngp
        self.Wngp = Wngp
        self.WMax = WMax
        self.Wgrid = np.linspace(self.WMin, self.WMax, self.Wngp)
        self.Dngp = Dngp
        self.DMax = DMax
        self.DMin = 0.0
        self.Dgrid = np.linspace(self.DMin, self.DMax, self.Dngp)
        self.cMax = self.WMax
        self.a = 1.5
        self.d = 0.1
        self.EV = np.empty((self.Dngp, self.Wngp, self.pngp, self.yngp))
        self.ixD1policy = np.empty((self.Dngp, self.Wngp, self.pngp, self.yngp), dtype=np.int32)
        self.ixW1policy = np.empty((self.Dngp, self.Wngp, self.pngp, self.yngp), dtype=np.int32)





    # Getter function for rho
    @property
    def rho(self):
        return self._rho

    # Setter function for rho
    @rho.setter
    def rho(self, value):
        self._rho = value
        self.isSet['rho'] = True
        if all([val for val in self.isSet.values()]):
            self.y, self.yprob = approx_markov(rho=self.rho, sigma_u=self.sigma_u, m=3, n=self.yngp)
            self.y = np.exp(self.y)
            # WMin = -yMin - yMin/R - yMin/(R**2) - ... = -yMin*R / (R - 1)
            self.WMin = -self.y[0]*self.R / (self.R - 1.0)

    # Getter function for sigma_u
    @property
    def sigma_u(self):
        return self._sigma_u

    # Setter function for sigma_u
    @sigma_u.setter
    def sigma_u(self, value):
        self._sigma_u = value
        self.isSet['sigma_u'] = True
        if all([val for val in self.isSet.values()]):
            self.y, self.yprob = approx_markov(rho=self.rho, sigma_u=self.sigma_u, m=3, n=self.yngp)
            self.y = np.exp(self.y)
            # WMin = -yMin - yMin/R - yMin/(R**2) - ... = -yMin*R / (R - 1)
            self.WMin = -self.y[0]*self.R / (self.R - 1.0)

    # Getter function for yngp
    @property
    def yngp(self):
        return self._yngp

    # Setter function for yngp
    @yngp.setter
    def yngp(self, value):
        self._yngp = value
        self.isSet['yngp'] = True
        self.isSet3['yngp'] = True
        if all([val for val in self.isSet.values()]):
            self.y, self.yprob = approx_markov(rho=self.rho, sigma_u=self.sigma_u, m=3, n=self.yngp)
            self.y = np.exp(self.y)
            # WMin = -yMin - yMin/R - yMin/(R**2) - ... = -yMin*R / (R - 1)
            self.WMin = -self.y[0]*self.R / (self.R - 1.0)
        if all([val for val in self.isSet3.values()]):
            self.EV = np.empty((self.Dngp, self.Wngp, self.pngp, self.yngp))
            self.policy = np.empty((self.Dngp, self.Wngp, self.pngp, self.yngp))


    # Getter function for phi
    @property
    def phi(self):
        return self._phi

    # Setter function for phi
    @phi.setter
    def phi(self, value):
        self._phi = value
        self.isSet2['phi'] = True
        if all([val for val in self.isSet2.values()]):
            self.p, self.pprob = approx_markov(rho=self.phi, sigma_u=self.sigma_v, m=3, n=self.pngp)
            self.p = np.exp(self.p)

    # Getter function for sigma_v
    @property
    def sigma_v(self):
        return self._sigma_v

    # Setter function for sigma_v
    @sigma_v.setter
    def sigma_v(self, value):
        self._sigma_v = value
        self.isSet2['sigma_v'] = True
        if all([val for val in self.isSet2.values()]):
            self.p, self.pprob = approx_markov(rho=self.phi, sigma_u=self.sigma_v, m=3, n=self.pngp)
            self.p = np.exp(self.p)

    # Getter function for pngp
    @property
    def pngp(self):
        return self._pngp

    # Setter function for pngp
    @pngp.setter
    def pngp(self, value):
        self._pngp = value
        self.isSet2['pngp'] = True
        self.isSet3['pngp'] = True
        if all([val for val in self.isSet2.values()]):
            self.p, self.pprob = approx_markov(rho=self.phi, sigma_u=self.sigma_v, m=3, n=self.pngp)
            self.p = np.exp(self.p)
        if all([val for val in self.isSet3.values()]):
            self.EV = np.empty((self.Dngp, self.Wngp, self.pngp, self.yngp))
            self.policy = np.empty((self.Dngp, self.Wngp, self.pngp, self.yngp))




    # Getter function for Dngp
    @property
    def Dngp(self):
        return self._Dngp

    # Setter function for Dngp
    @Dngp.setter
    def Dngp(self, value):
        self._Dngp = value
        self.isSet3['Dngp'] = True
        if all([val for val in self.isSet3.values()]):
            self.EV = np.empty((self.Dngp, self.Wngp, self.pngp, self.yngp))
            self.policy = np.empty((self.Dngp, self.Wngp, self.pngp, self.yngp))



    # Getter function for Wngp
    @property
    def Wngp(self):
        return self._Wngp

    # Setter function for Wngp
    @Wngp.setter
    def Wngp(self, value):
        self._Wngp = value
        self.isSet3['Wngp'] = True
        if all([val for val in self.isSet3.values()]):
            self.EV = np.empty((self.Dngp, self.Wngp, self.pngp, self.yngp))
            self.policy = np.empty((self.Dngp, self.Wngp, self.pngp, self.yngp))



    # Quadratic utility
    def quadraticUtil(self, c, D, DNext):
        return -0.5*((self.cMax - c)**2) - 0.5*self.a*((self.DMax - D)**2) - 0.5*self.d*((DNext - D)**2)



    # Initial guess for EV
    def guessSln(self):

        V = np.empty((self.Dngp, self.Wngp, self.pngp, self.yngp))

        # Calculate value function assuming shocks are known and that everything is consumed this period
        for ixD, D in enumerate(self.Dgrid):
            for ixW, W in enumerate(self.Wgrid):
                for ixp, p in enumerate(self.p):
                    for ixy, y in enumerate(self.y):
                        c = W + y + p*(1.0 - self.delta)*D
                        V[ixD, ixW, ixp, ixy] = self.quadraticUtil(c, D, 0.0)

        # Now integrate over uncertain shocks
        # See https://www.tutorialspoint.com/numpy/numpy_matmul.htm for how N-dimensional multiplication works
        # (self.Dngp, self.Wngp, self.pngp, self.yngp)
        self.EV[:,:,:,:] = self.pprob @ V @ self.yprob.T


    # Find next iteration
    def nextIter(self, EVNext):
    
        VNext = np.empty((self.Dngp, self.Wngp, self.pngp, self.yngp))
    
        # Calculate value function assuming shock is known
        for ixD, D in enumerate(self.Dgrid):
            for ixW, W in enumerate(self.Wgrid):
                for ixp, p in enumerate(self.p):
                    for ixy, y in enumerate(self.y):

                        # For each value of (p, y, D, W), work out optimal Dnext and Wnext
                        # 2D grids over Wnext and Dnext
                        Wnext = np.repeat(self.Wgrid[None,:], repeats=self.Dngp, axis=0)
                        Dnext = np.repeat(self.Dgrid[:,None], repeats=self.Wngp, axis=1)
                        # c implied by a 2D grid over D and W
                        c = W + y - (Wnext/self.R) - p*(Dnext - (1.0 - self.delta)*D)
                        objective = self.quadraticUtil(c, D, Dnext) + (self.beta*self.EV[:, :, ixp, ixy])
                        # Coords of optimal choice:
                        ixOpt = np.unravel_index(np.argmax(objective), objective.shape)
                        # Value at optimal choice
                        VNext[ixD, ixW, ixp, ixy] = objective[ixOpt]

        # Now integrate over uncertain shocks
        # (self.Dngp, self.Wngp, self.pngp, self.yngp)
        EVNext[:,:,:,:] = self.pprob @ VNext @ self.yprob.T


    # Find policy function using solution
    def findPolicy(self):

        for ixD, D in enumerate(self.Dgrid):
            for ixW, W in enumerate(self.Wgrid):
                for ixp, p in enumerate(self.p):
                    for ixy, y in enumerate(self.y):

                        # For each value of (p, y, D, W), work out optimal Dnext and Wnext
                        # 2D grids over Wnext and Dnext ("None" adds an extra axis)
                        Wnext = np.repeat(self.Wgrid[None,:], repeats=self.Dngp, axis=0)
                        Dnext = np.repeat(self.Dgrid[:,None], repeats=self.Wngp, axis=1)
                        # c implied by a 2D grid over D and W
                        c = W + y - (Wnext/self.R) - p*(Dnext - (1.0 - self.delta)*D)
                        objective = self.quadraticUtil(c, D, Dnext) + (self.beta*self.EV[:, :, ixp, ixy])
                        # Coords of optimal choice:
                        ixOpt = np.unravel_index(np.argmax(objective), objective.shape)

                        self.ixD1policy[ixD, ixW, ixp, ixy] = ixOpt[0]
                        self.ixW1policy[ixD, ixW, ixp, ixy] = ixOpt[1]


    # Solution
    def solve(self, iter=1000, verbose=False):

        self.guessSln()

        EVNext = np.empty((self.Dngp, self.Wngp, self.pngp, self.yngp))

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



    # Plot value function with W on horizontal axis
    def plotValueW(self, ixp=0, ixy=0):
    
        fig, ax = plt.subplots()
        ax.plot(self.Wgrid, self.EV[:,:,ixp, ixy].T)
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()


    # Plot value function with D on horizontal axis
    def plotValueD(self, ixp=0, ixy=0):
    
        fig, ax = plt.subplots()
        ax.plot(self.Dgrid, self.EV[:,:,ixp, ixy])
        ax.set_xlabel('D', fontsize=14)
        ax.set_ylabel('V', fontsize=14)
        ax.set_title('Value function')
        plt.show()


    # Plot W1 policy function
    def plotW1Policy(self, ixp=0, ixy=0):
    
        fig, ax = plt.subplots()
        ax.plot(self.Wgrid, self.Wgrid[self.ixW1policy[:,:,ixp, ixy]].T)
        ax.set_xlabel('W', fontsize=14)
        ax.set_ylabel('Optimal W1', fontsize=14)
        ax.set_title('Policy function for W1')
        plt.show()


    # Plot D1 policy function
    def plotD1Policy(self, ixp=0, ixy=0):
    
        fig, ax = plt.subplots()
        ax.plot(self.Dgrid, self.Dgrid[self.ixD1policy[:,:,ixp, ixy]])
        ax.set_xlabel('D', fontsize=14)
        ax.set_ylabel('Optimal D1', fontsize=14)
        ax.set_title('Policy function for D1')
        plt.show()


    def drawShocks(self, n=1000, horiz=100):
        
        sims = {'ixp':np.empty((n, horiz), dtype=np.int32), 'ixy':np.empty((n, horiz), dtype=np.int32)}
        sims['ixp'][:, 0] = np.random.choice(self.pngp, size=n)
        sims['ixy'][:, 0] = np.random.choice(self.yngp, size=n)

        for i in range(n):
            for h in range(1, horiz):

                sims['ixp'][i, h] = np.random.choice(self.pngp, p = self.pprob[sims['ixp'][i, h-1], :])
                sims['ixy'][i, h] = np.random.choice(self.yngp, p = self.yprob[sims['ixy'][i, h-1], :])

        return sims


    # Simulate the consumption and labour choices of a number of individuals
    def simulate(self, sims, n=1000, horiz=100):

        sims['ixD'] = np.empty((n, horiz), dtype=np.int32)
        sims['ixW'] = np.empty((n, horiz), dtype=np.int32)
        sims['ixD1'] = np.empty((n, horiz), dtype=np.int32)
        sims['ixW1'] = np.empty((n, horiz), dtype=np.int32)

        for i in range(n):

            sims['ixD'][i, 0] = np.random.choice(range(self.Dngp))
            sims['ixW'][i, 0] = np.random.choice(range(self.Wngp))
            sims['ixD1'][i, 0] = self.ixD1policy[sims['ixD'][i, 0], sims['ixW'][i, 0], sims['ixp'][i, 0], sims['ixy'][i, 0]]
            sims['ixW1'][i, 0] = self.ixW1policy[sims['ixD'][i, 0], sims['ixW'][i, 0], sims['ixp'][i, 0], sims['ixy'][i, 0]]

            for h in range(1, horiz):

                # Copy forward optimal D and W
                sims['ixD'][i, h] = sims['ixD1'][i, h-1]
                sims['ixW'][i, h] = sims['ixW1'][i, h-1]
                # Calculate optimal D1 and W1'
                sims['ixD1'][i, h] = self.ixD1policy[sims['ixD'][i, h], sims['ixW'][i, h], sims['ixp'][i, h], sims['ixy'][i, h]]
                sims['ixW1'][i, h] = self.ixW1policy[sims['ixD'][i, h], sims['ixW'][i, h], sims['ixp'][i, h], sims['ixy'][i, h]]
        

        # Now calculate other variables
        sims['p'] = self.p[sims['ixp']]
        sims['y'] = self.y[sims['ixy']]
        sims['D'] = self.Dgrid[sims['ixD']]
        sims['W'] = self.Wgrid[sims['ixW']]
        sims['D1'] = self.Dgrid[sims['ixD1']]
        sims['W1'] = self.Wgrid[sims['ixW1']]
        sims['e'] = sims['D1'] - (1.0 - self.delta)*sims['D']
        sims['c'] = sims['W'] + sims['y'] - (sims['W1']/self.R) - sims['p']*sims['e']

        return sims



