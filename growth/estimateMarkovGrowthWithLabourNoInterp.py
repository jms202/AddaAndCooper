
import numpy as np
from scipy.optimize import minimize

import warnings
warnings.filterwarnings("error")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


from markovGrowthWithLabourNoInterpTasteShocks import Model, objective, solveSimMoments, callback

# Simulate the truth
####################

# alpha, beta, gamma, delta, rho, sigma_u, B
truth = [0.75, 0.9, 1.5, 0.05, 0.9, 0.1, 2.]
model = Model()
sims = model.drawShocks()
targetMoments = solveSimMoments(truth, model, sims)

crash

# Try to recover the truth
##########################

sims = model.drawShocks()

guess = [0.6, 0.9, 1.5, 0.2, 0.9, 0.1, 1.8]
initial_simplex = np.array([[0.6, 0.9, 1.5, 0.2, 0.9, 0.1, 1.8],
                            [0.8, 0.9, 1.5, 0.2, 0.9, 0.1, 1.8],
                            [0.6, 0.95, 1.5, 0.2, 0.9, 0.1, 1.8],
                            [0.6, 0.9, 1.7, 0.2, 0.9, 0.1, 1.8],
                            [0.6, 0.9, 1.5, 0.3, 0.9, 0.1, 1.8],
                            [0.6, 0.9, 1.5, 0.2, 0.93, 0.1, 1.8],
                            [0.6, 0.9, 1.5, 0.2, 0.9, 0.13, 1.8],
                            [0.6, 0.9, 1.5, 0.2, 0.9, 0.1, 2.0]])
res = minimize(objective, x0=np.array(guess), args=(targetMoments, model, sims), method='Nelder-Mead', options={'initial_simplex':initial_simplex}, callback=callback)

finalMoments = solveSimMoments(res.x, model, sims)
print("finalMoments = ", finalMoments)

if (not res.success):
    raise ArithmeticError(res.message)
print(res)

crash



# Set up target moments
#######################

# US economy moments
targetMoments = [0.69, 1.35, 0.52, 1.14, 0.85, 0.6, 0.07, 0.76]



# Grid search for best guess
############################

bestObjective = 10e06
for alpha in np.linspace(0.05, 0.95, 5):
    for beta in [0.9, 0.95, 0.99]:
        for gamma in np.linspace(1.25, 2.0, 4):
            for delta in np.linspace(0.05, 0.45, 3):
                for rho in [0.5, 0.75, 0.9]:
                    for sigma_u in np.linspace(0.05, 0.15, 3):
                        for B in np.linspace(1.0, 5.0, 5):
                            print("Currently: alpha = %0.6f, beta = %0.6f, gamma = %0.6f, delta = %0.6f, rho = %0.6f, sigma_u = %0.6f, B = %0.6f" % (alpha, beta, gamma, delta, rho, sigma_u, B))
                            thisGuess = [alpha, beta, gamma, delta, rho, sigma_u, B]
                            thisObjective = objective(thisGuess, targetMoments, model, sims)
                            if (thisObjective < bestObjective):
                                bestGuess = thisGuess
                                bestObjective = thisObjective

print("bestGuess = ", bestGuess)
print("bestObjective = ", bestObjective)
print("bestMoments = ", solveSimMoments(bestGuess, model, sims))


initial_simplex = np.array([[0.6, 0.9, 1.5, 0.4, 0.9, 0.1, 1.8],
                            [0.8, 0.9, 1.5, 0.4, 0.9, 0.1, 1.8],
                            [0.6, 0.95, 1.5, 0.4, 0.9, 0.1, 1.8],
                            [0.6, 0.9, 1.7, 0.4, 0.9, 0.1, 1.8],
                            [0.6, 0.9, 1.5, 0.5, 0.9, 0.1, 1.8],
                            [0.6, 0.9, 1.5, 0.4, 0.93, 0.1, 1.8],
                            [0.6, 0.9, 1.5, 0.4, 0.9, 0.13, 1.8],
                            [0.6, 0.9, 1.5, 0.4, 0.9, 0.1, 2.0]])
# Use bestGuess as starting point for estimation
res = minimize(objective, x0=np.array(bestGuess), args=(targetMoments, model, sims), method='Nelder-Mead', options={'initial_simplex':initial_simplex}, callback=callback)

finalMoments = solveSimMoments(res.x, model, sims)
print("finalMoments = ", finalMoments)

if (not res.success):
    raise ArithmeticError(res.message)
print(res)








