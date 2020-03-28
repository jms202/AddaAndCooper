import numpy as np
from scipy.optimize import minimize

from labourDemand import Model, objective, solveSimMoments, callback


# Simulate the truth
####################

# w2, eta
truth = [0.03, 1.0]
model = Model()
sims = model.drawShocks()
targetMoments = solveSimMoments(truth, model, sims)


# Try to recover the truth
##########################

sims = model.drawShocks()

guess = [0.05, 0.75]
initial_simplex = np.array([[0.05, 0.75],
                            [0.09, 0.75],
                            [0.05, 1.0]])
res = minimize(objective, x0=np.array(guess), args=(targetMoments, model, sims), method='Nelder-Mead', options={'initial_simplex':initial_simplex}, callback=callback)

finalMoments = solveSimMoments(res.x, model, sims)
print("finalMoments = ", finalMoments)

if (not res.success):
    raise ArithmeticError(res.message)
print(res)

crash

