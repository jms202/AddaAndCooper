import numpy as np
import matplotlib.pyplot as plt
from investmentConvexAdj import Model as ModelConvex
from investmentNonConvexAdj import Model as ModelNonConvex

modelConvex = ModelConvex(Kngp = 50)
modelNonConvex = ModelNonConvex(Kngp = 50)

modelConvex.solve()
modelNonConvex.solve()

simsConvex = modelConvex.drawShocks()
simsNonConvex = modelNonConvex.drawShocks()

simsConvex = modelConvex.simulate(simsConvex)
simsNonConvex = modelNonConvex.simulate(simsNonConvex)

aggInvestConvex = np.mean(simsConvex['i'], axis = 0)
aggInvestNonConvex = np.mean(simsNonConvex['i'], axis = 0)

periods = len(aggInvestConvex)
start = 4
fig, ax = plt.subplots()
ax.plot(range(start, periods), aggInvestConvex[start:], label='Convex')
ax.plot(range(start, periods), aggInvestNonConvex[start:], label='Non-convex')
ax.set_xlabel('K', fontsize=14)
ax.set_ylabel('Optimal investment', fontsize=14)
ax.set_title('Aggregate investment: convex vs non-convex models')
plt.legend()
plt.show()


