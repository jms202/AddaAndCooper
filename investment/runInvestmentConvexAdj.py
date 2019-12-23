import numpy as np
from investmentConvexAdj import Model

model = Model(Kngp = 50)

model.solve()

model.plotInvestmentQbar()
