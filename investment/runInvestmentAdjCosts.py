import numpy as np
from investmentAdjCosts import Model

model = Model(Kngp = 50)

model.solve()

model.plotInvestmentQbar()
