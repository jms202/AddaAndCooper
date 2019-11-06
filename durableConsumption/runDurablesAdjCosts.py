import numpy as np
from durablesAdjCosts import Model
from sklearn.linear_model import LinearRegression

model = Model(Wngp = 15, Dngp = 15)
model.d = 10

model.solve()

sims = model.drawShocks()

sims = model.simulate(sims)


thisc = sims['c'][:, 1:].flatten()
prevc = sims['c'][:, :-1].reshape(-1, 1)

xmat = np.column_stack((sims['c'][:, :-1].flatten(), sims['D'][:, :-1].flatten()))


lr = LinearRegression()

lr.fit(prevc, thisc)
lr.coef_

lr.fit(xmat, thisc)
lr.coef_
