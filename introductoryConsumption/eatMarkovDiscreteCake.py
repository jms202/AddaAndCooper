# Script using markovDiscreteCake.py to solve a simple discrete
# consumption-savings problem with Markovian uncertainty

# Copyright (c) 2017, 2018 Jonathan Shaw

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import matplotlib.pyplot as plt


from markovDiscreteCake import Model

transMat = (np.ones(shape=(10,10))*0.05) + (np.eye(N=10)*0.5)
#transMat = np.ones(shape=(10,10))*0.1
model = Model(rho=0.95, p = transMat, eps=list(np.linspace(0.55, 1.45, 10)))


# Simulate some data
model.solve()
model.simulate()
d_data = np.array(model.d_sims)

# Calculate moments (mean and variance) of that data
# This is what we will try to replicate
model.calcMoments()
momentsData = np.array(model.moments)

print(momentsData)
results = model.estimate([0.95], momentsData)
print(results)

