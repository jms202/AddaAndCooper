# Script using iidDiscreteCake.py to solve a simple discrete IID stochastic
# consumption-savings problem

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


from iidDiscreteCake import Model

# Set up the model (we're assuming uniformly distributed shocks between 0.5 and 1.5)
model = Model(rho=0.95, p = [0.1] * 10, eps=list(np.linspace(0.55, 1.45, 10)))

# Simulate some data
model.solve()
model.simulate()
d_data = np.array(model.d_sims)


# Calculate the log likelihood for different values of theta (i.e. rho)
thetaVec = np.linspace(0.55, 0.99, 15)
loglVec = np.array([model.negLogLikelihood([theta], d_data) for theta in thetaVec])

# Plot graph
fig, ax = plt.subplots()
ax.plot(thetaVec, loglVec, 'r')
ax.set_xlabel('rho', fontsize=14)
ax.set_ylabel('Negative log likelihood', fontsize=14)
ax.set_title('Negative log likelihood for different values of rho')
plt.show()


results = model.estimate([0.9], d_data)
print(results)

bsStdErr = model.bootstrapStdErr([0.95], [0.9], reps=100)
print(bsStdErr)

