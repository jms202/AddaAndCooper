# Script using certainCake.py to solve a simple continuous non-stochastic
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



import sys
import numpy as np
import matplotlib.pyplot as plt
from certainCake import Model

def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)



# Plot a sequence of value function iterations
model = Model(uType='smooth')
seq = model.genSeq(n=50, guess=True)
model.plotValueSeq(seq)


# Check whether solving the euler equation or maximising the value function makes any difference to the solution
model1 = Model(approach='euler')
model1.solve()
model1.findPolicy()

model2 = Model(approach='value')
model2.solve()
model2.findPolicy()

fig, ax = plt.subplots()
ax.plot(model1.Wgrid, model1.policy, 'r', label='Solving Euler equation')
ax.plot(model2.Wgrid, model2.policy, 'b--', label='Maximising value function')
ax.set_xlabel('W', fontsize=14)
ax.set_ylabel('Optimal c', fontsize=14)
ax.set_title('Comparing policy functions', fontsize=14)
ax.legend(loc='best')
plt.show()


# Check whether having more grid points makes any difference to the solution
model3 = Model(approach='euler', Wngp=11)
model3.solve()
model3.findPolicy()

fig, ax = plt.subplots()
ax.plot(model1.Wgrid, model1.policy, 'r', label='101 grid points')
ax.plot(model3.Wgrid, model3.policy, 'b--', label='11 grid points')
ax.set_xlabel('W', fontsize=14)
ax.set_ylabel('Optimal c', fontsize=14)
ax.set_title('Comparing policy functions', fontsize=14)
ax.legend(loc='best')
plt.show()


# Check whether interpolating using inverse marginal utility makes any difference to the solution
model4 = Model(approach='euler', Wngp=11, interpTranform='inverse')
model4.solve()
model4.findPolicy()

fig, ax = plt.subplots()
ax.plot(model3.Wgrid, model3.policy, 'r', label='Interpolating marginal utility')
ax.plot(model4.Wgrid, model4.policy, 'b--', label='Interpolating inverse marginal utility')
ax.set_xlabel('W', fontsize=14)
ax.set_ylabel('Optimal c', fontsize=14)
ax.set_title('Comparing policy function', fontsize=14)
ax.legend(loc='best')
plt.show()


