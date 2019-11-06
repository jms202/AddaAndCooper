# Script comparing a consumption-savings problem with no uncertainty with
# consumption-savings problem with Markovian uncertainty using certainCake.py
# and markovCake.py 

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
from markovCake import Model as MarkovModel

def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

model = Model()
model.solve()
model.findPolicy()

markovModel = MarkovModel()
markovModel.solve()
markovModel.findPolicy()

fig, ax = plt.subplots()
ax.plot(model.Wgrid, model.policy, 'r', label='No uncertainty')
ax.plot(markovModel.Wgrid, markovModel.policy[0,:], 'b--', label='With uncertainty, low income')
ax.plot(markovModel.Wgrid, markovModel.policy[1,:], 'g--', label='With uncertainty, high income')
ax.set_xlabel('W', fontsize=14)
ax.set_ylabel('Optimal c', fontsize=14)
ax.set_title('Policy function')
ax.legend(loc='best')
plt.show()

fig, ax = plt.subplots()
ax.plot(model.Wgrid, model.V, 'r', label='No uncertainty')
ax.plot(markovModel.Wgrid, markovModel.EV[0,:], 'b--', label='With uncertainty, low income')
ax.plot(markovModel.Wgrid, markovModel.EV[1,:], 'g--', label='With uncertainty, high income')
ax.set_xlabel('W', fontsize=14)
ax.set_ylabel('Value function/expected value function', fontsize=14)
ax.set_title('Value function and expected value function')
ax.legend(loc='best')
plt.show()

