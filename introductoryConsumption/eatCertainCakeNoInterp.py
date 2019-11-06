# Script comparing value function iteration with interpolation with policy
# function iteration using certainCake.py and certainCakeNoInterp.py 

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
from certainCakeNoInterp import Model as ModelNoInterp

def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

model = Model()
model.solve()
model.findPolicy()

modelNoInterp = ModelNoInterp(Wngp=1001, method='PFI')
modelNoInterp.solve()

fig, ax = plt.subplots()
ax.plot(model.Wgrid, model.policy, 'r', label='Value function iteration with interpolation')
ax.plot(modelNoInterp.Wgrid, modelNoInterp.policy, 'b--', label='Policy function iteration (no interpolation)')
ax.set_xlabel('W', fontsize=14)
ax.set_ylabel('Optimal c', fontsize=14)
ax.set_title('Policy function')
ax.legend(loc='best')
plt.show()


fig, ax = plt.subplots()
ax.plot(model.Wgrid, model.V, 'r', label='Value function iteration with interpolation')
ax.plot(modelNoInterp.Wgrid, modelNoInterp.V, 'b--', label='Policy function iteration (no interpolation)')
ax.set_xlabel('W', fontsize=14)
ax.set_ylabel('V', fontsize=14)
ax.set_title('Value function')
ax.legend(loc='best')
plt.show()


