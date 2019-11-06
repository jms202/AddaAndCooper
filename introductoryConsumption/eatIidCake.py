# Script comparing a consumption-savings problem with no uncertainty with
# consumption-savings problem with Markovian uncertainty using iidCake.py 

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
from iidCake import Model

def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)



# Compare certain model and IID model
certainModel = Model(y=[5.0, 5.0])
certainModel.solve()
certainModel.findPolicy()

iidModel = Model()
iidModel.solve()
iidModel.findPolicy()

fig, ax = plt.subplots()
ax.plot(certainModel.Xgrid, certainModel.policy, 'r', iidModel.Xgrid, iidModel.policy, 'b--')
ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Optimal c', fontsize=14)
ax.set_title('Policy function')
plt.show()



