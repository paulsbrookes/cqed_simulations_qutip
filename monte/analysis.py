from qutip import *
import matplotlib.pyplot as plt
import numpy as np

trajectories = qload('test')
final_states = trajectories[:, trajectories.shape[1] - 1]
[cavity_levels, transmon_levels] = final_states[0].dims[0]
number_operator = tensor(num(cavity_levels), qeye(transmon_levels))
expected_photons = expect(number_operator, final_states)
probabilities, bins, patches = plt.hist(expected_photons, 50, normed=1, facecolor='green')
plt.show()