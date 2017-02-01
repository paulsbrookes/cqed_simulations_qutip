from qutip import *
import matplotlib.pyplot as plt
import numpy as np

folder_path = '/homes/pbrookes/PycharmProjects/cqed/cqed_simulations_qutip/results/crosscheck/2017-01-26--16-13-36'
expectations_path = folder_path + '/expectations'
expectations = qload(expectations_path)
probabilities = np.array([expectations[x] for x in range(0, len(expectations))]);
n_snaps = probabilities.shape[1]
c_levels = probabilities.shape[0]
levels_array = np.linspace(0, c_levels - 1, c_levels)
probabilities_array = probabilities[:, n_snaps - 1]

plt.subplot(2,1,1)
plt.bar(levels_array, probabilities_array)
plt.xlabel('Cavity level');
plt.ylabel('Probability');

plt.subplot(2, 1, 2)
plt.pcolor(probabilities)
plt.xlabel('Snapshot');
plt.ylabel('Photons');

plt.show()