from qutip import *
import matplotlib.pyplot as plt
import numpy as np

folder_path = '/homes/pbrookes/PycharmProjects/cqed/cqed_simulations_qutip/results/crosscheck2/2017-01-26--17-58-53'
expectations_path = folder_path + '/expectations'
expectations = qload(expectations_path)
abs_transmissions = np.absolute(expectations[0])
print abs_transmissions
plt.plot(abs_transmissions)
plt.show()