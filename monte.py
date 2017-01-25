import numpy as np
from qutip import *
from pylab import *
from scipy.fftpack import fft
import matplotlib.pyplot as plt

# runfile('C:/Users/User/Documents/Python Scripts/untitled3.py',
#                 wdir='C:/Users/User/Documents/Python Scripts',
#                 args="5450 5500")



def solution(eps, wd, cavity_levels, transmon_levels):

    wc = 10.3641 # 10.5665 cavity frequency
    wa = 9.49101  # atom frequency
    g = 0.202  # coupling strength
    kappa = 0.00146
    gamma = 0.000833  # atom dissipation rate
    pump = 0  # atom pump rate
    time = 50000
    chi = -0.097
    snaps = 100
    taus = linspace(0, time, snaps)
    a = tensor(destroy(cavity_levels), qeye(transmon_levels))
    sm = tensor(qeye(cavity_levels), destroy(transmon_levels))
    H = -(wc - wd) * a.dag() * a - (wa - wd) * sm.dag() * sm + chi * sm.dag() * sm * (sm.dag() * sm - 1) + g * (a.dag() * sm + a * sm.dag()) + eps * (
        a + a.dag())

    # n_th = 0  # bath temperature in terms of excitation number
    c_op_list = []

    n_th_a = 0  # zero temperature
    rate = kappa * (1 + n_th_a)
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * a)

    rate = kappa * n_th_a
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * a.dag())

    rate = gamma
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * sm)

    rate = pump
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * sm.dag())

    rho0 = tensor(basis(cavity_levels, 0), basis(transmon_levels, 0))

    projections = [tensor(fock_dm(cavity_levels, x), qeye(transmon_levels)) for x in range(cavity_levels)]
    output = mcsolve(H, rho0, taus, c_op_list, [], ntraj=100)

    return output


if __name__ == '__main__':
    cavity_levels = 200;
    transmon_levels = 4;
    output = solution(0.03, 10.4, cavity_levels, transmon_levels)
    qsave(output.states, 'test')