import numpy as np
from qutip import *
from pylab import *
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from datetime import datetime
import os

# runfile('C:/Users/User/Documents/Python Scripts/untitled3.py',
#                 wdir='C:/Users/User/Documents/Python Scripts',
#                 args="5450 5500")

class system_parameters:
    def __init__(self, fc, fq, g, chi, eps, fd, kappa, gamma):
        self.fc = fc
        self.fq = fq
        self.eps = eps
        self.g = g
        self.chi = chi
        self.eps = eps
        self.fd = fd
        self.gamma = gamma
        self.kappa = kappa

class hilbert_dimensions:
    def __init__(self, c_levels, t_levels):
        self.c_levels = c_levels
        self.t_levels = t_levels

class simulation_options:
    def __init__(self, end_time, n_snaps, m_ops, n_traj = 50):
        self.end_time = end_time
        self.n_snaps = n_snaps
        self.m_ops = m_ops
        self.n_traj = n_traj

def hamiltonian(sys_params, hilbert_dims):
    a = tensor(destroy(hilbert_dims.c_levels), qeye(hilbert_dims.t_levels))
    sm = tensor(qeye(hilbert_dims.c_levels), destroy(hilbert_dims.t_levels))
    H = -(sys_params.fc - sys_params.fd) * a.dag() * a - (sys_params.fq - sys_params.fd) * sm.dag() * sm \
        + sys_params.chi * sm.dag() * sm * (sm.dag() * sm - 1) + sys_params.g * (a.dag() * sm + a * sm.dag()) \
        + sys_params.eps * (a + a.dag())
    return H

def collapse_operators(sys_params, hilbert_dims):
    a = tensor(destroy(hilbert_dims.c_levels), qeye(hilbert_dims.t_levels))
    sm = tensor(qeye(hilbert_dims.c_levels), destroy(hilbert_dims.t_levels))
    c_ops = []
    c_ops.append(np.sqrt(sys_params.kappa) * a)
    c_ops.append(np.sqrt(sys_params.gamma) * sm)
    return c_ops

def solution(sys_params, hilbert_dims, initial_state, sim_options):
    H = hamiltonian(sys_params, hilbert_dims)
    c_ops = collapse_operators(sys_params, hilbert_dims)
    snapshot_times = linspace(0, sim_options.end_time, sim_options.n_snaps)
    output = mcsolve(H, initial_state, snapshot_times, c_ops, sim_options.m_ops, ntraj=sim_options.n_traj)
    return output

if __name__ == '__main__':
    sys_params = system_parameters(10.3641, 9.49101, 0.275, -0.097, 0.01, 10.44184, 0.00146, 0.000833)
    hilbert_dims = hilbert_dimensions(30, 4)
    #m_ops = [tensor(fock_dm(hilbert_dims.c_levels, x), qeye(hilbert_dims.t_levels)) \
    #         for x in range(hilbert_dims.c_levels)]
    a = tensor(destroy(hilbert_dims.c_levels), qeye(hilbert_dims.t_levels))
    m_ops = [a]
    sim_options = simulation_options(200000, 200, m_ops, 1000)
    initial_state = tensor(basis(hilbert_dims.c_levels, 0), basis(hilbert_dims.t_levels, 0))
    output = solution(sys_params, hilbert_dims, initial_state, sim_options)
    time = datetime.now()
    time_string = time.strftime('%Y-%m-%d--%H-%M-%S')
    folder = 'crosscheck2'
    path = '/homes/pbrookes/PycharmProjects/cqed/cqed_simulations_qutip/results/' + folder + '/' + time_string + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    qsave(output.expect, path + 'expectations')
    qsave(output.times, path + 'times')
    qsave(sys_params, path + 'params')
    qsave(hilbert_dims, path + 'hilbert_dims')
    qsave(sim_options, path + 'sim_options')