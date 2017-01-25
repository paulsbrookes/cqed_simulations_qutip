import numpy as np
from qutip import *
from pylab import *
from scipy.fftpack import fft
import matplotlib.pyplot as plt

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

def solution(sys_params, hilbert_dims, timings, initial_state, n_traj=50):
    end_time = timings[0]
    n_snaps = timings[1]
    H = hamiltonian(sys_params, hilbert_dims)
    c_ops = collapse_operators(sys_params, hilbert_dims)
    snapshot_times = linspace(0, end_time, n_snaps)
    output = mcsolve(H, initial_state, snapshot_times, c_ops, [], ntraj=n_traj)
    return output

if __name__ == '__main__':
    sys_params = system_parameters(10.3641, 9.49101, 0.202, -0.097, 0.03, 10.4, 0.00146, 0.000833)
    hilbert_dims = hilbert_dimensions(200, 4)
    timings = [500, 100]
    initial_state = tensor(basis(hilbert_dims.c_levels, 0), basis(hilbert_dims.t_levels, 0))
    output = solution(sys_params, hilbert_dims, timings, initial_state, 10)
    qsave(output.states, 'test2')