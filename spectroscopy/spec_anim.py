import numpy as np
from qutip import *
from pylab import *
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import yaml

class parameters:
    def __init__(self, wc, wq, eps, g, chi, kappa, gamma, t_levels, c_levels):
        self.wc = wc
        self.wq = wq
        self.eps = eps
        self.g = g
        self.chi = chi
        self.gamma = gamma
        self.kappa = kappa
        self.t_levels = t_levels
        self.c_levels = c_levels

def hamiltonian(params, wd):
    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    sm = tensor(qeye(params.c_levels), destroy(params.t_levels))
    H = - (params.wc - wd) * a.dag() * a - (params.wq - wd) * sm.dag() * sm \
        + params.chi * sm.dag() * sm * (sm.dag() * sm - 1) + params.g * (a.dag() * sm + a * sm.dag()) \
        + params.eps * (a + a.dag())
    return H

def sweep(params, wd_points):

    transmissions = parallel_map(transmission_calc, wd_points, (params,), num_cpus = 10)
    transmissions = np.array(transmissions)

    return transmissions

def transmission_calc(wd, params):

    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    sm = tensor(qeye(params.c_levels), destroy(params.t_levels))
    c_ops = []
    c_ops.append(np.sqrt(params.kappa) * a)
    c_ops.append(np.sqrt(params.gamma) * sm)
    H = hamiltonian(params, wd)
    rho_ss = steadystate(H, c_ops)
    transmission = expect(a, rho_ss)

    return transmission

def new_points(wd_points, transmissions, threshold):

    metric_vector = curvature_vector(wd_points, transmissions)
    indices = np.array([index for index, metric in enumerate(metric_vector) if metric > threshold]) + 1
    new_wd_points = generate_points(wd_points, indices)

    return new_wd_points

def generate_points(wd_points, indices):
    n_points = 6
    new_wd_points = np.array([])
    for index in indices:
        multi_section = np.linspace(wd_points[index - 1], wd_points[index + 1], n_points)
        new_wd_points = np.concatenate((new_wd_points, multi_section))
    unique_set = set(new_wd_points) - set(wd_points)
    new_wd_points_unique = np.array(list(unique_set))
    return new_wd_points_unique

def curvature_vector(wd_points, transmissions):

    is_ordered = all([wd_points[i] <= wd_points[i + 1] for i in xrange(len(wd_points) - 1)])
    assert is_ordered, "Vector of wd_points is not ordered."
    assert len(wd_points) == len(transmissions), "Vectors of wd_points and transmissions are not of equal length."

    metric_vector = []
    for index in range(len(wd_points) - 2):
        metric = curvature(wd_points[index:index + 3], transmissions[index:index + 3])
        metric_vector.append(metric)
    return metric_vector

def curvature(wd_triplet, transmissions_triplet):

    wd_are_floats = all([isinstance(wd_triplet[i], float) for i in xrange(len(wd_triplet) - 1)])
    assert wd_are_floats, "The vector wd_triplet contains numbers which are not floats."
    transmissions_are_floats = all([isinstance(transmissions_triplet[i], float) \
                                    for i in xrange(len(transmissions_triplet) - 1)])
    assert transmissions_are_floats, "The vector transmissions_triplet contains numbers which are not floats."


    wd_delta_0 = wd_triplet[1] - wd_triplet[0]
    wd_delta_1 = wd_triplet[2] - wd_triplet[1]
    transmissions_delta_0 = transmissions_triplet[1] - transmissions_triplet[0]
    transmissions_delta_1 = transmissions_triplet[2] - transmissions_triplet[1]
    metric = 2 * (wd_delta_1 * transmissions_delta_1 - wd_delta_0 * transmissions_delta_0) / (wd_delta_0 + wd_delta_1)
    abs_normalised_metric = np.absolute(metric / transmissions_triplet[1])
    return abs_normalised_metric

def y_lim_calc(y_points):
    buffer_fraction = 0.1
    y_max = np.amax(y_points)
    y_min = np.amin(y_points)
    range = y_max - y_min
    y_lim_u = y_max + buffer_fraction * range
    y_lim_l = y_min - buffer_fraction * range
    return np.array([y_lim_l, y_lim_u])


if __name__ == '__main__':

    #wc, wq, eps, g, chi, kappa, gamma, t_levels, c_levels
    params = parameters(10.3641, 9.4914, 0.0001, 0.389, -0.097, 0.00146, 0.000833, 2, 10)
    save = 1
    fidelity = 0.05
    wd_lower = 10.4
    wd_upper = 10.55
    wd_points = np.linspace(wd_lower, wd_upper, 10)
    transmissions = sweep(params, wd_points)
    abs_transmissions = np.absolute(transmissions)
    new_wd_points = new_points(wd_points, abs_transmissions, fidelity)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(wd_lower, wd_upper)
    y_limits = y_lim_calc(abs_transmissions)
    ax.set_ylim(y_limits[0], y_limits[1])
    ax.set_xlabel('Cavity drive frequency (GHz)')
    ax.set_ylabel('|<a>|')

    ax.hold(True)
    plt.show(False)
    plt.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)
    points = ax.plot(wd_points, abs_transmissions, 'o')[0]

    while (len(new_wd_points) > 0):
        new_transmissions = sweep(params, new_wd_points)
        new_abs_transmissions = np.absolute(new_transmissions)
        wd_points = np.concatenate([wd_points, new_wd_points])
        transmissions = concatenate([transmissions, new_transmissions])
        abs_transmissions = concatenate([abs_transmissions, new_abs_transmissions])
        sort_indices = np.argsort(wd_points)
        wd_points = wd_points[sort_indices]
        transmissions = transmissions[sort_indices]
        abs_transmissions = abs_transmissions[sort_indices]
        new_wd_points = new_points(wd_points, abs_transmissions, fidelity)
        points.set_data(wd_points, abs_transmissions)
        fig.canvas.restore_region(background)
        ax.draw_artist(points)
        fig.canvas.blit(ax.bbox)
        y_limits = y_lim_calc(abs_transmissions)
        ax.set_ylim(y_limits[0], y_limits[1])


    if save == 1:
        np.savetxt('results/abs_transmissions.csv', abs_transmissions, delimiter=',')
        np.savetxt('results/drive_frequencies.csv', wd_points, delimiter=',')

        params_dic = {'f_c': params.wc,
                      'f_q': params.wq,
                      'epsilon': params.eps,
                      'g': params.g,
                      'kappa': params.kappa,
                      'gamma': params.gamma,
                      'transmon_levels': params.t_levels,
                      'cavity_levels': params.c_levels}

        with open('results/parameters.yml', 'w') as outfile: yaml.dump(params_dic, outfile, default_flow_style = True)

    plt.scatter(wd_points, abs_transmissions)
    plt.show()