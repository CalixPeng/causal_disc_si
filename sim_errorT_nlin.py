import os
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'

import torch
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from datetime import datetime

from utils import System
from algos import learn_topo
from utils_dagma.nonlinear import DagmaMLP, DagmaNonlinear
from utils_dcdi.dcdi import fit_dcdi

linear = False
n_criterion, n_alg = 2, 3
n_case, n_worker = 48, 24

def compute_error(adj_true, adj):
    false_pos = np.sum((adj_true == 0) & (adj == 1))
    false_neg = np.sum((adj_true == 1) & (adj == 0))
    fnr = false_neg / np.sum(adj_true == 1)
    shd = false_pos + false_neg - np.sum((adj_true == 1) & (adj == 0) & (adj.T == 1))
    return fnr, shd


def one_case(t_list, sys):
    d, t_max = sys.d, t_list[-1]
    error = np.zeros((n_criterion, n_alg, t_list.size))
    adj_B_true, adj_BI_true = (sys.B!=0).astype(int), (sys.BI!=0).astype(int)
    # DAGMA
    X, act = np.zeros((d, t_max)), np.zeros((d, t_max), dtype=int)
    for t in range(t_max):
        act[:,t] = (t % 2) * np.ones(d)
        X[:,t] = sys.step(act[:,t])
    for i_t, t in enumerate(t_list):
        eq_model = DagmaMLP(dims=[d, 10, 1], bias=True)
        model = DagmaNonlinear(eq_model)
        adj_B = model.fit(np.array(X[:,0:t:2].T), lambda1=0.02, lambda2=0.005)
        adj_B[adj_B!=0] = 1
        eq_model = DagmaMLP(dims=[d, 10, 1], bias=True)
        model = DagmaNonlinear(eq_model)
        adj_BI = model.fit(np.array(X[:,1:t:2].T), lambda1=0.02, lambda2=0.005)
        adj_BI[adj_BI!=0] = 1
        fnr_0, shd_0 = compute_error(adj_B_true, adj_B)
        fnr_1, shd_1 = compute_error(adj_BI_true, adj_BI)
        error[:,0,i_t] = [(fnr_0 + fnr_1)/2, (shd_0 + shd_1)/2]
    # DCDI
    X, act = np.zeros((d, t_max)), np.zeros((d, t_max), dtype=np.int8)
    act_option = np.random.choice(2, size=(d, d), p=[0.8, 0.2])
    np.fill_diagonal(act_option, 1)
    act_option = np.tril(act_option)
    for t in range(t_max):
        if t % (d + 1) != 0:
            act[:,t] = act_option[:,t % (d + 1) - 1]
        X[:,t] = sys.step(act[:,t])
    for i_t, t in enumerate(t_list):
        adj_B = fit_dcdi(np.array(X[:,:t]), np.array(act[:,:t]))
        fnr_0, shd_0 = compute_error(adj_B_true, adj_B)
        error[:,1,i_t] = [fnr_0, shd_0]
    # CSL-SI
    X, act = np.zeros((d, t_max)), np.zeros((d, t_max), dtype=np.int8)
    for t in range(t_max):
        act[:,t] = np.random.randint(2, size=d)
        X[:,t] = sys.step(act[:,t])
    for i_t, t in enumerate(t_list):
        adj_B, adj_BI = learn_topo(np.array(X[:,:t]), np.array(act[:,:t]), sys.nu, linear)
        fnr_0, shd_0 = compute_error(adj_B_true, adj_B)
        fnr_1, shd_1 = compute_error(adj_BI_true, adj_BI)
        error[:,2,i_t] = [(fnr_0 + fnr_1)/2, (shd_0 + shd_1)/2]
    return error


def plot_result(t_list, error_all, save_fig=False):
    error_all = error_all[:,:,:t_list.size,:]
    error_avg = np.mean(error_all, axis=3)
    std = np.std(error_all, axis=3)
    error_low = np.maximum(0, error_avg - std)
    error_up = error_avg + std
    error_up[0,:,:] = np.minimum(1, error_up[0,:,:])
    c_list = ['#984ea3', '#377eb8', '#4daf4a', '#e41a1c', '#ff7f00', 
              '#ffff33', '#a65628', '#f781bf']
    m_list = ['o', '^', 'd', 's']
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 12})
    alpha = 0.3
    plt.figure(1, dpi=800)
    plt.plot(t_list, error_avg[0,0,:], color=c_list[1], marker=m_list[1], label='DAGMA')
    plt.fill_between(t_list, error_low[0,0,:], error_up[0,0,:], color=c_list[1], alpha=alpha)
    plt.plot(t_list, error_avg[0,1,:], color=c_list[2], marker=m_list[2], label='DCDI')
    plt.fill_between(t_list, error_low[0,1,:], error_up[0,1,:], color=c_list[2], alpha=alpha)
    plt.plot(t_list, error_avg[0,2,:], color=c_list[3], marker=m_list[3], label='CSL-SI')
    plt.fill_between(t_list, error_low[0,2,:], error_up[0,2,:], color=c_list[3], alpha=alpha)
    plt.xlabel('Time steps')
    plt.ylabel('False negative rate')
    plt.legend(ncol=2, columnspacing=0.2, labelspacing=0.2, loc='upper right', 
               bbox_to_anchor=(1, 1))
    plt.grid()
    if save_fig:
        plt.savefig('fnr_t_nl.pdf')
    plt.figure(2, dpi=800)
    plt.plot(t_list, error_avg[1,0,:], color=c_list[1], marker=m_list[1], label='DAGMA')
    plt.fill_between(t_list, error_low[1,0,:], error_up[1,0,:], color=c_list[1], alpha=alpha)
    plt.plot(t_list, error_avg[1,1,:], color=c_list[2], marker=m_list[2], label='DCDI')
    plt.fill_between(t_list, error_low[1,1,:], error_up[1,1,:], color=c_list[2], alpha=alpha)
    plt.plot(t_list, error_avg[1,2,:], color=c_list[3], marker=m_list[3], label='CSL-SI')
    plt.fill_between(t_list, error_low[1,2,:], error_up[1,2,:], color=c_list[3], alpha=alpha)
    plt.xlabel('Time steps')
    plt.ylabel('Structural Hamming distance')
    plt.legend(ncol=2, columnspacing=0.2, labelspacing=0.2, loc='upper right', 
               bbox_to_anchor=(1, 1))
    plt.grid()
    if save_fig:
        plt.savefig('hd_t_nl.pdf')
    plt.show()


if __name__ == '__main__':
    np.random.seed(2025)
    torch.manual_seed(2025)
    d, t_list = 10, np.arange(100, 401, 50)
    nu, sigma = np.ones(d), np.ones(d)
    func = partial(one_case, t_list)
    sys_list = [System(d, nu, sigma, linear) for _ in range(n_case)]
    error_all = np.zeros((n_criterion, n_alg, t_list.size, n_case))
    print('Start at ' + datetime.now().strftime('%H:%M:%S'))
    p = Pool(n_worker)
    n = 0
    for i_case, error in enumerate(p.imap(func, sys_list)):
        error_all[:,:,:,i_case] = error
        n += 1
        print(f'\r{n}/{n_case}, ' + datetime.now().strftime('%H:%M:%S'), 
              end='', flush=True)
    p.close()
    plot_result(t_list, error_all)
