import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from datetime import datetime

from utils import System
from algos import learn_topo, GA_LCB_SL
from utils_gies.gies import fit_bic
from utils_dagma.linear import DagmaLinear

import warnings
warnings.filterwarnings('ignore')

linear = True
n_criterion, n_alg = 2, 4
n_case, n_worker = 10, 10

def compute_error(Topo_true, Topo):
    false_pos = np.sum((Topo_true == 0) & (Topo == 1))
    false_neg = np.sum((Topo_true == 1) & (Topo == 0))
    fnr = false_neg / np.sum(Topo_true == 1)
    shd = false_pos + false_neg - np.sum((Topo_true == 1) & (Topo == 0) & (Topo.T == 1))
    return fnr, shd

def one_case(t_list, sys):
    d, t_max = sys.d, t_list[-1]
    Error = np.zeros((n_criterion, n_alg, t_list.size))
    Topo_B_true, Topo_BI_true = (sys.B!=0).astype(int), (sys.BI!=0).astype(int)
    # DAGMA
    X, Act = np.zeros((d, t_max)), np.zeros((d, t_max), dtype=np.int8)
    for tau in range(t_max):
        Act[:,tau] = (tau % 2) * np.ones(d)
        X[:,tau] = sys.step(Act[:,tau])
    for i_t, t in enumerate(t_list):
        model = DagmaLinear(loss_type='l2')
        Topo_B = model.fit(X[:,0:t:2].T, lambda1=0.02)
        Topo_B[Topo_B!=0] = 1
        model = DagmaLinear(loss_type='l2')
        Topo_BI = model.fit(X[:,1:t:2].T, lambda1=0.02)
        Topo_BI[Topo_BI!=0] = 1
        fnr_0, shd_0 = compute_error(Topo_B_true, Topo_B)
        fnr_1, shd_1 = compute_error(Topo_BI_true, Topo_BI)
        Error[:,0,i_t] = [(fnr_0 + fnr_1)/2, (shd_0 + shd_1)/2]
    # GIES
    X = np.zeros((d, t_max))
    Act = np.repeat(np.random.randint(2, size=(d, t_max//2)), 2, axis=1).astype(np.int8)
    Act[:,:2] = 0
    Id_act = np.array([int(''.join(map(str, Act[:,tau])),2) for tau in range(t_max)])
    for tau in range(t_max):
        X[:,tau] = sys.step(Act[:,tau])
    for i_t, t in enumerate(t_list):
        X_list, Act_list = [], []
        for id_act in set(Id_act[:t]):
            Idx = np.where(Id_act[:t] == id_act)[0]
            X_list.append(X[:,Idx].T)
            Act_list.append(list(np.where(Act[:, Idx[0]] == 1)[0]))
        Topo_B, _ = fit_bic(X_list, Act_list)
        fnr_0, shd_0 = compute_error(Topo_B_true, Topo_B)
        Error[:,1,i_t] = [fnr_0, shd_0]
    # GA_LCB_SL
    X, Act = np.zeros((d, t_max)), np.zeros((d, t_max), dtype=np.int8)
    for tau in range(t_max):
        i = (tau % (d + 1)) - 1
        if i >= 0:
            Act[i,tau] = 1
        X[:,tau] = sys.step(Act[:,tau])
    for i_t, t in enumerate(t_list):
        Topo_B = GA_LCB_SL(X[:,:t], Act[:,:t], sys.Nu)
        fnr_0, shd_0 = compute_error(Topo_B_true, Topo_B)
        Error[:,2,i_t] = [fnr_0, shd_0]
    # CSL-SI
    X = np.zeros((d, t_max))
    Act = np.random.randint(2, size=(d, t_max), dtype=np.int8)
    for tau in range(t_max):
        X[:,tau] = sys.step(Act[:,tau])
    for i_t, t in enumerate(t_list):
        Topo_B, Topo_BI = learn_topo(X[:,:t], Act[:,:t], sys.Nu, linear)
        fnr_0, shd_0 = compute_error(Topo_B_true, Topo_B)
        fnr_1, shd_1 = compute_error(Topo_BI_true, Topo_BI)
        Error[:,3,i_t] = [(fnr_0 + fnr_1)/2, (shd_0 + shd_1)/2]
    return Error

def plot_result(t_list, Error_all, save_fig=False):
    Error_all = Error_all[:,:,:t_list.size,:]
    Error_avg = np.mean(Error_all, axis=3)
    Std = np.std(Error_all, axis=3)
    Error_low = np.maximum(0, Error_avg - Std)
    Error_up = Error_avg + Std
    Error_up[0,:,:] = np.minimum(1, Error_up[0,:,:])
    C_list = ['#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00', 
              '#ffff33', '#a65628', '#f781bf']
    M_list = ['^', 'd', 's', 'o']
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 12})
    alpha = 0.3
    plt.figure(1)
    plt.plot(t_list, Error_avg[0,0,:], color=C_list[0], marker=M_list[0], label='DAGMA')
    plt.fill_between(t_list, Error_low[0,0,:], Error_up[0,0,:], color=C_list[0], alpha=alpha)
    plt.plot(t_list, Error_avg[0,1,:], color=C_list[1], marker=M_list[1], label='GIES')
    plt.fill_between(t_list, Error_low[0,1,:], Error_up[0,1,:], color=C_list[1], alpha=alpha)
    plt.plot(t_list, Error_avg[0,2,:], color=C_list[2], marker=M_list[2], label='GA-LCB-SL')
    plt.fill_between(t_list, Error_low[0,2,:], Error_up[0,2,:], color=C_list[2], alpha=alpha)
    plt.plot(t_list, Error_avg[0,3,:], color=C_list[3], marker=M_list[3], label='CSL-SI')
    plt.fill_between(t_list, Error_low[0,3,:], Error_up[0,3,:], color=C_list[3], alpha=alpha)
    plt.xlabel('Time steps')
    plt.ylabel('False negative rate')
    plt.legend(ncol=2, columnspacing=0.2, labelspacing=0.2, loc='upper right', 
               bbox_to_anchor=(1, 1))
    plt.grid()
    if save_fig:
        plt.savefig('fnr_t_lin.pdf')
    plt.figure(2)
    plt.plot(t_list, Error_avg[1,0,:], color=C_list[0], marker=M_list[0], label='DAGMA')
    plt.fill_between(t_list, Error_low[1,0,:], Error_up[1,0,:], color=C_list[0], alpha=alpha)
    plt.plot(t_list, Error_avg[1,1,:], color=C_list[1], marker=M_list[1], label='GIES')
    plt.fill_between(t_list, Error_low[1,1,:], Error_up[1,1,:], color=C_list[1], alpha=alpha)
    plt.plot(t_list, Error_avg[1,2,:], color=C_list[2], marker=M_list[2], label='GA-LCB-SL')
    plt.fill_between(t_list, Error_low[1,2,:], Error_up[1,2,:], color=C_list[2], alpha=alpha)
    plt.plot(t_list, Error_avg[1,3,:], color=C_list[3], marker=M_list[3], label='CSL-SI')
    plt.fill_between(t_list, Error_low[1,3,:], Error_up[1,3,:], color=C_list[3], alpha=alpha)
    plt.xlabel('Time steps')
    plt.ylabel('Structural Hamming distance')
    plt.legend(ncol=2, columnspacing=0.2, labelspacing=0.2, loc='upper right', 
               bbox_to_anchor=(1, 0.7))
    plt.grid()
    if save_fig:
        plt.savefig('hd_t_lin.pdf')
    plt.show()

if __name__ == '__main__':
    np.random.seed(2025)
    d, t_list = 10, np.arange(30, 101, 10)
    Nu, Sigma = np.ones(d), np.ones(d)
    func = partial(one_case, t_list)
    sys_list = [System(d, Nu, Sigma, linear) for _ in range(n_case)]
    Error_all = np.zeros((n_criterion, n_alg, t_list.size, n_case))
    print('Start at ' + datetime.now().strftime('%H:%M:%S'))
    p = Pool(n_worker)
    n = 0
    for i_case, Error in enumerate(p.imap(func, sys_list)):
        Error_all[:,:,:,i_case] = Error
        n += 1
        print(f'\r{n}/{n_case}, ' + datetime.now().strftime('%H:%M:%S'), 
              end='', flush=True)
    p.close()
    plot_result(t_list, Error_all)