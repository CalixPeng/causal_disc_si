import os
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

from utils import System
from algos import learn_topo, GA_LCB_SL
from utils_gies.gies import fit_bic
from utils_dagma.linear import DagmaLinear

import warnings
warnings.filterwarnings('ignore')

linear = True
N_criterion, N_alg = 2, 4
N_case, N_worker = 100, 24

def compute_error(Topo_true, Topo):
    N = Topo_true.shape[0]
    false_pos = np.sum((Topo_true == 0) & (Topo == 1))
    false_neg = np.sum((Topo_true == 1) & (Topo == 0))
    fnr = false_neg / np.sum(Topo_true == 1)
    SHD = false_pos + false_neg - np.sum((Topo_true == 1) & (Topo == 0) & (Topo.T == 1))
    return fnr, SHD/(N ** 2)

def one_case(T, S):
    N = S.N
    Error = np.zeros((N_criterion, N_alg))
    Topo_B_true, Topo_BI_true = (S.B!=0).astype(int), (S.BI!=0).astype(int)
    # DAGMA
    X, Act = np.zeros((N, T)), np.zeros((N, T), dtype=np.int8)
    for t in range(T):
        Act[:,t] = (t % 2) * np.ones(N)
        X[:,t] = S.step(Act[:,t])
    model = DagmaLinear(loss_type='l2')
    Topo_B = model.fit(X[:, 0::2].T, lambda1=0.02)
    Topo_B[Topo_B!=0] = 1
    model = DagmaLinear(loss_type='l2')
    Topo_BI = model.fit(X[:, 1::2].T, lambda1=0.02)
    Topo_BI[Topo_BI!=0] = 1
    fnr_0, SHD_0 = compute_error(Topo_B_true, Topo_B)
    fnr_1, SHD_1 = compute_error(Topo_BI_true, Topo_BI)
    Error[:, 0] = [(fnr_0 + fnr_1)/2, (SHD_0 + SHD_1)/2]
    # GIES
    X = np.zeros((N, T))
    Act = np.repeat(np.random.randint(2, size=(N, T//2)), 2, axis=1).astype(np.int8)
    Act[:, :2] = 0
    Id_act = np.array([int(''.join(map(str, Act[:,t])),2) for t in range(T)])
    for t in range(T):
        X[:, t] = S.step(Act[:, t])
    X_list, Act_list = [], []
    for id_act in set(Id_act):
        Idx = np.where(Id_act == id_act)[0]
        X_list.append(X[:, Idx].T)
        Act_list.append(list(np.where(Act[:, Idx[0]] == 1)[0]))
    Topo_B, _ = fit_bic(X_list, Act_list)
    fnr_0, SHD_0 = compute_error(Topo_B_true, Topo_B)
    Error[:, 1] = [fnr_0, SHD_0]
    # GA_LCB_SL
    X, Act = np.zeros((N, T)), np.zeros((N, T), dtype=np.int8)
    for t in range(T):
        i = (t % (N + 1)) - 1
        if i >= 0:
            Act[i,t] = 1
        X[:, t] = S.step(Act[:, t])
    Topo_B = GA_LCB_SL(X[:, :T], Act[:, :T], S.Nu)
    fnr_0, SHD_0 = compute_error(Topo_B_true, Topo_B)
    Error[:, 2] = [fnr_0, SHD_0]
    # CSL-SI
    X = np.zeros((N, T))
    Act = np.random.randint(2, size=(N, T), dtype=np.int8)
    for t in range(T):
        X[:, t] = S.step(Act[:, t])
    Topo_B, Topo_BI = learn_topo(X, Act, S.Nu, linear)
    fnr_0, SHD_0 = compute_error(Topo_B_true, Topo_B)
    fnr_1, SHD_1 = compute_error(Topo_BI_true, Topo_BI)
    Error[:, 3] = [(fnr_0 + fnr_1)/2, (SHD_0 + SHD_1)/2]
    return Error

def plot_result(N_list, Error_all, save_fig=False):
    Error_all = Error_all[:, :, :N_list.size, :]
    Error_avg = np.mean(Error_all, axis=3)
    Std = np.std(Error_all, axis=3)
    Error_low = np.maximum(0, Error_avg - Std)
    Error_up = np.minimum(1, Error_avg + Std)
    C_list = ['#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00', 
              '#ffff33', '#a65628', '#f781bf']
    M_list = ['^', 'd', 's', 'o']
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 12})
    alpha = 0.3
    plt.figure(1, dpi=800)
    plt.plot(N_list, Error_avg[0, 0, :], color=C_list[0], marker=M_list[0], label='DAGMA')
    plt.fill_between(N_list, Error_low[0, 0, :], Error_up[0, 0, :], color=C_list[0], alpha=alpha)
    plt.plot(N_list, Error_avg[0, 1, :], color=C_list[1], marker=M_list[1], label='GIES')
    plt.fill_between(N_list, Error_low[0, 1, :], Error_up[0, 1, :], color=C_list[1], alpha=alpha)
    # plt.plot(N_list, Error_avg[0, 2, :], color=C_list[2], marker=M_list[2], label='GA-LCB-SL')
    # plt.fill_between(N_list, Error_low[0, 2, :], Error_up[0, 2, :], color=C_list[2], alpha=alpha)
    plt.plot(N_list, Error_avg[0, 3, :], color=C_list[3], marker=M_list[3], label='CSL-SI')
    plt.fill_between(N_list, Error_low[0, 3, :], Error_up[0, 3, :], color=C_list[3], alpha=alpha)
    plt.xlabel('Number of Nodes')
    plt.ylabel('False Negative Rate')
    plt.legend(ncol=1, columnspacing=0.2, labelspacing=0.2, loc='upper right', 
               bbox_to_anchor=(1, 1))
    plt.grid()
    if save_fig:
        plt.savefig('fnr_d_lin.pdf')
    plt.figure(2, dpi=800)
    plt.plot(N_list, Error_avg[1, 0, :], color=C_list[0], marker=M_list[0], label='DAGMA')
    plt.fill_between(N_list, Error_low[1, 0, :], Error_up[1, 0, :], color=C_list[0], alpha=alpha)
    plt.plot(N_list, Error_avg[1, 1, :], color=C_list[1], marker=M_list[1], label='GIES')
    plt.fill_between(N_list, Error_low[1, 1, :], Error_up[1, 1, :], color=C_list[1], alpha=alpha)
    # plt.plot(N_list, Error_avg[1, 2, :], color=C_list[2], marker=M_list[2], label='GA-LCB-SL')
    # plt.fill_between(N_list, Error_low[1, 2, :], Error_up[1, 2, :], color=C_list[2], alpha=alpha)
    plt.plot(N_list, Error_avg[1, 3, :], color=C_list[3], marker=M_list[3], label='CSL-SI')
    plt.fill_between(N_list, Error_low[1, 3, :], Error_up[1, 3, :], color=C_list[3], alpha=alpha)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Normalized Structural Hamming Distance')
    plt.legend(ncol=1, columnspacing=0.2, labelspacing=0.2, loc='upper right', 
               bbox_to_anchor=(1, 1))
    plt.grid()
    if save_fig:
        plt.savefig('hd_d_lin.pdf')
    plt.show()

if __name__ == '__main__':
    np.random.seed(2025)
    N_list, T = np.arange(6, 21, 2), 100
    func = partial(one_case, T)
    Error_all = np.zeros((N_criterion, N_alg, N_list.size, N_case))
    print('Start at ' + datetime.now().strftime('%H:%M:%S'))
    p = Pool(N_worker)
    for i_N, N in enumerate(N_list):
        Nu, Sigma = np.ones(N), np.ones(N)
        S_list = [System(N, Nu, Sigma, linear) for _ in range(N_case)]
        n = 0
        for i_case, Error in enumerate(p.imap(func, S_list)):
            Error_all[:, :, i_N, i_case] = Error
            n += 1
            print(f'\rN = {N}, {n}/{N_case}, ' + datetime.now().strftime('%H:%M:%S'), 
                  end='', flush=True)
        plot_result(N_list[:i_N+1], Error_all)
        print('')
    p.close()
