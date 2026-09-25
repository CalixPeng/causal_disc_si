import itertools
import numpy as np
import networkx as nx
from copy import deepcopy
from sklearn.linear_model import LinearRegression, Lasso

import torch
from torch import nn

###############################################################################
#                               GA-LCB-SL
###############################################################################
def GA_LCB_SL(X, act, nu):
    d = X.shape[0]
    mean_est = np.zeros((d, d + 1))
    id_relate = np.where(act.sum(axis=0) == 0)[0]
    mean_est[:,0] = np.mean(X[:,id_relate], axis=1)
    for i in range(d):
        id_relate = np.where(act[i,:] == 1)[0]
        mean_est[:,i+1] = np.mean(X[:,id_relate], axis=1)
    # Get descendant and ancestor
    Des = [[] for i in range(d)]
    Anc = [[] for i in range(d)]
    for i in range(d):
        Des[i] = [j for j in range(d) if i != j and 
                  abs(mean_est[j,0] - mean_est[j,i+1]) > 0.5/2]
    for i in range(d):
        Anc[i] = [j for j in range(d) if i != j and i in Des[j]]
    # Get parant from Lasso
    adj = np.zeros((d, d))
    id_relate = np.where(act.sum(axis=0) == 0)[0]
    X_obs = X[:,id_relate]
    for i in range(d):
        if len(Anc[i]) == 0:
            continue
        lasso = Lasso(alpha=0.5, fit_intercept=False)
        lasso.fit(X_obs[Anc[i],:].T, X_obs[i,:]-nu[i])
        Id = [item for item, flag in zip(Anc[i], lasso.coef_!=0) if flag]
        adj[Id, i] = 1
    return adj


###############################################################################
#                                  CSL
###############################################################################
class Net(nn.Module):
    def __init__(self, d):
        torch.set_default_dtype(torch.double)
        super(Net, self).__init__()
        self.hidden = nn.Linear(d, 8, bias=False)
        self.output = nn.Linear(8, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        return self.output(x)


def fit_lin(X, y, parent):
    learning_set = np.where(parent==1)[0]
    X_reg = X[learning_set,:]
    reg = LinearRegression(fit_intercept=False).fit(X_reg.T, y)
    res = y - (X_reg.T @ reg.coef_)
    return np.sum(res ** 2)


def fit_nonl(X, y, parent, net_W=None):
    # hyper-parameters
    max_iter, delta = int(2e3), 1e-6
    model = Net(parent.size)
    if net_W != None:
        model.load_state_dict(net_W)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
    # training
    X_tr = torch.from_numpy(deepcopy(X).T)
    X_tr[:,np.where(parent!=1)[0]] = 0
    y_tr = torch.from_numpy(deepcopy(y)).view(-1, 1)
    loss_min, count, patience = np.inf, 0, 20
    for _ in range(max_iter):
        out = model(X_tr)
        loss = loss_func(out, y_tr) + 0.10 * \
            (torch.sum(torch.abs(model.hidden.weight)) + 
             torch.sum(torch.abs(model.output.weight)))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss)
        if loss.item() < loss_min - delta:
            loss_min = loss.item()
            count = 0
        else:
            count += 1
        if count >= patience:
            break
    with torch.no_grad():
        res_norm = torch.sum((y_tr - model(X_tr)) ** 2).item()
    return res_norm, model.state_dict()


def cal_ppl(X, act, nu, intv, i_pa, i_ch, adj, Res_norm, linear, Net_W):
    parent = deepcopy(adj)
    parent[i_ch] = 2
    id_relate = np.where(act[i_ch,:]==intv)[0]
    X_reg = X[:,id_relate]
    y = X_reg[i_ch,:] - nu[i_ch]
    # regress on all estimated parents
    key = ''.join(str(p) for p in parent)
    if (key not in Res_norm):
        if linear:
            Res_norm[key] = fit_lin(X_reg, y, parent)
        else:
            Res_norm[key], Net_W[key] = fit_nonl(X_reg, y, parent)
    res_norm_1 = Res_norm[key]
    net_W = Net_W[key] if not linear else None
    # regress on other parents
    if np.sum(parent == 1) == 1:
        return (np.sum(y ** 2) - res_norm_1) / id_relate.size
    parent[i_pa] = 0
    key = ''.join(str(p) for p in parent)
    if (key not in Res_norm):
        if linear:
            Res_norm[key] = fit_lin(X_reg, y, parent)
        else:
            Res_norm[key], Net_W[key] = fit_nonl(X_reg, y, parent, net_W)
    res_norm_2 = Res_norm[key]
    return (res_norm_2 - res_norm_1) / id_relate.size


def learn_topo(X, act, nu, linear):
    d = X.shape[0]
    adj_list = []
    for intv in [0, 1]:
        Res_norm, Net_W = {}, {}
        D_res = np.zeros((d, d))
        adj = (np.ones((d, d)) - np.eye(d)).astype(int)
        for i, j in itertools.product(range(d), range(d)):
            if i == j:
                continue
            D_res[i,j] = cal_ppl(X, act, nu, intv, i, j, adj[:,j], Res_norm, 
                                   linear, Net_W)
        D_pair = D_res - D_res.T
        D_pair = np.nan_to_num(D_pair, nan=0, posinf=0, neginf=0)
        while np.any(D_pair < 0):
            index = np.unravel_index(D_pair.argmin(), D_pair.shape)
            adj[index], D_res[index] = 0, np.inf
            j = index[1]
            for i in range(d):
                if adj[i,j] == 0:
                    continue
                D_res[i,j] = cal_ppl(X, act, nu, intv, i, j, adj[:,j], Res_norm, 
                                        linear, Net_W)
            D_pair = D_res - D_res.T
            D_pair = np.nan_to_num(D_pair, nan=0, posinf=0, neginf=0)
        np.fill_diagonal(D_res, np.inf)
        while not nx.is_directed_acyclic_graph(nx.DiGraph(adj)):
            index = np.unravel_index(D_res.argmin(), D_res.shape)
            adj[index], D_res[index] = 0, np.inf
            j = index[1]
            for i in range(d):
                if adj[i,j] == 0:
                    continue
                D_res[i,j] = cal_ppl(X, act, nu, intv, i, j, adj[:,j], Res_norm, 
                                       linear, Net_W)
        adj_list.append(adj)
    return adj_list