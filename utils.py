import numpy as np

###############################################################################
#                                  general
###############################################################################
class System:
    def __init__(self, d, nu, sigma, linear):
        self.d, self.nu, self.sigma = d, nu, sigma
        self.linear = linear
        while True:
            Exist = np.random.randint(2, size=(d, d))
            Sign = 2*np.random.randint(2, size=(d, d)) - 1
            B = np.triu(np.random.uniform(0.5, 2, size=(d, d)), k=1) * Exist * Sign
            Exist = np.random.randint(2, size=(d, d))
            Sign = 2*np.random.randint(2, size=(d, d)) - 1
            BI = np.triu(np.random.uniform(0.5, 2, size=(d, d)), k=1) * Exist * Sign
            if all([not np.array_equal(B[:,n], BI[:,n]) for n in range(1, d)]):
                break
        if not linear:
            B, BI = (B != 0).astype(int), (B != 0).astype(int)
        self.perm = np.random.permutation(d)
        self.inv_perm = np.argsort(self.perm)
        self.B_topo, self.BI_topo = B, BI
        self.B, self.BI = B[self.perm][:,self.perm], BI[self.perm][:,self.perm]
        if not linear:
            self.F_topo, self.FI_topo = [], []
            for n in range(d):
                num = np.sum(self.B_topo[:,n]).astype(int)
                if num > 0:
                    W = np.random.uniform(0.5, 2, size=(num + 1, 64))
                    W[np.random.rand(*W.shape) < 0.5] *= -1
                    self.F_topo.append(W)
                else:
                    self.F_topo.append(np.array([]))
                num = np.sum(self.BI_topo[:,n]).astype(int)
                if num > 0:
                    W = np.random.uniform(0.5, 2, size=(num + 1, 64))
                    W[np.random.rand(*W.shape) < 0.5] *= -1
                    self.FI_topo.append(W)
                else:
                    self.FI_topo.append(np.array([]))
    
    def step(self, act):
        act_topo = act[self.inv_perm]
        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        Ba = constr_post(act_topo, self.B_topo, self.BI_topo)
        Eps = self.nu + np.random.normal(size=self.d) * self.sigma
        X = np.zeros(self.d)
        for n in range(self.d):
            if self.linear:
                X[n] = np.dot(Ba[:,n], X) + Eps[n]
            else:
                W = self.F_topo[n] if act_topo[n] == 0 else self.FI_topo[n]
                if W.size > 0:
                    X[n] = sigmoid(X[Ba[:,n]!=0] @ W[:-1,:]) @ W[-1,:]
                X[n] += Eps[n]
        return X[self.perm]


def constr_post(act, B, BI):
    Ba = np.zeros(B.shape, dtype=B.dtype)
    for n in range(act.size):
        Ba[:,n] = B[:,n] if (act[n] == 0) else BI[:,n]
    return Ba