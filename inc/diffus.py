from inc.utils import *

SIR_STATES = Dict(S = 0, I = 1, R = 2)

@torch.no_grad()
def diffus_trans(edge_index, y, wI, wR): # ei: long(2, edges); y: long(nodes, samples); wI: long(edges, samples), wR: long(nodes, samples)
    # S-->I
    y = torch.maximum(y, pysc.scatter_max(src = torch.minimum(y[edge_index[0]], wI), dim = 0, index = edge_index[1], dim_size = y.size(0))[0])
    if wR is not None:
        # I-->R
        y = torch.where((y == SIR_STATES.I) & (wR != 0), SIR_STATES.R, y)
    return y

@torch.no_grad()
def diffus_sim(edge_index, y0, WI, WR = None): # y0: (nodes, samples); WI: (T, edges, samples); WR: (T, nodes, samples)
    n_nodes, n_samples = y0.size()
    T = WI.size(dim = 0)
    Y = torch.empty(size = (T + 1, n_nodes, n_samples), dtype = y0.dtype, device = y0.device)
    Y[0] = y0
    for t in range(T):
        Y[t + 1] = diffus_trans(edge_index, Y[t], WI[t], None if WR is None else WR[t])
    return Y # (T + 1, nodes, samples)

@torch.no_grad()
def diffus_gen(T, n_nodes, edge_index, I0, n_samples, pI, pR):
    n_edges = edge_index.size(dim = 1)
    idx = torch.ones(n_samples, n_nodes, device = edge_index.device).multinomial(I0, replacement = False).T # (I0, samples)
    y0 = torch.full((n_nodes, n_samples), SIR_STATES.S, dtype = torch.long, device = edge_index.device)
    y0.scatter_(index = idx, dim = 0, src = torch.full_like(idx, SIR_STATES.I))
    WI = (torch.rand(T, n_edges, n_samples, device = y0.device) < pI).long()
    WR = (torch.rand(T, n_nodes, n_samples, device = y0.device) < pR).long() if pR > 0 else None
    return diffus_sim(edge_index, y0, WI, WR) # (T+1, nodes, samples)

def diffus_liks(Y, edge_index, I0, coef, pI, pR): # Y: (T+1, nodes, samples) # assuming Y feasible
    log1pI = torch_log(1. - pI) if isinstance(pI, torch.Tensor) else math_log(1. - pI)
    log_pR = torch_log(pR)      if isinstance(pR, torch.Tensor) else math_log(pR)
    log1pR = torch_log(1. - pR) if isinstance(pR, torch.Tensor) else math_log(1. - pR)
    T, n_nodes, n_samples = Y.size(); T -= 1
    liks = -coef * (((Y[0] == SIR_STATES.I).float().sum() - I0).abs() + (Y[0] == SIR_STATES.R).float().sum()) # (samples,)
    zero = torch.tensor(0., dtype = torch.float, device = Y.device)
    for t in range(T):
        # S->I
        eI = pysc.scatter_sum(src = (Y[t, edge_index[0]] == SIR_STATES.I).float(), dim = 0, index = edge_index[1], dim_size = n_nodes) # (nodes, samples)
        qI = (1. - pI) ** eI # (nodes, samples)
        liks = liks + torch.where(Y[t] == SIR_STATES.S, torch.where(Y[t + 1] >= SIR_STATES.I, torch_log(1. - qI), eI * log1pI), zero).sum(dim = 0) # (samples,)
        # I->R
        liks = liks + torch.where(Y[t + 1] == SIR_STATES.R, torch.where(Y[t] <= SIR_STATES.I, log_pR, log1pR), zero).sum(dim = 0) # (samples,)
    return liks

class BPar(nn.Module):
    def __init__(self, pI, pR, device, pImax = 0.9999):
        super().__init__()
        self.pI = nn.Parameter(torch.tensor(pI, dtype = torch.float, device = device), requires_grad = True)
        self.pR = nn.Parameter(torch.tensor(pR, dtype = torch.float, device = device), requires_grad = True)
        self.pImax = pImax
    @torch.no_grad()
    def clamp_(self):
        self.pI.clamp_(0., self.pImax)
        self.pR.clamp_(0., 1.)
    def __repr__(self, digits = 4):
        return f'pI={round(self.pI.item(), digits)} pR={round(self.pR.item(), digits)}'
    def dict(self):
        return Dict(pI = self.pI.item(), pR = self.pR.item())

def b_lik(bpar, data): # yT: (nodes,)
    device = data.y.device
    n_nodes = data.num_nodes
    T = data.T.item()
    ei = data.edge_index
    I0 = (data.y[:, 0] == 1).sum()
    pI, pR = bpar.pI, bpar.pR
    lSs, lIs, lRs = [], [], []
    lIs.append(torch.full((n_nodes,), I0 / n_nodes, dtype = torch.float, device = device))
    lSs.append(torch.full((n_nodes,), 1. - I0 / n_nodes, dtype = torch.float, device = device))
    lRs.append(torch.zeros(n_nodes, dtype = torch.float, device = device))
    for t in range(T):
        aI = pysc.scatter_mul(src = (1. - lIs[-1] * pI)[ei[0]], dim = 0, index = ei[1], dim_size = n_nodes)
        lS = lSs[-1] * aI
        kI = lIs[-1] + lSs[-1] * (1. - aI)
        lI = kI * (1. - pR)
        lR = lRs[-1] + kI * pR
        lSs.append(lS)
        lIs.append(lI)
        lRs.append(lR)
    lik = torch.stack([lSs[-1], lIs[-1], lRs[-1]], dim = 0) # (states, nodes)
    yT = data.y[:, -1].unsqueeze(dim = 0) # (1, nodes)
    lik = lik.gather(dim = 0, index = yT) # (1, nodes)
    lik = torch_log(lik).mean()
    return lik

def b_estim(data, args):
    T = data.T.item()
    n_nodes = data.num_nodes
    n_edges = data.edge_index.size(dim = 1)
    n_cls = data.y[:, T].max().item() + 1
    pI, pR = args.b_pI0, (args.b_pR0 if n_cls == 3 else 0.)
    #print(f'[ini] pI={pI:.4f}, pR={pR:.4f}', flush = True)
    device = data.y.device
    bpar = BPar(pI = pI, pR = pR, device = device)
    bpar.train()
    opt = optim.AdamW(bpar.parameters(), lr = args.b_lr, betas = (0.5, 0.5))
    pbar = trange(1, args.b_steps + 1)
    for step in pbar:
        opt.zero_grad()
        loss = -b_lik(bpar, data)
        loss.backward()
        opt.step()
        bpar.clamp_()
        pbar.set_description(f'[step={step}] {bpar}')
    bpar.eval()
    return bpar.dict()
