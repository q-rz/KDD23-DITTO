from inc.diffus import *
from inc.nn import *
from inc.test import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, help = 'dataset name')
    parser.add_argument('--seed', type = int, help = 'random seed')
    parser.add_argument('--data_dir', type = str, help = 'dataset folder')
    parser.add_argument('--output', type = str, help = 'output file name')
    parser.add_argument('--device', type = torch.device, help = 'torch device')
    parser.add_argument('--b_pI0', type = float, help = 'initial infection rate in diffusion parameter estimation')
    parser.add_argument('--b_pR0', type = float, help = 'initial recovery rate in diffusion parameter estimation')
    parser.add_argument('--b_steps', type = int, help = 'optimization steps in diffusion parameter estimation')
    parser.add_argument('--b_lr', type = float, help = 'learning rate in diffusion parameter estimation')
    parser.add_argument('--q_steps', type = int, help = 'training steps for the proposal model')
    parser.add_argument('--q_lr', type = float, help = 'learning rate for the proposal model')
    parser.add_argument('--q_hid', type = int, help = 'hidden size of the proposal model')
    parser.add_argument('--q_gnn', type = int, help = 'number of layers of the GNN in the proposal model')
    parser.add_argument('--q_mlp', type = int, help = 'number of layers of the MLP in the proposal model')
    parser.add_argument('--q_samples', type = int, help = 'sample size to estimate the loss function of the proposal model')
    parser.add_argument('--q_zlim', type = int, help = 'a hyperparameter to stablize gradient')
    parser.add_argument('--p_coef', type = float, help = 'the coefficient gamma in the initial distribution P[y_0]')
    parser.add_argument('--t_samples', type = int, help = 'MCMC sample size')
    parser.add_argument('--t_steps', type = int, help = 'MCMC steps')
    parser.add_argument('--t_keep', type = float, help = 'moving average in MCMC')
    args = parser.parse_args()
    args = Dict(args)
    return args

class QNet(nn.Module):
    @classmethod
    def make(cls, data, args):
        return cls(
            eidx = data.edge_index,
            T = data.T.item(),
            hid = args.q_hid,
            gnn = args.q_gnn,
            mlp = args.q_mlp,
            n_nodes = data.num_nodes,
            zlim = args.q_zlim,
        ).to(args.device)
    def __init__(self, eidx, T, hid, gnn, mlp, n_nodes, zlim):
        super().__init__()
        self.eidx = eidx
        self.device = self.eidx.device
        self.n_nodes = n_nodes
        self.n_inf = self.n_nodes + 2
        self.n_edges = self.eidx.size(dim = 1)
        self.zlim = zlim
        self.T = T
        self.hid = int(hid)
        self.gnn_dep = int(gnn)
        self.mlp_dep = int(mlp)
        self.w = nn.Parameter(data = torch.randn((self.n_edges, self.hid), dtype = torch.float32, device = self.device), requires_grad = True)
        self.gnn = GNN(v_in = 1, e_in = self.hid, hid = self.hid, dep = self.gnn_dep)
        self.mlp = MLP([self.hid] * self.mlp_dep + [2 * self.T])
        self.rem = (pyg.utils.degree(self.eidx[1], num_nodes = self.n_nodes).long().unsqueeze(dim = 1) + 1).detach().clone() # (nodes, 1)
        self.neighbs = [[] for u in range(self.n_nodes)]
        for i in range(self.n_edges):
            self.neighbs[self.eidx[0, i].item()].append(self.eidx[1, i].item())
        for u in range(self.n_nodes):
            self.neighbs[u] = torch.tensor(self.neighbs[u], dtype = torch.long, device = self.device)
        self.zero = torch.tensor(0., dtype = torch.float, device = self.device)
    def clamp_z(self, z):
        return z.clamp(-self.zlim, self.zlim)
    def forward(self, y, orig = False): # y: (nodes, samples)
        n_nodes, n_samples = y.size()
        y = y.T.reshape((-1, 1)) # (samples*nodes, 1)
        eidx = (self.eidx.unsqueeze(dim = 1) + n_nodes * torch.arange(n_samples, dtype = torch.long, device = y.device).unsqueeze(dim = -1)).reshape((2, -1)) # (2, samples*edges)
        w = self.w.repeat(n_samples, 1) # (samples, hid)
        z, e = self.gnn(y.float(), eidx, w)
        z = self.mlp(z) # (samples*nodes, 2*T)
        z = z.T.reshape((2 * self.T, n_samples, -1)) # (2*T, samples, nodes)
        zI, zR = z[: self.T], z[self.T :] # (T, samples, nodes)
        zI, zR = zI.transpose(1, 2), zR.transpose(1, 2) # (T, nodes, samples)
        if orig:
            return zI, zR, self.clamp_z(zI), self.clamp_z(zR)
        else:
            return self.clamp_z(zI), self.clamp_z(zR)
    def lik(self, Y): # Y: (T+1, nodes, samples)
        n_samples = Y.size(dim = 2)
        zI0, zR0, zI, zR = self.forward(Y[-1], orig = True) # (T, nodes, samples)
        zI = zI.clone().detach().requires_grad_(True); zI.retain_grad()
        zR = zR.clone().detach().requires_grad_(True); zR.retain_grad()
        # R->I
        qR = torch.sigmoid(zR) # (T, nodes, samples) # prob of R->I
        lR1 = torch_log(qR) # (T, nodes, samples)
        lR0 = torch_log(1. - qR) # (T, nodes, samples)
        with torch.no_grad():
            mskR = (Y[1 :] == SIR_STATES.R) # (T, nodes, samples)
            trsR = (Y[: -1] != SIR_STATES.R) # (T, nodes, samples)
        # I->S
        zI_, uid = zI.sort(dim = 1, descending = True) # (T, nodes, samples)
        qI = torch.sigmoid(zI_) # (T, nodes, samples) # prob of I->S
        lI1 = torch_log(qI) # (T, nodes, samples)
        lI0 = torch_log(1. - qI) # (T, nodes, samples)
        with torch.no_grad():
            mskI = ((Y[1 :] >= SIR_STATES.I) & (Y[: -1] <= SIR_STATES.I)).flatten() # (T * nodes * samples)
            trsI = ((Y[: -1] != SIR_STATES.I)).flatten() # (T * nodes * samples)
            rem = torch.where(mskI, self.rem.expand(self.T, -1, n_samples).flatten(), self.n_inf) # (T * nodes * samples)
            ptr = torch.arange(self.T, dtype = torch.long, device = self.device).unsqueeze(dim = 1) * self.n_nodes # (T, 1)
            for i in range(uid.size(dim = 1)):
                uidi = (ptr + uid[:, i]).flatten() * n_samples # (T * samples)
                mski = mskI[uidi] # (T * samples)
                if mski.max():
                    trsi = trsI[uidi] # (T * samples)
                    vids, degi = [], [0]
                    for t in range(self.T):
                        for j in range(n_samples):
                            u = uid[t, i, j]
                            vids.append((t * self.n_nodes + u.unsqueeze(dim = 0)) * n_samples)
                            vid = self.neighbs[u.item()]
                            vids.append((t * self.n_nodes + vid) * n_samples)
                            degi.append(vid.size(dim = 0) + 1)
                    vids = torch.cat(vids, dim = 0) # (T * sum neighbs)
                    degi = torch.tensor(degi, dtype = torch.long, device = self.device) # (1 + T * samples)
                    indptr = degi.cumsum(dim = 0) # (1 + T * samples)
                    degi = degi[1 :] # (T * samples)
                    rems = rem.flatten()[vids] # (T * sum neighbs)
                    opti = (pysc.segment_min_csr(src = rems, indptr = indptr)[0] > 1) # (T * samples)
                    rem.flatten()[vids] = torch.where(mski.repeat_interleave(repeats = degi), torch.where(trsi.repeat_interleave(repeats = degi), rems - 1, self.n_inf), rems) # (T * sum neighbs)
                    mskI[uidi] &= opti # (T * samples)
        # likR + likI
        lik = (torch.where(mskR, torch.where(trsR, lR1, lR0), self.zero).view(-1, n_samples) + torch.where(mskI.view(-1, n_samples), torch.where(trsI.view(-1, n_samples), lI1.view(-1, n_samples), lI0.view(-1, n_samples)), self.zero)).sum(dim = 0) # (samples,)
        return lik, zI0, zR0, zI, zR # (samples,)
    @torch.no_grad()
    def clamp_grad(self, z0, grad):
        return torch.where(z0 < self.zlim, torch.where(z0 > -self.zlim, grad, F.relu(grad)), -F.relu(-grad))
    def backward(self, loss, zI0, zR0, zI, zR):
        loss.backward()
        z0 = torch.stack([zI0, zR0], dim = 0)
        z0.backward(torch.stack([self.clamp_grad(zI0, zI.grad), self.clamp_grad(zR0, zR.grad)], dim = 0))
    @torch.no_grad()
    def samp(self, y, zI, zR, n_samples, compute_lik = False): # y: (nodes,); zI, zR: (T, nodes, 1)
        zI, uid = zI.sort(dim = 1, descending = True) # (T, nodes, 1)
        uid = uid.squeeze(dim = 2) # (T, nodes)
        qI = torch.sigmoid(zI) # (T, nodes, 1) # prob of I->S
        xI = SIR_STATES.I - qI.expand(-1, -1, n_samples).bernoulli().long() # (T, nodes, samples) # 1 for I->S
        lI = torch_log(torch.where(xI != SIR_STATES.I, qI, 1. - qI)) # (T, nodes, samples)
        qR = torch.sigmoid(zR) # (T, nodes, 1) # prob of R->I
        xR = SIR_STATES.R - qR.expand(-1, -1, n_samples).bernoulli().long() # (T, nodes, samples) # 1 for R->I
        lR = torch_log(torch.where(xR != SIR_STATES.R, qR, 1. - qR)) # (T, nodes, samples)
        y = y.unsqueeze(dim = 1).expand(-1, n_samples) # (nodes, samples)
        Y = torch.empty(self.T, self.n_nodes, n_samples, dtype = torch.long, device = self.device) # (T, nodes, samples)
        if compute_lik:
            lik = self.zero
        for t in range(self.T - 1, -1, -1):
            # R->I
            msk = (y == SIR_STATES.R) # (nodes, samples)
            y = torch.where(msk, xR[t], y) # (nodes, samples)
            if compute_lik:
                lik = lik + torch.where(msk, lR[t], self.zero).sum(dim = 0) # (samples,)
            # I->S
            msk = (y == SIR_STATES.I) # (nodes, samples)
            rem = torch.where(msk, self.rem, self.n_inf) # (nodes, samples)
            for i, u in enumerate(uid[t]):
                if msk[u].max():
                    vid = self.neighbs[u.item()] # (neighbs,)
                    opt = (rem[u] > 1) & (rem[vid].min(dim = 0).values > 1) # (samples,)
                    msk_opt = msk[u] & opt
                    y[u] = torch.where(msk_opt, xI[t, i], y[u]) # (samples,)
                    trs = (y[u] != SIR_STATES.I) # (samples,)
                    rem[u] = torch.where(msk[u], torch.where(trs, rem[u] - 1, self.n_inf), rem[u]) # (samples,)
                    rem[vid] = torch.where(msk[u].unsqueeze(dim = 0), torch.where(trs.unsqueeze(dim = 0), rem[vid] - 1, self.n_inf), rem[vid]) # (neighbs, samples)
                    msk[u] = msk_opt
            Y[t] = y
            if compute_lik:
                lik = lik + torch.where(msk[uid[t]], lI[t], self.zero).sum(dim = 0) # (samples,)
        Y = Y.detach().clone()
        if compute_lik:
            lik = lik.detach().clone()
            return Y, lik
        else:
            return Y 

def q_loss(q_net, data, I0, bpar, n_samples):
    T = data.T.item()
    n_nodes = data.num_nodes
    Y = diffus_gen(T = T, n_nodes = n_nodes, edge_index = data.edge_index, I0 = I0, n_samples = n_samples, pI = bpar.pI, pR = bpar.pR) # (T+1, nodes, samples)
    q_liks, zI0, zR0, zI, zR = q_net.lik(Y = Y) # (samples,)
    return -q_liks.mean(), zI0, zR0, zI, zR

def q_train(data, bpar, args):
    I0 = (data.y[:, 0] == 1).long().sum().item()
    q_net = QNet.make(data, args)
    q_net.train()
    opt = optim.AdamW(q_net.parameters(), lr = args.q_lr)
    pbar = trange(1, args.q_steps + 1)
    for step in pbar:
        opt.zero_grad()
        loss, zI0, zR0, zI, zR = q_loss(q_net, data, I0, bpar, args.q_samples)
        pbar.set_description(f'[step={step}] loss={loss.item():.4f}')
        q_net.backward(loss, zI0, zR0, zI, zR)
        opt.step()
    q_net.eval()
    return q_net

@torch.no_grad()
def t_mcmc(data, bpar, q_net, args, keepdim = True):
    I0 = (data.y[:, 0] == 1).long().sum().item()
    zI, zR = q_net(data.y[:, -1 :]) # (T, nodes, 1)
    X, lqX = q_net.samp(data.y[:, -1], zI, zR, args.t_samples, compute_lik = True) # (T, nodes, samples)
    lpX = diffus_liks(Y = X, edge_index = data.edge_index, I0 = I0, coef = args.p_coef, pI = bpar.pI, pR = bpar.pR) # (samples,)
    tI_avg = data_make_t(X, SIR_STATES.I, dim = 0).float().mean(dim = 1, keepdim = keepdim) # (nodes, 1)
    tR_avg = data_make_t(X, SIR_STATES.R, dim = 0).float().mean(dim = 1, keepdim = keepdim) # (nodes, 1)
    pbar = trange(1, args.t_steps + 1)
    for step in pbar:
        Y, lqY = q_net.samp(data.y[:, -1], zI, zR, args.t_samples, compute_lik = True) # (T, nodes, samples)
        lpY = diffus_liks(Y = Y, edge_index = data.edge_index, I0 = I0, coef = args.p_coef, pI = bpar.pI, pR = bpar.pR) # (samples,)
        a = torch.rand(args.t_samples, device = args.device) <= torch.exp(lpY + lqX - lpX - lqY) # (samples,) # Hastings MCMC
        X = torch.where(a, Y, X) # (T, nodes, samples)
        lqX = torch.where(a, lqY, lqX) # (samples,)
        lpX = torch.where(a, lpY, lpX) # (samples,)
        tI = data_make_t(X, SIR_STATES.I, dim = 0).float().mean(dim = 1, keepdim = keepdim) # (nodes, 1)
        tR = data_make_t(X, SIR_STATES.R, dim = 0).float().mean(dim = 1, keepdim = keepdim) # (nodes, 1)
        tI_avg = args.t_keep * tI_avg + (1. - args.t_keep) * tI # (nodes, 1)
        tR_avg = args.t_keep * tR_avg + (1. - args.t_keep) * tR # (nodes, 1)
    return tI_avg, tR_avg # (nodes, 1)

def main(data):
    # estimate diffusion parameters
    bpar = b_estim(data, args)
    print(f'[est] pI={bpar.pI:.4f}, pR={bpar.pR:.4f}', flush = True)
    # train a proposal network
    q_net = q_train(data, bpar, args)
    # estimate transition times
    tI, tR = t_mcmc(data, bpar, q_net, args, keepdim = True) # (nodes, 1)
    T = data.T.item()
    tI = tI.round().long()
    tR = tR.round().long()
    # compose a history
    with torch.no_grad():
        y_pred = torch.zeros_like(data.y) # (nodes, T+1)
        y_pred.scatter_(dim = 1, index = torch.minimum(tI, data.T), src = torch.full_like(tI, 1))
        y_pred.scatter_(dim = 1, index = torch.minimum(tR, data.T), src = torch.full_like(tR, 2))
        y_pred = y_pred[:, : data.T.item()].cummax(dim = 1).values
        return y_pred

args = get_args()
tester = Tester(args.data_dir, args.device, main)
tester.test([args.dataset], seed = args.seed, rep = 1)
tester.save(args.output)
