from inc.diffus import *
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
    parser.add_argument('--lr', type = float, help = 'learning rate for GIN')
    parser.add_argument('--epochs', type = int, help = 'training epochs for GIN')
    parser.add_argument('--batch_size', type = int, help = 'batch size when training GIN')
    parser.add_argument('--units', type = int, help = 'hidden size of GIN')
    parser.add_argument('--layers', type = int, help = 'number of layers in GIN')
    parser.add_argument('--dropout', type = float, help = 'dropout rate in GIN')
    args = parser.parse_args()
    args = Dict(args)
    return args

class BPar(nn.Module):
    def __init__(self, pI, pR, device, pImax):
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
    device = data.y.device
    bpar = BPar(pI = pI, pR = pR, device = device, pImax = args.b_pImax)
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

def gin_run(data):
    bpar = b_estim(data, args)
    
    T = data.T.item()
    n_nodes = data.num_nodes
    n_cls = data.y.max().item() + 1
    model = gnn.GIN(1, args.units, args.layers, T * n_cls, args.dropout)
    model = model.to(args.device)

    # train
    model.train()
    I0 = (data.y[:, 0] == SIR_STATES.I).long().sum().item()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    pbar = trange(1, args.epochs + 1)
    for epoch in pbar:
        opt.zero_grad()
        labels = diffus_gen(T = data.T.item(), n_nodes = data.num_nodes, edge_index = data.edge_index, I0 = I0, n_samples = args.batch_size, pI = bpar.pI, pR = bpar.pR).transpose(0, 2) # (batch, nodes, T + 1)
        x = labels[:, :, T].reshape((-1, 1))
        edge_index = (data.edge_index.unsqueeze(dim = 2) + n_nodes * torch.arange(args.batch_size, dtype = torch.long, device = x.device)).flatten(start_dim = 1)
        logits = F.log_softmax(model(x.float(), edge_index).view(-1, n_cls), dim = -1)
        loss = F.nll_loss(logits, labels[:, :, : T].flatten())
        pbar.set_description(f'epoch={epoch} loss={loss.item():.4f}')
        opt.step()
    
    # infer
    with torch.no_grad():
        model.eval()
        y_pred = model(data.y[:, -1 :].float(), data.edge_index).view(n_nodes, T, n_cls).argmax(dim = 2)
        return y_pred.clone()

args = get_args()
seed_all(args.seed)
tester = Tester(args.data_dir, args.device, gin_run)
tester.test([args.dataset], rep = 1)
tester.save(args.output)
