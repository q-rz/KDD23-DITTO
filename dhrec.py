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
    args = parser.parse_args()
    args = Dict(args)
    return args

def pcdsvc_greedy(bpar, G, y):
    n = G.number_of_nodes()
    l1s = -np.log(1. - bpar.pI)
    lr = -np.log(bpar.pR)
    x = np.where(y == 2, 2, 0)
    we = l1s
    ws = np.zeros(n, dtype = np.float32)
    wi = np.zeros(n, dtype = np.float32)
    for u in range(n):
        if y[u] == 0:
            for v in G.neighbors(u):
                wi[v] += l1s
        elif y[u] == 1:
            ws[u] -= 1.
        else:
            ws[u] += lr - 1.
            wi[u] += lr
    # R --> I
    pbar = tqdm()
    while True:
        mvs = []
        for u in range(n):
            if y[u] >= 1 and x[u] != 1:
                cur = wi[u]
                for v in G.neighbors(u):
                    if y[v] >= 1 and x[v] != 1:
                        cur -= we
                mvs.append((cur, u))
        if len(mvs) == 0:
            break
        mv = min(mvs, key = lambda mv: mv[0])
        if mv[0] >= 0.:
            dom = True
            for u in range(n):
                if y[u] == 1:
                    dm = (x[u] == 1)
                    for v in G.neighbors(u):
                        dm |= (x[v] == 1)
                        if dm:
                            break
                    dom &= dm
            if dom:
                break
        x[mv[1]] = 1
        pbar.update(1)
    # I --> S
    for u in range(n):
        if x[u] == 2 and ws[u] < 0:
            dm = False
            for v in G.neighbors(u):
                dm |= (x[v] == 1)
                if dm:
                    break
            if dm:
                x[u] = 0
                pbar.update(1)
    pbar.close()
    return x

def pcdsvc_run(data):
    bpar = b_estim(data, args)
    with torch.no_grad():
        T = data.T.item()
        n_nodes = data.num_nodes
        n_cls = data.y.max().item() + 1
        obs = data.y[:, -1].cpu().detach().numpy()
        G = pyg.utils.to_networkx(data, to_undirected = True, remove_self_loops = True)
        y_pred = np.zeros((n_nodes, T + 1), dtype = np.int32)
        y_pred[:, -1] = obs
        for t in trange(T, 0, -1):
            y_pred[:, t - 1] = pcdsvc_greedy(bpar, G, y_pred[:, t])
        return torch.tensor(np.minimum(y_pred, n_cls - 1), dtype = torch.long, device = data.y.device)

args = get_args()
seed_all(args.seed)
tester = Tester(args.data_dir, args.device, pcdsvc_run)
tester.test([args.dataset], rep = 1)
tester.save(args.output)
