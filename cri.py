from inc.diffus import *
from inc.test import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, help = 'dataset name')
    parser.add_argument('--seed', type = int, help = 'random seed')
    parser.add_argument('--data_dir', type = str, help = 'dataset folder')
    parser.add_argument('--output', type = str, help = 'output file name')
    parser.add_argument('--device', type = torch.device, help = 'torch device')
    args = parser.parse_args()
    args = Dict(args)
    return args

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(SRC = None, DIST = None)
def cri_dist(G, s, t): # slower than pre-computing all but prevents OOM
    if cri_dist.SRC == t:
        s, t = t, s
    if cri_dist.SRC != s:
        cri_dist.SRC = s
        cri_dist.DIST = dict()
        for v, d in nx.single_source_shortest_path_length(G, s).items():
            cri_dist.DIST[v] = d
        cri_dist.DIST[s] = 0
    return cri_dist.DIST[t]

def cri_cluster(G, obs, T):
    n = obs.shape[0]
    VI = [u for u in range(n) if obs[u] == 1]
    diam, diam_u, diam_v = 0, None, None
    for u in tqdm(VI, desc = 'diam'):
        for v in VI:
            if cri_dist(G, u, v) > diam:
                diam, diam_u, diam_v = cri_dist(G, u, v), u, v
    INF = diam + 1
    B = [diam_u, diam_v]
    pbar = tqdm(desc = 'cluster')
    while max([min([cri_dist(G, u, s) for s in B]) for u in VI]) > T:
        B.append(max([u for u in VI if u not in B], key = lambda u: min([cri_dist(G, u, s) for s in B])))
        pbar.update(1)
    Vs = {s: [] for s in B}
    for u in VI:
        Vs[min(B, key = lambda s: cri_dist(G, u, s))].append(u)
    return [Vi for s, Vi in Vs.items() if len(Vi) > 0], VI

def cri_rev_infect(G, VI, Vi, y):
    n = G.number_of_nodes()
    ni = len(Vi)
    f = set()
    g = [dict() for u in range(n)]
    for x in Vi:
        f.add((x, x))
        for v in G[x]:
            f.add((v, x))
    t = 0
    while max([len(g[u]) for u in range(n)]) < ni:
        ff = set()
        for u, x in f:
            g[u][x] = t
            for v in G[u]:
                if x not in g[v]:
                    ff.add((v, x))
        f = ff
        t = t + 1
    s = min([u for u in range(n) if len(g[u]) == ni], key = lambda u: sum(g[u].values()))
    tI = g[s]
    for u in tI.items():
        y[u, t :] = 1

@torch.no_grad()
def cri_run(data):
    T = data.T.item()
    n_nodes = data.num_nodes
    n_cls = data.y.max().item() + 1
    obs = data.y[:, -1].cpu().detach().numpy() & 1 # S,R -> 0; I -> 1
    G = pyg.utils.to_networkx(data, to_undirected = True, remove_self_loops = True)
    Vs, VI = cri_cluster(G, obs, T)
    y_pred = np.zeros((n_nodes, T + 1), dtype = np.int32)
    for Vi in tqdm(Vs, desc = 'rev_infect'):
        cri_rev_infect(G, VI, Vi, y_pred)
    return torch.tensor(np.minimum(y_pred, n_cls - 1), dtype = torch.long, device = data.y.device)

args = get_args()
seed_all(args.seed)
tester = Tester(args.data_dir, args.device, cri_run)
tester.test([args.dataset], rep = 1)
tester.save(args.output)
