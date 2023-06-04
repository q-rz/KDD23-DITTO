from inc.header import *

class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def seed_np(seed):
    random.seed(seed)
    np.random.seed(seed + 1)

def seed_torch(seed):
    torch.manual_seed(seed + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + 3)

def seed_all(seed):
    seed_np(seed)
    seed_torch(seed)

def file_require(url, fdir, fname = None, z = None):
    if not osp.exists(fdir):
        os.makedirs(fdir)
    if url is not None:
        zpath = osp.join(fdir, osp.basename(url))
    if fname is None:
        fname = osp.basename(url)
        if z == 'gz':
            assert url.endswith(f'.{z}'), f'not .{z}'
            fname = fname[: -(1 + len(z))]
    fpath = osp.join(fdir, fname)
    if not osp.exists(fpath):
        os.system(f'wget {url} -P {fdir}')
        if z == 'gz':
            os.system(f'gzip -d {zpath}')
        elif z == 'zip':
            os.system(f'unzip -o {zpath} -d {fdir}')
        assert osp.exists(fpath), f'failed in acquiring {fname}'
    return fpath

def nx_layout(Gnx):
    if not hasattr(Gnx, 'layout'):
        Gnx.layout = nx.kamada_kawai_layout(Gnx)
    return Gnx.layout

def nx_plot(title, Gnx, y, nx_with_labels = False, plt_show = True):
    node_colors = np.full(Gnx.number_of_nodes(), '#444444')
    node_colors[y == STATES.S] = '#0000FF'
    node_colors[y == STATES.I] = '#FF0000'
    node_colors[y == STATES.R] = '#00FF00'
    nx.draw_networkx(Gnx, pos = nx_layout(Gnx), with_labels = nx_with_labels, node_size = 10, width = 0.2, node_color = node_colors)
    plt.title(title)
    if plt_show:
        plt.show()

def torch2np(x):
    return x.cpu().detach().numpy()

def torch_device(device = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)

def torch_reset(mod):
    if callable(getattr(mod, 'reset_parameters', None)):
        mod.reset_parameters()
    elif isinstance(mod, nn.Module):
        for child in mod.children():
            torch_reset(child)

def torch_log(x, log0 = -np.inf):
    x = x.log()
    if log0 is not None:
        x[x.isnan()] = log0
    return x

def math_log(x, log0 = -np.inf):
    return math.log(x) if x > 0 else log0
