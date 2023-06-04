from inc.utils import *

class MLP(nn.Module):
    def __init__(self, units_list, act_fn = F.silu):
        super().__init__()
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = act_fn
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
        return x

gnn_aggr_sum  = lambda src, index, dim = 0, dim_size = None: pysc.scatter_sum(src = src, index = index, dim = dim, dim_size = dim_size)
gnn_aggr_mean = lambda src, index, dim = 0, dim_size = None: pysc.scatter_mean(src = src, index = index, dim = dim, dim_size = dim_size)
gnn_aggr_max  = lambda src, index, dim = 0, dim_size = None: pysc.scatter_max(src = src, index = index, dim = dim, dim_size = dim_size)[0]
gnn_aggr_min  = lambda src, index, dim = 0, dim_size = None: pysc.scatter_min(src = src, index = index, dim = dim, dim_size = dim_size)[0]

class GNN(nn.Module):
    def __init__(self, v_in, e_in, hid, dep, act_fn = F.silu, agg_fn = gnn_aggr_mean):
        super().__init__()
        self.v_in = v_in
        self.e_in = e_in
        self.hid = hid
        self.dep = dep
        self.act_fn = act_fn
        self.agg_fn = agg_fn
        if self.v_in != self.hid:
            self.v_lin0 = nn.Linear(self.v_in, self.hid)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.hid, self.hid) for i in range(self.dep)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.hid, self.hid) for i in range(self.dep)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.hid, self.hid) for i in range(self.dep)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.hid, self.hid) for i in range(self.dep)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.hid) for i in range(self.dep)])
        if self.e_in != self.hid:
            self.e_lin0 = nn.Linear(self.e_in, self.hid)
        self.e_lins1 = nn.ModuleList([nn.Linear(self.hid, self.hid) for i in range(self.dep)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.hid) for i in range(self.dep)])
    def forward(self, node_attr, edge_index, edge_attr):
        n_nodes = node_attr.size(dim = 0)
        x = node_attr
        w = edge_attr
        if self.v_in != self.hid:
            x = self.v_lin0(x)
            x = self.act_fn(x)
        if self.e_in != self.hid:
            w = self.e_lin0(w)
            w = self.act_fn(w)
        for i in range(self.dep):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins1[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(src = w2 * x2[edge_index[1]], index = edge_index[0], dim = 0, dim_size = n_nodes)))
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return x, w
