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
    parser.add_argument('--lr', type = float, help = 'learning rate for GCN')
    parser.add_argument('--epochs', type = int, help = 'training epochs for GCN')
    parser.add_argument('--batch_size', type = int, help = 'batch size when training GCN')
    parser.add_argument('--units', type = int, help = 'hidden size of GCN')
    parser.add_argument('--layers', type = int, help = 'number of layers in GCN')
    parser.add_argument('--dropout', type = float, help = 'dropout rate in GCN')
    args = parser.parse_args()
    args = Dict(args)
    return args

def gcn_run(data):
    bpar = b_estim(data, args)
    
    T = data.T.item()
    n_nodes = data.num_nodes
    n_cls = data.y.max().item() + 1
    model = gnn.GCN(1, args.units, args.layers, T * n_cls, args.dropout)
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
tester = Tester(args.data_dir, args.device, gcn_run)
tester.test([args.dataset], rep = 1)
tester.save(args.output)
