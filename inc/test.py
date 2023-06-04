from inc.data import *

@torch.no_grad()
def test_fix_obs(data, y_pred):
    return torch.cat([y_pred[:, : data.T.item()], data.y[:, -1 :]], dim = 1)

@torch.no_grad()
def test_skm(skm_fn, data, y_pred, **kwargs):
    y_true = torch2np(data.y).flatten()
    y_pred = test_fix_obs(data, y_pred)
    y_pred = torch2np(y_pred).flatten()
    return float(skm_fn(y_true, y_pred, **kwargs))

@torch.no_grad()
def test_nrmse(data, y_pred):
    y_pred = test_fix_obs(data, y_pred)
    tI_pred = data_make_t(y_pred, SIR_STATES.I, dim = -1)
    mse = skm.mean_squared_error(torch2np(data.tI), torch2np(tI_pred))
    if hasattr(data, 'tR'):
        tR_pred = data_make_t(y_pred, SIR_STATES.R, dim = -1)
        mseR = skm.mean_squared_error(torch2np(data.tR), torch2np(tR_pred))
        mse = (mse + mseR) / 2.
    nrmse = np.sqrt(mse) / (data.T.item() + 1)
    return float(nrmse)
    

TEST_METRICS = {
    None: lambda data, y_pred: y_pred.tolist(),
    'acc': fnt.partial(test_skm, skm.accuracy_score),
    'prc': fnt.partial(test_skm, skm.precision_score, average = 'macro', zero_division = 0),
    'rec': fnt.partial(test_skm, skm.recall_score,    average = 'macro', zero_division = 0),
    'f1':  fnt.partial(test_skm, skm.f1_score,        average = 'macro', zero_division = 0),
    'nrmse': test_nrmse,
}

class Tester:
    def __init__(self, data_dir, device, model_fn):
        self.data_dir = data_dir
        self.device = device
        self.model_fn = model_fn
        self.res = dict()
    def test_once(self, dataset, seed = None):
        data = data_load(dataset, self.data_dir, self.device)
        if seed is not None:
            seed_all(seed)
        y_pred = self.model_fn(data)
        if dataset not in self.res:
            self.res[dataset] = dict()
        res = self.res[dataset]
        for metric, fn in TEST_METRICS.items():
            if metric not in res:
                res[metric] = list()
            self.res[dataset][metric].append(fn(data, y_pred))
    def test_dataset(self, dataset, seed = None, rep = 5, verbose = True):
        for i in range(rep):
            if verbose:
                print(f'[{dataset} #{i}]', flush = True)
            self.test_once(dataset, seed = None if seed is None else (seed ^ i))
            if verbose:
                print(f'[{dataset} #{i}]', ', '.join([f'{metric}={scores[-1]:.4f}' for metric, scores in self.res[dataset].items() if metric is not None]), flush = True)
    def test(self, datasets = None, seed = None, **kwargs):
        if datasets is None:
            datasets = DATASETS.keys()
        for dataset in datasets:
            self.test_dataset(dataset = dataset, seed = seed, **kwargs)
    def print(self, brief = True):
        for dataset, metrics in self.res.items():
            for metric, scores in metrics.items():
                if metric is not None:
                    print(f'{dataset} {metric}:', end = '')
                    if brief:
                        print(f' {np.mean(scores):.4f} ({np.std(scores):.4f})')
                    else:
                        for score in scores:
                            print(f' {score:.4f}', end = '')
                        print('')
    def save(self, f, verbose = True):
        return torch.save(self.res, f)
