from inc.diffus import *

def data_make_t(y, x, dim = -1):
    return torch.where(*(y == x).max(dim), y.size(dim))

def data_simulate(Gnx, seed, T, diffus, params):
    sir = (diffus == 'sir')
    cfg = ndmc.Configuration()
    for key, val in params.items():
        cfg.add_model_parameter(key, val)
    diffus = dict(
        si = ndep.SIModel,
        sir = ndep.SIRModel,
    )[diffus](Gnx, seed)
    diffus.set_initial_status(cfg)
    states = {u: [y] for u, y in diffus.status.items()}
    for t in range(1, T + 1):
        diffus.iteration()
        for u, y in diffus.status.items():
            states[u].append(y)
    nx.set_node_attributes(Gnx, values = states, name = 'y')
    data = pyg.utils.from_networkx(Gnx)
    data.T = torch.tensor(T, dtype = data.y.dtype, device = data.y.device)
    data.tI = data_make_t(data.y, SIR_STATES.I)
    if sir:
        data.tR = data_make_t(data.y, SIR_STATES.R)
    return data

def data_calc_tI(df):
    dg = df.loc[df.groupby('to').time.idxmin()].drop('fr', axis = 1)
    tI = {row.to: row.time for _, row in tqdm(dg.iterrows(), total = dg.shape[0])}
    for _, row in tqdm(df.iterrows(), total = df.shape[0]):
        if row.fr in tI:
            tI[row.fr] = min(tI[row.fr], row.time)
        else:
            tI[row.fr] = 0
    return tI

def data_calc_tIR(df, T):
    nodes = list(sorted(set(df.to.tolist()).union(set(df.fr.tolist()))))
    c0 = {u: 0 for u in nodes}
    t0 = {u: T + 1 for u in nodes}
    t1 = {u: -1 for u in nodes}
    for _, row in tqdm(df.iterrows(), total = df.shape[0]):
        c0[row.fr] += 2
        c0[row.to] += 1
        t0[row.fr] = min(t0[row.fr], row.time)
        t0[row.to] = min(t0[row.to], row.time)
        t1[row.fr] = max(t1[row.fr], min(T + 1, row.time + 1))
    c00 = min(c0.values())
    #print('c00:', c00)
    tI = {u: t0[u] if c0[u] > c00 else T + 1 for u in nodes}
    tR = {u: t1[u] if t1[u] >= 0 else min(T + 1, tI[u] + 1) for u in nodes}
    return tI, tR

@torch.no_grad()
def data_make_states(T, tI, tR = None):
    n = tI.size(dim = 0)
    Y = torch.zeros(T + 2, n, dtype = tI.dtype, device = tI.device)
    Y.scatter_(dim = 0, index = tI.unsqueeze(dim = 0), src = torch.tensor(1, dtype = tI.dtype, device = tI.device).expand(1, n))
    if tR is not None:
        Y.scatter_(dim = 0, index = tR.unsqueeze(dim = 0), src = torch.tensor(2, dtype = tI.dtype, device = tI.device).expand(1, n))
    Y = Y[: T + 1].cummax(dim = 0).values.T
    return Y.detach().clone()

def data_synthetic(graph, diffus, data_dir, device):
    data_dir = osp.join(data_dir, 'synthetic')
    f_data = osp.join(data_dir, f'{graph}-{diffus}.pt')
    if osp.exists(f_data):
        return torch.load(f_data, map_location = device)
    else:
        seed = 123456789
        T = 10
        Gnx = dict(
            ba = lambda seed: nx.barabasi_albert_graph(n = 1000, m = 4, seed = seed),
            er = lambda seed: nx.fast_gnp_random_graph(n = 1000, p = 0.008, directed = False, seed = seed),
        )[graph](seed)
        Gnx = Gnx.subgraph(max(nx.connected_components(Gnx), key = len)).copy() # largest component
        params = dict(
            si = dict(fraction_infected = 0.05, beta = 0.1),
            sir = dict(fraction_infected = 0.05, beta = 0.1, gamma = 0.1),
        )[diffus]
        data = data_simulate(Gnx, seed, T, diffus, params).to(device)
        torch.save(data, f_data)
        return data

def data_prost(diffus, data_dir, device):
    '''https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001109'''
    data_dir = osp.join(data_dir, 'prost')
    f_data = osp.join(data_dir, f'prost-{diffus}.pt')
    if osp.exists(f_data):
        return torch.load(f_data, map_location = device)
    else:
        seed = 123456789
        T = 15
        f_raw = file_require('https://doi.org/10.1371/journal.pcbi.1001109.s001', data_dir, 'journal.pcbi.1001109.s001')
        df = pd.read_csv(f_raw, delimiter = ';', header = None, comment = '#')
        df.columns = ['seller', 'buyer', 'date', 'grade', 'anal', 'oral', 'mouth']
        Gnx = nx.from_edgelist([(row.seller, row.buyer) for _, row in tqdm(df.iterrows(), total = df.shape[0])])
        Gnx = Gnx.subgraph(max(nx.connected_components(Gnx), key = len)).copy() # largest component
        params = dict(
            si = dict(fraction_infected = 0.1, beta = 0.1),
            sir = dict(fraction_infected = 0.1, beta = 0.1, gamma = 0.05),
        )[diffus]
        data = data_simulate(Gnx, seed, T, diffus, params).to(device)
        torch.save(data, f_data)
        return data

def data_oregon2(diffus, data_dir, device):
    '''http://snap.stanford.edu/data/Oregon-2.html'''
    data_dir = osp.join(data_dir, 'oregon2')
    f_data = osp.join(data_dir, f'oregon2-{diffus}.pt')
    if osp.exists(f_data):
        return torch.load(f_data, map_location = device)
    else:
        seed = 123456789
        T = 15
        f_raw = file_require('http://snap.stanford.edu/data/oregon2_010526.txt.gz', data_dir, z = 'gz')
        Gnx = nx.read_edgelist(f_raw, comments = '#', delimiter = '\t')
        Gnx = Gnx.subgraph(max(nx.connected_components(Gnx), key = len)).copy() # largest component
        params = dict(
            si = dict(fraction_infected = 0.1, beta = 0.1),
            sir = dict(fraction_infected = 0.1, beta = 0.1, gamma = 0.05),
        )[diffus]
        data = data_simulate(Gnx, seed, T, diffus, params).to(device)
        torch.save(data, f_data)
        return data

def data_farmers_si(data_dir, device):
    '''https://github.com/USCCANA/netdiffuseR/blob/master/data-raw/brfarmersDiffNet.R'''
    data_dir = osp.join(data_dir, 'farmers')
    f_data = osp.join(data_dir, 'farmers-si.pt')
    if osp.exists(f_data):
        return torch.load(f_data, map_location = device)
    else:
        f_raw = file_require(None, data_dir, 'brfarmers.rdata')
        df = pyreadr.read_r('farmers/brfarmers.rdata')['brfarmers']
        netvars = [col for col in df.columns if col.startswith('net')]
        df.village = df.village.astype(np.int32)
        df.toa = df.toa.astype(np.int32)
        df.id = df.id.astype(np.int32) + 100 * df.village
        for col in netvars:
            df[col] = df[col] + 100 * df.village
        df = df[['id', 'toa'] + netvars]
        df.toa -= df.toa.min()
        tI = {row.id: row.toa for _, row in df.iterrows()}
        nodes = set(df.id)
        edges = list({(min(row.id, row[col]), max(row.id, row[col])) for _, row in df.iterrows() for col in netvars if row[col] in nodes})
        Gnx = nx.from_edgelist(edges)
        nx.set_node_attributes(Gnx, values = tI, name = 'tI')
        Gnx = Gnx.subgraph(max(nx.connected_components(Gnx), key = len)).copy() # largest component
        data = pyg.utils.from_networkx(Gnx).to(device)
        data.tI -= data.tI.min()
        data.T = (data.tI.max() - 1).detach().clone()
        data.y = data_make_states(data.T.item(), data.tI)
        torch.save(data, f_data)
        return data

def data_pol_si(data_dir, device):
    '''https://networkrepository.com/rt-pol.php'''
    data_dir = osp.join(data_dir, 'pol')
    f_data = osp.join(data_dir, 'pol-si.pt')
    if osp.exists(f_data):
        return torch.load(f_data, map_location = device)
    else:
        f_edge = file_require('https://nrvis.com/download/data/rt/rt-pol.zip', data_dir, 'rt-pol.txt', z = 'zip')
        df_fr, df_to, df_time = [], [], []
        with open(f_edge, 'r') as fi:
            for line in fi.readlines():
                fr, to, time = map(int, line.split(','))
                df_fr.append(fr)
                df_to.append(to)
                df_time.append(time)
        df = pd.DataFrame(dict(fr = df_fr, to = df_to, time = df_time))
        df.time = ((df.time - df.time.min() + 1) / (24 * 60 * 60)).apply(lambda x: int(np.ceil(x))) # day
        tI = data_calc_tI(df)
        Gnx = nx.from_edgelist(df[['fr', 'to']].values.tolist())
        nx.set_node_attributes(Gnx, values = tI, name = 'tI')
        Gnx = Gnx.subgraph(max(nx.connected_components(Gnx), key = len)).copy() # largest component
        data = pyg.utils.from_networkx(Gnx).to(device)
        T = 40
        data.T = torch.tensor(T, dtype = data.tI.dtype, device = device)
        data.tI = torch.minimum(data.tI, data.T + 1)
        data.y = data_make_states(T, data.tI)
        torch.save(data, f_data)
        return data

def data_covid_sir(data_dir, device):
    '''https://data.cdc.gov/Public-Health-Surveillance/United-States-COVID-19-Community-Levels-by-County/3nnm-4jni'''
    data_dir = osp.join(data_dir, 'covid')
    f_data = osp.join(data_dir, 'covid-sir.pt')
    if osp.exists(f_data):
        return torch.load(f_data, map_location = device)
    else:
        COVID_KNN = 10
        f_s2a = file_require(None, data_dir, 'state2abbr.pyon')
        with open(f_s2a, 'r') as fi:
            STATE2ABBR = eval(fi.read())
        f_raw = file_require(None, data_dir, 'United_States_COVID-19_Community_Levels_by_County.csv')
        df = pd.read_csv(f_raw)
        df['level'] = df['covid-19_community_level'].apply(lambda lv: int(lv != 'Low'))
        df.date_updated = pd.to_datetime(df.date_updated)
        df['time'] = (df.date_updated.dt.year - 2019) * 12 + df.date_updated.dt.month - 1 # month
        df.time -= df.time.min()
        df = df[['state', 'county', 'time', 'level']]
        df = df.loc[df.groupby(['state', 'time'])['level'].idxmax()]
        f_geo = file_require(None, data_dir, 'us-counties.csv')
        GEO_COORDS = {(row['State'].strip('  '), row['County'].strip('  ')): (float(row['Latitude']), float(row['Longitude'])) for _, row in pd.read_csv(f_geo).iterrows()}
        def geo_abbr(state, county):
            state = STATE2ABBR[state] if state in STATE2ABBR else None
            county = county.replace('-', '-').replace('ñ', 'n').strip()
            words = county.split(' ')
            for m in range(len(words), 0, -1):
                ct = ' '.join(words[: m])
                if (state, ct) in GEO_COORDS:
                    return state, ct
            return state, None
        def county2node(state, county):
            state, county = geo_abbr(state, county)
            return f'{county}, {state}'
        unknown = {(row['state'], row['county']) for _, row in df.iterrows() if geo_abbr(row['state'], row['county']) not in GEO_COORDS}
        nodes = {county2node(row['state'], row['county']) for _, row in df.iterrows() if geo_abbr(row['state'], row['county']) in GEO_COORDS}
        coords = {county2node(row['state'], row['county']): GEO_COORDS[geo_abbr(row['state'], row['county'])] for _, row in df.iterrows() if geo_abbr(row['state'], row['county']) in GEO_COORDS}
        dists = {(u, v): geod.geodesic(coords[u], coords[v]).km for u in tqdm(nodes) for v in nodes}
        df = df.loc[[(geo_abbr(row['state'], row['county']) in GEO_COORDS) for _, row in df.iterrows()]]
        df['node'] = [county2node(row['state'], row['county']) for _, row in df.iterrows()]
        df = df.drop('state', axis = 1).drop('county', axis = 1)
        T = df.time.max()
        tI = {u: T + 1 for u in nodes}
        tR = {u: -1 for u in nodes}
        for _, row in df.iterrows():
            if row['level'] == 1:
                tI[row['node']] = min(tI[row['node']], row['time'])
                tR[row['node']] = max(tR[row['node']], row['time'])
        for u in nodes:
            tR[u] = max(tI[u], min(T + 1, tR[u] + 1))
        nodes = list(nodes)
        edges = sum([sorted([(min(u, v), max(u, v)) for v in nodes if v != u], key = lambda e: dists[e])[: COVID_KNN] for u in nodes], [])
        Gnx = nx.from_edgelist(edges)
        nx.set_node_attributes(Gnx, values = tI, name = 'tI')
        nx.set_node_attributes(Gnx, values = tR, name = 'tR')
        Gnx = Gnx.subgraph(max(nx.connected_components(Gnx), key = len)).copy() # largest component
        data = pyg.utils.from_networkx(Gnx).to(device)
        data.T = torch.tensor(T, dtype = data.tI.dtype, device = device)
        data.y = data_make_states(T, data.tI, data.tR)
        torch.save(data, f_data)
        return data

def data_heb_sir(data_dir, device):
    '''https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0230811'''
    data_dir = osp.join(data_dir, 'heb')
    f_data = osp.join(data_dir, 'heb-sir.pt')
    if osp.exists(f_data):
        return torch.load(f_data, map_location = device)
    else:
        f_edge = file_require(url = None, fdir = data_dir, fname = 'DS1_NON_VIRAL_Gtw.tsv')
        df = pd.read_csv(f_edge, sep = '\t', header = None, names = ['time', 'to', 'fr'], dtype = dict(time = str, to = int, fr = int), parse_dates = ['time'])
        df.time = df.time.astype(int) / 1e9
        df.time = ((df.time - df.time.min()) / (7 * 24 * 60 * 60)).apply(lambda x: int(np.ceil(x))) # week
        T = df.time.max()
        tI, tR = data_calc_tIR(df, T)
        Gnx = nx.from_edgelist(df[['fr', 'to']].values.tolist())
        nx.set_node_attributes(Gnx, values = tI, name = 'tI')
        nx.set_node_attributes(Gnx, values = tR, name = 'tR')
        Gnx = Gnx.subgraph(max(nx.connected_components(Gnx), key = len)).copy() # largest component
        data = pyg.utils.from_networkx(Gnx).to(device)
        data.T = data.tI.max()
        data.y = data_make_states(T, data.tI, data.tR)
        torch.save(data, f_data)
        return data

DATASETS = {
    'ba-si': fnt.partial(data_synthetic, 'ba', 'si'),
    'er-si': fnt.partial(data_synthetic, 'er', 'si'),
    'prost-si': fnt.partial(data_prost, 'si'),
    'oregon2-si': fnt.partial(data_oregon2, 'si'),
    'farmers-si': data_farmers_si,
    'pol-si': data_pol_si,
    'ba-sir': fnt.partial(data_synthetic, 'ba', 'sir'),
    'er-sir': fnt.partial(data_synthetic, 'er', 'sir'),
    'oregon2-sir': fnt.partial(data_oregon2, 'sir'),
    'prost-sir': fnt.partial(data_prost, 'sir'),
    'covid-sir': data_covid_sir,
    'heb-sir': data_heb_sir,
}

def data_load(dataset, data_dir, device):
    assert dataset in DATASETS, 'unknown dataset'
    data = DATASETS[dataset](data_dir, device)
    data.name = dataset
    return data
