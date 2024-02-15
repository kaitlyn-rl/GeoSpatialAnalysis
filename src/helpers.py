import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import networkx as nx
import momepy
import shapely
import numpy as np
import scipy as sp
import pandas as pd

def gen_data(edge_pth, node_pth, gs_pth):
    # read data from geojson files
    # !! change the path to the data folder according your local directory !!
    edges = gpd.read_file(edge_pth)
    nodes = gpd.read_file(node_pth)
    green_space = gpd.read_file(gs_pth)
    
    # set NaN in pop_per_node and pop_den to 0
    nodes.pop_per_node.fillna(0, inplace=True)
    nodes.pop_den.fillna(0, inplace=True)

    # convert edges into networkx graph
    G = momepy.gdf_to_nx(edges, approach='primal', multigraph=False)

    # assign node attributes to a dictionary
    nc = list(nodes.columns)
    node_attrs = {n.geometry.coords[0]: {key: n[key] for key in nc} for _, n in nodes.iterrows()}
    for key1, value1 in node_attrs.items():
        for key2, value2 in value1.items():
            if key2 == 'gs_id':
                node_attrs[key1][key2] = value2.split(',')
    nx.set_node_attributes(G, node_attrs) # assign node attributes to the graph
    # check nodes without any attributes and remove them
    nodes_remove = [n[0] for n in G.nodes(data=True) if 'gs_bool' not in n[1]]
    G.remove_nodes_from(nodes_remove)
    G = G.subgraph(max(nx.connected_components(G), key=len)) # remove unconnected nodes and edges

    gs_dict = {gs.id: 0 for _, gs in green_space.iterrows()} # dictionary for number nodes belonging to each green space
    for n in list(G.nodes(data=True)):
        for g in n[1]['gs_id']:
            if g in list(gs_dict.keys()):
                gs_dict[g] += 1
    gs_empty_list = [key for key, value in gs_dict.items() if value == 0] # list of green spaces with no nodes
    green_space = green_space[~green_space.id.isin(gs_empty_list)] # remove green spaces with no nodes
    
    return G, green_space, gs_dict, nodes, edges

def datazone(dz_pth, boundary_pth, pop_pth):
    DataZone = gpd.read_file(dz_pth)
    boundaries = gpd.read_file(boundary_pth)
    edinburgh = boundaries.loc[boundaries["local_authority"]=='City of Edinburgh', "geometry"].values[0]
    DataZone["geometry"] = DataZone.intersection(edinburgh)
    DataZone = DataZone.loc[~DataZone.geometry.is_empty]

    # add population data
    population = pd.read_csv(pop_pth, header=8, names=['link', 'area name', 'population'])
    population['DataZone'] = population['link'].apply(lambda x: x.split('/')[-1])
    DataZone = DataZone.merge(population[['DataZone', 'population']],  how='left', on='DataZone')# , right_on='DataZone')
    return DataZone

def gen_subG(G, green_space, centre_gs, buffer=1000):
    # create graph centred around Princes Street Gardens
    # !! set buffer to change how much to include in the graph !!
    gs_sub = green_space[green_space['distName1'] == centre_gs]
    gs_list = [gs.id for _, gs in green_space.iterrows() if gs.geometry.distance(gs_sub.geometry.values[0]) < buffer]
    gs = green_space[green_space.id.isin(gs_list)]
    boundary = shapely.MultiPolygon([gs.geometry for _, gs in green_space.explode().iterrows() if gs.id in gs_list]).convex_hull
    subG = G.copy()
    for e in list(G.edges(data=True)):
        if not boundary.covers(e[2]['geometry'] or boundary.exterior.distance(e[2]['geometry']) > 200):
            subG.remove_edge(e[0], e[1])
    subG.remove_nodes_from(list(nx.isolates(subG)))
    subG = subG.subgraph(max(nx.connected_components(subG), key=len)) # remove unconnected nodes and edges
    return subG, gs
    
def gen_L(G): 
    """Compute variables associated with graphs

    Args:
        G (nx.Graph): Graph of interest

    Returns:
        W (sp.csr_sparse_array): sparse array of weights
        D (sp.csr_sparse_array): sparse array of out degree
        L (sp.csr_sparse_array): sparse array of unnormalised graph Laplacian
        gnodes (dict): dictionary of green space nodes and their index in the graph
    """
    W = nx.to_scipy_sparse_array(G) 
    D = sp.sparse.csr_array(np.diag(np.array(np.sum(W, axis=1))))
    L = D - W

    #getting only the park nodes to compute shortest path from
    gnodes = {}
    for idx, n in enumerate(list(G.nodes(data = True))):
        if n[1]['gs_entrance'] == 1:
            gnodes[n[0]] = idx
    return W, D, L, gnodes

def directed_graph(G, green_space, centre_gs, weight_inverse=True, buffer=1000):
    """ Create directed graph, with edges connected to green space entrance directing only to green space 
        entrance, also remove nodes within green space (except green space entrance nodes)
    

    Args:
        G (nx.Graph): graph of interest.
        green_space (pd.DataFrame): dataframe of green spaces read from geojson file.
        weight_inverse (bool, optional): Set weight to 1/length. Defaults to True.
        centre_gs (str, optional): green space to centre graph around. 
        buffer (int, optional): buffer from . Defaults to 1000.

    Returns:
        _type_: _description_
    """
    gs_sub = green_space[green_space['distName1'] == centre_gs]
    gs_list = [gs.id for idx, gs in green_space.iterrows() if gs.geometry.distance(gs_sub.geometry.values[0]) < buffer]
    gs = green_space[green_space.id.isin(gs_list)]
    boundary = shapely.MultiPolygon([gs.geometry for idx, gs in green_space.explode().iterrows() if gs.id in gs_list]).convex_hull
    sub_G = G.copy()
    for e in list(G.edges(data=True)):
        if not boundary.covers(e[2]['geometry'] or boundary.exterior.distance(e[2]['geometry']) > 200):
            sub_G.remove_edge(e[0], e[1])
    sub_G.remove_nodes_from(list(nx.isolates(sub_G)))
    sub_G.remove_nodes_from([n[0] for n in list(sub_G.nodes(data=True)) if (n[1]['gs_bool'] == True and n[1]['gs_entrance'] == False)])
    sub_G_d = sub_G.to_directed()
    for n in list(sub_G_d.nodes(data=True)):
        if n[1]['gs_entrance'] == True:
            sub_G_d.remove_edges_from([(e[0], e[1]) for e in list(sub_G_d.edges(n[0])) if sub_G_d.nodes(data=True)[e[1]]['gs_bool'] == False])
    sub_G_d.remove_nodes_from(list(nx.isolates(sub_G_d)))
    gs_dict = {gs_id: 0 for gs_id in gs.id.unique()}
    for n in list(sub_G_d.nodes(data=True)):
        if n[1]['gs_entrance'] == True:
            for g in n[1]['gs_id']:
                if g in gs_dict:
                    gs_dict[g] += 1
    if weight_inverse:
        for e in list(sub_G_d.edges(data=True)):
            e[2]['weight'] = 1/e[2]['length']
    sub_G_d = sub_G_d.subgraph(max(nx.weakly_connected_components(sub_G_d), key=len)) # remove unconnected nodes and edges
    return sub_G_d, gs #, gs_dict

# random perturbation of graph, not fully implemented
def rand_pert(G, mode, num_G=10):
    Gs = []
    for seed in range(num_G):
        np.random.seed(seed*10)
        G_tmp = G.copy()
        # remove 1% of edges randomly (if chosen edge is less than 100m long)
        num_edges = len(G.edges)
        if mode == 'repurpose':
            edge_node_list = []
            while len(G_tmp.edges) > 0.99*num_edges:
                e_idx = np.random.randint(0, len(G_tmp.edges))
                e_tmp = list(G_tmp.edges)[e_idx]
                if e_tmp[0] not in edge_node_list and e_tmp[1] not in edge_node_list and G_tmp.edges[e_tmp]['length'] < 100:
                    G_tmp.remove_edge(e_tmp[0], e_tmp[1])
                    edge_node_list.append(e_tmp[0])
                    edge_node_list.append(e_tmp[1])
            num_mod_edges = 0
            
            G_tmp = G_tmp.subgraph(max(nx.connected_components(G_tmp), key=len)) # remove unconnected nodes and edges
            Gs.append(G_tmp)
        # G_tmp2 = G_tmp.copy()
        # remove 2.5% of nodes randomly
        # num_nodes = len(G_tmp.nodes)
        # node_list = []
        # while len(G_tmp2.nodes) > 0.975*num_nodes:
        #     n_idx = np.random.randint(0, len(G_tmp2.nodes))
        #     n = list(G_tmp2.nodes(data=True))[n_idx]
        #     if n[0] not in node_list and n[0] not in edge_node_list and n[1]['gs_bool'] == False:
        #         G_tmp2.remove_node(n[0])
        #         node_list.append(n[0])
                
        elif mode == 'inaccuracy':
            edges_to_add = int(np.floor(num_edges*0.01))
            edges_added = 0
            edge_add_list = []
            # add 1% of edges randomly (if chosen edge is less than 100m long)
            while edges_added < edges_to_add:
                n_idx1 = np.random.randint(0, len(G_tmp.nodes))
                n_1 = list(G_tmp.nodes)[n_idx1]
                n_idx2 = np.random.randint(0, len(G_tmp.nodes))
                n_2 = list(G_tmp.nodes)[n_idx2]
                if n_1 in edge_add_list or n_2 in edge_add_list:
                    continue
                line_tmp = shapely.LineString([n_1, n_2])
                if line_tmp.length < 100:
                    G_tmp.add_edge(n_1, n_2, length=line_tmp.length, weight=line_tmp.length, geometry = line_tmp)
                    edges_added += 1
                    edge_add_list.append(n_1)
                    edge_add_list.append(n_2)
            # randomly modify 2.5% of edges by +/- 25% of original length
            while num_mod_edges < 0.025*num_edges:
                e_idx = np.random.randint(0, len(G_tmp.edges))
                e_tmp = list(G_tmp.edges)[e_idx]
                G_tmp.edges[e_tmp]['length'] *= 1 + np.random.uniform(-0.75, 0.25)
            G_tmp = G_tmp.subgraph(max(nx.connected_components(G_tmp), key=len)) # remove unconnected nodes and edges
            Gs.append(G_tmp)
    return Gs
        