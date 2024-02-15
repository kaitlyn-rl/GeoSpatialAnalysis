import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import networkx as nx
import pandas as pd
import shapely
import momepy
import multiprocessing as mp

def get_data(street_pth, gs_pth, datazone_pth, boundaries_pth, population_pth):
    # load data
    streets = gpd.read_file(street_pth)
    green_space_original = gpd.read_file(gs_pth)
    green_space = green_space_original.explode()
    # Preprocess streets data
    streets = streets.explode()
    # From Tim:
    # streets["geometry"] = streets.intersection(shape[0], align=False)
    streets = streets[~streets.is_empty] # remove empty geometries
    buffered_streets = streets.copy() # make a copy
    buffered_streets["geometry"] = buffered_streets.geometry.buffer(1) # extend streets by 1m for intersection
    un = streets.geometry.unary_union # connect them all together into one big multistring
    geom = [i for i in un.geoms] # separate it out into linestrings
    id = [j for j in range(len(geom))] # generate ids for each linestring
    unary = gpd.GeoDataFrame({"id":id,"geometry":geom}, crs= streets.crs).set_index("id") 
    streets = gpd.sjoin(unary, buffered_streets, how="inner",predicate='within')#.drop(columns="index_right") # join the original streets back to the extended ones

    walkable = ['residential', 'primary', 'secondary', 'tertiary', 'unclassified', 'service', 'pedestrian', 'footway', 'cycleway', 'track', 'steps', 'path', 'living_street', 'bridleway', 'corridor']
    streets = streets[streets.highway.isin(walkable)] # filter out non-walkable44 streets
    streets = streets.reset_index(drop=True)
    
    # iteratively check if the last coordinates of any linestring repeats its first coordinates, if so remove the last coordinates
    for idx, row in streets.iterrows():
        if row.geometry.coords[0] == row.geometry.coords[-1]:
            streets.loc[idx, "geometry"] = shapely.geometry.LineString(row.geometry.coords[:-1])
            
    # generate datazone
    DataZone = gpd.read_file(datazone_pth)
    # DataZone = gpd.read_file('/Users/tc/Documents/UoE/2223/S2/geospatial_analysis/data/SG_DataZoneBdry_2011')
    boundaries = gpd.read_file(boundaries_pth)
    # boundaries = gpd.read_file("/Users/tc/Documents/UoE/2223/S2/geospatial_analysis/data/local_authorities.geojson")
    edinburgh = boundaries.loc[boundaries["local_authority"]=='City of Edinburgh', "geometry"].values[0]
    DataZone["geometry"] = DataZone.intersection(edinburgh)
    DataZone = DataZone.loc[~DataZone.geometry.is_empty]

    # add population data
    population = pd.read_csv(population_pth, header=8, names=['link', 'area name', 'population'])
    # population = pd.read_csv('/Users/tc/Documents/UoE/2223/S2/geospatial_analysis/data/population-estimates-2011-datazone-linked-dataset.csv', header=8, names=['link', 'area name', 'population'])
    population['DataZone'] = population['link'].apply(lambda x: x.split('/')[-1])
    DataZone = DataZone.merge(population[['DataZone', 'population']],  how='left', on='DataZone')# , right_on='DataZone')
    DataZone['buffered1'] = DataZone.buffer(5)
    return streets, green_space, DataZone

def gen_graph(streets):
    G = momepy.gdf_to_nx(streets, approach="primal", length="length", multigraph=False) # Construct nodes and edges of the primal graph
    for n in list(G.nodes(data=True)):
        n[1]['gs_bool'] = False
        n[1]['gs_id'] = []
        n[1]['gs_entrance'] = False
    return G

def iden_gs_nodes(gs, G):
    buffer = 2
    gs_node_ct = 0
    node_dict = {}
    while True:
        for n in list(G.nodes(data=True)):
            if gs.geometry.buffer(buffer).covers(shapely.Point(n[0])):
                if n[0] not in node_dict:
                    node_dict[n[0]] = {'gs_bool': True, 'gs_id': [gs.id], 'gs_entrance': False}
                elif n[0] in node_dict and gs.id not in node_dict[n[0]]['gs_id']:
                    node_dict[n[0]]['gs_id'].append(gs.id)
                    node_dict[n[0]]['gs_bool'] = True
                gs_node_ct += 1
        if gs_node_ct > 0:
            break
        else:
            buffer += 0.5
        if buffer > 6:
            break
    return node_dict

def check_its(e, gdf):
    return_lines = []
    return_coords = []
    return_gs = []
    for idx, gs in gdf.iterrows():
        if gs.geometry.buffer(5).exterior.intersects(e[2]['geometry']):
            its = gs.geometry.buffer(5).exterior.intersection(e[2]['geometry'])
            if its.geom_type == 'Point':
                its_list = [(list(its.coords)[0][0], list(its.coords)[0][1])]
            elif 'Multi' in its.geom_type:
                its_list = [(list(pt.coords)[0][0], list(pt.coords)[0][1]) for pt in list(its.geoms)]
            else:
                print('WTF?')
            line = e[2]['geometry']
            # First coords of line (start + end)
            coords = [line.coords[0], line.coords[-1]] 
            coords_edge = list(e[2]['geometry'].coords)
            # Add the coords from the points
            for point in its_list:
                coords += shapely.Point(point).coords
                coords_edge += shapely.Point(point).coords
            # Calculate the distance along the line for each point
            dists = [line.project(shapely.Point(p)) for p in coords]
            # sort the coordinates
            coords = [p for (d, p) in sorted(zip(dists, coords))]
            lines = [([coords[i], coords[i+1]]) for i in range(len(coords)-1)]
            # Calculate the distance along the line for each point
            dists_edge = [line.project(shapely.Point(p)) for p in coords_edge]
            # sort the coordinates
            coords_edge = [p for (d, p) in sorted(zip(dists_edge, coords_edge))]
            return_lines.append(lines)
            return_coords.append(coords_edge)
            return_gs.append(gs.id)
    return e, return_lines, return_coords, return_gs

def gs_entrance(n, G):
    gs_entrance = False
    if n[1]['gs_bool'] == True:
        for e in list(G.edges(n[0])):
            if G.nodes(data=True)[e[1]]['gs_bool'] == False:
                gs_entrance = True
                break
    return n[0], gs_entrance

def assign_dataZone(n, DataZone):
    for idx, d in DataZone.iterrows():
        if d['geometry'].covers(shapely.Point(n[0])):
            DataZoneID = d['DataZone']
            break
        elif d['buffered1'].covers(shapely.Point(n[0])):
            DataZoneID = d['DataZone']
            break
        else:
            DataZoneID = None
    return n[0], DataZoneID

if __name__ == '__main__':
    street_pth = './data/highways.geojson'
    gs_pth = './data/green_spaces.geojson'
    datazone_pth = './data/SG_DataZoneBdry_2011'
    boundaries_pth = './data/local_authorities.geojson'
    population_pth = './data/population-estimates-2011-datazone-linked-dataset.csv'
    
    streets, green_space, DataZone = get_data(street_pth, 
                                              gs_pth, 
                                              datazone_pth, 
                                              boundaries_pth, 
                                              population_pth)
    G = gen_graph(streets)
    with mp.Pool(mp.cpu_count()) as pool:
        for results in pool.starmap(check_its, [(e, green_space) for e in G.edges(data=True)]):
            e = results[0]
            for idx in range(len(results[3])):
                lines = results[1][idx]
                coords_edge = results[2][idx]
                gs_id = results[3][idx]
                if G.has_edge(e[0], e[1]):
                    # print("removing for ", e[0], ", ", e[1])
                    G.remove_edge(e[0], e[1])
                for lin in lines:
                    for n in lin:
                        if not G.has_node(n):
                            G.add_node(n, gs_bool=True, gs_id=[gs_id])
                        else:
                            G.nodes(data=True)[n]['gs_bool'] = True
                            if gs_id not in G.nodes(data=True)[n]['gs_id']:
                                G.nodes(data=True)[n]['gs_id'].append(gs_id)
                    e_dict_tmp = e[2].copy()
                    start = coords_edge.index(lin[0])
                    end = coords_edge.index(lin[1])
                    # print(coords_edge[start:end+1], start, end)
                    if start == end:
                        continue
                    else:
                        e_dict_tmp['geometry'] = shapely.LineString(coords_edge[start:end+1])
                        e_dict_tmp['length'] = e_dict_tmp['geometry'].length
                        e_dict_tmp['weight'] = e_dict_tmp['length']
                        G.add_edge(lin[0], lin[1])
                        nx.set_edge_attributes(G, {(lin[0], lin[1]): e_dict_tmp})
                    
    with mp.Pool(mp.cpu_count()) as pool:
        for results in pool.starmap(iden_gs_nodes, [(gs, G) for _, gs in green_space.iterrows()]):
            for n, n_dict in results.items():
                G.nodes(data=True)[n]['gs_bool'] = n_dict['gs_bool']
                for gs_id in n_dict['gs_id']:
                    if gs_id not in G.nodes(data=True)[n]['gs_id']:
                        G.nodes(data=True)[n]['gs_id'].append(gs_id)

    with mp.Pool(mp.cpu_count()) as pool:
        for results in pool.starmap(assign_dataZone, [(n, DataZone) for n in G.nodes(data=True)]):
            # nodes.loc[nodes.nodeID==results[0], 'DataZone'] = results[1]
            G.nodes(data=True)[results[0]]['DataZone'] = results[1]
                                
    nodes, edges, W = momepy.nx_to_gdf(G, spatial_weights=True)
    
    nodes = nodes[nodes.DataZone.notnull()]
    for d in nodes.DataZone.unique():
        ns = nodes[nodes.DataZone == d]
        ns = ns[ns.gs_bool == False]
        pop = DataZone[DataZone.DataZone == d].population.values[0]
        pop_per_node = pop / len(ns)
        area = DataZone[DataZone.DataZone == d].StdAreaKm2.values[0]
        pop_den = pop / area
        DataZone.loc[DataZone.DataZone == d, 'pop_per_node'] = pop_per_node
        DataZone.loc[DataZone.DataZone == d, 'pop_den'] = pop_den
        
    for idx, n in nodes.iterrows():
        if n['gs_bool'] == False:
            nodes.loc[idx, 'pop_per_node'] = DataZone[DataZone.DataZone == n.DataZone].pop_per_node.values[0]
            nodes.loc[idx, 'pop_den'] = DataZone[DataZone.DataZone == n.DataZone].pop_den.values[0]
            
    G2 = momepy.gdf_to_nx(edges, approach='primal', multigraph=False)
    node_attrs = {}
    for idx, n in nodes.iterrows():
        # print(n.geometry.coords[0])
        # break
        node_attrs[n.geometry.coords[0]] = {}
        for key in list(nodes.columns):
            # if key == 'gs_id':
            #     node_attrs[n.geometry.coords[0]][key] = n[key].split(',')
            #     # print(n[key].split(','))
            # else:
            node_attrs[n.geometry.coords[0]][key] = n[key]
    nx.set_node_attributes(G2, node_attrs)
    nodes_remove = []
    for n in G2.nodes(data=True):
        if 'gs_bool' not in n[1]:
            nodes_remove.append(n[0])
    G2.remove_nodes_from(nodes_remove)

    gs_dict_tmp = {} # dictionary for number nodes belonging to each green space
    for idx_g, gs in green_space.iterrows():
        gs_dict_tmp[gs.id] = 0
    for n in list(G2.nodes(data=True)):
        for g in n[1]['gs_id']:
            if g in list(gs_dict_tmp.keys()):
                gs_dict_tmp[g] += 1
    gs_empty_list = [] # list of green spaces with no nodes
    for gs_key, gs_val in gs_dict_tmp.items():
        if gs_val == 0:
            gs_empty_list.append(gs_key)
            
    green_space_empty = green_space[green_space.id.isin(gs_empty_list)]
    
    with mp.Pool(mp.cpu_count()) as pool:
        for results in pool.starmap(iden_gs_nodes, [(gs, G2) for _, gs in green_space_empty.iterrows()]):
            for n, n_dict in results.items():
                G2.nodes(data=True)[n]['gs_bool'] = n_dict['gs_bool']
                for gs_id in n_dict['gs_id']:
                    if gs_id not in G2.nodes(data=True)[n]['gs_id']:
                        G2.nodes(data=True)[n]['gs_id'].append(gs_id)
                        
    with mp.Pool(mp.cpu_count()) as pool:
        for results in pool.starmap(gs_entrance, [(n, G2) for n in list(G2.nodes(data=True))]):
            G2.nodes(data=True)[results[0]]['gs_entrance'] = results[1]
    
    nodes, edges, W = momepy.nx_to_gdf(G2, spatial_weights=True)
    
    for idx, e in edges.iterrows():
        edges.loc[idx, 'weight'] = e.length
    
    for idx, row in nodes.iterrows():
        if type(row['gs_id']) == list:
            nodes.loc[idx,'gs_id'] = ','.join(row['gs_id']) # geojson does not allow list
    output_pth = './graphs/'
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)
    nodes.to_file(os.path.join(output_pth, "/nodes_test.geojson"), driver="GeoJSON")
    edges.to_file(os.path.join(output_pth, "/edges_test.geojson"), driver="GeoJSON")