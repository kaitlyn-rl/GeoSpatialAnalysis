from Graph import Graph

if __name__ == '__main__':
    """ - This part is necessary for all metrics computation. 
        - Modify the path to each file in full_graph() as necessary.
        - Comment out g.sub_graph() if you want to compute metrics for the full graph.
        (Note that it currently only work for shortest path for full graph.)
        - Modify the centre_gs and buffer in sub_graph() as necessary.
    """
    g = Graph()
    g.full_graph(edge_pth='./graphs/edges.geojson', 
                 node_pth='./graphs/nodes.geojson', 
                 gs_pth='./data/green_spaces.geojson')
    g.sub_graph(centre_gs='West Princes Street Gardens', buffer=1000)
    
    """ Shortest path computation
    """
    # dist_matrix, dists = g.shortest_path()
    # mean, var = g.shortest_path_stats(dists)
    # print(mean, var)
    # g.viz_dist(attr_name='shortest_path',
    #            save=True)
    
    """ Commute distance computation
    """
    # minComm = g.comm_dist()
    # mean, var = g.comm_dist_stats(minComm)
    # print(mean, var)
    # g.viz_dist('commute_distance')

    """ Network diffusion computation
    # """
    sol = g.diffusion(alpha=1,
                      steps=100,
                      dt=1,
                      unit='m'
                      )
    g.viz_diff(sol, save=True)
    g.viz_diff_pop_ex(sol, save=True)
    
    """ Happiness diffusion computation
    """
    # g.happiness()
    # g.viz_dist('happiness',
    #            save = True)
    