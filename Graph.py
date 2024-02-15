import networkx as nx
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

from helpers import gen_data, gen_subG, gen_L, directed_graph


class Graph:

    def __init__(self, ):
        pass

    def full_graph(self, edge_pth, node_pth, gs_pth):
        """generate weighted undirected graph from data

        Args:
            edge_pth (str): path to edge data
            node_pth (str): path to node data
            gs_pth (str): path to green space data
        """
        self.G, self.green_space, self.gs_dict, _, _ = gen_data(edge_pth,
                                                                node_pth,
                                                                gs_pth)
        self.name = 'full'

    def sub_graph(self, centre_gs, buffer=2500):
        self.subG, self.subgs = gen_subG(self.G,
                                         self.green_space,
                                         centre_gs=centre_gs,
                                         buffer=buffer)
        self.centre_gs = centre_gs
        self.buffer = buffer
        self.G = self.subG.copy()
        self.green_space = self.subgs.copy()
        self.name = '_'.join(centre_gs.split(' ')) + '_' + str(buffer)
        self.name_alt = centre_gs + ' with buffer of ' + str(buffer) + 'm'
        del self.subG, self.subgs

    def graph_viz(self, figsize=(20, 20), node_size=5):
        node_map = ['orange' if n[1]['gs_entrance'] else 'green' if n[1]
                    ['gs_bool'] else 'red' for n in list(self.G.nodes(data=True))]
        positions = {n: [n[0], n[1]] for n in list(self.G.nodes())}
        ax = self.green_space.plot(figsize=figsize)  # plot green spaces
        # plot graph onto green space plot
        nx.draw(self.G, positions, node_size=node_size, node_color=node_map)
        plt.show()

    def shortest_path(self):
        """Compute shortest path of each node to their closest green space

        Returns:
            dist_matrix (np.ndarray):   shortest path distance matrix of size (m, n) where m is the number 
                                        of green spaces and n is the number of all nodes
            dists (np.ndarray):     shortest path distance vector of size (n) of each node to their closest 
                                    green space
        """
        W = nx.to_scipy_sparse_array(self.G)
        gnodes = {}
        for idx, n in enumerate(list(self.G.nodes(data=True))):
            if n[1]['gs_entrance'] == 1:
                gnodes[n[0]] = idx
        gs_nodes = [i for i in gnodes.values()]
        dist_matrix = sp.sparse.csgraph.shortest_path(
            W, directed=False, indices=gs_nodes)
        dists = dist_matrix.min(axis=0)
        for idx, n in enumerate(list(self.G.nodes(data=True))):
            self.G.nodes[n[0]]['shortest_path'] = dists[idx]
        return dist_matrix, dists

    def shortest_path_stats(self, dists):
        """Compute mean and variance of shortest path distance of each node to their closest green space

        Args:
            dists (np.ndarray): shortest path distance vector of size (n) of each node to their closest 
                                green space, computed by shortest_path()

        Returns:
            mean (np.float): mean of shortest path distance
            var (np.float): variance of shortest path distance
        """

        pop = np.array([n[1]['pop_per_node'] for n in self.G.nodes(data=True)])
        mean = np.sum(dists*pop) / np.sum(pop)
        var = np.sum(pop*(dists - mean)**2)/np.sum(pop)

        return mean, var

    def comm_dist(self):
        """Compute commute distance between each green space node and all nodes in the graph

        Returns:
            minComm (np.ndarray): minimum commute distance vector of size (n) of each node
        """
        W, _, L, gnodes = gen_L(self.G)
        gn_idx = list(gnodes.values())
        vol_V = np.sum(W)
        L_pinv = np.linalg.pinv(L.todense(), hermitian=True)
        # to just get the gspace info
        comm_dist = np.zeros((len(gnodes), L_pinv.shape[1]))
        for i in range(comm_dist.shape[0]):
            for j in range(comm_dist.shape[1]):
                comm_dist[i, j] = vol_V * (L_pinv[gn_idx[i], gn_idx[i]] -
                                           2 * L_pinv[gn_idx[i], j] + L_pinv[j, j])
        minComm = comm_dist.min(axis=0)
        for idx, n in enumerate(list(self.G.nodes(data=True))):
            self.G.nodes[n[0]]['commute_distance'] = minComm[idx]
        return minComm

    def comm_dist_stats(self, minComm):
        """Compute mean and variance of the minimum commute distance of each node to a green space

        Args:
            minComm (np.ndarray): minimum commute distance vector of size (n) of each node

        Returns:
            mean (np.float): mean of minimum commute distance
            variance (np.float): variance of minimum commute distance
        """
        pop = np.array([n[1]['pop_per_node'] for n in self.G.nodes(data=True)])
        mean = np.sum(minComm*pop) / np.sum(pop)
        var = np.sum(pop*(minComm - mean)**2)/np.sum(pop)
        return mean, var

    def viz_dist(self, attr_name, figsize=(20, 20), node_size=1, save=False, save_pth=None):
        """Visualise result of shortest path or commute distance on the graph of interest, also works
            for happiness diffusion

        Args:
            attr_name (str): shortest_path or commute_distance
            figsize (tuple, optional): Defaults to (20, 20).
            node_size (int, optional): Defaults to 1.
            save (bool, optional): Defaults to False.
            save_pth (_type_, optional): Defaults to None.
        """
        node_map = [n[1][attr_name] for n in list(self.G.nodes(data=True))]
        edge_map = [(self.G.nodes[e[0]][attr_name] + self.G.nodes[e[1]]
                     [attr_name])/2 for e in list(self.G.edges())]
        positions = {n: [n[0], n[1]] for n in list(self.G.nodes())}
        ax = self.green_space.plot(
            figsize=figsize, color='green')  # plot green spaces
        vmin = np.min(node_map)
        vmax = np.max(node_map)
        norm = plt.Normalize(vmin, vmax)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", ["red", "orange", "yellow", "green"][::-1])
        nx.draw(self.G, positions, ax=ax, node_size=node_size, node_color=node_map,
                edge_color=edge_map, cmap=cmap, edge_cmap=cmap, width=1)  # plot graph onto green space plot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, fraction=0.042, pad=0.04)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(' '.join('_'.split(attr_name)),
                       rotation=90, labelpad=10, fontsize=15)

        if save:
            fig = plt.gcf()
            fig_title = 'for full graph' if self.name == 'full' else 'centred at {}'.format(
                self.name_alt)
            fig.suptitle('{} {}'.format(' '.join(attr_name.split('_')), fig_title), fontsize=20)
            if save_pth == None:
                save_pth = './results'
                if not os.path.exists(save_pth):
                    os.makedirs(save_pth)
                    print('Created directory: {}'.format(save_pth))
            fig.savefig(os.path.join(
                save_pth, '{}_{}.png'.format(attr_name, self.name)))

    def get_path(Pr, i, j):
        """Get shortest path from any node i to green space node j

        Args:
            Pr (np.ndarray): predecessors matrix of size (n, n), n = number of nodes
            i (int): starting node
            j (int): destination green space node

        Returns:
            list: list of indices of nodes in the shortest path form i to j
        """
        path = [j]
        k = j
        while Pr[i, k] != -9999:
            path.append(Pr[i, k])
            k = Pr[i, k]
        return path[::-1]

    def diffusion(self, alpha=1., steps=500, dt=1, unit='s', save=False, save_pth=None):
        """Simulation of 'population' diffusion across graph

        Args:
            alpha (float, optional):    weight of pure diffusion in Laplacian used, between 0 and 1. 
                                        Defaults to 1. beta = 1 - alpha.
            steps (int, optional): number of steps to solve in diffusion. Defaults to 500.
            dt (int, optional): time step. Defaults to 1.
            unit (str, optional): unit of time step. Defaults to 's'. must be 's', 'm' or 'h'
            save (bool, optional): Defaults to False.
            save_pth (str, optional): Defaults to None.


        Returns:
            sol (np.ndarray): solution containing population distribution at all nodes at each time step saved.
        """
        self.alpha = alpha
        self.beta = 1 - self.alpha
        self.G_d, _ = directed_graph(self.G,
                                     self.green_space,
                                     self.centre_gs,
                                     True,
                                     self.buffer)
        self.steps = steps
        W, _, L, gnodes = gen_L(self.G_d)
        if alpha < 1:
            W = W.todense()
            gs_nodes = [i for i in gnodes.values()]
            dist_matrix, predecessors = sp.sparse.csgraph.shortest_path(nx.to_scipy_sparse_array(self.G),
                                                                        directed=False,
                                                                        indices=gs_nodes,
                                                                        return_predecessors=True)

            closest_gs = [gs_nodes[list(dist_matrix[:, j]).index(np.min([i for i in list(
                dist_matrix[:, j]) if i > 0]))] for j in range(dist_matrix.shape[1])]
            W_drift = np.zeros(W.shape)
            for i in range(W_drift.shape[0]):
                if i not in gs_nodes:
                    pth_tmp = self.get_path(
                        predecessors, gs_nodes.index(closest_gs[i]), i)
                    W_drift[i, pth_tmp[-2]] = W[i, pth_tmp[-2]]
            L_drift = np.diag(np.sum(W_drift, axis=1)) - W_drift
            L = alpha*L.todense() + (self.beta)*L_drift
        else:
            L = L.todense()

        if unit == 's':
            time_mul = 1
        elif unit == 'm':
            time_mul = 60
        elif unit == 'h':
            time_mul = 3600
        else:
            raise ValueError(
                'unit must be one of s(econds), m(inutes), h(ours)')

        def deriv(t, p):
            return -1.4*time_mul*np.dot(L.T, p)

        p0 = [n[1]['pop_per_node'] for n in self.G_d.nodes(data=True)]
        self.t_end = steps*dt
        t_span = (0, self.t_end)
        save_times = list(range(0, self.t_end, int(dt)))
        MA_d = sp.integrate.solve_ivp(
            deriv, t_span, p0, method='BDF', t_eval=save_times, atol=1e-9, rtol=1e-9)
        sol = MA_d.y
        if save:
            if save_pth == None:
                save_pth = './results'
                if not os.path.exists(save_pth):
                    os.makedirs(save_pth)
                    print('Created directory: {}'.format(save_pth))
            np.save(
                save_pth + 'diff_sol_a{}_b{}.npy'.format(int(alpha*100), int((self.beta)*100)), sol)
        return sol

    def viz_diff(self, sol, t=-1, figsize=(20, 20), save=False, save_pth=None):
        """Visualise population distribution at time t

        Args:
            sol (np.ndarray): solution matrix of size (n, n) from diffusion
            t (int, optional): time step at when to plot the population. Defaults to -1.
            save (bool, optional): Defaults to False.
            save_pth (_type_, optional): Defaults to None.
        """
        gs = {id: 0 for id in self.green_space.id}
        sol_t = sol[:, t]
        G_u = self.G_d.to_undirected()
        for idx_n, n in enumerate(list(G_u.nodes(data=True))):
            if n[1]['gs_bool'] == True:
                for i in n[1]['gs_id']:
                    if i in gs:
                        gs[i] += sol_t[idx_n]/len(n[1]['gs_id'])

        vmin = 0
        vmax = np.max(list(gs.values()))
        norm = plt.Normalize(vmin, vmax)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", ["red", "orange", "yellow", "green"])

        gs_pop_list = np.array(list(gs.values()))
        gs_pop_list /= vmax
        gs_color = [matplotlib.colors.rgb2hex(cmap(i)) for i in gs_pop_list]
        ax = self.green_space.plot(color=gs_color, figsize=figsize)
        positions = {n: [n[0], n[1]] for n in list(self.G.nodes())}
        nx.draw(G_u, positions, ax=ax, node_size=0.1,
                node_color='black', edge_color='black', width=0.25)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, fraction=0.042, pad=0.04)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label('populaiton', rotation=90, labelpad=10, fontsize=15)
        fig = plt.gcf()
        title_time = sol.shape[1] if t == -1 else t
        fig_title = 'for full graph' if self.name == 'full' else 'centred at {}'.format(self.name)
        fig.suptitle(r'$Diffusion with \alpha$ = {}, $\beta$ = {} at time = {}'.format(self.alpha, self.beta, title_time)
                     + '\n' + '{}'.format(fig_title), fontsize=20)

        if save:
            if save_pth == None:
                save_pth = './results'
                if not os.path.exists(save_pth):
                    os.makedirs(save_pth)
                    print('Created directory: {}'.format(save_pth))
            fig.savefig(os.path.join(save_pth, 'diffusion_a{}_b{}_t{}_{}.png'.format(
                int(self.alpha*100), int(self.beta*100), title_time, '_'.join(self.name.split(' ')))))
            
    def find_nearest(self, array, value):
        """_summary_

        Args:
            array (nd.array): vector of population
            value (int): value to find the nearest index of from the array

        Returns:
            int: index of the nearest value in the array
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
            
    def viz_diff_pop_ex(self, sol, save=False, save_pth=None):
        """Visualise exchange of population between green space and non-green space during diffusion

        Args:
            sol (nd.array): solution matrix of size (n, n) from diffusion
            save (bool, optional): Defaults to False.
            save_pth (_type_, optional): Defaults to None.
        """
        G_u = self.G_d.to_undirected()
        save_times = list(range(0, self.t_end, int(self.t_end/self.steps)))
        perc = np.array([0.25, 0.5, 0.75])
        perc_col = ['red', 'blue', 'purple']

        pop_tol = np.sum(sol[:,0])
        pop_perc = pop_tol * perc
        gs_pop = np.zeros(len(save_times))
        ngs_pop = np.zeros(len(save_times))
        for idx_t, t in enumerate(save_times):
            for idx_n, n in enumerate(list(G_u.nodes(data=True))):
                if n[1]['gs_bool'] == True:
                    gs_pop[idx_t] += sol[idx_n, t]
                else:
                    ngs_pop[idx_t] += sol[idx_n, t]    
                    
        idx_perc = [self.find_nearest(gs_pop, i) for i in pop_perc]
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(1, 1, figsize=(12.5, 10))
        ax.plot(save_times, gs_pop, label='Green space population', color='green')
        ax.plot(save_times, ngs_pop, label='Non-green space population', color='orange')
        ax.set_xlabel('Time')
        ax.set_ylabel('Population')
        ax.set_xticks(list(np.arange(0, self.t_end+1, self.t_end/10)))
        ax.set_xlim((0, self.t_end))
        ax.set_ylim((0, pop_tol))
        ax2 = ax.twinx()
        ax2.set_ylabel('% Population')
        ax2.set_ylim((0, 100))
        ax2.plot([], [])
        for idx_j, j in enumerate(idx_perc):
            ax.axvline(x = j, ymax=pop_perc[idx_j]/pop_tol, color=perc_col[idx_j], linestyle='--', label='{}% population in green spaces'.format(int(perc[idx_j]*100)))
            ax.axhline(y = pop_perc[idx_j], xmin=j/save_times[-1], color=perc_col[idx_j], linestyle='--')
        ax.legend()
        if save:
            if save_pth == None:
                save_pth = './results'
                if not os.path.exists(save_pth):
                    os.makedirs(save_pth)
                    print('Created directory: {}'.format(save_pth))
            fig.savefig(os.path.join(save_pth, 'diffusion_population_a{}_b{}.png'.format(int(self.alpha*100), int(self.beta*100))))
            
    def set_sources(self):
        """Initialise weight of source at each node

        Returns:
            list: Source weight of all nodes
        """
        allnodes = list(self.G.nodes(data=True))
        totalpop = np.sum([n[1]['pop_per_node']
                          for n in self.G.nodes(data=True)])
        totalarea = np.sum(
            [n['geometry'].area for i, n in self.green_space.iterrows()])
        pop_to_area = totalpop/totalarea

        for i, park in self.green_space.iterrows():
            self.green_space.loc[i, 'pop_share'] = pop_to_area * \
                (park['geometry'].area)

        # making a dictionary to tell each greenspace how many nodes it has
        node_dict = {gs: 0 for gs in self.green_space.id}
        # gs_list = self.green_space.id.unique()
        # for gs in gs_list:
        #     node_dict[gs] = 0
        for n in self.G.nodes(data=True):
            if (n[1]['gs_bool']) == True:
                for gs_id in n[1]['gs_id']:
                    if gs_id in node_dict:  # for using subgraphs
                        node_dict[gs_id] += 1
                # elif type(n[1]['gs_id']) == str and len(n[1]['gs_id']) > 0:
                #     node_dict[n[1]['gs_id']] += 1

        # telling the greenspaces what share of the population per node
        for i, park in self.green_space.iterrows():
            if node_dict[park.id] == 0:
                self.green_space.loc[i, 'pop_per_node'] = 0
            else:
                self.green_space.loc[i, 'pop_per_node'] = park['pop_share']/node_dict[park.id]

        for n in list(self.G.nodes(data=True)):
            if n[1]['gs_bool'] == True:
                for gs_id in n[1]['gs_id']:
                    if gs_id in node_dict:  # for using subgraphs
                        n[1]['pop_per_node'] += self.green_space.loc[self.green_space.id ==
                                                                     gs_id, 'pop_per_node'].values[0]

        sourceweight = []
        for i in range(len(allnodes)):
            if allnodes[i][1]['gs_bool'] == True:
                sourceweight.append(allnodes[i][1]['pop_per_node'])
            else:
                # sink if its a city node
                sourceweight.append(-1*allnodes[i][1]['pop_per_node'])

        return sourceweight

    def happiness(self):
        """Compute equilibrium state solution of happiness diffusion
        """
        sourceweight = np.array(self.set_sources())
        if sum(sourceweight) > .1:
            print('Bad source setting. Sourcesum is ', sum(sourceweight))
            return
        _, _, L, _ = gen_L(self.G)
        equil = np.linalg.lstsq(-L.todense(), sourceweight, rcond=None)

        for idx, n in enumerate(list(self.G.nodes(data=True))):
            n[1]['happiness'] = equil[0][idx]
