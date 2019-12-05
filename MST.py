
#list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]
#https://gist.github.com/Peng-YM/84bd4b3f6ddcb75a147182e6bdf281a6

import numpy as np

#kruskal alog -> returns adjancy matrix of the mst
def kruskal(matrix):
    # initialize MST
    
    n_vertices = len(matrix)
    MST = np.full((n_vertices, n_vertices), 0)
    
    # collect all edges from graph G
    edges = set()
    for j in range(n_vertices):
        for k in range(j):
            if matrix[j][k] != "x": 
                edges.add((j,k))

    # sort all edges in graph G by weights from smallest to largest
    sorted_edges = sorted(edges, key = lambda e: matrix[e[0]][e[1]])
    
    uf = UF(n_vertices)

    for e in sorted_edges:
        u, v = e
        # if u, v already connected, abort this edge
        if uf.connected(u, v):
            continue
        # if not, connect them and add this edge to the MST
        uf.union(u, v)

        MST[u, v] = matrix[u, v]
        MST[v, u] = matrix[u, v]

        if (np.count_nonzero(MST) == 2*(n_vertices - 1)):
        	break

	return MST
    #NOTE: MST is adjancy matrix with "x" replaced by 0's



class UF:
    def __init__(self, N):
        self._id = [i for i in range(N)]

    # judge two node connected or not
    def connected(self, p, q):
        return self._find(p) == self._find(q)

    # quick union two component
    def union(self, p, q):
        p_root = self._find(p)
        q_root = self._find(q)
        if p_root == q_root:
            return
        self._id[p_root] = q_root

    # find the root of p
    def _find(self, p):
        while p != self._id[p]:
            p = self._id[p]
        return p
