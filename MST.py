
#list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]
#https://gist.github.com/Peng-YM/84bd4b3f6ddcb75a147182e6bdf281a6

import numpy as np
import networkx as nx

#delete k-1 most expensive edges to create k clusters
#return the centers of each cluster
def clusterify(matrix, k, list_of_homes):
	mst = kruskal(matrix)
	n = len(matrix)

	sorted_edges = collect_edges(mst, 0, lambda e: -1 * matrix[e[0]][e[1]])

	counter = 0

	for e in sorted_edges:
		u, v = e
		mst[u][v] = 0
		mst[v][u] = 0
		counter += 1

		if counter == k-1:
			break


	#clusters is a list of sets, ith entry is nodes in ith cluster
	clusters = [set() for i in range(k)]
	#checks if vertex has been visited or not
	visited = set()
	#queue for DFS
	fringe = []

	start = 0
	k = -1
	
	while(len(visited) < n):
		#fringe empty means done exploring, move onto next cluster
		if len(fringe) == 0:
			k += 1
			while(start in visited):
				start += 1
			fringe.append(start)

		curr = fringe.pop()

		if curr not in visited:
			clusters[k].add(curr)
			visited.add(curr)
			for i in range(n):
				if mst[curr][i] != 0:
					fringe.append(i)

	#create adjacency matrix for each cluster
	cluster_matrices = [np.full((n,n), 0.0) for i in range(k)]
	for c in range(k):
		curr_matrix = cluster_matrices[c]
		curr_cluster = clusters[c]

		for node1 in curr_cluster:
			for node2 in curr_cluster:
				curr_matrix[node1][node2] = matrix[node1][node2]

	#find which homes belong to which cluster
	cluster_homes = [set() for i in range(k)]
	for h in list_of_homes:
		c = 0
		while h not in clusters[c]:
			c += 1
		cluster_homes[c].add(h)

	#create graphs for each cluster
	graphs = [Graph(cluster_matrices[c]) for c in range(k)]

	#find shortest distance from nodes to homes in each cluster
	centers = [0 for c in range(k)]
	
	for c in range(k):

		min_dist = float("inf")
		min_cen = centers[c]
		graph = graphs[c]

		for cen in clusters[c]:
			#can nx find distance from node to itself?
			distance = sum([nx.dijkstra_path(graph, cen, h)] for h in homes[c])

			if distance < min_dist:
				min_dist = distance
				min_cen = cen

		centers[c] = min_cen

	return centers


#kruskal alog -> returns adjancy matrix of the mst
def kruskal(matrix):
    # initialize MST
    print(matrix)
    n_vertices = len(matrix)
    mst = np.full((n_vertices, n_vertices), 0.0)
    
    # sort all edges in graph G by weights from smallest to largest
    sorted_edges = collect_edges(matrix, "x", lambda e: matrix[e[0]][e[1]])
    
    uf = UF(n_vertices)

    for e in sorted_edges:
        u, v = e
        # if u, v already connected, abort this edge
        if uf.connected(u, v):
            continue
        # if not, connect them and add this edge to the MST
        uf.union(u, v)

        mst[u][v] = matrix[u][v]
        mst[v][u] = matrix[u][v]

        if (np.count_nonzero(mst) == 2*(n_vertices - 1)):
        	break
    return mst
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


#collects edges of matrix and returns set of ordered edges in form ((u1,v1)...)
def collect_edges(matrix, zero, sort):
	n_vertices = len(matrix)
	edges = set()

	for j in range(n_vertices):
		for k in range(j):
			if matrix[j][k] != zero:
				edges.add((j,k))

	return sorted(edges, key = sort)
