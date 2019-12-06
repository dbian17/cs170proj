
#list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]
#https://gist.github.com/Peng-YM/84bd4b3f6ddcb75a147182e6bdf281a6

import numpy as np
import networkx as nx

#return the centers of each cluster
def clusterify(mst, matrix, k, list_of_homes):
	n = len(mst)
	sorted_edges = collect_edges(mst, 0.0, lambda e: -1 * mst[e[0]][e[1]])

	counter = 0

	for i in range(n):
		for j in range(n):
			if matrix[i][j] == "x":
				matrix[i][j] = 0.0

	matrix = np.array(matrix)
	#print(matrix)

	#delete k-1 most expensive edges to create k clusters
	for e in sorted_edges:
		u, v = e
		mst[u][v] = 0
		mst[v][u] = 0
		counter += 1
		#print("removed", (u,v))

		if counter == k-1:
			break

	#print(mst)
	#clusters is a list of sets, ith entry is nodes in ith cluster
	clusters = [[] for i in range(k)]
	#checks if vertex has been visited or not
	visited = set()
	#queue for DFS
	fringe = []

	start = 0
	c = -1
	
	while(len(visited) < n):
		#fringe empty means done exploring, move onto next cluster
		if len(fringe) == 0:
			c += 1
			while(start in visited):
				start += 1
			fringe.append(start)

		curr = fringe.pop()

		if curr not in visited:
			clusters[c].append(curr)
			visited.add(curr)
			for i in range(n):
				if mst[curr][i] != 0:
					fringe.append(i)

	#print(clusters)
	#create adjacency matrix for each cluster
	cluster_matrices = [np.full((n,n), 0.0) for i in range(k)]
	for c in range(k):
		curr_matrix = cluster_matrices[c]
		curr_cluster = clusters[c]

		for node1 in curr_cluster:
			for node2 in curr_cluster:
				curr_matrix[node1][node2] = matrix[node1][node2]

	#find which homes belong to which cluster
	cluster_homes = [[] for i in range(k)]
	#print(cluster_homes)
	for h in list_of_homes:
		c = 0
		while h not in clusters[c]:
			c += 1
		cluster_homes[c].append(h)

	#print(cluster_homes)

	#create graphs for each cluster
	graphs = [nx.Graph(cluster_matrices[c]) for c in range(k)]

	#find shortest distance from nodes to homes in each cluster
	centers = [0 for c in range(k)]
	
	for c in range(k):

		min_dist = float("inf")
		min_cen = centers[c]
		graph = graphs[c]

		for cen in clusters[c]:
			#can nx find distance from node to itself?
			distance = sum([nx.dijkstra_path_length(graph, cen, h) for h in cluster_homes[c]])

			if distance < min_dist:
				min_dist = distance
				min_cen = cen

		centers[c] = min_cen

	#calculate distances between centers
	main_graph = nx.Graph(matrix)
	c_distances = [[0 for i in centers] for i in centers]

	for i in range(len(centers)):
		for j in range(i):
			d = nx.dijkstra_path_length(main_graph, centers[i], centers[j])
			c_distances[i][j] = d
			c_distances[j][i] = d

	return c_distances, centers, {centers[c]:cluster_homes[c] for c in range(k)}


#kruskal alog -> returns adjancy matrix of the mst
def kruskal(matrix):
    # initialize MST
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
