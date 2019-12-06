import numpy as np
import random
import networkx as nx
from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import student_utils as stu

#something stupid for later
def mega_runner(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    best_stops = -1
    best_cost = float("inf")
    best_dict = {}
    best_route = []
    for i in range(1, len(list_of_homes)+1):
        cr, do, cost = runner(list_of_locations, list_of_homes,
                              starting_car_location, i, adjacency_matrix, 'average')
        print("with " + str(i) + " stops our cost is " + str(cost))
        if best_cost > cost:
            best_stops = i
            best_cost = cost
            best_dict = do
            best_route = cr
    return best_route, best_dict, best_cost



'''
params: list of locations, list of homes, string for starting, number of bus stop clusters, adjacency matrix
returns: a list that represents the vehicle route and a dictionary that represents dropoffs and home locations
'''

def runner(list_of_locations, list_of_homes, starting_car_location, num_clusters, adjacency_matrix, linkage):
    g, msg = adjacency_matrix_to_graph(adjacency_matrix)
    home_indices = home_names_to_indices(list_of_homes, list_of_locations)
    #print('number of homes is '+ str(len(home_indices)))
    home_distance_matrix = make_home_distance_matrix(g, home_indices)
    distance_matrix = make_distance_matrix(g, len(list_of_locations))
    clustering = make_clusters(num_clusters, linkage, home_distance_matrix)
    #print('number of clustering is '+ str(len(home_indices)))
    clusters = key_to_clusters(clustering, home_indices)
    bstops = all_bus_stop(clusters, distance_matrix)
    #print('we will have ' + str(len(bstops)) + ' bus stops')
    starting_index = index_of_start(starting_car_location, list_of_locations)
    stops = [starting_index] + bstops
    distance_matrix_of_stops =  make_home_distance_matrix(g, stops)
    bus_route = route(distance_matrix_of_stops, 0)
    for i in range(len(bus_route)):
        bus_route[i] = stops[bus_route[i]]
    complete_route = bus_stop_routing_to_complete_routing(bus_route, g)
    dropoffs = dropoff_dict(clusters, bstops)
    cost, m = stu.cost_of_solution(g, complete_route, dropoffs)
    return complete_route, dropoffs, cost


'''
params: list of homes and locations
returns: list of homes in terms of indices
'''
def home_names_to_indices(homes, locs):
    doritos = []
    for home in homes:
        doritos.append(locs.index(home))
    return doritos

def index_of_start(start, locs):
    return locs.index(start)

'''
params: adjacency matrix
returns: an error message and a graph
'''
def adjacency_matrix_to_graph(adjacency_matrix):
    node_weights = [adjacency_matrix[i][i] for i in range(len(adjacency_matrix))]
    adjacency_matrix_formatted = [[0 if entry == 'x' else entry for entry in row] for row in adjacency_matrix]

    for i in range(len(adjacency_matrix_formatted)):
        adjacency_matrix_formatted[i][i] = 0

    G = nx.convert_matrix.from_numpy_matrix(np.matrix(adjacency_matrix_formatted))

    message = ''

    for node, datadict in G.nodes.items():
        if node_weights[node] != 'x':
            message += 'The location {} has a road to itself. This is not allowed.\n'.format(node)
        datadict['weight'] = node_weights[node]

    return G, message

'''
params: raoG, home_indices_in_locations - a list that tells you the index of each home in list of locations
returns: distance matrix of homes only
'''
def make_home_distance_matrix(raoG, home_indices_in_locations):
    all_vertex_shortest_distances=list(nx.all_pairs_dijkstra_path_length(raoG))
    n = len(home_indices_in_locations)
    distances=np.zeros((n,n))
    # distances[m, k] is the length of the shortest path between i and j
    #print(home_indices_in_locations)
    for i in all_vertex_shortest_distances:
        if i[0] in home_indices_in_locations:
            for j in range(n):
                vertex_i = home_indices_in_locations.index(i[0])
                vertex_j = home_indices_in_locations[j]
                this_dict = i[1]
                distances[vertex_i][j] = this_dict[vertex_j]
    return distances

'''
params: raoG, num_locs (number of locations)
returns: distance matrix of homes only
'''
def make_distance_matrix(raoG, num_locs):
    n = num_locs #this is number of locations
    pcc=list(nx.all_pairs_dijkstra_path_length(raoG))
    distances=np.zeros((n,n))
    # distances[i, j] is the length of the shortest path between i and j
    for i in pcc:
        for key in i[1]:
            m = i[0] # this is the i
            k = key #this is j
            dist = i[1][key]
            distances[m,k] = dist
    return distances

'''
params: num_clusters, type of linkage, distance matrix
linkage - {“ward”, “complete”, “average”, “single”}, default is average
returns: a list with the list of all the clusters
'''
def make_clusters(num_clusters, linkage_type, distance_matrix):
    clustering = AgglomerativeClustering(num_clusters,
                                         linkage = linkage_type,
                                         affinity='precomputed').fit_predict(distance_matrix)
    return clustering

'''
params: cluster key list [0, 0, 0, 1], home indices in locations [2, 3, 4, 5]
returns: list of lists organized by cluster
'''
def key_to_clusters(clustering, home_indices_in_locations):
    mydict = {}
    for i in range(len(clustering)):
        cluster = clustering[i]
        if cluster in mydict:
            mydict[cluster].append(home_indices_in_locations[i])
        else:
            mydict[cluster] = [home_indices_in_locations[i]]
    alist = []
    for key in mydict:
        alist.append(mydict[key])
    return alist

'''
params: distances-distance matrix, cluster - cluster we lookin at
returns: best bus stop for our cluster
'''
def best_bus_stop(distances, cluster):
    mine = float("inf") #min distance
    min_index = -1 #best bus stop
    for row in range(len(distances)):
        nom = 0;
        for home in cluster:
            nom += distances[row][home]
        if nom < mine:
            min_index = row
            mine = nom
    #print('min travel distance is ' + str(mine))
    #print('the bus stop is '+ str(min_index))
    return min_index

'''
params: clusters - a list of list of clusters, distances - distance matrix
returns: a list of all bus stops
'''
def all_bus_stop(clusters, distances):
    bstops = []
    for cluster in clusters:
        stop = best_bus_stop(distances, cluster)
        bstops.append(stop)
    return bstops

'''
params: bus stop routing from tsp and graph
returns: complete routing
'''
def bus_stop_routing_to_complete_routing(bus_stop_routing, raoG):
    path = dict(nx.all_pairs_dijkstra_path(raoG))
    complete_route = [bus_stop_routing[0]]
    #path[u][v] gives the shortest path from u to v
    #for every pair of routing append the path
    for i in range(len(bus_stop_routing)-1):
        start = bus_stop_routing[i]
        end = bus_stop_routing[i+1]
        route = path[start][end]
        route = route[1:]
        complete_route.extend(route)
    return complete_route

'''
params: total amount of clusters
returns: bus stop nums
'''
def num_dropoffs(clusters):
    return len(clusters)

'''
params: clusters - list of cluster, bus stops - list of bus stops
returns: dictionary of drop off locations + hoesm
'''
def dropoff_dict(clusters, bus_stops):
    #print(clusters)
    #print(bus_stops)
    #print('we haave cluster amt '+ str(len(clusters)))
    #print('we have bus stop amt ' + str(len(bus_stops)))
    mydict = {}
    for i in range(len(clusters)):
        if bus_stops[i] in mydict:
            mydict[bus_stops[i]].extend(clusters[i])
        else:
            mydict[bus_stops[i]] = clusters[i]
    #print(mydict)
    #print('dic is size + ' + str(len(mydict)))
    return mydict


def create_data_model(bus_stop_matrix, starting_point):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = bus_stop_matrix
    data['num_vehicles'] = 1
    data['depot'] = starting_point
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    doritos = []
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            doritos.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        doritos.append(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        #print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    #print('Maximum of the route distances: {}m'.format(max_route_distance))
    return doritos, route_distance


'''
params: bus stop matrix (includes starting point), index of starting/ending point
returns: a route IN TERMS OF bus stop matrix
'''
def route(bus_stop_matrix, starting_point):
    """Solve the CVRP problem."""

    # Instantiate the data problem.
    data = create_data_model(bus_stop_matrix, starting_point)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        9223372036854775807,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)


    # Print solution on console.
    if solution:
        route_sol, memes = print_solution(data, manager, routing, solution)
        return route_sol

    return None
