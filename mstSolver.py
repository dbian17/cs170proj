import os
import sys
from MST import *
import argparse
import utils
from outputs import dfs_output, create_output
from agglo import *
from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """
    #runner(list_of_locations, list_of_homes, starting_car_location, num_clusters, adjacency_matrix, linkage):
    
    starting_index = [i for i in range(len(list_of_locations)) if list_of_locations[i] == starting_car_location][0]
    home_indices = [i for i in range(len(list_of_locations)) if list_of_locations[i] in list_of_homes]

    #k = 10

    min_span_tree = kruskal(adjacency_matrix)
    min_cost = float("inf")
    min_drop = {}
    min_stops = []
    min_path = []
    cost = 0

    for k in range(1, len(list_of_homes) + 1):
        distance_matrix_of_stops, stops, dropoffs, g = clusterify(np.copy(min_span_tree), adjacency_matrix, k, home_indices, starting_index)  
        #print(distance_matrix_of_stops, starting_index[0])
        bus_route = route(distance_matrix_of_stops, 0)
        #print("go")

        #print(bus_route)
        #print(stops)
        #convert bus_route index numbers to the location index numbers
        for i in range(len(bus_route)):
            bus_route[i] = stops[bus_route[i]]

        complete_route = bus_stop_routing_to_complete_routing(bus_route, g)
        cost, m = stu.cost_of_solution(g, complete_route, dropoffs)

        if(cost < min_cost):
            min_cost = cost
            min_drop = dropoffs
            min_stops = stops
            min_path = complete_route
        print("with " + str(i) + " stops our cost is " + str(cost))

    print("-----",'\n' , len(min_stops), "stops produces", min_cost,'\n', "-----")    
    return min_path, min_drop

"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    #print("path", path, '\n')
    #print("drop", dropoff_mapping, '\n')
    #print("list_loc", list_locs, '\n')
    string = ''
    for node in path:
        #print(len(list_locs))
        #print(node)
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
