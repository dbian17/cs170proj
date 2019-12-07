import os
import sys
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
    #route, dropoffs, cost = agglo.runner(list_of_locations, list_of_homes, starting_car_location, 44, adjacency_matrix, 'average')

    #best_stops = -1
    
    
    size = len(list_of_homes)
     
    #maps # clusters to its solution
    cluster_sol = [[] for c in range(1, size + 1)]
    #initialize start of search
    best_cluster = size//2
    k = best_cluster
    #keeps track of which # clusters has best solution
    rankings = [k]

    best_route, best_dict, best_cost = runner(list_of_locations, list_of_homes,
                                              starting_car_location, k, adjacency_matrix, 'average')
    cluster_sol[k] = [best_route, best_dict, best_cost]
    visited = set()
    visited.add(k)
    iterations = 0

    while (k > 1 and k < size and len(visited) < size and iterations < 6):
        if len(rankings) == 0:
            break

        #search thru most opitmal local_min for next iteration
        print(rankings)
        if len(rankings) > size//10:
            #sort local optima based on cost
            #rankings.sort(reverse = True, key = lambda i: cluster_sol[i][2])
            rankings.pop(0)
        #k = rankings[len(rankings) - 1]
        print(k)

        local_best_route = []
        local_best_dict = {}
        local_best_cost = float("inf")

        neighborhood = [i for i in range(max(1, k - size//10), min(k + size//10, size))]
        print(neighborhood)

        #finds best neighbor
        for neighbor in neighborhood:
            if (neighbor not in rankings):
                
                #if the cluster solution already found
                if len(cluster_sol[neighbor]) > 0:
                    curr_route = cluster_sol[neighbor][0]
                    curr_drop = cluster_sol[neighbor][1]
                    curr_cost = cluster_sol[neighbor][2]
                else:
                    curr_route, curr_drop, curr_cost = runner(list_of_locations, list_of_homes,
                                                  starting_car_location, neighbor, adjacency_matrix, 'average')
                    cluster_sol[neighbor] = [curr_route, curr_drop, curr_cost] 

                if (curr_cost < local_best_cost):
                    local_best_route = curr_route
                    local_best_dict = curr_drop
                    local_best_cost = curr_cost
                    local_best_cluster = neighbor
        #if neighbors provided better cost, add it to rankings
        rankings.append(local_best_cluster)
        k = local_best_cluster
        cluster_sol[local_best_cluster] = [local_best_route, local_best_dict, local_best_cost]
        
        if local_best_cost < best_cost:
            best_route = local_best_route
            best_dict = local_best_dict
            best_cost = local_best_cost
            best_cluster = local_best_cluster
        
        iterations += 1

    print("cost", best_cost)
    return best_route, best_dict
    #return cluster_sol[best_cluster][0], cluster_sol[best_cluster][1]


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
