#outputs must include:
#cycle that car took
#list of drop-off locations
#the number of distinct locations at which TAs were dropped off,
#the homes of the TAs who got off at each drop-off location

import numpy as np
import string
import random

dfs = []
dfs_result = []

def dfs_output(input_matrix, start):
    visited = []
    #unmark all vertices
    visited.extend([False]*len(input_matrix))

    #call dfs helper for halp >w<
    dfs_helper(input_matrix, visited, input_matrix[start], start, dfs, dfs_result)
    dfs.append(start)
    print(sum(dfs_result))
    return dfs

def dfs_helper(input, visited, vertex, vertex_i, dfs, dfs_result):
    visited[vertex_i] = True
    dfs.append(vertex_i)
    if all(visited):
        return dfs
    for i in range(len(vertex)):
        if vertex[i] != 0:
            if (not visited[i]):
                dfs_result.append(vertex[i])
                dfs_helper(input, visited, input[i], i, dfs, dfs_result)
                dfs_result.append(vertex[i])
                dfs.append(i)
    return dfs

def create_output(loc_names, loc_size, home_size, homes, path):
    filename = str(loc_size) + ".out"
    f = open(filename, "w")

    #path the car took
    for location in path:
        drop_off = loc_names[location]
        f.write(drop_off)
        f.write(" ")
    f.write("\n")

    #num of unique drop off locations
    f.write(str(home_size))
    f.write("\n")

    #homes of TAs dropped off at each location
    for home in homes[:len(homes)-1]:
        f.write(home)
        f.write(" ")
        f.write(home)
        f.write("\n")

    f.close()
