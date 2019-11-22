#num locations
#num homes
#names of locations
#names of homes
import numpy as np
import string
import random
from outputs import dfs_output, create_output

def make_input(loc_size, home_size):

    loc_names = []
    #makes some random locations
    for x in range(loc_size):
        loc_names.append(''.join(random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k=10)))
    home_names = np.random.choice(loc_names, home_size + 1, replace = False)
    #home_names[home_size] will be the starting location
    #b = np.random.random_integers(0,1,size=(loc_size,loc_size))
    #b_symm = (b + b.T)//2

    a = 99999
    b = 50000

    #start with no edges
    s = (loc_size, loc_size)
    b_symm = np.zeros(s)

    #makes a tree that is connected
    for x in range(1, loc_size):
        #pick a random node smaller than x
        random_v = np.random.randint(0, x)
        #makes a random edge length
        rand_edge = np.random.randint(50000, 99999)
        #puts it in matwix such that it is symmetric
        b_symm[x][random_v] = rand_edge
        b_symm[random_v][x] = rand_edge

    #adds some random ass edges
    for x in range(loc_size):
        v1 = np.random.randint(0, loc_size)
        v2 = np.random.randint(0, loc_size)
        if v1 == v2:
            continue
        rand_edge = np.random.randint(50000, 99999)
        b_symm[v1][v2] = rand_edge
        b_symm[v2][v1] = rand_edge

    path = dfs_output(b_symm) #output creation
    create_output(loc_names, loc_size, home_size, home_names, path)

    filename = str(loc_size) + ".in"
    f = open(filename, "w")

    #number of list_locations
    f.write(str(loc_size))
    f.write("\n")
    #number of homes
    f.write(str(home_size))
    f.write("\n")
    #list of locations, separated by spaces
    for loc_name in loc_names:
        f.write(loc_name)
        f.write(" ")
    f.write("\n")
    #list of homes, separated by spaces
    for i in range(home_size):
        f.write(home_names[i])
        f.write(" ")
    f.write("\n")
    #starting/ending location
    f.write(home_names[home_size])
    f.write("\n")
    #dumbass grid
    for gridrow in b_symm:
        for edge in gridrow:
            if edge > 0:
                f.write(str(edge))
                f.write(" ")
            else:
                f.write("x ")
        f.write("\n")

    f.close()

make_input(50, 25)
