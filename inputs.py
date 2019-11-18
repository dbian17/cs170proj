#num locations
#num homes
#names of locations
#names of homes
import numpy as np
import string
import random

def make_input(loc_size, home_size):

    loc_names = []
    #makes some random locations
    for x in range(loc_size):
        loc_names.append(''.join(random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k=10)))
    home_names = np.random.choice(loc_names, home_size, replace = False)
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

    print(loc_size)
    print(home_size)
    print(loc_names)
    print(home_names)
    print(b_symm)