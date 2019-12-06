from MST import *

matrix = [["x",5,"x","x",6],
		  [5,"x",4,"x","x"],
		  ["x",4,"x",3.6,3.5],
		  ["x","x",3.6,"x",1],
		  [6,"x",3.5,1,"x"]]

print(kruskal(matrix))