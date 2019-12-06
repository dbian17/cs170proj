from MST import *

matrix = [["x",7,"x",5,"x","x","x"],
		  [7,"x",8,9,7,"x","x"],
		  ["x",8,"x","x",5,"x","x"],
		  [5,9,"x","x",15,6,"x"],
		  ["x",7,5,15,"x",8,9],
		  ["x","x","x",6,8,"x",11],
		  ["x","x","x","x",9,11,"x"]
		  ]
clusterify(kruskal(matrix), matrix, 3, [0,2,6])