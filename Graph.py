'''A class to represent graphs'''
class Graph:

    def __init__(self):
        self.numVertices = None # number of vertices in the graph
        self.adjMat= None # adjacency matrix
        self.weightMat= None # weight matrix
        self.adjList = None # adjacency lists

    def GenerateRandomComplete(self, n, squareSide = 100, coord = None):
        ''' Generates a random, undirected, and complete graph with n vertices
                
        n vertices will be scattered on a 100x100 plane
        the entries in the weight matrix will be the Euclidean distances between the respective vertices	
        '''
        import random
        import math

        self.numVertices = n
        self.adjMat = [[0 if j == i else 1 for j in range(n)] for i in range(n)]
        self.adjList = [[j for j in range(n) if j != i] for i in range(n)]

        if not coord:
            self.coord = [(random.random()*squareSide, random.random()*squareSide) for i in range(n)]
        else:
            self.coord = coord[:]

        d = lambda p1, p2: math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        p = self.coord
        self.weightMat = [[0 for j in range(n)] for i in range(n)]

        for i in range(n):
            for j in range(i+1, n):
                self.weightMat[i][j] = self.weightMat[j][i] = d(p[i], p[j])
