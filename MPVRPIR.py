import random
import math
from Graph import Graph

class MPVRPIR:
    def __init__(self, 
            numVertices = 0, 
            numServices = 1, 
            numTasks = 1, # list with the number of tasks in each service
            numTeams = 1, 
            availableTime = 8, 
            timeSet = [0.5, 1, 1.5, 2], # set for drawing standard task times
            skillLevel = [0.5, 1, 2], # set for drawing skill levels of the teams on the tasks
            gridSize = 100, # size of the grid from which customers will be sampled
            gridDist = 0.1, # distance betwen grids in Km
            speed = 40, # vehicle speed in Km/h
            type = 'A' # instance type
            ):

        if numVertices != 0:
            self.numVertices = numVertices
            self.numServices = numServices
            self.numTasks = numTasks
            self.numTeams = numTeams
            self.availableTime = availableTime
            self.timeSet = timeSet
            self.skillLevel = skillLevel
            self.gridSize = gridSize
            self.gridDist = gridDist
            self.speed = speed

            self.GenerateRandomServices()
            self.GenerateCustomerGraph()
            self.GenerateRequests()

            if type == 'A':
                self.GenerateTaskTimesA()
            elif type == 'B':
                self.GenerateTaskTimesB()
            else:
                self.GenerateTaskTimesC()

    def GenerateRandomServices(self):
        '''Generates a random dependence graph for each of the services '''

        self.Dependencies = []
        for s in range(self.numServices):
            self.Dependencies.append(self.GenerateServiceGraph(self.numTasks[s]))

    def GenerateServiceGraph(self, n):
        '''Generates a random, directed, and acyclic graph '''

        # distribute the vertices over 3 levels
        levels = [[] for x in range(3)]
        for v in range(n):
            random.choice(levels).append(v)

        # remove empty levels
        levels = [ l for l in levels if l != [] ]

        # construct the graph
        adjList = [[] for x in range(n)]
        
        # each task depends on every task from the previous level
        # u -> v means that u depends on v
        for x in range(1, len(levels)):
            for u in levels[x]:
                for v in levels[x-1]:
                    adjList[u].append(v)

        return adjList

    def GenerateCustomerGraph(self):
        gridSize = self.gridSize
        speed = self.speed
        gridDist = self.gridDist
        n = self.numVertices

        '''
        graph = Graph()
        graph.GenerateRandomComplete(n)
        '''

        x = [random.randint(0, gridSize) for v in range(n)]
        y = [random.randint(0, gridSize) for v in range(n)]

        self.travelTime = [[0 for u in range(n)] for v in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                self.travelTime[u][v] = self.travelTime[v][u] = gridDist*(abs(x[u]-x[v]) + abs(y[u]-y[v]))/speed;


    def GenerateRequests(self):
        '''Generates a random request for each customer '''

        self.request = [random.choice(range(self.numServices)) for u in range(self.numVertices)]
        self.request[0] = None

    def GenerateTaskTimesA(self):
        timeSet = self.timeSet
        skillLevel = self.skillLevel
        times = [ [random.choice(timeSet) for a in range(self.numTasks[s])] for s in range(self.numServices)] 
        self.stimes = times
        skills = [[ [random.choice(skillLevel) for a in range(self.numTasks[s])] for s in range(self.numServices)] for t in range(self.numTeams)]

        self.taskTimes = [ [None] + [ [ 0 for a in range(self.numTasks[self.request[c]])] for c in range(1,self.numVertices) ] for u in range(self.numTeams) ]

        for u in range(self.numTeams):
            for c in range(1, self.numVertices):
                s = self.request[c]
                for a in range(self.numTasks[s]):
                    self.taskTimes[u][c][a] = times[s][a]/skills[u][s][a]

    def GenerateTaskTimesB(self):
        timeSet = self.timeSet
        self.timeSet = timeSet+[1000]

        self.GenerateTaskTimesA()
        self.timeSet = timeSet

        oneTeam = { (s, a):(random.randint(0, self.numTeams-1), random.choice(self.timeSet)*random.choice(self.skillLevel)) for s in range(self.numServices) for a in range(self.numTasks[s]) }

        for c in range(1, self.numVertices):
            s = self.request[c]
            for a in range(self.numTasks[s]):
                u, t = oneTeam[(s,a)]
                self.taskTimes[u][c][a] = t

    def GenerateTaskTimesC(self):
        oneTeam = { (s, a):(a, random.choice(self.timeSet)*random.choice(self.skillLevel)) for s in range(self.numServices) for a in range(self.numTasks[s]) }

        self.taskTimes = [ [None] + [ [ 1000 for a in range(self.numTasks[self.request[c]])] for c in range(1,self.numVertices) ] for u in range(self.numTeams) ]

        for c in range(1, self.numVertices):
            s = self.request[c]
            for a in range(self.numTasks[s]):
                u, t = oneTeam[(s,a)]        
                self.taskTimes[u][c][a] = t

    def SaveToFile(self, fileName):
        f = open(fileName, 'w')

        f.write('Number_of_customers: {}\n'.format(self.numVertices))
        f.write('Number_of_teams: {}\n'.format(self.numTeams))
        f.write('Number_of_services: {}\n'.format(self.numServices))
        f.write('Customers_requested_services:\n')
        f.write( ' '.join( [ str(x) for x in self.request[1:] ] ) + '\n')
        f.write('Daily_available_time: {}\n'.format(self.availableTime))
        for s in range(self.numServices):
            f.write('Service {}:\n'.format(s))
            f.write('Number_of_tasks: {}\n'.format(self.numTasks[s]))
            f.write('Dependencies:\n')
            for a in range(self.numTasks[s]):
                f.write(str(a) + ': ' + ' '.join([str(x) for x in self.Dependencies[s][a]]) + '\n')
        f.write('Travel_times:\n')
        for v in range(self.numVertices):
            f.write(' '.join(['{0:.2f}'.format(x) for x in self.travelTime[v]]) + '\n')
        f.write('Team_tasks_times:\n')
        for v in range(self.numTeams):
            f.write('Team {}:\n'.format(v))
            for c in range(1,self.numVertices):
                f.write(str(c) + ': ' + ' '.join(['{0:.2f}'.format(x) for x in self.taskTimes[v][c] ]) + '\n')

        f.close()

    def ReadInstance(self, fileName):
        f = open(fileName, 'r')

        s = f.readline().split()
        self.numVertices = int(s[1])

        s = f.readline().split()
        self.numTeams = int(s[1])

        s = f.readline().split()
        self.numServices = int(s[1])

        s = f.readline().split()
        self.request = [None] + [int(x) for x in f.readline().split()]

        s = f.readline().split()
        self.availableTime = int(s[1])

        self.numTasks = []
        self.Dependencies = []
        for i in range(self.numServices):
            s = f.readline()
            s = f.readline().split()
            self.numTasks.append(int(s[1]))

            self.Dependencies.append([])
            s = f.readline()
            for a in range(self.numTasks[i]):
                adjList = [int(x) for x in f.readline().split()[1:]]
                self.Dependencies[-1].append(adjList)

        self.travelTime = []
        s = f.readline()
        for a in range(self.numVertices):
            self.travelTime.append([float(x) for x in f.readline().split()])

        self.taskTimes = [[None] + [ [] for s in range(1,self.numVertices)] for a in range(self.numTeams)]
        s = f.readline()
        for a in range(self.numTeams):
            s = f.readline()
            for b in range(1,self.numVertices):
                self.taskTimes[a][b] = [float(x) for x in f.readline().split()[1:]]

        f.close()

