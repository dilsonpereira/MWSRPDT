import Formigueiro
import random
from MPVRPIR import *
from heapq import heappush, heappop, heapify
import math

team_day_v1_v2 = 1
team_v1_v2 = 2
team_v2 = 3
v1_v2 = 4

def GetAntCls(AlgType = Formigueiro.MMAS_Ant):
    class TSPAnt(AlgType):
        def __init__(self, instance, componentType = team_v1_v2, **kwargs):
            self.instance = instance
            self.componentType = componentType

            super().__init__(**kwargs)

        def getComponentCost(self, component):
            v, a = component[-2], component[-1]
            return self.end(v, a)
        
        def constructSolution(self):
            instance = self.instance
            
            # facilitating access to some structures
            teams = range(instance.numTeams)
            customers = range(1, instance.numVertices)
            tasks = [None] + [range(instance.numTasks[instance.request[v]]) for v in customers]
            dependencies = [None] + [[ instance.Dependencies[ instance.request[v] ][a] for a in tasks[v]] for v in customers]
            taskTimes = instance.taskTimes
            travelTime = instance.travelTime

            # a is dependent of b (a in dependent[b]) if b in dependencies[a]
            dependent = [None] + [[ [] for a in tasks[v]] for v in customers] 
            for v in customers:
                for a in tasks[v]:
                    for b in dependencies[v][a]:
                        dependent[v][b].append(a)

            # opening time of each task (day, hour)
            openingTime = [None] + [[(math.inf, math.inf) if dependencies[v][a] else (0,0) for a in tasks[v]] for v in customers]

            # execution information for each task
            # (team, execution day, starting hour, ending hour) 
            execution = [None] + [[(None, math.inf, math.inf, math.inf) for a in tasks[v]] for v in customers]

            toExecute = {(v, a) for v in customers for a in tasks[v]}

            solution = []

            currentDay = 0
            while toExecute:
                # available tasks
                # a task is available if its dependencies were executed and the task itself was not executed
                available = {(v, a) for v in customers for a in tasks[v] if openingTime[v][a] != (math.inf, math.inf) and (v, a) in toExecute}

                # tasks executed by each team on currentDay
                # list of (vertex, task, starting time) for each team
                solution.append( [[(0, None, 0)] for k in teams] )

                # heap with events, an event is a (time of completion of some task, team)
                events = [(0, k) for k in teams]

                while events:
                    # next team completing a task, team is now free
                    (t, k) = heappop(events)

                    start = lambda v, a: max(t + travelTime[ solution[currentDay][k][-1][0] ][v], openingTime[v][a][1] if openingTime[v][a][0] == currentDay else 0)
                    end = lambda v, a: start(v,a) + taskTimes[k][v][a]
                    self.end = end
                    v_, a_, _ = solution[currentDay][k][-1]

                    # choose an available task for the team
                    if self.componentType == team_v1_v2:
                        availableForTeam = [(k, v_, a_, v, a) for (v, a) in available if end(v, a) + travelTime[v][0] <= instance.availableTime]
                    elif self.componentType == team_day_v1_v2:
                        availableForTeam = [(k, currentDay, v_, a_, v, a) for (v, a) in available if end(v, a) + travelTime[v][0] <= instance.availableTime]
                    elif self.componentType == team_v2:
                        availableForTeam = [(k, v, a) for (v, a) in available if end(v, a) + transTime[v][0] <= instance.availableTime]
                    else:
                        availableForTeam = [(v_, a_, v, a) for (v, a) in available if end(v, a) + transTime[v][0] <= instance.availableTime]

                    if not availableForTeam:
                        solution[-1][k].append((0, None, t + travelTime[ solution[currentDay][k][-1][0] ][0]))
                        continue

                    dec = self.makeDecision(availableForTeam)
                    v, a = dec[-2], dec[-1]

                    available.remove((v,a))
                    toExecute.remove((v,a))

                    execution[v][a] = (k, currentDay, start(v, a), end(v, a))

                    # update opening times of those tasks that depend on a
                    for b in dependent[v][a]:
                        openingTime[v][b] = max( [ (execution[v][c][1], execution[v][c][3]) for c in dependencies[v][b] ] )
                        if openingTime[v][b] != (math.inf, math.inf):
                            available.add((v,b))

                    heappush(events, (end(v, a), k))
                    solution[-1][k].append((v,a, start(v,a)))
                currentDay += 1

            self.solution = solution
            
        def getSolutionValue(self):
            M = []
            for k in range(self.instance.numTeams):
                #M.append(max([x[2] for x in self.solution[-1][k]]))
                M.append(self.solution[-1][k][-1][2])

            #print(M)
            return len(self.solution)-1 + max(M)/8
            #return len(self.solution)

    return TSPAnt

def Solve(problem):
    antArgs  = {'rho': 0.020247742915913844, 'alpha': 6.471059083737413, 'tau0': 8.880886363687786, 'maxPheromone': 5.694118387554422, 'beta': 5.786631522658782, 'minPheromone': 0.028678791658380387, 'Q': 9.966879981510193, 'componentType': 1}
    bestAnt = Formigueiro.Solve(antCls = GetAntCls(Formigueiro.MMAS_Ant), instance = problem, numIterations = 100, numAnts = 100, **antArgs)

    return bestAnt.solution


