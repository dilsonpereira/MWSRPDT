from MPVRPIR import MPVRPIR
from heapq import heappush, heappop, heapify
import math

def ChooseTaskSmallestEndingTime(team, partialSolution, availableTasks, start, end):
    return min( availableTasks, key = lambda va: end(*va) )

def Constructive(P, ChooseTask = ChooseTaskSmallestEndingTime):
    # facilitating access to some structures
    teams = range(P.numTeams)
    customers = range(1, P.numVertices)
    tasks = [None] + [range(P.numTasks[P.request[v]]) for v in customers]
    dependencies = [None] + [[ P.Dependencies[ P.request[v] ][a] for a in tasks[v]] for v in customers]
    activityTime = P.taskTimes
    travelTime = P.travelTime

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
            end = lambda v, a: start(v,a) + activityTime[k][v][a]

            # choose an available task for the team
            availableForTeam = [(v, a) for (v, a) in available if end(v, a) + travelTime[v][0] <= P.availableTime]
            if not availableForTeam:
                solution[-1][k].append((0, None, t + travelTime[ solution[currentDay][k][-1][0] ][0]))
                continue

            (v, a) = ChooseTask(k, solution, availableForTeam, start, end)

            available.remove((v,a))
            toExecute.remove((v,a))

            execution[v][a] = (k, currentDay, start(v, a), end(v, a))

            # update opening times of those actitities that depend on a
            for b in dependent[v][a]:
                openingTime[v][b] = max( [ (execution[v][c][1], execution[v][c][3]) for c in dependencies[v][b] ] )
                if openingTime[v][b] != (math.inf, math.inf):
                    available.add((v,b))

            heappush(events, (end(v, a), k))
            solution[-1][k].append((v,a, start(v,a)))
        currentDay += 1

    return solution


