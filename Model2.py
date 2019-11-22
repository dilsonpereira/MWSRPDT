from MPVRPIR import *
from math import ceil
from collections import namedtuple
import random

import cplex
from cplex.exceptions import CplexError
from cplex.callbacks import HeuristicCallback, MIPInfoCallback

import Constructive

from collections import defaultdict

import os
import psutil

class MyInfo(MIPInfoCallback):
    def __init__(self, env):
        MIPInfoCallback.__init__(self, env)
        self.info = defaultdict(lambda: 0)
    def __call__(self):
        self.info['lb'] = self.get_best_objective_value()
        self.info['numnodes'] = self.get_num_nodes()
        if self.get_num_nodes() == 0:
            self.info['lbroot'] = self.get_best_objective_value()

        process = psutil.Process(os.getpid())

        if process.memory_percent() > 90:
            print('out of memory')
            self.abort()

def SolveCplexModel2(P, heu = None, useHeuristicCallback = True, integer = True):
    prob = cplex.Cplex()
		
    prob.objective.set_sense(prob.objective.sense.minimize)

    # build extended graph
    # vertices
    Vext = [(0, None)] + [(v, a) for v in range(1, P.numVertices) for a in range(P.numTasks[P.request[v]])]
    Vidx = {v:k for (k, v) in enumerate(Vext)}
    numVext = len(Vext)
    # adjacency lists
    # AdjEx = [[u for v in range(len(Vext)) if u != v] for v in range(len(Vext))]

    # lambda function for getting activity times on the extended graph
    activityTime = lambda k, v: P.taskTimes[k][Vext[v][0]][Vext[v][1]]
    # lambda function for getting translation times on the extended graph
    transTime = lambda u, v: 0 if Vext[u][0] == Vext[v][0] else P.travelTime[Vext[u][0]][Vext[v][0]]

    # creating the variables
    Variable = namedtuple('Variable', ['name', 'obj', 'lb', 'ub', 'type'])

    # x[k,h,u,v] in {0,1}: 1 if team k goes from u to v on day h
    # x is a list of tuples, one for each variable x, each tuple consists of (name, obj, lb, ub, type)
    xname = lambda k,h,u,v: 'x_{}_{}_{}_{}'.format(k,h,u,v)
    x = [ Variable(name=xname(k,h,u,v), obj=0, lb=0, ub=1, type='I') for k in range(P.numTeams) for h in range(P.maxDays) for u in range(numVext) for v in range(numVext) ]

    name, obj, lb, ub, types = list(zip(*x))
    if integer:
        prob.variables.add(names = name, obj = obj, lb = lb, ub = ub, types = types)
    else:
        prob.variables.add(names = name, obj = obj, lb = lb, ub = ub)

    # q[k,h,u,v] in R+: moment in which team k arrives at vertex v comming from u on day h
    qname = lambda k,h,u,v: 'q_{}_{}_{}_{}'.format(k,h,u,v)
    q = [ Variable(name=qname(k,h,u,v), obj=0, lb=0, ub=cplex.infinity, type='C') for k in range(P.numTeams) for h in range(P.maxDays) for u in range(numVext) for v in range(numVext) ]
    name, obj, lb, ub, types = list(zip(*q))
    if integer:
        prob.variables.add(names = name, obj = obj, lb = lb, ub = ub, types = types)
    else:
        prob.variables.add(names = name, obj = obj, lb = lb, ub = ub)

    # y[k,h,i,v] in {0,1}: 1 if team k executes activity i of customer v on day h
    yname = lambda k,h,i,v: 'y_{}_{}_{}_{}'.format(k,h,i,v)
    y = [ Variable(name=yname(k,h,i,v), obj=0, lb=0, ub=1, type='I') for k in range(P.numTeams) for h in range(P.maxDays) for v in range(1, P.numVertices) for i in range(P.numTasks[P.request[v]])]

    name, obj, lb, ub, types = list(zip(*y))
    # setting branching priority
    if integer:
        prob.variables.add(names = name, obj = obj, lb = lb, ub = ub, types = types)
        prob.order.set([(var.name, 10, prob.order.branch_direction.default) for var in y])
    else:
        prob.variables.add(names = name, obj = obj, lb = lb, ub = ub)

    # p in Z+: number of days
    p = [ Variable(name='p', obj = 1, lb = 0, ub = cplex.infinity, type = 'C' )]
    if integer:
        prob.variables.add(names = ['p'], obj = [1], lb = [0], ub = [cplex.infinity], types = ['C'])
    else:
        prob.variables.add(names = ['p'], obj = [1], lb = [0], ub = [cplex.infinity])

    #prob.order.set([('p', 100, prob.order.branch_direction.default)])

    # Every activity of the service request by a custumer must must be executed 
    for v in range(1, P.numVertices):
        for i in range(P.numTasks[P.request[v]]):
            vars, coefs = [], []
            for k in range(P.numTeams):
                for h in range(P.maxDays):
                    vars.append(yname(k,h,i,v))
                    coefs.append(1)
            prob.linear_constraints.add(lin_expr = [[vars,coefs]], senses = ['E'], rhs = [1])

    # An activity can only be executed in a place if the activity on which it depends has been completed before
    for v in range(1, P.numVertices):
        for i in range(P.numTasks[P.request[v]]):
            for j in P.Dependencies[ P.request[v] ][i]:
                vars, coefs = [], []
                for h in range(P.maxDays):
                    for k in range(P.numTeams):
                        vars.append(yname(k, h, i, v))
                        coefs.append(P.availableTime*h)

                        vars.append(yname(k, h, j, v))
                        coefs.append(-P.availableTime*h - activityTime(k, Vidx[(v, j)]))
                        for u in range(numVext):
                            if Vext[u] != (v, i):
                                vars.append(qname(k, h, u, Vidx[(v, i)]))
                                coefs.append(1)
                            if Vext[u] != (v, j):
                                vars.append(qname(k, h, u, Vidx[(v, j)]))
                                coefs.append(-1)
                            
                prob.linear_constraints.add(lin_expr=[[vars, coefs]], senses = ['G'], rhs = [0] )

    # If an activity is executed in a place on a period, then the corresponding team has to visit the place
    for v in range(1, numVext):
        for k in range(P.numTeams):
            for h in range(P.maxDays):
                vars, coefs = [yname(k,h,Vext[v][1],Vext[v][0])], [-1]

                for u in range(numVext):
                    if u == v: 
                        continue
                    vars.append(xname(k,h,u,v))
                    coefs.append(1)

                prob.linear_constraints.add(lin_expr=[[vars,coefs]], senses = ['E'], rhs = [0])
    # If an activity is executed in a place on a period, then the corresponding team has to visit the place
    for v in range(1, numVext):
        for k in range(P.numTeams):
            for h in range(P.maxDays):
                vars, coefs = [yname(k,h,Vext[v][1],Vext[v][0])], [-1]

                for u in range(numVext):
                    if u == v: 
                            continue
                    vars.append(xname(k,h,v,u))
                    coefs.append(1)

                prob.linear_constraints.add(lin_expr=[[vars,coefs]], senses = ['E'], rhs = [0])

    # A team cannot return to depot and leave again
    for k in range(P.numTeams):
        for h in range(P.maxDays):
            vars, coefs = [], []
            for v in range(1, numVext):
                vars.append(xname(k,h,0,v))
                coefs.append(1)
            prob.linear_constraints.add(lin_expr = [[vars, coefs]], senses = ['L'], rhs = [1])
				
    # Correct value of p
    for k in range(P.numTeams):
        for h in range(P.maxDays):
            vars, coefs = ['p'], [-1]
            for v in range(1, numVext):
                vars.append(xname(k, h, 0, v))
                coefs.append(h+1)
                #vars.append(qname(k, h, v, 0))
                #coefs.append(1/P.availableTime)
            prob.linear_constraints.add(lin_expr=[[vars, coefs]], senses=['L'], rhs= [0])

    # flow out of the depot
    for v in range(1, numVext):
        for k in range(P.numTeams):
            for h in range(P.maxDays):
                vars, coefs = [xname(k,h,0,v), qname(k, h, 0, v)], [-transTime(0, v), 1]

                prob.linear_constraints.add(lin_expr=[[vars,coefs]], senses = ['G'], rhs = [0])

    # flow conservation
    for v in range(1, numVext):
        for k in range(P.numTeams):
            for h in range(P.maxDays):
                vars, coefs = [], []
                for u in range(numVext):
                    if u == v:
                        continue
                    vars.extend( [qname(k, h, u, v), xname(k, h, v, u), qname(k, h, v, u)] )
                    coefs.extend( [1, transTime(v, u), -1] )

                vars.append(yname(k, h, Vext[v][1], Vext[v][0]))
                coefs.append(activityTime(k, v))

                prob.linear_constraints.add(lin_expr=[[vars,coefs]], senses = ['L'], rhs = [0])

    # flow capacity
    for v in range(numVext):
        for u in range(numVext):
            if u == v:
                continue
            for k in range(P.numTeams):
                for h in range(P.maxDays):
                    vars, coefs = [xname(k,h,u,v), qname(k, h, u, v)], [-P.availableTime, 1]

                    prob.linear_constraints.add(lin_expr=[[vars,coefs]], senses = ['L'], rhs = [0])
                    #prob.linear_constraints.advanced.add_lazy_constraints(lin_expr=[[vars,coefs]], senses = ['L'], rhs = [0])

    # create mipstart solution
    if integer and heu:
        mipstart = Model2MipStartFromHeuristic(P, heu, x+q+y+p, xname, qname, yname, Vidx)
        prob.MIP_starts.add(mipstart, prob.MIP_starts.effort_level.check_feasibility)
    
    # register heuristic callback
    if integer and useHeuristicCallback:
        heuCallback = prob.register_callback(Model2HeuristicCallback)
        heuCallback.P = P
        heuCallback.xname, heuCallback.yname, heuCallback.qname = xname, yname, qname
        heuCallback.Vidx = Vidx
        heuCallback.vars = x+y+q+p

    if integer:
        ic = prob.register_callback(MyInfo)
        info = ic.info
    else:
        info = {}

    prob.set_log_stream(None)
    prob.set_results_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)

    prob.parameters.timelimit.set(3600)

    prob.solve()
    
    ub = prob.solution.get_objective_value()

    if integer:
        info['ub'] = ub
        info['ubheu'] = len(heu)
        info['numnodes'] += 1
        info['status'] = prob.solution.status[prob.solution.get_status()]
    else:
        info['lb'] = ub

    if integer:
        ub = int(ub)
        solution = []
        for h in range(ub):
            solution.append( [[(0, None, 0)] for k in range(P.numTeams)] )

            for k in range(P.numTeams):
                s = 0
                while True: 
                    t = 0
                    for v in range(numVext):
                        vx = prob.solution.get_values(xname(k,h,s,v))
                        if vx > 0:
                            t = v
                            break
                    q = prob.solution.get_values(qname(k, h, s, t))
                    s = t
                    solution[-1][k].append((Vext[s][0], Vext[s][1], q))
                    if s == 0:
                        break
    else:
        solution = None

    return info, solution

# Heuristic callback class for model 2               
class Model2HeuristicCallback(HeuristicCallback):
    def __call__(self):
        cur = self.get_incumbent_objective_value()
        
        varnames = [var.name for var in self.vars]
        varvalues = self.get_values(varnames)
        
        vardict = dict(zip(varnames, varvalues))
        def ChooseTaskHeuristicCallback(team, partialSolution, availableTasks, start, end):
            currentDay = len(partialSolution)-1
            if currentDay >= cur:
                return availableTasks[0]
            lastva = partialSolution[-1][team][-1][0], partialSolution[-1][team][-1][1]
            xval = lambda v, a: vardict[ self.xname(team, currentDay, self.Vidx[lastva], self.Vidx[v,a]) ] 
            yval = lambda v, a: vardict[ self.yname(team, currentDay, a, v) ]
            # return min( availableTasks, key = lambda va: end(*va)*(2 -xval(*va) - yval(*va)) )
            return max( availableTasks, key = lambda va: xval(*va) + yval(*va) )
            #return max( availableTasks, key = lambda va: xval(*va) )
            #return random.choice(sorted( availableTasks, key = lambda va: xval(*va) * yval(*va), reverse=True )[:3])

        heu = Constructive.Constructive(self.P, ChooseTaskHeuristicCallback)
        if len(heu) < cur:
            print('heuristic callback found new incumbent: ', len(heu))
            solution = Model2MipStartFromHeuristic( self.P, heu, self.vars, self.xname, self.qname, self.yname, self.Vidx)
            self.set_solution(solution)

# Creates a starting solution for the model given the heuristic solution                
def Model2MipStartFromHeuristic(P, heu, vars, xname, qname, yname, Vidx):
    # heu is a list of days
    # each day is a list of teams
    # each team is a list of activities
    # each activity is a tuple (vertex, activity, starting time)
                  
    mipdict = {var.name:0 for var in vars}              
    mipdict['p'] = len(heu)
    for h in range(len(heu)):
        for k in range(len(heu[h])):
            for (v1, a1, s1), (v2, a2, s2) in zip(heu[h][k], heu[h][k][1:]):
                mipdict[xname(k, h, Vidx[v1, a1], Vidx[v2, a2])] = 1 
                mipdict[qname(k, h, Vidx[v1, a1], Vidx[v2, a2])] = s2
                if v2 != 0:
                    mipdict[yname(k, h, a2, v2)] = 1
    
    return list(zip(*mipdict.items()))
                  
def Solve(problem, initialSolution, integer = True):

    problem.maxDays = len(initialSolution)

    return SolveCplexModel2(problem, initialSolution, useHeuristicCallback = True, integer = integer)


