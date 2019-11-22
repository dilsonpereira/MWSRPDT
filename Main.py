import Model2

import time
import random
from MPVRPIR import *
import Constructive
import ACO.Main

def SolveProblem(problem, constructive = True, aco = True, Model = False):
    nt = len([(v,i) for v in range(1, problem.numVertices) for i in range(problem.numTasks[problem.request[v]])])

    print('num tasks', nt)
    if constructive:
        ts = time.time()
        sol = Constructive.Constructive(problem)
        te = time.time()
        print('Constructive heuristic')
        print('  Solution:', len(sol), 'days')
        print('  Time:', te-ts)

    if aco:
        ts = time.time()
        sol = ACO.Main.Solve(problem)
        te = time.time()
        print('Ant Colony Algorithm')
        print('  Solution:', len(sol), 'days')
        print('  Time:', te-ts)

    if Model:
        print('Model')


        info, _ = Model2.Solve(problem, sol, integer=False)
        lbroot = info['lb']
        
        ts = time.time()
        info, sol = Model2.Solve(problem, sol, integer=True)
        te = time.time()


        print('  Upper bound:', info['ub'])
        if info['status'] == 'MIP_optimal':
            print('  Lower bound:', info['ub'])
        else:
            print('  Lower bound:', info['lb'])

        print('  Root lower bound:', lbroot)
        print('  Root lower bound (with cplex cuts):', info['lbroot'])
        print('  Number of nodes:', info['numnodes'])
        print('  Time', te-ts)

def GeneralExperiments():
    for type in ['A', 'B', 'C']:
        #for n in [20, 25, 30, 35]:
        for n in [40, 50, 60, 70, 80, 90, 100]:
            print('N', n)
            for i in range(10):
                print('Instance', i)
                random.seed(i)

                if type == 'C':
                    instanceArgs = {'numVertices':n, 'numServices':1, 'numTeams':3, 'numTasks':[3], 'type':type}
                else:
                    instanceArgs = {'numVertices':n, 'numServices':3, 'numTeams':3, 'numTasks':[1,3,5], 'type':type}

                # Generate a random instance
                instance = MPVRPIR(**instanceArgs) 
                #instance.SaveToFile('{}_{}_{}.txt'.format(n, type, i))

                # Load an instance from a file
                #instance = MPVRPIR() # Create an empty instance
                #instance.ReadInstance('{}_{}_{}.txt'.format(n, type, i))

                SolveProblem(problem = instance, constructive = True, aco = True, Model = True)

def OneInstance(type, n, i):
        print('Instance', i)
        random.seed(i)

        if type == 'C':
            instanceArgs = {'numVertices':n, 'numServices':1, 'numTeams':3, 'numTasks':[3], 'type':type}
        else:
            instanceArgs = {'numVertices':n, 'numServices':3, 'numTeams':3, 'numTasks':[1,3,5], 'type':type}

        instance = MPVRPIR(**instanceArgs) 
        SolveProblem(problem = instance, constructive = True, aco = True, Model = True)


if __name__ == '__main__':
    import sys
    OneInstance(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

    #GeneralExperiments()   

    #instanceArgs = {'numVertices':10, 'numServices':3, 'numTeams':3, 'numTasks':[1,3,5], 'type':'A'}
    #instance = MPVRPIR(**instanceArgs) 
    #SolveProblem(problem = instance, constructive = True, aco = False, Model = False)


