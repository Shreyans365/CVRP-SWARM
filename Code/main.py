from input_info import InputInfo
from input_info import Route
from input_info import Solution
from genetic_algorithm import GeneticAlgorithm
import os
import time
import signal
import sys
import math
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy


capacity = -1
ci_obj = None

def mapSolution(sol):
    coords = ci_obj.coords[1:]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1,0.1,0.75,0.75])
    plt.scatter([x[0] for x in coords], [x[1] for x in coords], color='black', s=50, zorder=2)
    plt.scatter([coords[0][0]], [coords[0][1]], s=200, color='r', zorder=2)
    patches = []
    for i, route in enumerate(sol.routes):
        color = numpy.random.random(3)
        r_ = route.route
        patch_ = mpatches.Patch(color=color, label=("Vehicle "+str(i+1)))
        patches.append(patch_)
        for i in range(0,len(r_)-1):
            x0 = ci_obj.coords[r_[i]][0]
            y0 = ci_obj.coords[r_[i]][1]
            x1 = ci_obj.coords[r_[i+1]][0]
            y1 = ci_obj.coords[r_[i+1]][1]
            plt.plot([x0, x1], [y0, y1], c=color, linewidth=1, zorder=1)

    ax.grid()
    plt.show()

def outputSolution(sol):
    ci_obj.set_variables(sol)
    final_routes = []

    repaired_solution = Solution()

    for route in sol.routes:
        if len(route.route) <= 2:
            continue 

        if route.demand <= capacity:
            repaired_solution.routes.append(route)
            continue

        routes_sep = []
        dem_till_now = 0
        curr_route = []
        for i in range(1,len(route.route)-1):
            if dem_till_now+ci_obj.demand[route.route[i]] > capacity:
                routes_sep.append(curr_route)
                curr_route = [route.route[i]]
                dem_till_now = ci_obj.demand[route.route[i]]
            else:
                curr_route.append(route.route[i])
                dem_till_now += ci_obj.demand[route.route[i]]

        routes_sep.append(curr_route)
        
        for r in routes_sep:
            r_this = [1] + r + [1]
            new_route = Route(route=r_this)
            repaired_solution.routes.append(new_route)

    ci_obj.set_variables(repaired_solution)
    total_cost = 0
    os.system('clear')
    print(" ")
    table = []
    for i, route in enumerate(repaired_solution.routes):
        route_str = " -> ".join([str(x) for x in route.route])
        cost_this = route.cost
        total_cost += cost_this
        table.append(["Vehicle "+str(i+1), "Demand: "+str(route.demand), "Cost: "+str(round(route.cost,2)), route_str])

    print(tabulate(table, tablefmt="pretty"))
    print(" ")
    print("Total cost is: " + str(int(math.ceil(total_cost))))
    print(" ")
    mapSolution(repaired_solution)
    return


class MainRunner(object):

    def __init__(self, algorithm,  iterations):
        self.algorithm = algorithm
        self.print_cycle = 10
        self.num_iter = iterations
        self.iter = 0

    def run(self):
        self.start_time = time.time()
        curr_best = -1
        while self.iter < self.num_iter:
            best = self.algorithm.step()
            self.best = best
            if self.iter == 0 :
                curr_best = best.cost
            if self.iter % self.print_cycle == 0:
                os.system('clear')
                print("Generation: {0} Leader_Penalty:{1}".format(self.iter, self.best.cost))
            if self.iter%100 == 0:
                if best.cost == curr_best and self.iter != 0:
                    break
                else:
                    curr_best = best.cost
            self.iter += 1
            if time.time() - self.start_time > 1800:
                break
  
        outputSolution(best)
        

if __name__ == "__main__":
    arg_list = sys.argv
    print(len(sys.argv))
    nodes = arg_list[1]
    ci_obj = InputInfo("inputs/"+nodes+"nodes.txt", debug=True)
    capacity = ci_obj.capacity
    cvrp = MainRunner(GeneticAlgorithm(ci_obj, 1, 200000), 200000)
    cvrp.run()

