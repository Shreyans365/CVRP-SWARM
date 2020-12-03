import os
import math
import random
import threading
from PIL import Image, ImageDraw

class Route(object):

    # Constructor for the Route Object
    def __init__(self, route=[], cost=0, validity=False, demand=0):
        self.validity = validity
        self.route = route
        self.cost = cost
        self.demand = demand


    # function to insert a subroute in a route at a particular index
    def insert_subroute(self, index, route):
        self.validity = False
        self.route = self.route[:index + 1] + route + self.route[index + 1:]

    # function to append a node at the end of a route
    def append_node(self, node):
        self.validity = False
        self.route = self.route[:-1] + [node] + [1]

    # function to remove a particular node from a route if it exists in it
    def remove_node(self, x):
        self.validity = False
        del self.route[self.route.index(x)]

    # function to map a Route object to a String object
    def __repr__(self):
        debug_str = ", cost = " + str(self.cost) + ", demand = " + str(self.demand)
        ret_str = "->".join(str(n) for n in self.route)
        print(self.demand)
        return ret_str + (debug_str if False else "")

class Solution(object):

    # Constructor for the Solution Object
    def __init__(self, routes=[], cost=0, validity=False, demand=0):
        self.validity = validity
        self.routes = routes
        self.cost = cost
        self.demand = demand
        self.penalty = 0


    # function to remove a node from the Solution
    def remove_node(self, x):
        for route in self.routes:
            if x in route.route:
                route.remove_node(x)
        self.validity = False

    # function to insert a subroute in a Solution
    def insert_subroute(self, route_id, route_index, route):
        self.routes[route_id].insert_subroute(route_index, route)
        self.validity = False

        
    # function to get a random subroute which exists in the Solution
    def random_subroute(self):
        route_index = random.randrange(0, len(self.routes))
        while len(self.routes[route_index].route) == 2:
            route_index = random.randrange(0, len(self.routes))
        start_index = random.randrange(1, len(self.routes[route_index].route))
        end_index = start_index
        while end_index == start_index:
            end_index = random.randrange(1, len(self.routes[route_index].route))
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        return self.routes[route_index].route[start_index:end_index]

    def hash(self):
        return hash("-".join([",".join(str(x) for x in x.route) for x in self.routes]))

    def __repr__(self):
        return "\n".join([str(route) for route in self.routes])

    # This function is used for facilitating heap comparisons.
    def __lt__(self, other):
        if len(self.routes) == len(other.routes):
            return True
        return len(self.routes) < len(other.routes)

class InputInfo(object):

    # This object is the root object... it contains all information about the current run
    def __init__(self, data_file, debug=False):
        self.read_data(data_file)
        self.make_dist_matrix()
        self.hub = 1
        self.debug = debug
        self.max_route_len = 10
        random.seed()

    # function to read data from the input file 
    def read_data(self, data_file):
        with open(data_file) as f:
            content = [line.rstrip("\n") for line in f.readlines()]
        self.dimension = int(content[0].split()[-1])
        self.capacity = int(content[1].split()[-1])

        self.demand = [-1 for _ in range(self.dimension + 1)]
        self.coords = [(-1, -1) for _ in range(self.dimension + 1)]

        for i in range(3, self.dimension + 3):
            nid, xc, yc = [float(x) for x in content[i].split()]
            self.coords[int(nid)] = (xc, yc)
        for i in range(self.dimension + 4, 2 * (self.dimension + 2)):
            nid, dem = [int(x) for x in content[i].split()]
            self.demand[nid] = dem

    # function to compute euclidean distance between two nodes
    def compute_dist(self, node1, node2):
        node1 = self.coords[node1]
        node2 = self.coords[node2]
        return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
        

    # function to construct the distance matrix
    def make_dist_matrix(self):
        self.dist = [list([-1 for _ in range(self.dimension + 1)]) \
                        for _ in range(self.dimension + 1)]
        for xi in range(self.dimension + 1):
            for yi in range(self.dimension + 1):
                self.dist[xi][yi] = self.compute_dist(xi, yi)

                
    # function to find the bounding box of a set of nodes
    def bounding_box(self, route):
        x_min = min(self.coords[node][0] for node in route)
        x_max = max(self.coords[node][0] for node in route)
        y_min = min(self.coords[node][1] for node in route)
        y_max = max(self.coords[node][1] for node in route)
        return x_min, x_max, y_min, y_max

        
    # function to make a Solution object from a list of Route Objects
    def make_solution(self, routes):
        cost = 0
        demand = 0
        validity = True
        visited = set()
        for route in routes:
            if not route.validity:
                validity = False
            for x in route.route:
                visited.add(x)
            cost += route.cost
            demand += route.demand
        if len(visited) != self.dimension:
            print("THERE WAS AN ERROR WITH ONE OF THE ROUTES !!!!")
            print(visited)
        sol = Solution(cost=cost, demand=demand, validity=validity, routes=routes)
        return sol


    # function to make Route object from a list of nodes
    def make_route(self, node_list):
        if node_list[0] != self.hub:
            return None
        cost = 0
        demand = 0
        validity = True
        for i in range(1, len(node_list)):
            node1, node2 = node_list[i - 1], node_list[i]
            cost += self.dist[node1][node2]
            demand += self.demand[node2]
        if demand > self.capacity:
            validity = False

        route = Route(cost=cost, demand=demand, validity=validity, route=node_list)
        return route
        
    # funtion to Generate a Random Solution... this is used to generate initial population
    def make_random_solution(self, greedy=False):
        unvisited = [i for i in range(2, self.dimension + 1)]
        random.shuffle(unvisited)
        routes = []
        curr_route = [1]
        route_demand = 0
        route_length = 0
        while unvisited:
            i = 0
            node = unvisited[i]
            if route_length <= self.max_route_len and route_demand + self.demand[node] <= self.capacity:
                curr_route += [node]
                route_length += 1
                route_demand += self.demand[node]
                del unvisited[i]
                continue
            curr_route += [1]
            routes += [self.make_route(curr_route)]
            curr_route = [1]
            route_demand = 0
            route_length = 0
        routes += [self.make_route(curr_route + [1])]
        return self.make_solution(routes)

    def set_variables(self, solution):
        solution.cost, solution.demand = 0, 0
        for j,route_obj in enumerate(solution.routes):
            route = route_obj.route
            solution.routes[j].demand, solution.routes[j].cost = 0, 0
            for i in range(1, len(route) - 1):
                solution.routes[j].demand += self.demand[route[i]]
            
            for i in range(0, len(route) -1):
                solution.routes[j].cost += self.dist[route[i]][route[i + 1]]

            solution.cost += solution.routes[j].cost
            solution.demand += solution.routes[j].demand
            if route_obj.demand > self.capacity:
                route_obj.validity = False
                solution.validity = False

    def enrich_route(self, route):
        savings = 1
        iters = 0
        while savings > 0:
            savings = 0
            if iters > 1000:
                return route
            for A_index in range(len(route) - 2):
                for C_index in range(len(route) - 2):
                    if C_index != A_index and C_index != A_index + 1 and C_index + 1 != A_index:
                        A = route[A_index]
                        B = route[A_index + 1]
                        D = route[C_index + 1]
                        C = route[C_index]
                        diff = self.dist[A][B] + self.dist[C][D] - self.dist[B][D] - self.dist[A][C]
                        if diff > savings:
                            savings = diff
                            Abest = A_index
                            Cbest = C_index
            if savings > 0:
                route[Abest+1], route[Cbest] = route[Cbest], route[Abest+1]
            iters += 1
        return route

    def enrich_solution(self, solution):
        new_routes = []
        for route in solution.routes:
            route = self.enrich_route(route.route)
            new_routes += [self.make_route(route)]
        return self.make_solution(new_routes)

