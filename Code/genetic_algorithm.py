from input_info import InputInfo
import random
import copy
import threading

from heapq import *

class Population(object):
    # constructor for creating Population Object
    def __init__(self, info, total_iters):
        self.info = info
        self.info.max_route_len = 10
        self.chromosomes = []
        for x in [self.info.enrich_solution(self.info.make_random_solution(greedy=False)) for _ in range(800)]:
            heappush(self.chromosomes, (x.cost, x))
        self.leader = self.chromosomes[0][1]
        self.iters = 0
        self.total_iters = total_iters
        self.prob_unbiased_mutation = 0.25
        random.seed()

    # single step for the genetic algorithm
    def step(self):
        index_chromsome_to_replace = -1
        for i in range(12):
            for j in range(i + 1, 12):
                par1, par2 = self.chromosomes[i][1], self.chromosomes[j][1]
                if random.uniform(0, 1) < 0.2:
                    par2 = self.chromosomes[random.randrange(10, len(self.chromosomes) - 1)][1]
                offspring = self.guided_crossover(par1, par2)
                if random.uniform(0, 1) < 0.95:
                    for _ in range(3):
                        c = self.guided_crossover(par1, offspring)
                        self.info.set_variables(c)
                        if c.cost < offspring.cost:
                            offspring = c
                else:
                    for _ in range(3):
                        c = self.exhaustive_crossover(par1, offspring)
                        self.info.set_variables(c)
                        if c.cost < offspring.cost:
                            offspring = c
                self.info.set_variables(offspring)
                self.mutation(offspring)
                self.info.set_variables(offspring)
                self.heal(offspring)
                self.info.set_variables(offspring)
                self.info.enrich_solution(offspring)
                self.info.set_variables(offspring)
                self.chromosomes[index_chromsome_to_replace] = (self.fitness(offspring), offspring)
                index_chromsome_to_replace -= 1
        heapify(self.chromosomes)
        self.iters += 1
        if self.chromosomes[0][1].cost < self.leader.cost:
            self.leader = self.chromosomes[0][1]
        return self.leader

    # calculate fitness of a chromosome
    def fitness(self, chromosome):
        penalty = self.penalty(chromosome)
        return chromosome.cost + penalty

    # calculate penalty for a chromosome
    def penalty(self, chromosome):
        penalty_sum = 0
        for route in chromosome.routes:
            penalty_sum += max(0, route.demand - self.info.capacity)**2
        mnv = sum(self.info.demand[i] for i in range(self.info.dimension)) / self.info.capacity
        alpha = self.leader.cost / ((1 / (self.iters + 1)) * (self.info.capacity * mnv / 2)**2 + 0.00001)
        penalty = alpha * penalty_sum * self.iters / self.total_iters
        chromosome.penalty = penalty
        return penalty

    # returns true when a healing was needed, false otherwise (heals if needed)
    def heal(self, chromosome):
        routes = chromosome.routes
        route_max_index = max((i for i in range(len(routes))), key = lambda i: routes[i].demand)
        route_min_index = min((i for i in range(len(routes))), key = lambda i: routes[i].demand)
        if routes[route_max_index].demand > self.info.capacity:
            rint = random.randrange(1, len(routes[route_max_index].route) - 1)
            routes[route_min_index].append_node(routes[route_max_index].route[rint])
            routes[route_max_index].remove_node(routes[route_max_index].route[rint])
            return True
        return False

    def exhaustive_crossover(self, chrom1, chrom2):
        child = copy.deepcopy(chrom1)
        sub_route = chrom2.random_subroute()
        for x in sub_route:
            child.remove_node(x)
        route_index, n_id = self.insert_subroute_optimally_chromosome(child, sub_route)
        child.insert_subroute(route_index, n_id, sub_route)
        return child

    def guided_crossover(self, c1, c2):
        child = copy.deepcopy(c1)
        sub_route = c2.random_subroute()
        routes = []
        for x in sub_route:
            child.remove_node(x)
        for i, route in enumerate(child.routes):
            x0_min, x0_max, y0_min, y0_max = self.info.bounding_box(route.route)
            x1_min, x1_max, y1_min, y1_max = self.info.bounding_box(sub_route)
            x_overlap = max(0, min(x0_max, x1_max) - max(x0_min, x1_min))
            y_overlap = max(0, min(y0_max, y1_max) - max(y0_min, y1_min))
            heappush(routes, (x_overlap * y_overlap, i))
        top6 = nlargest(6, routes)
        min_i = min((i[1] for i in top6), key = lambda x: child.routes[x].demand)
        _, optimal = self.insert_subroute_optimally_route(sub_route, child.routes[min_i].route)
        child.insert_subroute(min_i, optimal, sub_route)
        return child

    

    # function for unbiased mutation... (insert node optimally in the same route from which it was taken)
    def unbiased_mutation(self, chromosome, route_chosen, node):
        _, optimal = self.insert_subroute_optimally_route([node], chromosome.routes[route_chosen].route)
        index_optimal_route = (route_chosen, optimal)
        return index_optimal_route 

    # function for biased mutation... (deliberately insert node optimally in other route)
    def biased_mutation(self, chromosome, route_chosen, node):
        route_except_chosen = route_chosen
        while route_chosen == route_except_chosen:
            route_except_chosen = random.randrange(0, len(chromosome.routes))
        _, optimal = self.insert_subroute_optimally_route([node], chromosome.routes[route_except_chosen].route)
        index_optimal_route = (route_except_chosen, optimal)
        return index_optimal_route

    # function to carry out mutation
    def mutation(self, chromosome):
        route_chosen = random.randrange(0, len(chromosome.routes))
        while(len(chromosome.routes[route_chosen].route) == 2):
            route_chosen = random.randrange(0, len(chromosome.routes))
        c_i = random.randrange(1, len(chromosome.routes[route_chosen].route) - 1)
        node = chromosome.routes[route_chosen].route[c_i]
        chromosome.remove_node(node)
        if random.uniform(0, 1) < self.prob_unbiased_mutation:
            index_optimal_route = self.unbiased_mutation(chromosome, route_chosen, node)
        else:
            index_optimal_route = self.biased_mutation(chromosome, route_chosen, node)
        chromosome.insert_subroute(index_optimal_route[0], index_optimal_route[1], [node])

    # finds the index where the route is optimal inserted
    def insert_subroute_optimally_route(self, sub_route, route):
        start = sub_route[0]
        end = sub_route[-1]
        optimal_savings, index_optimal_route = 0, 0
        dist = self.info.dist
        i = 0
        for i in range(0, len(route) - 1):
            init_cost = dist[route[i]][route[i + 1]]
            savings = init_cost - dist[route[i]][start] - dist[end][route[i + 1]]
            if savings > optimal_savings:
                optimal_savings, index_optimal_route = savings, i
        return optimal_savings, i

    # finds the optimal route index, and node index where the route should go
    def insert_subroute_optimally_chromosome(self, child, sub_route):
        optimal_savings, optimal_route_index, optimal_node_index = -1, 0, 0
        for route_index, route in enumerate(child.routes):
            route = route.route
            subopt_optimal, n_id = self.insert_subroute_optimally_route(sub_route, route)
            if subopt_optimal > optimal_savings:
                optimal_savings, optimal_route_index, optimal_node_index = subopt_optimal, route_index, n_id
        return optimal_route_index, optimal_node_index

class Algorithm(object):
    def __init__(self, info):
        self.info = info
        self.leader = None


class GeneticAlgorithm(Algorithm):
    def __init__(self, info, num_populations, total_iters):
        super(GeneticAlgorithm, self).__init__(info)

        self.populations = [Population(self.info, total_iters) for _ in range(num_populations)]
        self.pop_optimals = [0 for _ in range(num_populations)]
    def step(self):
        for i, pop in enumerate(self.populations):
            self.pop_optimals[i] = pop.step()
        self.leader = min(self.pop_optimals, key = lambda x: x.cost)
        return self.leader

