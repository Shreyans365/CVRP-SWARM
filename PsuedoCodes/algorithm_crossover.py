def crossover(H, P, GL, index_par1, index_par2):
        
        if randNum(0, 1) < 0.2:
            # choose a weaker individual as second parent
            index_par2 = randNum(10, P-1)
        

        par1 = H[index_par1], par2 = H[index_par2]
        par1_chromosome = par1[1], par2_chromosome = par2[1]

        # use guided crossover to produce initial child
        child_chromosome = guidedCrossover(par1_chromosome, par2_chromosome)
        child_cost = child_chromsome.child_cost

        if randNum(0, 1) < GL:
            # use exhaustive crossover to improve child
            for iter in range(3):
                child_improved_chromosome = exhaustiveCrossover(par1_chromosome, child_chromosome)
                if child_improved_chromosome.cost < child_chromosome.cost:
                    child_chromosome = child_improved_chromosome
                    child_cost = child_improved_chromosome.cost
        else:
            # use guided crossover to improve child
            for iter in range(3):
                child_improved_chromosome = guidedCrossover(par1_chromosome, child_chromosome)
                if child_improved_chromosome.cost < child_chromosome.cost:
                    child_chromosome = child_improved_chromosome
                    child_cost = child_improved_chromosome.cost

        return child_chromosome



def guidedCrossover(par1_chromosome, par2_chromosome):
    # copy par1 into child and pop a random subroute
    child_chromosome = par1_chromosome
    subroute = randomSubroute(par2_chromosome)
    child_chromsome.remove(subroute)

    # use reactangular overlap heuristic to find best route for subroute insertion
    route_list = [ (-rectangularOverlap(route, subroute), route.demand, i) for i, route in enumerate(child_chromosome.routes) ]
    heapify(route_list)

    best_route_index = route_list.top()[2]

    # return the child after inserting subroute optimally
    return insertOptimally(child_chromosome, best_route_index, subroute)




def exhaustiveCrossover(par1_chromsome, par2_chromosome):
    # copy par1 into child and pop a random subroute
    child_chromosome = par1_chromosome
    subroute = randomSubroute(par2_chromosome)
    child_chromsome.remove(subroute)

    child_to_return = None
    cost = INFINITY

    # try inserting subroute into every route to find best route for subroute insertion
    for i, route in enumerate(child_chromosome.routes):
        child = insertOptimally(child_chromosome, i, subroute)
        if child.cost < cost:
            child_to_return = child
            cost = child.cost

    # return the newly created child
    return child_to_return



    





