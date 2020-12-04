def fitness(K, D, I, TI, population, chromosome):

    capacity_overflow = 0
    for route in chromosome.routes:
        capacity_overflow += max(0, route.demand - K)**2

    flynn_adjustment = (I*(I+1)) / TI

    peer_reliability = bestSolution(population).cost

    p_const = (D/2)**2

    penalty = p_const * (peer_reliability * capacity_overflow * flynn_adjustment)

    return 1/(chromosome.cost + penalty)



