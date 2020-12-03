def mutation(chromsome, BM):

    if randNum(0, 1) < BM:
        return biasedMutation(chromosome)
    
    else:
        return unbiasedMutation(chromosome)


def unbiasedMutation(chromosome):
    # shift a single node within it's route to improve overall cost
    i, subroute = randomSingularSubroute(chromosome)
    chromosome.remove(subroute)
    return insertOptimally(chromosome, i, subroute)

def biasedMuation(chromsome):
    # shift a single node between routes to improve overall cost
    i, subroute = randomSingularSubroute(chromosome)
    chromosome.remove(subroute)

    chromosome_to_return = None
    cost = INFINTY

    for i, route in enumerate(chromosome.routes):
        chromosome_mutated = insertOptimally(chromosome, i, subroute)

        if chromosome_mutated.cost < cost:
            chromosome_to_return = chromosome_mutated
            cost = chromosome_mutated.cost

    return chromosome_to_return








