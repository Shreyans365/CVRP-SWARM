def healing(chromosome, K):

    route_max_demand_index = findRouteWithMaxDemand(chromosome)
    route_min_demand_index = findRouteWithMinDemand(chromosome)

    if chromosome.routes[route_max_demand_index].demand > K :

        subroute = getSingularSubrouteFromRoute(chromosome, route_max_demand_index)
        chromosome.remove(subroute)

        return insertOptimally(chromosome, route_min_demand_index, subroute)

    return chromosome




    