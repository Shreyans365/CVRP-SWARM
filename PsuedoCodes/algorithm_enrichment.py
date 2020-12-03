def enrichment(chromosome):
    # enrich a chromosome by improving each route iteratively
    for route in chromosome.routes:
        for iter in range(1000):
            for i in range(1, len(route)-2):
                for j in range(i+2, len(route)-2):

                    distance = dist(route[i], route[i+1]) + dist(route[j], route[j+1])
                    distance_ = dist(route[i], route[j]) + dist(route[i+1], route[j+1])

                    if distance_ < distance :
                        route[i+1], route[j] = route[j], route[i+1]
                        route = reverse(route, i+2, j-1)

    return chromosome


    