from queue import PriorityQueue

from data.Constants import COMPONENT_IDS
from data.datamodel.Grammar import UAVGrammar


class Graph:
    """ Graph class

    Author: Scott Robinson et. al
    Source:
    https://stackabuse.com/courses/graphs-in-python-theory-and-implementation/lessons/dijkstras-algorithm/

    """
    def __init__(self, num_of_vertices):
        self.v = num_of_vertices
        self.edges = [[-1 for i in range(num_of_vertices)] for j in range(num_of_vertices)]
        self.visited = []

    def add_edge(self, u, v, weight):
        self.edges[u][v] = weight
        self.edges[v][u] = weight


class UAVGraph(Graph):
    """
    Extension of the graph class to construct UAV graphs

    """

    def __init__(self, uav_str: str):
        parser = UAVGrammar()
        components, connections, _, _ = parser.parse(uav_str)
        num_of_vertices = len(components)
        super(UAVGraph, self).__init__(num_of_vertices)

        for c in connections:
            c_orig = COMPONENT_IDS.index(c[0])
            c_dest = COMPONENT_IDS.index(c[1])
            self.add_edge(c_orig, c_dest, 1)


def dijkstra(graph, start_vertex):
    """
    Implementation of Dijkstra's algorithm with time complexity O(|E|+|V|log|V|) according to the authors

    Author: Scott Robinson et. al
    Source:
    https://stackabuse.com/courses/graphs-in-python-theory-and-implementation/lessons/dijkstras-algorithm/

    Adapted to encode infinite distances in output as -1

    """
    D = {v: float('inf') for v in range(graph.v)}
    D[start_vertex] = 0

    pq = PriorityQueue()
    pq.put((0, start_vertex))

    while not pq.empty():
        (dist, current_vertex) = pq.get()
        graph.visited.append(current_vertex)

        for neighbor in range(graph.v):
            if graph.edges[current_vertex][neighbor] != -1:
                distance = graph.edges[current_vertex][neighbor]
                if neighbor not in graph.visited:
                    old_cost = D[neighbor]
                    new_cost = D[current_vertex] + distance
                    if new_cost < old_cost:
                        pq.put((new_cost, neighbor))
                        D[neighbor] = new_cost

    for v, d in D.items():
        if d == float('inf'):
            D[v] = -1
    return D
