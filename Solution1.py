import sys
import numpy as np
from collections import defaultdict
from heapq import *


class Graph(object):

    def __init__(self, input_matrix):
        A = np.matrix(input_matrix)
        A = np.pad(A, 1, pad_with, padder=0)
        mask = np.zeros(A.shape, dtype=np.int)
        mask[A == '_'] = 1
        num_traverse_cells = np.sum(mask) + 1
        mask[mask == 1] = np.asarray(range(1, num_traverse_cells))
        mask[A == '*'] = -100
        mask[A == 'X'] = -2
        mask[A == 'R'] = -1
        self.__graph_dict = generate_neighbor(mask)
        self.edges = []
        for vertex in self.__graph_dict:
            for neighbour in self.__graph_dict[vertex]:
                i, j = np.argwhere(mask == neighbour)[0]
                i1, j1 = np.argwhere(mask == vertex)[0]
                if i1 != i and j1 != j:
                    cost = 10
                elif i1 == i and j1 != j:
                    cost = 5
                elif i1 != i and j1 == j:
                    cost = 6
                else:
                    cost = 0
                if (vertex, neighbour, cost) not in self.edges:
                    self.edges.append((vertex, neighbour, cost))

    def dijkstra(self, start, goal):
        g = defaultdict(list)
        for vertex, neighbor, c in self.edges:
            g[vertex].append((c, neighbor))

        q, seen, mins = [(0, start, ())], set(), {start: 0}
        while q:
            (cost, v1, path) = heappop(q)
            if v1 not in seen:
                seen.add(v1)
                path = (v1, path)
                if v1 == goal:
                    return cost

                for c, v2 in g.get(v1, ()):
                    if v2 in seen:
                        continue
                    prev = mins.get(v2, None)
                    next = cost + c
                    if prev is None or next < prev:
                        mins[v2] = next
                        heappush(q, (next, v2, path))
        return "No path found!"


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def neighbor_dict(A, i, j):
    neighbor_list = [A[i, j - 1], A[i, j + 1],
                     A[i - 1, j - 1], A[i - 1, j], A[i - 1, j + 1],
                     A[i + 1, j - 1], A[i + 1, j], A[i + 1, j + 1]]
    list = neighbor_list.copy()
    for value in neighbor_list:
        if value == -100 or value == 0:
            list.remove(value)
    neighbor_list = list.copy()
    return neighbor_list


def optimize_dict(graph_dict):
    temp_dict = graph_dict.copy()
    for key, value in graph_dict.items():
        if not value:
            del temp_dict[key]
    graph_dict = temp_dict.copy()
    return graph_dict


def generate_neighbor(A):
    graph_dict = {}
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            # print(A[i][j])
            if A[i][j] != 0 and A[i][j] != -100:
                # print(neighbor_dict(A, i, j))
                graph_dict[A[i][j]] = neighbor_dict(A, i, j)
    graph_dict = optimize_dict(graph_dict)
    return graph_dict


def solve_task1(input_matrix):
    # Enter your code here.
    # Return the minimum cost or return No path found!
    graph = Graph(input_matrix)
    return graph.dijkstra(-1, -2)


# Use as many helper functions as you like


# Get input from command prompt and run the program
input_arg = sys.argv[1]


def run_program(file_name=input_arg):
    # Read the input matrix
    input_matrix = np.genfromtxt(file_name, dtype='str')

    # Your main function to solve the matrix
    print(solve_task1(input_matrix))


run_program()

# To test the result yourself,
# Open the command line tool, navigate to the folder and execute:
# python Solution1.py input_for_task1.txt
