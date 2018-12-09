import sys
import numpy as np
import math

mincost = math.inf


class Graph(object):

    def __init__(self, graph_dict):
        self.__graph_dict = graph_dict

    def find_path(self, start_vertex, end_vertex, path=None):
        if path == None:
            path = []
        graph = self.__graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        if start_vertex not in graph:
            return None
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_path = self.find_path(vertex,
                                               end_vertex,
                                               path)
                if extended_path:
                    return extended_path
        return None

    def find_all_paths(self, start_vertex, end_vertex, path=[]):
        graph = self.__graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]
        if start_vertex not in graph:
            return []
        paths = []
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_paths = self.find_all_paths(vertex,
                                                     end_vertex,
                                                     path)
                for p in extended_paths:
                    paths.append(p)
        return paths


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
    graph = Graph(graph_dict)
    print(graph.find_path(-1, -2, []))
    print(graph.find_all_paths(-1, -2, []))
    print("I'm done")


def solve_task1(input_matrix):
    # Enter your code here.
    # Return the minimum cost or return No path found!
    A = np.matrix(input_matrix)
    A = np.pad(A, 1, pad_with, padder=0)
    mask = np.zeros(A.shape, dtype=np.int)
    mask[A == '_'] = 1
    num_traverse_cells = np.sum(mask) + 1
    mask[mask == 1] = np.asarray(range(1, num_traverse_cells))
    mask[A == '*'] = -100
    mask[A == 'X'] = -2
    mask[A == 'R'] = -1
    generate_neighbor(mask)
    return 1


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
