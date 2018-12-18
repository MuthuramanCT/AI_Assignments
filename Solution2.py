import sys
import numpy as np


class Graph(object):

    def __init__(self, input_matrix):
        A = np.matrix(input_matrix)
        A = np.pad(A, 1, pad_with, padder=0)
        self.mask = np.zeros(A.shape, dtype=np.int)
        self.mask[A == '.'] = 1
        num_traverse_cells = np.sum(self.mask) + 1
        self.mask[self.mask == 1] = np.asarray(range(1, num_traverse_cells))
        self.mask[A == 'X'] = 0
        self.__graph_dict = generate_neighbor(self.mask)
        self.__value_on_edge = self.value_on_edge()
        #print(self.mask)

    def value_on_edge(self):
        L1 = list(self.mask[1, :])
        L2 = list(self.mask[:, self.mask.shape[1]-2])
        L3 = list(self.mask[self.mask.shape[0]-2, :])
        L4 = list(self.mask[:, 1])
        L = L1+L2+L3+L4
        temp = L.copy()
        for value in L:
            if value == 0:
                temp.remove(value)
        L = temp.copy()
        return L

    def dfs(self, start, visited=None):
        graph = self.__graph_dict
        if visited is None:
            visited = []
        visited.append(start)
        iterator = [n for n in graph[start] if n not in visited]
        for next in iterator:
            if next not in visited:
                self.dfs(next, visited)
        return visited

    def generate_output(self):
        for value in self.__value_on_edge:
            path = self.dfs(value)
            #print(path)
            for item in path:
                m, n = np.argwhere(self.mask == item)[0]
                if item not in self.__value_on_edge:
                    self.mask[m, n] = -100
            self.__graph_dict = generate_neighbor(self.mask)
        for value in self.__value_on_edge:
            self.mask[self.mask == value] = -100
        output = np.empty(self.mask.shape, dtype=str)
        output[self.mask >= 0] = 'X'
        output[self.mask == -100] = '.'
        output = output[1:self.mask.shape[0]-1,1:self.mask.shape[1]-1]
        return output


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def neighbor_dict(A, i, j):
    neighbor_list = [A[i, j - 1], A[i, j + 1],
                     A[i - 1, j],A[i + 1, j]]
    if all([i == 0 for i in neighbor_list]):
        A[i, j] = 0
    list = neighbor_list.copy()
    for value in neighbor_list:
        if value == 0 or value == -100:
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
    #print(graph_dict)
    return graph_dict


def solve_task2(input_matrix):
    # Enter your code here.
    A = np.matrix(input_matrix)
    graph = Graph(input_matrix)
    A = graph.generate_output()
    #print(graph.mask)
    return A

# Use as many helper functions as you like


# Get input from command prompt and run the program
input_arg = sys.argv[1]


def run_program(filename = input_arg):
    # Read the input matrix
    input_matrix = np.genfromtxt(filename, dtype='str')

    # Your main function to solve the matrix
    output = solve_task2(input_matrix)

    # print the matrix to a txt file
    np.savetxt('output_for_task2.txt', output, fmt="%s")


run_program()


# To test the result yourself,
# Open the command line tool, navigate to the folder and execute:
# python Solution2.py input_for_task2.txt
