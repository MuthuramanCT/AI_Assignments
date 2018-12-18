import sys
import numpy as np


class Graph(object):

    def __init__(self, input_matrix):
        A = np.matrix(input_matrix)
        A = np.pad(A, 1, pad_with, padder=0)
        mask = np.zeros(A.shape, dtype=np.int)
        mask[A == '.'] = 1
        num_traverse_cells = np.sum(mask) + 1
        mask[mask == 1] = np.asarray(range(1, num_traverse_cells))
        mask[A == 'X'] = 0
        print(mask)
        self.__graph_dict = generate_neighbor(mask)
        print(mask)
        print(self.__graph_dict)


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def neighbor_dict(A, i, j):
    neighbor_list = [A[i, j - 1], A[i, j + 1],
                     A[i - 1, j - 1], A[i - 1, j], A[i - 1, j + 1],
                     A[i + 1, j - 1], A[i + 1, j], A[i + 1, j + 1]]
    if all([i == 0 for i in neighbor_list]):
        A[i, j] = 0
    return neighbor_list


def generate_neighbor(A):
    graph_dict = {}
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            # print(A[i][j])
            if A[i][j] != 0 and A[i][j] != -1:
                # print(neighbor_dict(A, i, j))
                graph_dict[A[i][j]] = neighbor_dict(A, i, j)
    #graph_dict = optimize_dict(graph_dict)
    return graph_dict


def solve_task2(input_matrix):
    # Enter your code here.
    A = np.matrix(input_matrix)
    graph = Graph(input_matrix)
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
