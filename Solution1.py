import sys
import numpy as np
import math

mincost = math.inf


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def cost(dict, AddCost):
    list = dict.copy()
    for i in dict.keys():
        #print(i, list)
        if i == 'RU' or i == 'LU' or i == 'LD' or i == 'RD':
            if -100 not in list[i] and 0 not in list[i]:
                list[i].append(10+AddCost)
            else:
                list[i].append(math.inf)
        elif i == 'U' or i == 'D':
            if -100 not in list[i] and 0 not in list[i]:
                list[i].append(6+AddCost)
            else:
                list[i].append(math.inf)
        elif i == 'L' or i == 'R':
            if -100 not in list[i] and 0 not in list[i]:
                list[i].append(5+AddCost)
            else:
                list[i].append(math.inf)
    return list


def neighbor_dict(A, value, AddCost):
    i, j = np.argwhere(A == value)[0]
    neighbor_list = {'LU': [A[i-1, j-1]], 'U': [A[i-1, j]], 'RU': [A[i-1, j+1]], 'L': [A[i, j-1]], 'R': [A[i, j+1]], 'LD': [A[i+1, j-1]], 'D': [A[i+1, j]], 'RD': [A[i+1, j+1]]}
    neighbor_list = cost(neighbor_list, AddCost)
    dict = neighbor_list.copy()
    for key, value in neighbor_list.items():
        if math.inf in value or -1 in value:
            del dict[key]
    neighbor_list = dict.copy()
    return neighbor_list


def search(mask, cost, path):
    global mincost
    # i, j = np.argwhere(mask == -1)[0]
    neighbor = neighbor_dict(mask, -1, cost)
    print(neighbor)
    while(1):
        for key, value in neighbor.items():
            if -2 in value:
                mincost = neighbor[key][1]
                return
        for k in neighbor.keys():
            print(neighbor[k])
            dict = neighbor_dict(mask, (neighbor[k])[0], neighbor[k][1])
            neighbor[k] = dict
        print(neighbor)


def solve_task1(input_matrix):
    # Enter your code here.
    # Return the minimum cost or return No path found!
    A = np.matrix(input_matrix)
    A = np.pad(A, 1, pad_with, padder=0)
    mask = np.zeros(A.shape, dtype=np.int)
    mask[A == '_'] = 1
    num_traverse_cells = np.sum(mask)+1
    mask[mask == 1] = np.asarray(range(1, num_traverse_cells))
    mask[A == '*'] = -100
    mask[A == 'X'] = -2
    mask[A == 'R'] = -1
    search(mask, 0, [])
    return 1

# Use as many helper functions as you like


# Get input from command prompt and run the program
input_arg = sys.argv[1]


def run_program(file_name = input_arg):
    # Read the input matrix
    input_matrix = np.genfromtxt(file_name, dtype='str')
    
    # Your main function to solve the matrix
    print(solve_task1(input_matrix))


run_program()

# To test the result yourself,
# Open the command line tool, navigate to the folder and execute:
# python Solution1.py input_for_task1.txt


