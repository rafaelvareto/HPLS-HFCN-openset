import os
import pickle
import numpy as np

"""
Reading from string file and saving into python binary
"""
for file_name in os.listdir('.'):
    if file_name.endswith('.txt'):
        print file_name
        file_data = open(file_name, 'r')
        lines = file_data.readlines()

        matrix_z = [] # image location
        matrix_y = [] # image label
        matrix_x = [] # image feature

        for line in lines:
            line = line.replace('[', '')
            line = line.replace(']', '')
            line = line.replace(',', '')

            split = line.split(' ')
            matrix_z.append(split[0])
            matrix_y.append(split[1])

            array_x = []
            for index in range(2, len(split)):
                value = float(split[index])
                array_x.append(value)
            matrix_x.append(array_x)

        assert len(matrix_x) == len(matrix_y) == len(matrix_z)
        out_matrix = [matrix_z, matrix_y, matrix_x]
        with open(file_name.replace('.txt', '.bin'), 'wb') as out_file:
            pickle.dump(out_matrix, out_file, protocol=pickle.HIGHEST_PROTOCOL)

        del matrix_x[:]
        del matrix_y[:]
        del matrix_z[:]
        del out_matrix[:]


"""
Reading from binary into python code
"""
# file_name = 'FRGC-SET-1-DEEP-FEATURE-VECTORS.bin'
# with open(file_name, 'rb') as infile:
#     matrix_z, matrix_y, matrix_x = pickle.load(infile)
