import matplotlib
import numpy as np
import os
import pickle

matplotlib.use('Agg')

from matplotlib import pyplot

# Fixing known set size and varuying number of models
# def main():
groups = ['set_1', 'set_2', 'set_4']
substring = '_0.1_'
path_files = os.listdir('.')

dict_prs = {}
dict_rocs = {}
list_files = []
for file in path_files:
    if file.endswith('.file'):
        if substring in file:
            files.append(file)
            with open(file) as infile:
                file_prs, file_rocs = pickle.load(infile)
                dict_prs[file] = file_prs
                dict_rocs[file] = file_rocs

    

    


# if __name__ == "__main__":
#     main()