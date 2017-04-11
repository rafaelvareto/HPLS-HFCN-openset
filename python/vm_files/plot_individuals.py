import matplotlib
import numpy as np
import os
import pickle

matplotlib.use('Agg')

from matplotlib import pyplot as plt

def main():
    """
    Fixing known_set_size and varying number of hashing functions
    Fixing number of hashing functions and varying known_set_size
    """
    substring = '_100_'
    path_files = os.listdir('.')

    prs_dict = {}
    rocs_dict = {}
    files_list = []
    for file in path_files:
        if file.endswith('.file'):
            if substring in file:
                files_list.append(file)
                print(file)
                with open(file) as infile:
                    file_prs, file_rocs = pickle.load(infile)
                    prs_dict[file] = file_prs
                    rocs_dict[file] = file_rocs

    instance_dict = {}
    for item in rocs_dict.iteritems():
        try:
            instance_dict[item[0][5:16]].append(item[1])
        except KeyError:
            instance_dict[item[0][5:16]] = [item[1]]
    # print(instance_dict.keys())

    auc_dict = {}
    for key in instance_dict.keys():
        group_list = instance_dict[key]
        for outer in group_list:
            auc_list = []
            for inner in outer:
                auc_list.append(inner['auc'])
            try:
                auc_dict[key].append((np.mean(auc_list), np.std(auc_list)))
            except KeyError:
                auc_dict[key] = [(np.mean(auc_list), np.std(auc_list))]
    # print(auc_dict)

    plt.clf()
    for key in auc_dict.keys():
        x_axis = [value for value in range(100, 401, 100)]
        y_axis = [item[0] for item in auc_dict[key]]
        e_axis = [item[1] for item in auc_dict[key]] 
        plt.errorbar(x_axis, y_axis, e_axis, label=key,linestyle='-', marker='o')

    plt.legend(loc='lower left')
    plt.title('Variable hashing functions')
    plt.xlabel('#Known subjects')
    plt.ylabel('AUC')
    plt.xlim([50.0, 450.0])
    plt.ylim([0.7, 1.05])
    plt.grid()
    # plt.show()
    plt.savefig('./individuals' + substring + '_'.join(auc_dict.keys()) + '.png')
        
if __name__ == "__main__":
    main()