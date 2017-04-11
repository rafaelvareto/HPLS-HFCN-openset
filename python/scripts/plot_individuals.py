import matplotlib
import numpy as np
import os
import pickle

matplotlib.use('Agg')

from matplotlib import pyplot as plt

def main():
    """
    Fixing number of hashing functions and varying known_set_size
    """
    descriptor = 'hog'
    hashing_functions = ['100', '200', '300', '400']
    enclosed_folder = '../files/'
    path_files = os.listdir(enclosed_folder)

    # Each iteration plots a chart
    for function in hashing_functions:
        prs_dict = {}
        rocs_dict = {}
        files_list = []
        for file in path_files:
            if file.endswith('.file'):
                if descriptor in file and function in file:
                    files_list.append(file)
                    print(file)
                    with open(enclosed_folder + file) as infile:
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
            print('Plotting %s' % key)
            x_values = [value for value in range(10, 91, 40)]
            e_axis = [item[1] for item in auc_dict[key]] 
            y_axis = [item[0] for item in auc_dict[key]]
            x_axis = x_values[0:len(y_axis)]
            
            print(len(x_axis), len(y_axis), len(e_axis))
            assert len(x_axis) == len(y_axis) == len(e_axis)
            plt.errorbar(x_axis, y_axis, e_axis, label=key, linestyle='-', marker='o')

        plt.legend(loc='lower left')
        plt.title('Variable known individuals: %s %s models' % (descriptor.upper(), function))
        plt.xlabel('#Known Individuals')
        plt.ylabel('AUC')
        plt.xlim([0.0, 100])
        plt.ylim([0.7, 1.05])
        plt.grid()
        # plt.show()
        plt.savefig('./individuals' + '_' + descriptor + '_' + function + 
            '_' + '_'.join(auc_dict.keys()).replace('label_','') + '.png')
        
if __name__ == "__main__":
    main()