import matplotlib
import numpy as np
import os
import pickle

matplotlib.use('Agg')

from itertools import cycle
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

def plot_precision_recall(prs, extra_name=None):
    # Setup plot details
    color_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    color_names = [name for name, color in color_dict.items()]
    colors = cycle(color_names)
    lw = 2

    # Plot Precision-Recall curve
    plt.clf()
    for index, color in zip(range(len(prs)), colors):
        pr = prs[index]
        plt.plot(pr['recall'], pr['precision'], lw=lw, color=color, label='PR curve %d (area = %0.2f)' % (index+1, pr['avg_precision']))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid()
    if extra_name == None:
        plt.show()
    else:
        plt.savefig('./precision_recall_' + extra_name + '.png')


def plot_roc_curve(rocs, extra_name=None):
    # Setup plot details
    color_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    color_names = [name for name, color in color_dict.items()]
    colors = cycle(color_names)
    lw = 2

    # Plot Receiver Operating Characteristic curve
    plt.clf()
    for index, color in zip(range(len(rocs)), colors):
        roc = rocs[index]
        plt.plot(roc['fpr'], roc['tpr'], color=color, lw=lw, label='ROC curve %d (area = %0.2f)' % (index+1, roc['auc']))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    if extra_name == None:
        plt.show()
    else:
        plt.savefig('./roc_curve_' + extra_name + '.png')

def main():
    path_files = os.listdir('.')

    for file in path_files:
        if file.endswith('.file'):
            with open(file) as infile:
                file_prs, file_rocs = pickle.load(infile)

            plot_precision_recall(file_prs, file.replace('.file',''))
            plot_roc_curve(file_rocs, file.replace('.file',''))

if __name__ == "__main__":
    main()