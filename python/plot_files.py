import os
import pickle

from auxiliar import generate_precision_recall, plot_precision_recall
from auxiliar import generate_roc_curve, plot_roc_curve

path_files = os.listdir('./files/*.file')
print path_files

for file in path_files:
    if file.endswith('.file'):
        file_path = './files/' + file
        
        with open(file_path) as infile:
            file_prs, file_rocs = pickle.load(infile)

        plot_precision_recall(file_prs, file_path.replace('.file','.jpg'))
        plot_roc_curve(file_rocs, file_path.replace('.file','.jpg'))