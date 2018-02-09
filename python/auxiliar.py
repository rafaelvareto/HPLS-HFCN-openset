import cv2 as cv
import matplotlib
import numpy as np
import random

matplotlib.use('Agg')

from itertools import cycle
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from pls_classifier import PLSClassifier
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import detection_error_tradeoff
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from sklearn.svm import SVR


def load_txt_file(file_name):
    this_file = open(file_name, 'r')
    this_list = []
    for line in this_file:
        line = line.rstrip()
        components = line.split()
        this_list.append(components)
    return this_list


def set_maximum_samples(complete_tuple_list, number_of_samples):
    # Establish dict with limited values
    tuple_dict = dict()
    for (path, label) in complete_tuple_list:
        if label in tuple_dict and len(tuple_dict[label]) < number_of_samples:
            tuple_dict[label].append(path)
        elif label not in tuple_dict:
            tuple_dict[label] = [path]
    # Transform dict to list
    tuple_list = list()
    for key in tuple_dict.keys():
        for value in tuple_dict[key]:
            tuple_list.append((value, key))
    return tuple_list


def split_known_unknown_sets(complete_tuple_list, known_set_size=0.5):
    label_set = {label for (path, label) in complete_tuple_list}
    # Sample individuals to compose known_set
    known_set = set(random.sample(label_set, int(known_set_size * len(label_set))))
    unknown_set = label_set - known_set
    # Select remaining individuals to compose unknown_set
    known_tuple = [(path, label) for (path, label) in complete_tuple_list if label in known_set]
    unknown_tuple = [(path, label) for (path, label) in complete_tuple_list if label in unknown_set]
    return known_tuple, unknown_tuple


def split_train_test_sets(complete_tuple_list, train_set_size=0.5):
    label_set = {label for (path, label) in complete_tuple_list}
    train_tuple = list()
    test_tuple = list()
    for label in label_set:
        # Sample images to compose train_set
        path_set = {path for (path, target) in complete_tuple_list if label == target}
        train_set = set(random.sample(path_set, int(train_set_size * len(path_set))))
        test_set = path_set - train_set
        # Put together labels and paths
        train_tuple.extend([(path, label) for path in train_set])
        test_tuple.extend([(path, label) for path in test_set])
    return train_tuple, test_tuple


def load_images(path, image_list, display=False):
    for name in image_list:
        image = cv.imread(path + '/' + name, cv.IMREAD_COLOR)
        if display:
            cv.imshow('img', image)
            cv.waitKey(20)
    if display:
        cv.destroyAllWindows()


def augment_gallery_set(image_sample):
    image_samples = []
    image_samples.append(image_sample)
    image_samples.append(cv.flip(image_sample, 1)) # vertical-axis flip
    rows,cols,dep = image_sample.shape
    for angle in range(-5,6,10):
        rot_matrix = cv.getRotationMatrix2D((rows/2, cols/2), angle, 1.1)
        image_samples.append(cv.warpAffine(image_sample, rot_matrix,(cols, rows)))
    return image_samples


def sliding_window(image, window_size, step_size):
    for y in xrange(0, image.shape[0], step_size):
        for x in xrange(0, image.shape[1], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def generate_pos_neg_dict(labels):
    to_shuffle = [item for item in labels]
    np.random.shuffle(to_shuffle)
    neg_set = map(lambda neg: (neg, -1), to_shuffle[0:(len(labels) / 2)])
    pos_set = map(lambda pos: (pos, +1), to_shuffle[(len(labels) / 2):len(labels)])
    full_set = neg_set + pos_set
    full_dict = dict((key, val) for key, val in full_set)
    return full_dict


def generate_oaa_splits(targets):
    oaa_splits = list()
    unique_labels = {target for target in targets}
    for label in unique_labels:
        indices = [idx for (idx,value) in enumerate(targets) if value == label]
        binary_y = np.ones(len(targets)) * (-1)
        binary_y[indices] = +1
        oaa_splits.append((label, binary_y))
    return oaa_splits


def split_into_chunks(full_list, num_chunks):
    split_list = []
    chunk_size = int(len(full_list) / num_chunks) + 1
    for index in range(0, len(full_list), chunk_size):
        split_list.append(full_list[index:index+chunk_size])
    return split_list

    
def learn_plsh_model(matrix_x, matrix_y, split):
    classifier = PLSClassifier()
    boolean_label = [split[key] for key in matrix_y]
    model = classifier.fit(np.array(matrix_x), np.array(boolean_label))
    return (model, split)


def learn_oaa_pls(matrix_x, split):
    classifier = PLSClassifier()
    label, boolean_labels = split
    model = classifier.fit(np.array(matrix_x), np.array(boolean_labels))
    return (model, label)


def learn_svmh_model(matrix_x, matrix_y, split):
    # classifier = SVR(C=1.0,kernel='linear')
    # boolean_label = [split[key] for key in matrix_y]
    # model = classifier.fit(np.array(matrix_x), np.array(boolean_label))
    # return (model, split)

    boolean_label = [split[key] for key in matrix_y]
    classifier = cv.ml.SVM_create()
    classifier.setKernel(cv.ml.SVM_LINEAR)
    classifier.setType(cv.ml.SVM_C_SVC)
    classifier.setC(1.0)
    classifier.setGamma(4.5)
    classifier.train(np.float32(matrix_x), cv.ml.ROW_SAMPLE, np.array(boolean_label))
    return (classifier, split)


def generate_det_curve(y_label_list, y_score_list):
    """
    DET curves typically feature missed detection rate on the Y axis, and false positive rate on the X axis. 
    This means that the bottom left corner of the plot is the ideal point - a false positive rate of zero, and a missed detection rate of zero as well. 
    This is not very realistic, but it does mean that a smaller area under the curve (AUC) is usually better.
    """
    # Prepare input data
    label_list = []
    score_list = []
    for line in y_label_list:
        temp_list = [item[1] for item in line]
        label_list.append(temp_list)
    for line in y_score_list:
        temp_list = [item[1] for item in line]
        score_list.append(temp_list)
    label_array = np.array(label_list)
    score_array = np.array(score_list)

    # Compute micro-average DET curve and DET area
    det = dict()
    det['fpr'], det['fnr'], det['thresh'] = detection_error_tradeoff(label_array.ravel(), score_array.ravel())
    det['auc']  = auc(det['fpr'], det['fnr'])
    return det


def generate_probe_histogram(individuals, values, extra_name):
    plt.clf()
    plt.bar(range(len(individuals)), values)
    if sample_name == result[0][0]:
        plt.savefig('plots/' + extra_name + '_' + str(NUM_HASH) + '_' + str(counter) + '_' + sample_name + '_' + result[0][0])
    else:
        plt.savefig('plots/' + extra_name + '_' + str(NUM_HASH) + '_' + str(counter) + '_' + sample_name + '_' + result[0][0] + '_ERROR')


def generate_precision_recall(y_label_list, y_score_list):
    """
    A system with high recall but low precision returns many resaucsults, but most of its predicted labels are incorrect when compared to the training labels. 
    A system with high precision but low recall is just the opposite, returning very few results, but most of its predicted labels are correct when compared to the training labels. 
    An ideal system with high precision and high recall will return many results, with all results labeled correctly.
    """
    # Prepare input data
    label_list = []
    score_list = []
    for line in y_label_list:
        temp_list = [item[1] for item in line]
        label_list.append(temp_list)
    for line in y_score_list:
        temp_list = [item[1] for item in line]
        score_list.append(temp_list)
    label_array = np.array(label_list)
    score_array = np.array(score_list)

    average_precision = dict()
    precision = dict()
    recall = dict()
    thresh = dict()

    # Compute micro-average ROC curve and ROC area
    pr = dict()
    pr['precision'], pr['recall'], pr['thresh'] = precision_recall_curve(label_array.ravel(), score_array.ravel())
    pr['avg_precision'] = average_precision_score(label_array, score_array, average="micro")
    return pr


def compute_fscore(pr):
    precision = list(pr['precision'])
    recall = list(pr['recall'])
    del precision[-1]
    del recall[-1]
    fscores = [2 * (pre * rec) / (pre + rec) for (pre,rec) in zip(precision, recall)]
    complete_zip = zip(pr['thresh'], fscores, pr['precision'], pr['recall'])
    complete_zip.sort(key=lambda tup: tup[1], reverse=True)
    return complete_zip[0]


def mean_results(zipped_list):
    threshs = [item[0] for item in zipped_list]
    fscores = [item[1] for item in zipped_list]
    precics = [item[2] for item in zipped_list]
    recalls = [item[3] for item in zipped_list]
    results = [('Thresh', np.mean(threshs), np.std(threshs)),('FSCore', np.mean(fscores), np.std(fscores)),('Precic', np.mean(precics), np.std(precics)),('Recall', np.mean(recalls), np.std(recalls))]
    return results


def generate_roc_curve(y_label_list, y_score_list):
    """
    ROC curves typically feature true positive rate on the Y axis, and false positive rate on the X axis. 
    This means that the top left corner of the plot is the ideal point - a false positive rate of zero, and a true positive rate of one. 
    This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better.
    """
    # Prepare input data
    label_list = []
    score_list = []
    for line in y_label_list:
        temp_list = [item[1] for item in line]
        label_list.append(temp_list)
    for line in y_score_list:
        temp_list = [item[1] for item in line]
        score_list.append(temp_list)
    label_array = np.array(label_list)
    score_array = np.array(score_list)

    # Compute micro-average ROC curve and ROC area
    roc = dict()
    roc['fpr'], roc['tpr'], roc['thresh'] = roc_curve(label_array.ravel(), score_array.ravel())
    roc['auc']  = auc(roc['fpr'], roc['tpr'])
    return roc


def plot_cmc_curve(os_scores, oaa_scores, extra_name=None):
    """
    The CMC shows how often the biometric subject template appears in the ranks (1, 5, 10, 100, etc.), based on the match rate.
    It is a method of showing measured accuracy performance of a biometric system operating in the closed-set identification task. 
    Templates are compared and ranked based on their similarity.
    """

    # Compute mean values
    os_mean = np.mean(os_scores, axis=0)
    oaa_mean = np.mean(oaa_scores, axis=0)
    x_axis = range(len(os_mean))
    os_auc = auc(x_axis, os_mean)
    ooa_auc = auc(x_axis, oaa_mean)
    
    # Plot Cumulative Matching Characteristic curve
    plt.clf()
    plt.plot(x_axis, os_mean, color='blue', linestyle='--', label='Open-set HPLS (%0.3f)' % (os_auc / len(os_scores[0])))
    plt.plot(x_axis, oaa_mean, color='red', linestyle='-', label='Closed-set OAA-PLS (%0.3f)' % (ooa_auc / len(os_scores[0])))
    plt.xlim([0, len(os_scores[0])])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Rank')
    plt.ylabel('Accuracy Rate')
    plt.title('Cumulative Matching Characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    if extra_name == None:
        plt.show()
    else:
        plt.savefig('./plots/CMC_' + extra_name + '.pdf')


def plot_det_curve(dets, extra_name=None):
    # Setup plot details
    color_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    color_names = [name for name, color in color_dict.items()]
    colors = cycle(color_names)
    lw = 2

    # Plot Receiver Operating Characteristic curve
    plt.clf()
    aucs = []
    for index, color in zip(range(len(dets)), colors):
        det = dets[index]
        plt.plot(det['fpr'], det['fnr'], color=color, lw=lw, label='DET curve %d (area = %0.3f)' % (index+1, det['auc']))
        aucs.append(det['auc'])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.xscale('linear') # linear, log, logit, symlog 
    plt.yscale('linear')
    plt.title('Detection Error Trade-off')
    plt.legend(loc="upper right")
    plt.grid()
    if extra_name == None:
        plt.show()
    else:
        plt.savefig('./plots/DET_' + extra_name + '.pdf')


def plot_precision_recall(prs, extra_name=None):
    # Setup plot details
    color_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    color_names = [name for name, color in color_dict.items()]
    colors = cycle(color_names)
    lw = 2

    # Plot Precision-Recall curve
    plt.clf()
    avgs = []
    for index, color in zip(range(len(prs)), colors):
        pr = prs[index]
        plt.plot(pr['recall'], pr['precision'], lw=lw, color=color, label='PR curve %d (area = %0.3f)' % (index+1, pr['avg_precision']))
        avgs.append(pr['avg_precision'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve (%0.3f - %0.3f)' % (np.mean(avgs),np.std(avgs)))
    plt.legend(loc="lower left")
    plt.grid()
    if extra_name == None:
        plt.show()
    else:
        plt.savefig('./plots/PR_' + extra_name + '.pdf')


def plot_roc_curve(rocs, extra_name=None):
    # Setup plot details
    color_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    color_names = [name for name, color in color_dict.items()]
    colors = cycle(color_names)
    lw = 2

    # Plot Receiver Operating Characteristic curve
    plt.clf()
    aucs = []
    for index, color in zip(range(len(rocs)), colors):
        roc = rocs[index]
        plt.plot(roc['fpr'], roc['tpr'], color=color, lw=lw, label='ROC curve %d (area = %0.3f)' % (index+1, roc['auc']))
        aucs.append(roc['auc'])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (%0.3f - %0.3f)' % (np.mean(aucs),np.std(aucs)))
    plt.legend(loc="lower right")
    plt.grid()
    if extra_name == None:
        plt.show()
    else:
        plt.savefig('./plots/ROC_' + extra_name + '.pdf')
